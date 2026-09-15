"""
Risk Persistence Mapper for FastAPI Fraud Detection & Risk Intelligence Platform.

Converts live prediction requests, response payloads, explanation dataclasses, and
optional HTTP context metadata into validated, strongly-typed `PersistRiskEvaluationCommand`
instances for the asynchronous persistence layer.

Design Principles:
- Zero Database Access: Pure in-memory transformation.
- Strict Type Safety: Converts floats to Decimals, parses timestamps, and maps domain enums safely.
- Explicit Validation: Validates numerical finiteness, bounds, required identifiers, and enum memberships.
- Deterministic: Produces identical command outputs for identical inputs.
- No Fabricated Business Facts: Rejects missing required domain identifiers rather than inventing defaults.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from backend.app.db.models.enums import (
    AttributionDirection,
    AuditActorType,
    DecisionAction,
    PolicyMode,
    ReasonSeverity,
    ReasonSource,
    RiskTier,
    RuleOutcome,
    RuleType,
)
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    TransactionPredictRequest,
)
from backend.app.services.persistence_service import (
    AuditLogData,
    FeatureAttributionData,
    PersistRiskEvaluationCommand,
    ReasonCodeData,
    RiskEvaluationData,
    RuleMatchData,
    TransactionData,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS


# ==============================================================================
# Context Dataclass
# ==============================================================================

@dataclass(frozen=True)
class RiskEvaluationContext:
    """
    Optional request lifecycle context metadata passed into the persistence mapper.
    """
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None
    actor_id: Optional[str] = "fastapi_predict_api"
    actor_type: Union[AuditActorType, str] = AuditActorType.SYSTEM
    evaluation_latency_ms: Optional[Union[Decimal, float, str]] = None
    external_transaction_id: Optional[str] = None
    account_id: Optional[str] = None
    merchant_id: Optional[str] = None
    currency: Optional[str] = "USD"
    transaction_timestamp: Optional[Union[datetime, str]] = None


# ==============================================================================
# Validation and Helper Functions
# ==============================================================================

def _ensure_finite_number(val: Any, field_name: str) -> float:
    """Validate that a numeric value is finite and not NaN or Inf."""
    if val is None:
        raise ValueError(f"Numeric field '{field_name}' must not be None.")
    try:
        f_val = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Field '{field_name}' must be a valid number, got {val!r}") from e
    if not math.isfinite(f_val):
        raise ValueError(f"Field '{field_name}' must be finite, got {val!r}")
    return f_val


def _parse_iso_or_datetime(val: Union[datetime, str], field_name: str = "timestamp") -> datetime:
    """
    Parse an ISO formatted string or timezone-aware/naive datetime into a UTC datetime.

    Raises:
        ValueError: If val is None, empty string, or cannot be parsed.
    """
    if val is None:
        raise ValueError(f"Datetime field '{field_name}' must not be None.")
    if isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            raise ValueError(f"Datetime field '{field_name}' must not be empty.")
        try:
            dt = datetime.fromisoformat(val_clean)
        except Exception as e:
            raise ValueError(f"Invalid datetime format in '{field_name}': {val!r}") from e
    elif isinstance(val, datetime):
        dt = val
    else:
        raise ValueError(f"Field '{field_name}' must be a datetime or ISO string, got {type(val).__name__}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt



def _to_decimal(
    val: Union[Decimal, float, int, str, None],
    field_name: str,
    precision: Optional[int] = None,
) -> Optional[Decimal]:
    """Safely convert numeric values to Decimal with finiteness and precision validation."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        if not val.is_finite():
            raise ValueError(f"Decimal field '{field_name}' must be finite, got {val}")
        return val
    f_val = _ensure_finite_number(val, field_name)
    if precision is not None:
        return Decimal(str(round(f_val, precision)))
    return Decimal(str(val))


def _to_enum(enum_cls: Any, val: Any, field_name: str) -> Any:
    """Convert a value to an Enum instance, raising descriptive ValueError on mismatch."""
    if val is None:
        raise ValueError(f"Enum field '{field_name}' must not be None.")
    if isinstance(val, enum_cls):
        return val
    if isinstance(val, str):
        val_clean = val.strip()
        try:
            return enum_cls(val_clean)
        except ValueError:
            valid_vals = [e.value for e in enum_cls]
            raise ValueError(
                f"Invalid value '{val}' for enum {enum_cls.__name__} in '{field_name}'. "
                f"Allowed values: {valid_vals}"
            )
    raise TypeError(
        f"Field '{field_name}' must be a {enum_cls.__name__} or string, got {type(val).__name__}"
    )


def _validate_currency(currency: Optional[str]) -> str:
    """Validate ISO 4217 3-letter currency code."""
    if currency is None or not currency.strip():
        return "USD"
    curr = currency.strip().upper()
    if len(curr) != 3 or not curr.isalpha():
        raise ValueError(f"Invalid currency code '{currency}'. Must be a 3-letter ISO 4217 code (e.g. 'USD').")
    return curr


def _extract_raw_dict(
    request: Union[TransactionPredictRequest, Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract a dictionary representation from request payload."""
    if isinstance(request, dict):
        return dict(request)
    if hasattr(request, "model_dump"):
        return request.model_dump()
    raise TypeError(
        f"Request must be a TransactionPredictRequest or dict, got {type(request).__name__}"
    )


def _extract_features_snapshot(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate a complete JSONB-serializable snapshot containing
    all 55 canonical predictive features in exact deterministic specification.
    """
    missing: List[str] = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in raw_dict]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required predictive feature column(s) for feature snapshot: {missing[:5]}"
        )

    snapshot: Dict[str, Any] = {}
    for col in PREDICTIVE_FEATURE_COLUMNS:
        val = raw_dict[col]
        if isinstance(val, (int, bool)):
            snapshot[col] = val
        elif isinstance(val, float):
            if not math.isfinite(val):
                raise ValueError(f"Predictive feature '{col}' contains non-finite value: {val!r}")
            snapshot[col] = val
        elif isinstance(val, Decimal):
            if not val.is_finite():
                raise ValueError(f"Predictive feature '{col}' contains non-finite Decimal: {val!r}")
            snapshot[col] = float(val)
        elif isinstance(val, str):
            snapshot[col] = val
        else:
            # Cast primitive scalars
            try:
                f_val = float(val)
                if not math.isfinite(f_val):
                    raise ValueError(f"Predictive feature '{col}' contains non-finite value: {val!r}")
                snapshot[col] = f_val
            except (TypeError, ValueError):
                snapshot[col] = str(val)

    return snapshot


# ==============================================================================
# Main Persistence Mapper
# ==============================================================================

class RiskPersistenceMapper:
    """
    Hardened adapter service converting prediction inputs, outputs, and context into
    strongly-typed `PersistRiskEvaluationCommand` instances.
    """

    @classmethod
    def map_prediction_to_command(
        cls,
        request: Union[TransactionPredictRequest, Dict[str, Any]],
        response: PredictionResponse,
        context: Optional[RiskEvaluationContext] = None,
    ) -> PersistRiskEvaluationCommand:
        """
        Map a `TransactionPredictRequest` and `PredictionResponse` into a `PersistRiskEvaluationCommand`.

        Args:
            request: Validated transaction request schema or feature dictionary.
            response: Completed PredictionResponse returned by RiskService.
            context: Optional HTTP request/audit context metadata.

        Returns:
            PersistRiskEvaluationCommand ready for FraudPersistenceService.

        Raises:
            ValueError: If input validation fails on bounds, required identifiers, or finiteness.
        """
        if request is None:
            raise ValueError("Request payload must not be None.")
        if response is None:
            raise ValueError("PredictionResponse must not be None.")

        ctx = context or RiskEvaluationContext()
        raw_dict = _extract_raw_dict(request)

        # 1. Map TransactionData
        tx_data = cls._map_transaction_data(raw_dict, response, ctx)

        # 2. Map RiskEvaluationData
        eval_data = cls._map_risk_evaluation_data_from_response(response, ctx)

        # 3. Map RuleMatchData list
        rule_matches = cls._map_rule_matches_from_response(response.rule_matches)

        # 4. Map ReasonCodeData list
        reason_codes = cls._map_reason_codes_from_response(response.reason_codes)

        # 5. Map FeatureAttributionData list
        feature_attributions = cls._map_attributions_from_response(
            top_risk=response.top_risk_factors,
            top_mitigating=response.top_mitigating_factors,
        )

        # 6. Map AuditLogData
        audit_data = cls._map_audit_log_data(response, tx_data, ctx)

        return PersistRiskEvaluationCommand(
            transaction=tx_data,
            evaluation=eval_data,
            rule_matches=rule_matches,
            reason_codes=reason_codes,
            feature_attributions=feature_attributions,
            audit=audit_data,
        )

    @classmethod
    def map_explanation_to_command(
        cls,
        request: Union[TransactionPredictRequest, Dict[str, Any]],
        explanation: Any,
        context: Optional[RiskEvaluationContext] = None,
    ) -> PersistRiskEvaluationCommand:
        """
        Map a `TransactionPredictRequest` and `TransactionExplanation` dataclass
        into a `PersistRiskEvaluationCommand`.

        Args:
            request: Validated transaction request schema or feature dictionary.
            explanation: Completed TransactionExplanation dataclass from RiskEvaluator.
            context: Optional HTTP request/audit context metadata.

        Returns:
            PersistRiskEvaluationCommand ready for FraudPersistenceService.

        Raises:
            ValueError: If input validation fails on bounds, required identifiers, or finiteness.
        """
        if request is None:
            raise ValueError("Request payload must not be None.")
        if explanation is None:
            raise ValueError("TransactionExplanation must not be None.")

        ctx = context or RiskEvaluationContext()
        raw_dict = _extract_raw_dict(request)

        # 1. Map TransactionData
        tx_data = cls._map_transaction_data(raw_dict, explanation, ctx)

        # 2. Map RiskEvaluationData from explanation
        eval_data = cls._map_risk_evaluation_data_from_explanation(explanation, ctx)

        # 3. Map RuleMatchData list
        rule_matches = cls._map_rule_matches_from_domain(explanation.rule_matches)

        # 4. Map ReasonCodeData list
        reason_codes = cls._map_reason_codes_from_domain(explanation.reason_codes)

        # 5. Map FeatureAttributionData list
        feature_attributions = cls._map_attributions_from_domain(
            top_risk=explanation.top_risk_factors,
            top_mitigating=explanation.top_mitigating_factors,
        )

        # 6. Map AuditLogData
        audit_data = cls._map_audit_log_data_from_explanation(explanation, tx_data, ctx)

        return PersistRiskEvaluationCommand(
            transaction=tx_data,
            evaluation=eval_data,
            rule_matches=rule_matches,
            reason_codes=reason_codes,
            feature_attributions=feature_attributions,
            audit=audit_data,
        )

    # --------------------------------------------------------------------------
    # Sub-component Mappers
    # --------------------------------------------------------------------------

    @classmethod
    def _map_transaction_data(
        cls,
        raw_dict: Dict[str, Any],
        source: Any,
        ctx: RiskEvaluationContext,
    ) -> TransactionData:
        """Map transaction fields into TransactionData DTO with strict boundary validation."""
        ext_tx_id = (
            ctx.external_transaction_id
            or raw_dict.get("transaction_id")
            or getattr(source, "transaction_id", None)
        )

        # Account ID resolution: must be provided via request or context
        raw_account_id = ctx.account_id or raw_dict.get("account_id")
        if raw_account_id is None or not str(raw_account_id).strip():
            raise ValueError(
                "Missing required 'account_id'. An account identifier must be provided in request payload or context."
            )
        account_id = str(raw_account_id).strip()

        # Merchant ID resolution
        raw_merchant_id = ctx.merchant_id or raw_dict.get("merchant_id")
        merchant_id = str(raw_merchant_id).strip() if raw_merchant_id is not None else None

        # Currency resolution
        currency = _validate_currency(ctx.currency or raw_dict.get("currency"))

        # Amount validation
        amount_val = _ensure_finite_number(raw_dict.get("amount"), "amount")
        if amount_val < 0.0:
            raise ValueError(f"Transaction amount must be non-negative (>= 0.0), got {amount_val}")
        amount = Decimal(str(round(amount_val, 2)))

        # Coordinate validations
        cardholder_lat = _ensure_finite_number(raw_dict.get("cardholder_lat"), "cardholder_lat")
        if not (-90.0 <= cardholder_lat <= 90.0):
            raise ValueError(f"cardholder_lat must be in [-90.0, 90.0], got {cardholder_lat}")

        cardholder_long = _ensure_finite_number(raw_dict.get("cardholder_long"), "cardholder_long")
        if not (-180.0 <= cardholder_long <= 180.0):
            raise ValueError(f"cardholder_long must be in [-180.0, 180.0], got {cardholder_long}")

        merchant_lat = _ensure_finite_number(raw_dict.get("merchant_lat"), "merchant_lat")
        if not (-90.0 <= merchant_lat <= 90.0):
            raise ValueError(f"merchant_lat must be in [-90.0, 90.0], got {merchant_lat}")

        merchant_long = _ensure_finite_number(raw_dict.get("merchant_long"), "merchant_long")
        if not (-180.0 <= merchant_long <= 180.0):
            raise ValueError(f"merchant_long must be in [-180.0, 180.0], got {merchant_long}")

        # City pop validation
        city_pop_val = _ensure_finite_number(raw_dict.get("city_pop", 0), "city_pop")
        if city_pop_val < 0:
            raise ValueError(f"city_pop must be non-negative (>= 0), got {city_pop_val}")
        city_pop = int(city_pop_val)

        # Categoricals
        merchant_cat = str(raw_dict.get("merchant_category", "")).strip()
        if not merchant_cat:
            raise ValueError("merchant_category must be a non-empty string.")

        job_cat = str(raw_dict.get("job_category", "")).strip()
        if not job_cat:
            raise ValueError("job_category must be a non-empty string.")

        # Timestamp resolution
        # Clearly distinguish:
        # 1. Original transaction timestamp (from payload or context)
        # 2. Ingestion-time fallback (UTC now) used only when the client payload omits an event timestamp
        #    to fulfill the database schema `NOT NULL` constraint on `transactions.transaction_timestamp`.
        raw_ts = ctx.transaction_timestamp or raw_dict.get("timestamp")
        if raw_ts is not None and (not isinstance(raw_ts, str) or raw_ts.strip()):
            ts = _parse_iso_or_datetime(raw_ts, "transaction_timestamp")
        else:
            ts = datetime.now(timezone.utc)

        # Snapshot of 55 predictive features
        features_snapshot = _extract_features_snapshot(raw_dict)

        return TransactionData(
            external_transaction_id=str(ext_tx_id).strip() if ext_tx_id is not None and str(ext_tx_id).strip() else None,
            account_id=account_id,
            merchant_id=merchant_id,
            merchant_category=merchant_cat,
            job_category=job_cat,
            amount=amount,
            currency=currency,
            cardholder_lat=Decimal(str(round(cardholder_lat, 6))),
            cardholder_long=Decimal(str(round(cardholder_long, 6))),
            merchant_lat=Decimal(str(round(merchant_lat, 6))),
            merchant_long=Decimal(str(round(merchant_long, 6))),
            city_pop=city_pop,
            transaction_timestamp=ts,
            features_snapshot=features_snapshot,
        )

    @classmethod
    def _map_risk_evaluation_data_from_response(
        cls,
        response: PredictionResponse,
        ctx: RiskEvaluationContext,
    ) -> RiskEvaluationData:
        """Map PredictionResponse into RiskEvaluationData DTO with bounds validation."""
        policy_mode = _to_enum(PolicyMode, response.policy_mode, "policy_mode")
        risk_tier = _to_enum(RiskTier, response.risk_tier, "risk_tier")
        decision_action = _to_enum(DecisionAction, response.decision_action, "decision_action")

        rule_action: Optional[RuleOutcome] = None
        if response.rule_action is not None:
            rule_action = _to_enum(RuleOutcome, response.rule_action, "rule_action")

        # Score validation
        model_score_val = _ensure_finite_number(response.model_score, "model_score")
        if not (0.0 <= model_score_val <= 1.0):
            raise ValueError(f"model_score must be in [0.0, 1.0], got {model_score_val}")
        model_score = Decimal(str(round(model_score_val, 6)))

        risk_score = int(response.risk_score)
        if not (0 <= risk_score <= 100):
            raise ValueError(f"risk_score must be an integer in [0, 100], got {risk_score}")

        model_ver = response.model_version
        if model_ver is None or not str(model_ver).strip():
            raise ValueError("model_version is required for risk evaluation persistence provenance.")

        baseline_action = decision_action if not response.is_overridden else None
        eval_ts = (
            _parse_iso_or_datetime(response.evaluated_at, "evaluated_at")
            if response.evaluated_at
            else datetime.now(timezone.utc)
        )

        return RiskEvaluationData(
            model_version=str(model_ver).strip(),
            policy_mode=policy_mode,
            model_score=model_score,
            risk_score=risk_score,
            risk_tier=risk_tier,
            decision_action=decision_action,
            baseline_action=baseline_action,
            is_overridden=bool(response.is_overridden),
            rule_action=rule_action,
            decision_reason=str(response.reason or ""),
            output_margin=None,
            base_value=None,
            evaluation_latency_ms=_to_decimal(ctx.evaluation_latency_ms, "evaluation_latency_ms", 2),
            correlation_id=ctx.correlation_id,
            evaluated_at=eval_ts,
        )

    @classmethod
    def _map_risk_evaluation_data_from_explanation(
        cls,
        explanation: Any,
        ctx: RiskEvaluationContext,
    ) -> RiskEvaluationData:
        """Map TransactionExplanation into RiskEvaluationData DTO with bounds validation."""
        policy_mode = _to_enum(PolicyMode, explanation.policy_mode, "policy_mode")
        risk_tier = _to_enum(RiskTier, explanation.risk_tier, "risk_tier")
        decision_action = _to_enum(DecisionAction, explanation.action, "action")

        baseline_action: Optional[DecisionAction] = None
        if getattr(explanation, "baseline_action", None) is not None:
            baseline_action = _to_enum(DecisionAction, explanation.baseline_action, "baseline_action")
        else:
            baseline_action = decision_action

        rule_action: Optional[RuleOutcome] = None
        if getattr(explanation, "rule_action", None) is not None:
            rule_action = _to_enum(RuleOutcome, explanation.rule_action, "rule_action")

        # Score validation
        model_score_val = _ensure_finite_number(explanation.model_score, "model_score")
        if not (0.0 <= model_score_val <= 1.0):
            raise ValueError(f"model_score must be in [0.0, 1.0], got {model_score_val}")
        model_score = Decimal(str(round(model_score_val, 6)))

        risk_score = int(explanation.risk_score)
        if not (0 <= risk_score <= 100):
            raise ValueError(f"risk_score must be an integer in [0, 100], got {risk_score}")

        model_ver = getattr(explanation, "model_version", None)
        if model_ver is None or not str(model_ver).strip():
            raise ValueError("model_version is required for risk evaluation persistence provenance.")

        output_margin = _to_decimal(getattr(explanation, "output_margin", None), "output_margin", 6)
        base_value = _to_decimal(getattr(explanation, "base_value", None), "base_value", 6)

        # Build standard reason description
        if explanation.is_overridden:
            rules_str = ", ".join(getattr(explanation, "rules_triggered", ()))
            decision_reason = (
                f"Baseline ML action overridden to {decision_action.value} by rule(s): {rules_str}."
            )
        else:
            decision_reason = (
                f"Policy mode {policy_mode.value} evaluated action {decision_action.value} "
                f"with normalized risk score {explanation.risk_score}/100."
            )

        return RiskEvaluationData(
            model_version=str(model_ver).strip(),
            policy_mode=policy_mode,
            model_score=model_score,
            risk_score=risk_score,
            risk_tier=risk_tier,
            decision_action=decision_action,
            baseline_action=baseline_action,
            is_overridden=bool(explanation.is_overridden),
            rule_action=rule_action,
            decision_reason=decision_reason,
            output_margin=output_margin,
            base_value=base_value,
            evaluation_latency_ms=_to_decimal(ctx.evaluation_latency_ms, "evaluation_latency_ms", 2),
            correlation_id=ctx.correlation_id,
            evaluated_at=datetime.now(timezone.utc),
        )

    @classmethod
    def _map_rule_matches_from_response(
        cls, rule_matches: Sequence[RuleMatchResponse]
    ) -> List[RuleMatchData]:
        """Map RuleMatchResponse list to RuleMatchData list."""
        result: List[RuleMatchData] = []
        for rm in rule_matches:
            outcome = _to_enum(RuleOutcome, rm.outcome, f"rule_matches[{rm.rule_id}].outcome")
            rule_type = _to_enum(RuleType, rm.rule_type, f"rule_matches[{rm.rule_id}].rule_type")
            priority = int(rm.priority)
            if priority < 0:
                raise ValueError(f"Rule priority must be >= 0, got {priority}")
            result.append(
                RuleMatchData(
                    rule_id=str(rm.rule_id),
                    description=str(rm.description),
                    feature_name=str(rm.feature_name),
                    operator=str(rm.operator),
                    comparison_value=str(rm.comparison_value),
                    outcome=outcome,
                    rule_type=rule_type,
                    priority=priority,
                )
            )
        return result

    @classmethod
    def _map_rule_matches_from_domain(
        cls, rule_matches: Sequence[Any]
    ) -> List[RuleMatchData]:
        """Map domain RuleMatch list to RuleMatchData list."""
        result: List[RuleMatchData] = []
        for rm in rule_matches:
            op_val = rm.operator.value if hasattr(rm.operator, "value") else str(rm.operator)
            outcome = _to_enum(RuleOutcome, rm.outcome, f"rule_matches[{rm.rule_id}].outcome")
            rule_type = _to_enum(RuleType, rm.rule_type, f"rule_matches[{rm.rule_id}].rule_type")
            priority = int(rm.priority)
            if priority < 0:
                raise ValueError(f"Rule priority must be >= 0, got {priority}")
            result.append(
                RuleMatchData(
                    rule_id=str(rm.rule_id),
                    description=str(rm.description),
                    feature_name=str(rm.feature_name),
                    operator=op_val,
                    comparison_value=str(rm.comparison_value),
                    outcome=outcome,
                    rule_type=rule_type,
                    priority=priority,
                )
            )
        return result

    @classmethod
    def _map_reason_codes_from_response(
        cls, reason_codes: Sequence[ReasonCodeResponse]
    ) -> List[ReasonCodeData]:
        """Map ReasonCodeResponse list to ReasonCodeData list."""
        result: List[ReasonCodeData] = []
        for rc in reason_codes:
            source = _to_enum(ReasonSource, rc.source, f"reason_codes[{rc.code}].source")
            severity = _to_enum(ReasonSeverity, rc.severity, f"reason_codes[{rc.code}].severity")
            rank = int(rc.rank)
            if rank < 1:
                raise ValueError(f"Reason code rank must be >= 1, got {rank}")
            result.append(
                ReasonCodeData(
                    code=str(rc.code),
                    headline=str(rc.headline),
                    description=str(rc.description),
                    category=str(rc.category),
                    source=source,
                    severity=severity,
                    rank=rank,
                )
            )
        return result

    @classmethod
    def _map_reason_codes_from_domain(
        cls, reason_codes: Sequence[Any]
    ) -> List[ReasonCodeData]:
        """Map domain ReasonCodeDetail list to ReasonCodeData list."""
        result: List[ReasonCodeData] = []
        for rc in reason_codes:
            source = _to_enum(ReasonSource, rc.source, f"reason_codes[{rc.code}].source")
            severity = _to_enum(ReasonSeverity, rc.severity, f"reason_codes[{rc.code}].severity")
            rank = int(rc.rank)
            if rank < 1:
                raise ValueError(f"Reason code rank must be >= 1, got {rank}")
            result.append(
                ReasonCodeData(
                    code=str(rc.code),
                    headline=str(rc.headline),
                    description=str(rc.description),
                    category=str(rc.category),
                    source=source,
                    severity=severity,
                    rank=rank,
                )
            )
        return result

    @classmethod
    def _map_attributions_from_response(
        cls,
        top_risk: Sequence[FeatureAttributionResponse],
        top_mitigating: Sequence[FeatureAttributionResponse],
    ) -> List[FeatureAttributionData]:
        """Map top risk and mitigating FeatureAttributionResponse lists to FeatureAttributionData list."""
        result: List[FeatureAttributionData] = []
        combined = list(top_risk) + list(top_mitigating)
        for fa in combined:
            direction = _to_enum(AttributionDirection, fa.direction, f"attribution[{fa.feature_name}].direction")
            shap_val = _ensure_finite_number(fa.shap_value, f"attribution[{fa.feature_name}].shap_value")
            pct_val = _ensure_finite_number(
                fa.relative_contribution_pct, f"attribution[{fa.feature_name}].relative_contribution_pct"
            )
            if not (0.0 <= pct_val <= 100.0):
                raise ValueError(
                    f"relative_contribution_pct must be in [0.0, 100.0], got {pct_val} for {fa.feature_name}"
                )
            rank = int(fa.rank)
            if rank < 1:
                raise ValueError(f"Feature attribution rank must be >= 1, got {rank}")

            result.append(
                FeatureAttributionData(
                    feature_name=str(fa.feature_name),
                    display_name=str(fa.display_name),
                    raw_value=fa.raw_value,
                    shap_value=Decimal(str(round(shap_val, 6))),
                    direction=direction,
                    relative_contribution_pct=Decimal(str(round(pct_val, 4))),
                    rank=rank,
                )
            )
        return result

    @classmethod
    def _map_attributions_from_domain(
        cls,
        top_risk: Sequence[Any],
        top_mitigating: Sequence[Any],
    ) -> List[FeatureAttributionData]:
        """Map domain FeatureAttribution lists to FeatureAttributionData list."""
        result: List[FeatureAttributionData] = []
        combined = list(top_risk) + list(top_mitigating)
        for fa in combined:
            direction = _to_enum(AttributionDirection, fa.direction, f"attribution[{fa.feature_name}].direction")
            shap_val = _ensure_finite_number(fa.shap_value, f"attribution[{fa.feature_name}].shap_value")
            pct_val = _ensure_finite_number(
                fa.relative_contribution_pct, f"attribution[{fa.feature_name}].relative_contribution_pct"
            )
            if not (0.0 <= pct_val <= 100.0):
                raise ValueError(
                    f"relative_contribution_pct must be in [0.0, 100.0], got {pct_val} for {fa.feature_name}"
                )
            rank = int(fa.rank)
            if rank < 1:
                raise ValueError(f"Feature attribution rank must be >= 1, got {rank}")

            result.append(
                FeatureAttributionData(
                    feature_name=str(fa.feature_name),
                    display_name=str(fa.display_name),
                    raw_value=fa.raw_value,
                    shap_value=Decimal(str(round(shap_val, 6))),
                    direction=direction,
                    relative_contribution_pct=Decimal(str(round(pct_val, 4))),
                    rank=rank,
                )
            )
        return result

    @classmethod
    def _map_audit_log_data(
        cls,
        response: PredictionResponse,
        tx_data: TransactionData,
        ctx: RiskEvaluationContext,
    ) -> AuditLogData:
        """Map AuditLogData DTO from PredictionResponse with JSONB payload serializability."""
        actor_type = _to_enum(AuditActorType, ctx.actor_type, "audit.actor_type")
        actor_id = ctx.actor_id or "fastapi_predict_api"

        return AuditLogData(
            event_type="RISK_EVALUATION_PERSISTED",
            action="PERSIST_EVALUATION",
            actor_type=actor_type,
            actor_id=str(actor_id),
            correlation_id=ctx.correlation_id,
            client_ip=ctx.client_ip,
            payload={
                "external_transaction_id": tx_data.external_transaction_id,
                "model_score": float(response.model_score),
                "risk_score": int(response.risk_score),
                "decision_action": str(response.decision_action),
                "is_overridden": bool(response.is_overridden),
            },
            event_timestamp=(
                _parse_iso_or_datetime(response.evaluated_at, "audit.event_timestamp")
                if response.evaluated_at
                else datetime.now(timezone.utc)
            ),
        )

    @classmethod
    def _map_audit_log_data_from_explanation(
        cls,
        explanation: Any,
        tx_data: TransactionData,
        ctx: RiskEvaluationContext,
    ) -> AuditLogData:
        """Map AuditLogData DTO from TransactionExplanation with JSONB payload serializability."""
        actor_type = _to_enum(AuditActorType, ctx.actor_type, "audit.actor_type")
        actor_id = ctx.actor_id or "fastapi_predict_api"
        action_val = (
            explanation.action.value
            if hasattr(explanation.action, "value")
            else str(explanation.action)
        )
        return AuditLogData(
            event_type="RISK_EVALUATION_PERSISTED",
            action="PERSIST_EVALUATION",
            actor_type=actor_type,
            actor_id=str(actor_id),
            correlation_id=ctx.correlation_id,
            client_ip=ctx.client_ip,
            payload={
                "external_transaction_id": tx_data.external_transaction_id,
                "model_score": float(explanation.model_score),
                "risk_score": int(explanation.risk_score),
                "decision_action": action_val,
                "is_overridden": bool(explanation.is_overridden),
            },
            event_timestamp=datetime.now(timezone.utc),
        )


# ==============================================================================
# Module Convenience Functions
# ==============================================================================

def map_prediction_to_command(
    request: Union[TransactionPredictRequest, Dict[str, Any]],
    response: PredictionResponse,
    context: Optional[RiskEvaluationContext] = None,
) -> PersistRiskEvaluationCommand:
    """Convenience functional wrapper around `RiskPersistenceMapper.map_prediction_to_command`."""
    return RiskPersistenceMapper.map_prediction_to_command(request, response, context)


def map_explanation_to_command(
    request: Union[TransactionPredictRequest, Dict[str, Any]],
    explanation: Any,
    context: Optional[RiskEvaluationContext] = None,
) -> PersistRiskEvaluationCommand:
    """Convenience functional wrapper around `RiskPersistenceMapper.map_explanation_to_command`."""
    return RiskPersistenceMapper.map_explanation_to_command(request, explanation, context)


__all__ = [
    "RiskPersistenceMapper",
    "RiskEvaluationContext",
    "map_prediction_to_command",
    "map_explanation_to_command",
]
