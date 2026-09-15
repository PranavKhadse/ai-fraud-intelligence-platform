"""
Risk Service Layer for FastAPI Fraud Detection & Risk Intelligence Platform.

Encapsulates the frozen RiskEvaluator, DecisionPolicyEngine, RuleEngine, and
TreeSHAP explainability modules, providing dependency injection hooks and structured
prediction mapping.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

from ml.risk_engine.evaluator import RiskEvaluator
from ml.risk_engine.rules import RuleEngine
from ml.risk_engine.catalog import get_standard_rule_catalog
from ml.risk_engine.config import DecisionPolicyConfig, PolicyMode
from backend.app.core.config import settings
from backend.app.schemas.predict import (
    TransactionPredictRequest,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    FeatureAttributionResponse,
)
from backend.app.schemas.health import HealthResponse

logger = logging.getLogger("fraud_api.service")


class RiskService:
    """
    Singleton service managing the lifecycle and execution of the production RiskEvaluator.
    """

    def __init__(
        self,
        model_path=settings.MODEL_PATH,
        preprocessor_path=settings.PREPROCESSOR_PATH,
        metadata_path=settings.METADATA_PATH,
        policy_mode: str = settings.DEFAULT_POLICY_MODE,
        review_threshold: float = settings.DEFAULT_REVIEW_THRESHOLD,
        block_threshold: float = settings.DEFAULT_BLOCK_THRESHOLD,
    ) -> None:
        """
        Initialize the RiskEvaluator with frozen artifacts and the standard 6-rule catalog.
        """
        logger.info(f"Initializing RiskEvaluator with model at '{model_path}'...")
        policy_config = DecisionPolicyConfig(
            policy_mode=PolicyMode(policy_mode),
            review_threshold=review_threshold,
            block_threshold=block_threshold,
        )
        rule_engine = RuleEngine(rules=get_standard_rule_catalog())

        self.evaluator = RiskEvaluator(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metadata_path=metadata_path,
            policy_config=policy_config,
            rule_engine=rule_engine,
        )
        self.rules_count = len(rule_engine)
        self.model_version = self.evaluator.model_version
        logger.info(
            f"RiskEvaluator initialized successfully (Model Version: {self.model_version}, Rules: {self.rules_count})."
        )

    def predict_transaction(
        self,
        payload: TransactionPredictRequest,
        top_k: int = settings.DEFAULT_TOP_K_RISK_FACTORS,
        top_mitigating: int = settings.DEFAULT_TOP_K_MITIGATING_FACTORS,
        max_reasons: int = settings.DEFAULT_MAX_REASON_CODES,
    ) -> PredictionResponse:
        """
        Evaluate a single transaction request and generate an audit-ready prediction
        with TreeSHAP feature attributions and deterministic rule explanations.

        Args:
            payload: Validated TransactionPredictRequest.
            top_k: Number of positive TreeSHAP risk factors.
            top_mitigating: Number of negative TreeSHAP mitigating factors.
            max_reasons: Maximum plain-English reason codes.

        Returns:
            PredictionResponse: Complete structured risk evaluation response.
        """
        raw_dict = payload.model_dump()
        tx_id = payload.transaction_id

        # Execute evaluation and TreeSHAP explainability in unified pass
        explanation = self.evaluator.explain_transaction(
            df_or_series=raw_dict,
            top_k=top_k,
            top_mitigating=top_mitigating,
            max_reasons=max_reasons,
        )

        # Build policy explanation reason string
        if explanation.is_overridden:
            rules_str = ", ".join(explanation.rules_triggered)
            decision_reason = (
                f"Baseline ML action overridden to {explanation.action.value} by rule(s): {rules_str}."
            )
        else:
            decision_reason = (
                f"Policy mode {explanation.policy_mode.value} evaluated action {explanation.action.value} "
                f"with normalized risk score {explanation.risk_score}/100."
            )

        # Map rule matches
        rule_matches: List[RuleMatchResponse] = [
            RuleMatchResponse(
                rule_id=m.rule_id,
                description=m.description,
                feature_name=m.feature_name,
                operator=m.operator.value if hasattr(m.operator, "value") else str(m.operator),
                comparison_value=m.comparison_value,
                outcome=m.outcome.value if hasattr(m.outcome, "value") else str(m.outcome),
                rule_type=m.rule_type.value if hasattr(m.rule_type, "value") else str(m.rule_type),
                priority=m.priority,
            )
            for m in explanation.rule_matches
        ]

        # Map reason codes
        reason_codes: List[ReasonCodeResponse] = [
            ReasonCodeResponse(
                code=r.code,
                headline=r.headline,
                description=r.description,
                category=r.category,
                source=r.source.value if hasattr(r.source, "value") else str(r.source),
                severity=r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                rank=r.rank,
            )
            for r in explanation.reason_codes
        ]

        # Map top risk factors
        top_risk_factors: List[FeatureAttributionResponse] = [
            FeatureAttributionResponse(
                feature_name=f.feature_name,
                display_name=f.display_name,
                raw_value=f.raw_value,
                shap_value=f.shap_value,
                direction=f.direction.value if hasattr(f.direction, "value") else str(f.direction),
                relative_contribution_pct=f.relative_contribution_pct,
                rank=f.rank,
            )
            for f in explanation.top_risk_factors
        ]

        # Map top mitigating factors
        top_mitigating_factors: List[FeatureAttributionResponse] = [
            FeatureAttributionResponse(
                feature_name=f.feature_name,
                display_name=f.display_name,
                raw_value=f.raw_value,
                shap_value=f.shap_value,
                direction=f.direction.value if hasattr(f.direction, "value") else str(f.direction),
                relative_contribution_pct=f.relative_contribution_pct,
                rank=f.rank,
            )
            for f in explanation.top_mitigating_factors
        ]

        evaluated_at = datetime.now(timezone.utc).isoformat()

        return PredictionResponse(
            transaction_id=tx_id,
            model_score=round(float(explanation.model_score), 6),
            risk_score=int(explanation.risk_score),
            risk_tier=explanation.risk_tier.value,
            decision_action=explanation.action.value,
            policy_mode=explanation.policy_mode.value,
            reason=decision_reason,
            is_overridden=explanation.is_overridden,
            rule_action=explanation.rule_action.value if explanation.rule_action else None,
            rules_triggered=list(explanation.rules_triggered),
            rule_matches=rule_matches,
            reason_codes=reason_codes,
            top_risk_factors=top_risk_factors,
            top_mitigating_factors=top_mitigating_factors,
            model_version=explanation.model_version,
            evaluated_at=evaluated_at,
        )

    def get_health_status(self) -> HealthResponse:
        """
        Produce health check telemetry indicating model readiness and rule count.
        """
        return HealthResponse(
            status="healthy",
            app_name=settings.APP_NAME,
            version=settings.VERSION,
            model_loaded=self.evaluator.model is not None,
            model_version=self.model_version,
            rules_loaded_count=self.rules_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# Global singleton instance holder
_risk_service_instance: Optional[RiskService] = None


def get_risk_service() -> RiskService:
    """
    FastAPI dependency provider for the RiskService singleton.
    """
    global _risk_service_instance
    if _risk_service_instance is None:
        _risk_service_instance = RiskService()
    return _risk_service_instance


def set_risk_service(service: Optional[RiskService]) -> None:
    """
    Set or reset the global RiskService singleton (useful in lifespan or testing fixtures).
    """
    global _risk_service_instance
    _risk_service_instance = service
