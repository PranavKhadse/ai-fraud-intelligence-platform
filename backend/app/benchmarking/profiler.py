"""
Seven-Stage Pipeline Profiler for Phase 10 Benchmarking.

Provides high-resolution micro-benchmarking across all seven lifecycle stages of the
real-time fraud detection pipeline without modifying production code paths or duplicating logic.

Stages:
1. Request Validation (Pydantic v2 payload parsing & validation)
2. Feature Preparation (Predictive feature extraction & preprocessor transformation)
3. ML Inference (XGBoost champion probability estimation)
4. TreeSHAP Attribution (Native TreeSHAP marginal contributions & waterfall)
5. Rule Engine / Policy (Deterministic rules, reason code synthesis & policy resolution)
6. Persistence Mapping (In-memory mapping to PersistRiskEvaluationCommand)
7. PostgreSQL Persistence (Atomic transaction boundary persistence via FraudPersistenceUnitOfWork)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import pandas as pd

from backend.app.benchmarking.metrics import PipelineStage
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    TransactionPredictRequest,
)
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.risk_persistence_mapper import (
    RiskEvaluationContext,
    RiskPersistenceMapper,
)
from backend.app.services.risk_service import RiskService, get_risk_service
from ml.explainability.reason_codes import ReasonCodeGenerator
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
from ml.risk_engine.evaluator import RiskEvaluator

logger = logging.getLogger("fraud_api.profiler")


@dataclass(frozen=True)
class ProfileResult:
    """
    Detailed latency profiling breakdown and output payload for a single transaction.
    """
    transaction_id: Optional[str]
    stage_latencies_ms: Dict[PipelineStage, float]
    total_profiled_latency_ms: float
    model_score: float
    risk_score: int
    action: str
    is_overridden: bool
    is_persisted: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile result to a JSON-serializable dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "stage_latencies_ms": {
                stage.value: lat for stage, lat in self.stage_latencies_ms.items()
            },
            "total_profiled_latency_ms": self.total_profiled_latency_ms,
            "model_score": self.model_score,
            "risk_score": self.risk_score,
            "action": self.action,
            "is_overridden": self.is_overridden,
            "is_persisted": self.is_persisted,
        }


class PipelineProfiler:
    """
    Component-level micro-profiler measuring individual stage latencies in isolation.
    """

    def __init__(
        self,
        risk_service: Optional[RiskService] = None,
        persistence_service: Optional[FraudPersistenceService] = None,
    ) -> None:
        """
        Initialize profiler with existing RiskService and optional FraudPersistenceService.
        """
        self.risk_service = risk_service or get_risk_service()
        self.evaluator: RiskEvaluator = self.risk_service.evaluator
        self.persistence_service = persistence_service

    def profile_transaction(
        self,
        payload: Union[Dict[str, Any], TransactionPredictRequest],
    ) -> ProfileResult:
        """
        Synchronously profile in-memory stages 1 through 6 for a single transaction payload.

        Args:
            payload: Raw dictionary or validated TransactionPredictRequest.

        Returns:
            ProfileResult with high-resolution latency measurements per stage.
        """
        stage_timings: Dict[PipelineStage, float] = {}

        # ----------------------------------------------------------------------
        # Stage 1: Request Validation (Pydantic parsing)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        if isinstance(payload, TransactionPredictRequest):
            req = payload
            raw_dict = req.model_dump()
        else:
            raw_dict = dict(payload)
            req = TransactionPredictRequest.model_validate(raw_dict)
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.REQUEST_VALIDATION] = max(0.0, (t1 - t0) / 1_000_000.0)

        # ----------------------------------------------------------------------
        # Stage 2: Feature Preparation & Preprocessor Transformation
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        df_row = pd.DataFrame([raw_dict])
        X = self.evaluator._extract_and_validate_features(df_row)
        X_trans = self.evaluator.preprocessor.transform(X)
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.FEATURE_PREPARATION] = max(0.0, (t1 - t0) / 1_000_000.0)

        # ----------------------------------------------------------------------
        # Stage 3: ML Inference (XGBoost champion model predict_proba)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        probabilities = self.evaluator.model.predict_proba(X_trans)
        if hasattr(probabilities, "ndim") and probabilities.ndim == 2:
            raw_prob = float(probabilities[0, 1])
        elif hasattr(probabilities, "ndim") and probabilities.ndim == 1:
            raw_prob = float(probabilities[0])
        elif isinstance(probabilities, (list, tuple)):
            raw_prob = float(probabilities[0])
        else:
            raw_prob = float(probabilities)
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.ML_INFERENCE] = max(0.0, (t1 - t0) / 1_000_000.0)

        # ----------------------------------------------------------------------
        # Stage 4: Native TreeSHAP Attribution & Margin Waterfall
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = self.evaluator.explainer.explain_features(
            raw_features=raw_dict,
            preprocessed_row=X_trans,
            top_k=5,
            top_mitigating=3,
        )
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.TREESHAP_EXPLAINABILITY] = max(0.0, (t1 - t0) / 1_000_000.0)

        # ----------------------------------------------------------------------
        # Stage 5: Rule Engine, Reason Code Synthesis & Decision Policy
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        # 5a. Evaluate deterministic rules
        rule_matches: Tuple[Any, ...] = ()
        if self.evaluator._rule_engine is not None:
            rule_matches = self.evaluator._rule_engine.evaluate(raw_dict)

        # 5b. Baseline decision policy
        decision = self.evaluator.evaluate_transaction(df_row)

        # 5c. Reason code synthesis
        reason_codes = ReasonCodeGenerator.generate_reason_codes(
            top_risk_factors=top_risk,
            rule_matches=rule_matches,
            is_overridden=decision.is_overridden,
            rule_action=decision.rule_action,
            max_reasons=5,
        )
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.RULE_ENGINE_POLICY] = max(0.0, (t1 - t0) / 1_000_000.0)

        # ----------------------------------------------------------------------
        # Stage 6: Persistence Command Mapping
        # ----------------------------------------------------------------------
        t0 = time.perf_counter_ns()
        # Build PredictionResponse representation
        pred_response = PredictionResponse(
            transaction_id=req.transaction_id,
            model_score=round(float(decision.model_score), 6),
            risk_score=int(decision.risk_score),
            risk_tier=decision.risk_tier.value,
            decision_action=decision.action.value,
            policy_mode=decision.policy_mode.value,
            reason="Profiled evaluation",
            is_overridden=decision.is_overridden,
            rule_action=decision.rule_action.value if decision.rule_action else None,
            rules_triggered=list(decision.rules_triggered),
            top_risk_factors=[
                FeatureAttributionResponse(
                    feature_name=f.feature_name,
                    display_name=f.display_name,
                    raw_value=f.raw_value,
                    shap_value=f.shap_value,
                    direction=f.direction.value if hasattr(f.direction, "value") else str(f.direction),
                    relative_contribution_pct=f.relative_contribution_pct,
                    rank=f.rank,
                )
                for f in top_risk
            ],
            top_mitigating_factors=[
                FeatureAttributionResponse(
                    feature_name=f.feature_name,
                    display_name=f.display_name,
                    raw_value=f.raw_value,
                    shap_value=f.shap_value,
                    direction=f.direction.value if hasattr(f.direction, "value") else str(f.direction),
                    relative_contribution_pct=f.relative_contribution_pct,
                    rank=f.rank,
                )
                for f in top_mitigating
            ],
            reason_codes=[
                ReasonCodeResponse(
                    code=r.code,
                    headline=r.headline,
                    description=r.description,
                    category=r.category,
                    source=r.source.value if hasattr(r.source, "value") else str(r.source),
                    severity=r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                    rank=r.rank,
                )
                for r in reason_codes
            ],
            rule_matches=[
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
                for m in rule_matches
            ],
            model_version=self.evaluator.model_version,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

        context = RiskEvaluationContext(
            actor_id="profiler",
            evaluation_latency_ms=sum(stage_timings.values()),
            external_transaction_id=req.transaction_id,
            account_id=req.account_id,
            merchant_id=req.merchant_id,
            transaction_timestamp=req.timestamp,
        )

        command = RiskPersistenceMapper.map_prediction_to_command(
            request=req,
            response=pred_response,
            context=context,
        )
        t1 = time.perf_counter_ns()
        stage_timings[PipelineStage.PERSISTENCE_MAPPING] = max(0.0, (t1 - t0) / 1_000_000.0)

        total_lat = round(sum(stage_timings.values()), 4)
        rounded_timings = {k: round(v, 4) for k, v in stage_timings.items()}

        return ProfileResult(
            transaction_id=req.transaction_id,
            stage_latencies_ms=rounded_timings,
            total_profiled_latency_ms=total_lat,
            model_score=decision.model_score,
            risk_score=decision.risk_score,
            action=decision.action.value,
            is_overridden=decision.is_overridden,
            is_persisted=False,
        )

    async def profile_transaction_async(
        self,
        payload: Union[Dict[str, Any], TransactionPredictRequest],
        skip_persistence: bool = False,
    ) -> ProfileResult:
        """
        Asynchronously profile all seven stages including PostgreSQL persistence.
        """
        # Execute in-memory stages 1 through 6
        result = self.profile_transaction(payload)
        stage_timings = dict(result.stage_latencies_ms)
        is_persisted = False

        # Stage 7: PostgreSQL Persistence
        if not skip_persistence and self.persistence_service is not None:
            # Reconstruct command
            req = payload if isinstance(payload, TransactionPredictRequest) else TransactionPredictRequest.model_validate(payload)
            pred_response = PredictionResponse(
                transaction_id=req.transaction_id,
                model_score=result.model_score,
                risk_score=result.risk_score,
                risk_tier="HIGH" if result.risk_score >= 60 else "LOW",
                decision_action=result.action,
                policy_mode="TRI_TIER",
                reason="Async profiled evaluation",
                is_overridden=result.is_overridden,
                rules_triggered=[],
                top_risk_factors=[],
                top_mitigating_factors=[],
                reason_codes=[],
                rule_matches=[],
                model_version=self.evaluator.model_version,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
            )
            context = RiskEvaluationContext(
                actor_id="profiler_async",
                evaluation_latency_ms=result.total_profiled_latency_ms,
                external_transaction_id=req.transaction_id,
                account_id=req.account_id,
                merchant_id=req.merchant_id,
                transaction_timestamp=req.timestamp,
            )
            command = RiskPersistenceMapper.map_prediction_to_command(
                request=req,
                response=pred_response,
                context=context,
            )

            t0 = time.perf_counter_ns()
            await self.persistence_service.persist_evaluation(command)
            t1 = time.perf_counter_ns()
            stage_timings[PipelineStage.POSTGRES_PERSISTENCE] = max(0.0, (t1 - t0) / 1_000_000.0)
            is_persisted = True

        total_lat = round(sum(stage_timings.values()), 4)
        rounded_timings = {k: round(v, 4) for k, v in stage_timings.items()}

        return ProfileResult(
            transaction_id=result.transaction_id,
            stage_latencies_ms=rounded_timings,
            total_profiled_latency_ms=total_lat,
            model_score=result.model_score,
            risk_score=result.risk_score,
            action=result.action,
            is_overridden=result.is_overridden,
            is_persisted=is_persisted,
        )
