"""
Operational Decision Evaluator for Phase 14.3.

Reuses the existing RiskEvaluator, DecisionPolicyEngine, RuleEngine, risk-score normalization,
and risk tiers to compute real-world routing and operational decision metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics
from ml.models.config import TARGET_COLUMN
from ml.risk_engine.config import DecisionAction, DecisionPolicyConfig, PolicyMode, RiskTier
from ml.risk_engine.evaluator import RiskEvaluator
from ml.risk_engine.rules import RuleEngine


class OperationalDecisionEvaluator:
    """
    Evaluator that executes the full Phase 6/7 hybrid decision pipeline
    (ML Model + DecisionPolicyEngine + RuleEngine) on transaction feature partitions.
    """

    def evaluate_operational_decisions(
        self,
        df: pd.DataFrame,
        model_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
        operating_threshold: float,
        metadata_path: Optional[Union[str, Path]] = None,
        policy_mode: PolicyMode = PolicyMode.TRI_TIER,
        rule_engine: Optional[RuleEngine] = None,
    ) -> OperationalDecisionMetrics:
        """
        Evaluate a transaction dataset DataFrame through the full RiskEvaluator pipeline.

        Args:
            df: Input DataFrame containing 55 predictive features and optional ground truth 'is_fraud'.
            model_path: Path to serialized model artifact.
            preprocessor_path: Path to serialized preprocessor artifact.
            operating_threshold: Frozen candidate operating threshold tau*.
            metadata_path: Optional path to manifest/metadata JSON.
            policy_mode: Policy mode (TRI_TIER or BINARY_AUTO).
            rule_engine: Optional RuleEngine instance for business rule overrides.

        Returns:
            OperationalDecisionMetrics capturing routing counts, percentages, and queue purities.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pd.DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("Input DataFrame cannot be empty for operational evaluation.")
        if not (0.0 < operating_threshold < 1.0):
            raise ValueError(f"Operating threshold {operating_threshold} out of valid bounds (0, 1).")

        # Configure DecisionPolicyConfig using existing platform semantics
        if policy_mode == PolicyMode.BINARY_AUTO:
            policy_cfg = DecisionPolicyConfig(
                policy_mode=PolicyMode.BINARY_AUTO,
                review_threshold=operating_threshold,
                block_threshold=operating_threshold,
            )
        else:
            # Tri-tier mode: block_threshold is set to the calibrated operating threshold tau*
            # review_threshold defaults to 0.35 (or min(0.35, operating_threshold - 0.05))
            rev_t = min(0.35, round(operating_threshold - 0.05, 4)) if operating_threshold > 0.35 else round(operating_threshold / 2.0, 4)
            policy_cfg = DecisionPolicyConfig(
                policy_mode=PolicyMode.TRI_TIER,
                review_threshold=rev_t,
                block_threshold=operating_threshold,
            )

        evaluator = RiskEvaluator(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metadata_path=metadata_path,
            policy_config=policy_cfg,
            rule_engine=rule_engine,
        )

        decision_results = evaluator.evaluate_dataframe(df)
        total = len(decision_results)

        # Action counts & percentages
        action_counts = {a.value: 0 for a in DecisionAction}
        for r in decision_results:
            action_counts[r.action.value] += 1
        action_percentages = {
            k: round((v / total) * 100.0, 4) for k, v in action_counts.items()
        }

        # Risk tier counts & percentages
        tier_counts = {t.value: 0 for t in RiskTier}
        for r in decision_results:
            tier_counts[r.risk_tier.value] += 1
        tier_percentages = {
            k: round((v / total) * 100.0, 4) for k, v in tier_counts.items()
        }

        # Score distributions
        model_scores = [r.model_score for r in decision_results]
        risk_scores = [r.risk_score for r in decision_results]

        model_score_stats = {
            "mean": round(float(np.mean(model_scores)), 6),
            "min": round(float(np.min(model_scores)), 6),
            "max": round(float(np.max(model_scores)), 6),
        }
        risk_score_stats = {
            "mean": round(float(np.mean(risk_scores)), 4),
            "min": float(np.min(risk_scores)),
            "max": float(np.max(risk_scores)),
        }

        # Rule overrides & trigger counts
        overridden_count = sum(1 for r in decision_results if r.is_overridden)
        rule_trigger_counts: Dict[str, int] = {}
        if rule_engine is not None:
            rule_trigger_counts = {r.rule_id: 0 for r in rule_engine.rules}
        for r in decision_results:
            for r_id in r.rules_triggered:
                rule_trigger_counts[r_id] = rule_trigger_counts.get(r_id, 0) + 1

        # Cross-reference with ground truth if TARGET_COLUMN is available
        fraud_in_block = 0
        fraud_in_review = 0
        fraud_in_approve = 0
        legit_in_block = 0
        legit_in_review = 0
        legit_in_approve = 0

        has_target = TARGET_COLUMN in df.columns
        if has_target:
            y_arr = df[TARGET_COLUMN].to_numpy()
            for res, y_val in zip(decision_results, y_arr):
                is_positive = bool(y_val == 1)
                action = res.action

                if is_positive:
                    if action == DecisionAction.BLOCK:
                        fraud_in_block += 1
                    elif action == DecisionAction.REVIEW:
                        fraud_in_review += 1
                    else:
                        fraud_in_approve += 1
                else:
                    if action == DecisionAction.BLOCK:
                        legit_in_block += 1
                    elif action == DecisionAction.REVIEW:
                        legit_in_review += 1
                    else:
                        legit_in_approve += 1

        total_review = fraud_in_review + legit_in_review
        total_block = fraud_in_block + legit_in_block

        review_queue_purity = (
            round(float(fraud_in_review / total_review), 5) if total_review > 0 else None
        )
        block_precision = (
            round(float(fraud_in_block / total_block), 5) if total_block > 0 else None
        )

        return OperationalDecisionMetrics(
            action_counts=action_counts,
            action_percentages=action_percentages,
            risk_tier_counts=tier_counts,
            risk_tier_percentages=tier_percentages,
            model_score_stats=model_score_stats,
            risk_score_stats=risk_score_stats,
            policy_mode=policy_cfg.policy_mode.value,
            operating_threshold=round(operating_threshold, 4),
            fraud_in_block_count=fraud_in_block,
            fraud_in_review_count=fraud_in_review,
            fraud_in_approve_count=fraud_in_approve,
            legit_in_block_count=legit_in_block,
            legit_in_review_count=legit_in_review,
            legit_in_approve_count=legit_in_approve,
            review_queue_purity=review_queue_purity,
            block_precision=block_precision,
            rule_overridden_count=overridden_count,
            rule_trigger_counts=rule_trigger_counts,
        )
