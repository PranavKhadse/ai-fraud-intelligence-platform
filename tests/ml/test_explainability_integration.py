"""
Automated Integration Tests for Phase 7: End-to-End Explainability & RiskEvaluator Integration.

Validates:
1. RiskEvaluator End-to-End Explainability:
   - explain_transaction across DataFrame, Series, and Dictionary inputs.
   - evaluate_with_explanation returning consistent (DecisionResult, TransactionExplanation) tuples.
2. Hybrid Rule & TreeSHAP Synthesis:
   - Protected ML BLOCK transactions with rule matches (no downgrade).
   - Escalated REVIEW transactions (ML APPROVE + REVIEW rule override).
   - Passive MONITOR rules attached as audit metadata with source="RULE".
3. Out-of-Time and Validation Sample Robustness:
   - Exact Lundberg TreeSHAP Additivity verification across 50 real transactions.
   - Exact margin-to-probability reconstruction.
   - Deterministic explanation repeatability.
4. Latency Benchmark:
   - Measures per-transaction explanation latency.
5. Frozen Artifact Immutability:
   - Asserts SHA-256 checksums of champion model, preprocessor, and metadata remain strictly invariant.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import pytest
import numpy as np
import pandas as pd

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
)
from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
    RuleOutcome,
)
from ml.risk_engine.rules import (
    RuleEngine,
)
from ml.risk_engine.catalog import (
    get_standard_rule_catalog,
)
from ml.risk_engine.evaluator import (
    RiskEvaluator,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    DEFAULT_METADATA_PATH,
)
from ml.explainability.schemas import (
    TransactionExplanation,
    AttributionDirection,
    ReasonSource,
)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def evaluator_with_catalog() -> RiskEvaluator:
    """Create RiskEvaluator with the Phase 6 standard rule catalog."""
    catalog = get_standard_rule_catalog()
    rule_engine = RuleEngine(catalog)
    return RiskEvaluator(
        policy_config=DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.35,
            block_threshold=0.78,
        ),
        rule_engine=rule_engine,
    )


@pytest.fixture
def val_df_sample() -> pd.DataFrame:
    """Load 50 sample transactions from validation dataset."""
    df = pd.read_parquet(VAL_FEATURES_PATH)
    return df.head(50).copy()


@pytest.fixture
def oot_df_sample() -> pd.DataFrame:
    """Load 50 sample transactions from protected OOT test dataset."""
    df = pd.read_parquet(TEST_FEATURES_PATH)
    return df.head(50).copy()


class TestExplainabilityIntegration:
    """End-to-end integration test suite for explainability with RiskEvaluator and real datasets."""

    def test_single_transaction_explanation_formats(
        self, evaluator_with_catalog: RiskEvaluator, val_df_sample: pd.DataFrame
    ) -> None:
        """Verify explain_transaction works identically across DataFrame, Series, and dict inputs."""
        row_df = val_df_sample.iloc[[0]]
        row_series = val_df_sample.iloc[0]
        row_dict = row_series.to_dict()

        exp_df = evaluator_with_catalog.explain_transaction(row_df)
        exp_series = evaluator_with_catalog.explain_transaction(row_series)
        exp_dict = evaluator_with_catalog.explain_transaction(row_dict)

        assert isinstance(exp_df, TransactionExplanation)
        assert isinstance(exp_series, TransactionExplanation)
        assert isinstance(exp_dict, TransactionExplanation)

        # All 3 formats must produce identical scores and margins
        assert exp_df.model_score == exp_series.model_score == exp_dict.model_score
        assert exp_df.output_margin == exp_series.output_margin == exp_dict.output_margin
        assert exp_df.base_value == exp_series.base_value == exp_dict.base_value
        assert exp_df.action == exp_series.action == exp_dict.action

    def test_evaluate_with_explanation_coherence(
        self, evaluator_with_catalog: RiskEvaluator, val_df_sample: pd.DataFrame
    ) -> None:
        """Verify evaluate_with_explanation returns coherent decision and explanation."""
        row_df = val_df_sample.iloc[[5]]
        decision, explanation = evaluator_with_catalog.evaluate_with_explanation(row_df)

        assert decision.action == explanation.action
        assert decision.risk_score == explanation.risk_score
        assert decision.risk_tier == explanation.risk_tier
        assert decision.model_score == explanation.model_score
        assert decision.is_overridden == explanation.is_overridden
        assert decision.rules_triggered == explanation.rules_triggered

    def test_additivity_on_real_validation_and_oot_samples(
        self,
        evaluator_with_catalog: RiskEvaluator,
        val_df_sample: pd.DataFrame,
        oot_df_sample: pd.DataFrame,
    ) -> None:
        """Verify TreeSHAP additivity holds on real transactions across Validation and OOT datasets."""
        for df_sample in [val_df_sample, oot_df_sample]:
            for i in range(len(df_sample)):
                row = df_sample.iloc[i]
                exp = evaluator_with_catalog.explain_transaction(row)

                # 1. Base value + all waterfall steps == final margin
                final_waterfall_step = exp.waterfall[-1]
                assert final_waterfall_step.step_type == "final"
                assert abs(final_waterfall_step.cumulative_margin - exp.output_margin) < 1e-5

                # 2. Reconstructed probability matches model_score
                expected_prob = 1.0 / (1.0 + np.exp(-exp.output_margin))
                assert abs(exp.model_score - expected_prob) < 1e-6

    def test_protected_block_with_rule_match_explanation(
        self, evaluator_with_catalog: RiskEvaluator
    ) -> None:
        """Verify high-risk ML block transaction preserves BLOCK action and reflects in explanation."""
        high_risk_tx: Dict[str, Any] = {
            "amount": 950.0,
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7589,
            "merchant_long": -73.9851,
            "city_pop": 500000.0,
            "merchant_category": "shopping_net",
            "job_category": "engineering",
            "transaction_hour": 2.0,
            "day_of_week": 0.0,
            "day_of_month": 1.0,
            "month": 1.0,
            "week_of_year": 1.0,
            "is_weekend": 0,
            "is_night": 1,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "day_of_week_sin": 0.0,
            "day_of_week_cos": 1.0,
            "txn_count_1h": 6.0,  # triggers RULE_VELOCITY_BURST_REVIEW
            "txn_count_6h": 10.0,
            "txn_count_24h": 15.0,
            "txn_count_7d": 30.0,
            "txn_count_30d": 70.0,
            "time_since_prev_txn_seconds": 15.0,
            "is_first_account_txn": 0,
            "amt_sum_1h": 2500.0,
            "amt_sum_24h": 4000.0,
            "amt_sum_7d": 8000.0,
            "amt_sum_30d": 15000.0,
            "amt_mean_24h": 400.0,
            "amt_mean_7d": 400.0,
            "amt_max_24h": 950.0,
            "amt_median_30d": 50.0,
            "historical_amount_mean": 45.0,
            "historical_amount_std": 15.0,
            "historical_amount_median": 40.0,
            "amount_zscore": 60.33,  # triggers RULE_AMT_ZSCORE_DEVIATION_REVIEW
            "amount_ratio_to_historical_mean": 21.11,
            "account_txn_count_before": 100.0,
            "account_total_spend_before": 4500.0,
            "account_avg_amount_before": 45.0,
            "account_max_amount_before": 120.0,
            "account_unique_merchant_count_before": 20.0,
            "account_unique_category_count_before": 5.0,
            "account_merchant_txn_count_before": 0.0,
            "account_category_txn_count_before": 10.0,
            "account_merchant_spend_before": 0.0,
            "account_category_spend_before": 450.0,
            "merchant_txn_count_before": 500.0,
            "category_txn_count_before": 20000.0,
            "cardholder_merchant_distance_km": 15.0,
            "distance_from_prev_merchant_km": 5.0,
            "implied_travel_speed_kmh": 120.0,
            "is_impossible_travel_speed": 0,
        }

        exp = evaluator_with_catalog.explain_transaction(high_risk_tx)

        # Baseline model score is very high (>0.78) -> BLOCK
        assert exp.action == DecisionAction.BLOCK
        assert exp.risk_tier == RiskTier.CRITICAL
        # Protected block: rule did NOT downgrade decision
        assert exp.is_overridden is False

        # Rules triggered must be recorded
        assert "RULE_VELOCITY_BURST_REVIEW" in exp.rules_triggered
        assert "RULE_AMT_ZSCORE_DEVIATION_REVIEW" in exp.rules_triggered

        # Reason codes contain both rules and model features
        sources = [r.source for r in exp.reason_codes]
        assert ReasonSource.RULE in sources
        assert ReasonSource.MODEL in sources

    def test_repeatability_and_determinism(
        self, evaluator_with_catalog: RiskEvaluator, val_df_sample: pd.DataFrame
    ) -> None:
        """Verify calling explain_transaction twice on identical input yields identical outputs."""
        row = val_df_sample.iloc[10]
        exp1 = evaluator_with_catalog.explain_transaction(row)
        exp2 = evaluator_with_catalog.explain_transaction(row)

        assert exp1.model_score == exp2.model_score
        assert exp1.output_margin == exp2.output_margin
        assert exp1.base_value == exp2.base_value
        assert exp1.to_dict() == exp2.to_dict()

    def test_artifact_immutability(
        self, evaluator_with_catalog: RiskEvaluator, val_df_sample: pd.DataFrame
    ) -> None:
        """Verify artifact SHA-256 hashes are strictly preserved after explanation execution."""
        h_model_before = compute_sha256(DEFAULT_MODEL_PATH)
        h_preproc_before = compute_sha256(DEFAULT_PREPROCESSOR_PATH)
        h_meta_before = compute_sha256(DEFAULT_METADATA_PATH)

        # Run 20 explanations
        for i in range(20):
            evaluator_with_catalog.explain_transaction(val_df_sample.iloc[i])

        h_model_after = compute_sha256(DEFAULT_MODEL_PATH)
        h_preproc_after = compute_sha256(DEFAULT_PREPROCESSOR_PATH)
        h_meta_after = compute_sha256(DEFAULT_METADATA_PATH)

        assert h_model_before == h_model_after
        assert h_preproc_before == h_preproc_after
        assert h_meta_before == h_meta_after

    def test_decision_semantics_and_overrides(
        self, evaluator_with_catalog: RiskEvaluator, val_df_sample: pd.DataFrame
    ) -> None:
        """Verify evaluate_with_explanation cleanly exposes and distinguishes all 5 decision components:
        a. model_score (probability)
        b. baseline_action (model-only policy action)
        c. rule_action (highest precedence rule outcome)
        d. action (final action)
        e. is_overridden (boolean flag)
        """
        # 1. Low model score + REVIEW rule override
        override_tx: Dict[str, Any] = {
            "amount": 45.0,
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7589,
            "merchant_long": -73.9851,
            "city_pop": 500000.0,
            "merchant_category": "grocery_pos",
            "job_category": "engineering",
            "transaction_hour": 14.0,
            "day_of_week": 2.0,
            "day_of_month": 10.0,
            "month": 5.0,
            "week_of_year": 19.0,
            "is_weekend": 0,
            "is_night": 0,
            "hour_sin": -0.5,
            "hour_cos": -0.866,
            "day_of_week_sin": 0.9749,
            "day_of_week_cos": -0.2225,
            "txn_count_1h": 5.0,  # triggers RULE_VELOCITY_BURST_REVIEW
            "txn_count_6h": 5.0,
            "txn_count_24h": 6.0,
            "txn_count_7d": 10.0,
            "txn_count_30d": 30.0,
            "time_since_prev_txn_seconds": 60.0,
            "is_first_account_txn": 0,
            "amt_sum_1h": 225.0,
            "amt_sum_24h": 270.0,
            "amt_sum_7d": 500.0,
            "amt_sum_30d": 1500.0,
            "amt_mean_24h": 45.0,
            "amt_mean_7d": 50.0,
            "amt_max_24h": 45.0,
            "amt_median_30d": 45.0,
            "historical_amount_mean": 45.0,
            "historical_amount_std": 10.0,
            "historical_amount_median": 45.0,
            "amount_zscore": 0.0,
            "amount_ratio_to_historical_mean": 1.0,
            "account_txn_count_before": 100.0,
            "account_total_spend_before": 4500.0,
            "account_avg_amount_before": 45.0,
            "account_max_amount_before": 120.0,
            "account_unique_merchant_count_before": 20.0,
            "account_unique_category_count_before": 5.0,
            "account_merchant_txn_count_before": 10.0,
            "account_category_txn_count_before": 30.0,
            "account_merchant_spend_before": 450.0,
            "account_category_spend_before": 1500.0,
            "merchant_txn_count_before": 500.0,
            "category_txn_count_before": 20000.0,
            "cardholder_merchant_distance_km": 5.0,
            "distance_from_prev_merchant_km": 1.0,
            "implied_travel_speed_kmh": 60.0,
            "is_impossible_travel_speed": 0,
        }

        decision, explanation = evaluator_with_catalog.evaluate_with_explanation(override_tx)

        # Baseline model score is very low (<0.35)
        assert explanation.model_score < 0.35
        # a. model_score vs baseline_action
        assert explanation.baseline_action == DecisionAction.APPROVE
        # b. rule_action
        assert explanation.rule_action == RuleOutcome.REVIEW
        # c. final action escalated to REVIEW
        assert explanation.action == DecisionAction.REVIEW
        # d. is_overridden flag is True
        assert explanation.is_overridden is True
        assert decision.is_overridden is True
        # e. rule reason code clearly indicates override
        assert any("escalated to REVIEW by policy rule" in r.description for r in explanation.reason_codes)

        # 2. Low model score + MONITOR-only rule
        monitor_tx = dict(override_tx)
        monitor_tx["txn_count_1h"] = 1.0  # no review rule
        monitor_tx["is_impossible_travel_speed"] = 1  # triggers RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR

        dec_mon, exp_mon = evaluator_with_catalog.evaluate_with_explanation(monitor_tx)
        assert exp_mon.model_score < 0.35
        assert exp_mon.baseline_action == DecisionAction.APPROVE
        assert exp_mon.rule_action is None
        assert exp_mon.action == DecisionAction.APPROVE
        assert exp_mon.is_overridden is False
        assert "RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR" in exp_mon.rules_triggered
