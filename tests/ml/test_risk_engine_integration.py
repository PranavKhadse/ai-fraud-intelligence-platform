"""
Automated Integration Tests for Phase 6 Increment 2 & 5B: RiskEvaluator & Hybrid ML + Rule Decision Pipeline.

Validates:
1. RiskEvaluator Initialization: Default and custom paths/configs, RuleEngine attachment, FileNotFoundError on missing paths.
2. Synthetic End-to-End Evaluation: Scalar (DataFrame, Series, Dict) and batch DataFrame scoring.
3. Defensive Input Validation: Missing columns, non-finite values, empty inputs, type errors.
4. Input Immutability: Asserts input DataFrames are not mutated during evaluation.
5. Authoritative Phase 5 Policy Consistency: Validates that RiskEvaluator in BINARY_AUTO mode with
   block_threshold=0.78 on Validation data dynamically matches authoritative Phase 5 confusion matrix.
6. Increment 3 Hardening & Provenance: Metadata tolerance, model version provenance, BatchDecisionSummary.
7. Increment 5B Hybrid Decision Integration:
   - ML APPROVE + BLOCK rule -> final BLOCK (is_overridden=True, rule_action=BLOCK)
   - ML REVIEW + BLOCK rule -> final BLOCK (is_overridden=True, rule_action=BLOCK)
   - ML APPROVE + REVIEW rule -> final REVIEW (is_overridden=True, rule_action=REVIEW)
   - ML REVIEW + REVIEW rule -> final REVIEW (is_overridden=False, rule_action=None)
   - ML BLOCK + REVIEW rule -> final BLOCK (is_overridden=False, rule_action=None, no downgrade)
   - ML BLOCK + BLOCK rule -> final BLOCK (is_overridden=False, rule_action=None)
   - ML action + MONITOR rule -> unchanged action (is_overridden=False, rule_action=None, rule recorded)
   - Multiple matched rules with deterministic precedence hierarchy (BLOCK > REVIEW > MONITOR)
   - Multiple matched rules with deterministic rules_triggered ordering
   - Exact baseline preservation when no rules supplied or empty RuleEngine
   - Single-transaction hybrid evaluation across DataFrame, Series, and dict formats
   - Batch DataFrame hybrid evaluation with accurate overridden_count and rule_trigger_counts
   - Missing rule feature tolerance without breaking ML scoring
   - Score and reason invariance: model_score, risk_score, reason, and reason_codes unchanged after overrides
8. Artifact Immutability: Guarantees zero artifact modification across Phase 4 and Phase 5 artifacts.
"""

import json
import hashlib
from pathlib import Path
from types import MappingProxyType
import pytest
import numpy as np
import pandas as pd

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
)
from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
    DecisionReasonCode,
    RuleOutcome,
    RuleType,
    RuleOperator,
)
from ml.risk_engine.evaluator import (
    RiskEvaluator,
    BatchDecisionSummary,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
)
from ml.risk_engine.rules import (
    RiskRule,
    RuleEngine,
)


# Baseline SHA-256 Hashes for Artifact Immutability Verification
BASELINE_ARTIFACT_HASHES = {
    "ml/models/artifacts/champion_model.joblib": "5598fc3c9267c3fc14efe83cefd4678bb1fd42fef966200fccd69fd37508611d",
    "ml/models/artifacts/champion_preprocessor.joblib": "24f4783cdb0141ff13fb1f1036f6dc5d15c32002f915cb33964489c65594231b",
    "ml/models/artifacts/model_metadata.json": "deb0f0e8e8b5436fb38c84e0d9554de1f6ce8097b3b57af70aea02abefc35c27",
    "ml/models/artifacts/threshold_analysis.json": "dd95d5c0a6e666227c46159d8a2b2cea497f07e245a15556e2ff1afd76ad742b",
    "ml/cost_optimization/artifacts/oot_evaluation_results.json": "9935ad4c1b7d2321b7225e72bcfcca3934d27ca50fb96b8a4816a3b59fe44fce",
    "ml/cost_optimization/artifacts/sensitivity_analysis.json": "1ec5cc6103e9d5c8c1b20da31fa5d233a1f0dce639ebca75d9e0418eb6cc488e",
    "ml/cost_optimization/artifacts/threshold_comparison.json": "f1463107d37d5a949cc1e9e6b2f467fc64a0fa8894822f930bd22aac0d008d6d",
    "ml/cost_optimization/artifacts/threshold_optimization_results.json": "84df4446560f399e066a7b8dd4416d0686e7fc37a20676fcbe87df0a7f8c5686",
}


@pytest.fixture(scope="module")
def evaluator() -> RiskEvaluator:
    """Create a module-scoped RiskEvaluator instance loaded from frozen artifacts without rules."""
    return RiskEvaluator()


@pytest.fixture
def synthetic_feature_row() -> pd.DataFrame:
    """Construct a single synthetic 55-column feature DataFrame with realistic ranges."""
    rng = np.random.default_rng(42)
    data = {}
    for col in PREDICTIVE_FEATURE_COLUMNS:
        if col == "merchant_category":
            data[col] = ["shopping_net"]
        elif col == "job_category":
            data[col] = ["Engineer"]
        else:
            data[col] = [float(rng.uniform(0.1, 100.0))]
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_feature_batch() -> pd.DataFrame:
    """Construct a synthetic 10-row 55-column feature DataFrame."""
    rng = np.random.default_rng(123)
    n = 10
    data = {}
    for col in PREDICTIVE_FEATURE_COLUMNS:
        if col == "merchant_category":
            data[col] = rng.choice(["shopping_net", "grocery_pos", "gas_transport"], size=n)
        elif col == "job_category":
            data[col] = rng.choice(["Engineer", "Doctor", "Teacher"], size=n)
        else:
            data[col] = rng.uniform(0.1, 100.0, size=n).astype(np.float64)
    return pd.DataFrame(data)


# =====================================================================
# 1. RiskEvaluator Initialization & Configuration Tests
# =====================================================================

class TestRiskEvaluatorInit:
    """Test suite for RiskEvaluator setup and path handling."""

    def test_default_initialization(self, evaluator: RiskEvaluator) -> None:
        """Verify default evaluator setup with frozen model, preprocessor, and no rule engine."""
        assert evaluator.model is not None
        assert evaluator.preprocessor is not None
        assert evaluator.config.policy_mode == PolicyMode.TRI_TIER
        assert evaluator.config.review_threshold == 0.35
        assert evaluator.config.block_threshold == 0.78
        assert evaluator.rule_engine is None

    def test_custom_policy_config(self) -> None:
        """Verify evaluator respects custom policy configuration."""
        custom_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.BINARY_AUTO,
            block_threshold=0.65,
        )
        evaluator = RiskEvaluator(policy_config=custom_cfg)
        assert evaluator.config.policy_mode == PolicyMode.BINARY_AUTO
        assert evaluator.config.block_threshold == 0.65

    def test_custom_rule_engine_attachment(self) -> None:
        """Verify evaluator accepts a validated RuleEngine instance."""
        rule = RiskRule(
            rule_id="R_HIGH_AMT",
            description="Block transactions > 5000",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=5000.0,
            outcome=RuleOutcome.BLOCK,
        )
        engine = RuleEngine([rule])
        evaluator = RiskEvaluator(rule_engine=engine)
        assert evaluator.rule_engine is engine
        assert len(evaluator.rule_engine) == 1

    def test_invalid_rule_engine_type_rejection(self) -> None:
        """Verify passing non-RuleEngine to rule_engine raises TypeError."""
        with pytest.raises(TypeError, match="rule_engine must be an instance of RuleEngine"):
            RiskEvaluator(rule_engine="INVALID_ENGINE")  # type: ignore

    def test_missing_model_file_raises_error(self, tmp_path: Path) -> None:
        """Verify non-existent model artifact path raises FileNotFoundError."""
        missing_path = tmp_path / "non_existent_model.joblib"
        with pytest.raises(FileNotFoundError, match="Champion model artifact not found"):
            RiskEvaluator(model_path=missing_path)

    def test_missing_preprocessor_file_raises_error(self, tmp_path: Path) -> None:
        """Verify non-existent preprocessor artifact path raises FileNotFoundError."""
        missing_path = tmp_path / "non_existent_preproc.joblib"
        with pytest.raises(FileNotFoundError, match="Champion preprocessor artifact not found"):
            RiskEvaluator(preprocessor_path=missing_path)


# =====================================================================
# 2. Synthetic End-to-End Evaluation Tests
# =====================================================================

class TestRiskEvaluatorSynthetic:
    """Test suite for end-to-end evaluation using synthetic fixtures."""

    def test_evaluate_single_transaction_dataframe(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """Verify evaluate_transaction on a 1-row DataFrame."""
        result = evaluator.evaluate_transaction(synthetic_feature_row)
        assert isinstance(result.model_score, float)
        assert 0.0 <= result.model_score <= 1.0
        assert isinstance(result.risk_score, int)
        assert 0 <= result.risk_score <= 100
        assert result.risk_score == int(round(100.0 * result.model_score))
        assert isinstance(result.risk_tier, RiskTier)
        assert isinstance(result.action, DecisionAction)
        assert result.policy_mode == PolicyMode.TRI_TIER
        assert result.rules_triggered == ()
        assert result.is_overridden is False
        assert result.rule_action is None

    def test_evaluate_single_transaction_series(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """Verify evaluate_transaction on a pd.Series."""
        series = synthetic_feature_row.iloc[0]
        result = evaluator.evaluate_transaction(series)
        assert isinstance(result.model_score, float)
        assert 0 <= result.risk_score <= 100

    def test_evaluate_single_transaction_dict(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """Verify evaluate_transaction on a dict."""
        record_dict = synthetic_feature_row.iloc[0].to_dict()
        result = evaluator.evaluate_transaction(record_dict)
        assert isinstance(result.model_score, float)
        assert 0 <= result.risk_score <= 100

    def test_evaluate_batch_dataframe(
        self, evaluator: RiskEvaluator, synthetic_feature_batch: pd.DataFrame
    ) -> None:
        """Verify evaluate_dataframe on a multi-row DataFrame preserves order and count."""
        results = evaluator.evaluate_dataframe(synthetic_feature_batch)
        assert len(results) == len(synthetic_feature_batch)
        for res in results:
            assert isinstance(res.model_score, float)
            assert 0.0 <= res.model_score <= 1.0
            assert isinstance(res.risk_score, int)
            assert 0 <= res.risk_score <= 100
            assert isinstance(res.risk_tier, RiskTier)
            assert isinstance(res.action, DecisionAction)
            assert res.rules_triggered == ()
            assert res.is_overridden is False
            assert res.rule_action is None

    def test_input_dataframe_immutability(
        self, evaluator: RiskEvaluator, synthetic_feature_batch: pd.DataFrame
    ) -> None:
        """Verify evaluator does NOT mutate input DataFrame columns, shape, or values."""
        df_copy = synthetic_feature_batch.copy(deep=True)
        _ = evaluator.evaluate_dataframe(synthetic_feature_batch)
        pd.testing.assert_frame_equal(synthetic_feature_batch, df_copy)


# =====================================================================
# 3. Defensive Validation & Error Handling Tests
# =====================================================================

class TestRiskEvaluatorValidation:
    """Test suite for input validation and defensive error handling."""

    def test_invalid_type_to_evaluate_dataframe(self, evaluator: RiskEvaluator) -> None:
        """Verify non-DataFrame input to evaluate_dataframe raises TypeError."""
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            evaluator.evaluate_dataframe([1, 2, 3])  # type: ignore

    def test_empty_dataframe_raises_error(self, evaluator: RiskEvaluator) -> None:
        """Verify empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame(columns=PREDICTIVE_FEATURE_COLUMNS)
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            evaluator.evaluate_dataframe(empty_df)

    def test_missing_column_raises_error(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """Verify DataFrame missing any required predictor column raises ValueError."""
        incomplete_df = synthetic_feature_row.drop(columns=["amount"])
        with pytest.raises(ValueError, match="Input DataFrame is missing"):
            evaluator.evaluate_dataframe(incomplete_df)

    def test_non_finite_numeric_feature_raises_error(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """Verify NaN or infinite numeric values raise ValueError."""
        df_nan = synthetic_feature_row.copy()
        df_nan.loc[0, "amount"] = np.nan
        with pytest.raises(ValueError, match="contains NaN or non-finite values"):
            evaluator.evaluate_dataframe(df_nan)

        df_inf = synthetic_feature_row.copy()
        df_inf.loc[0, "amount"] = np.inf
        with pytest.raises(ValueError, match="contains NaN or non-finite values"):
            evaluator.evaluate_dataframe(df_inf)

    def test_multi_row_df_to_evaluate_transaction_raises_error(
        self, evaluator: RiskEvaluator, synthetic_feature_batch: pd.DataFrame
    ) -> None:
        """Verify passing multi-row DataFrame to evaluate_transaction raises ValueError."""
        with pytest.raises(ValueError, match="expects a 1-row DataFrame"):
            evaluator.evaluate_transaction(synthetic_feature_batch)

    def test_invalid_type_to_evaluate_transaction(self, evaluator: RiskEvaluator) -> None:
        """Verify invalid types to evaluate_transaction raise TypeError."""
        with pytest.raises(TypeError, match="must be a 1-row DataFrame, Series, or dict"):
            evaluator.evaluate_transaction(12345)  # type: ignore


# =====================================================================
# 4. Authoritative Phase 5 Policy Consistency Test
# =====================================================================

class TestPhase5PolicyConsistency:
    """
    Test suite validating consistency with authoritative Phase 5 validation results.
    Loads validation features and threshold comparison JSON artifact in read-only mode.
    NEVER accesses test_features.parquet.
    """

    def test_binary_auto_validation_consistency(self) -> None:
        """
        Evaluate validation partition in BINARY_AUTO mode with block_threshold=0.78
        and verify exact match against authoritative Phase 5 validation confusion matrix.
        """
        auth_path = Path("ml/cost_optimization/artifacts/threshold_comparison.json")
        val_path = Path("data/processed/features/val_features.parquet")

        assert auth_path.exists(), f"Authoritative Phase 5 artifact missing: {auth_path}"
        assert val_path.exists(), f"Validation features file missing: {val_path}"

        # 1. Read Authoritative Phase 5 Results (Read-Only)
        with open(auth_path, "r", encoding="utf-8") as f:
            auth_data = json.load(f)
        auth_cm = auth_data["cost_optimal_threshold"]["confusion_matrix"]
        expected_threshold = auth_data["cost_optimal_threshold"]["threshold"]
        assert expected_threshold == 0.78

        # 2. Load Validation Features (Read-Only)
        df_val = pd.read_parquet(val_path)
        y_val = df_val["is_fraud"].values

        # 3. Instantiate RiskEvaluator in BINARY_AUTO mode at 0.78
        config = DecisionPolicyConfig(
            policy_mode=PolicyMode.BINARY_AUTO,
            block_threshold=0.78,
        )
        evaluator = RiskEvaluator(policy_config=config)

        # 4. Execute Batch Evaluation
        results = evaluator.evaluate_dataframe(df_val)
        assert len(results) == len(df_val)

        # 5. Calculate Confusion Matrix using boolean masks
        is_block = np.array([r.action == DecisionAction.BLOCK for r in results], dtype=bool)
        is_approve = np.array([r.action == DecisionAction.APPROVE for r in results], dtype=bool)

        tp = int(np.sum((y_val == 1) & is_block))
        fp = int(np.sum((y_val == 0) & is_block))
        fn = int(np.sum((y_val == 1) & is_approve))
        tn = int(np.sum((y_val == 0) & is_approve))

        # 6. Verify Dynamic Match with Authoritative Phase 5 Artifact
        assert tp == auth_cm["tp"], f"TP mismatch: calculated {tp} vs authoritative {auth_cm['tp']}"
        assert fp == auth_cm["fp"], f"FP mismatch: calculated {fp} vs authoritative {auth_cm['fp']}"
        assert fn == auth_cm["fn"], f"FN mismatch: calculated {fn} vs authoritative {auth_cm['fn']}"
        assert tn == auth_cm["tn"], f"TN mismatch: calculated {tn} vs authoritative {auth_cm['tn']}"


# =====================================================================
# 5. Increment 3 Hardening & Provenance Tests
# =====================================================================

class TestRiskEngineHardening:
    """
    Test suite for Phase 6 Increment 3: Risk Engine Hardening requirements.
    Validates decision provenance, metadata column tolerance, is_fraud isolation,
    scalar/batch consistency, and BatchDecisionSummary metrics.
    """

    def test_metadata_column_tolerance_and_target_isolation(
        self, evaluator: RiskEvaluator, synthetic_feature_row: pd.DataFrame
    ) -> None:
        """
        Verify RiskEvaluator accepts extra metadata columns (transaction_id, account_id,
        timestamp, is_fraud) on DataFrame, Series, and dict without mutating inputs,
        and verify that changing is_fraud has zero effect on model score or decision.
        """
        # Create row with metadata
        row_with_meta = synthetic_feature_row.copy(deep=True)
        row_with_meta["transaction_id"] = "tx_12345"
        row_with_meta["account_id"] = "acc_98765"
        row_with_meta["timestamp"] = "2026-09-12T12:00:00"
        row_with_meta["is_fraud"] = 0

        # Snapshot for immutability check
        original_copy = row_with_meta.copy(deep=True)

        # 1. Evaluate DataFrame with is_fraud=0
        res0 = evaluator.evaluate_transaction(row_with_meta)
        pd.testing.assert_frame_equal(row_with_meta, original_copy)

        # 2. Evaluate DataFrame with is_fraud=1
        row_with_meta["is_fraud"] = 1
        res1 = evaluator.evaluate_transaction(row_with_meta)

        # 3. Assert target isolation: is_fraud value must NOT influence scores or action
        assert res0.model_score == res1.model_score
        assert res0.risk_score == res1.risk_score
        assert res0.action == res1.action
        assert res0.risk_tier == res1.risk_tier

        # 4. Evaluate Series with metadata
        series_meta = row_with_meta.iloc[0]
        res_series = evaluator.evaluate_transaction(series_meta)
        assert res_series.model_score == res0.model_score

        # 5. Evaluate Dict with metadata
        dict_meta = row_with_meta.iloc[0].to_dict()
        res_dict = evaluator.evaluate_transaction(dict_meta)
        assert res_dict.model_score == res0.model_score

    def test_model_version_provenance(self, evaluator: RiskEvaluator) -> None:
        """Verify model_version is extracted from metadata ('1.0.0') and passed to results."""
        assert evaluator.model_version == "1.0.0"

        synthetic_row = pd.DataFrame({
            col: ["shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 10.0]
            for col in PREDICTIVE_FEATURE_COLUMNS
        })
        result = evaluator.evaluate_transaction(synthetic_row)
        assert result.model_version == "1.0.0"
        assert result.thresholds_applied == {"review_threshold": 0.35, "block_threshold": 0.78}
        assert result.to_dict()["model_version"] == "1.0.0"

    def test_missing_metadata_version_defaults_to_none(self, tmp_path: Path) -> None:
        """Verify that when metadata is missing or does not have a version, model_version is None."""
        evaluator_no_meta = RiskEvaluator(metadata_path=None)
        assert evaluator_no_meta.model_version is None

        synthetic_row = pd.DataFrame({
            col: ["shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 10.0]
            for col in PREDICTIVE_FEATURE_COLUMNS
        })
        result = evaluator_no_meta.evaluate_transaction(synthetic_row)
        assert result.model_version is None
        assert result.to_dict()["model_version"] is None

    def test_scalar_vs_batch_consistency(
        self, evaluator: RiskEvaluator, synthetic_feature_batch: pd.DataFrame
    ) -> None:
        """Verify evaluate_dataframe and evaluate_transaction produce identical results per row."""
        batch_results = evaluator.evaluate_dataframe(synthetic_feature_batch)
        assert len(batch_results) == len(synthetic_feature_batch)

        for i in range(len(synthetic_feature_batch)):
            single_row = synthetic_feature_batch.iloc[[i]]
            single_res = evaluator.evaluate_transaction(single_row)

            assert single_res.model_score == batch_results[i].model_score
            assert single_res.risk_score == batch_results[i].risk_score
            assert single_res.action == batch_results[i].action
            assert single_res.risk_tier == batch_results[i].risk_tier
            assert single_res.policy_mode == batch_results[i].policy_mode
            assert single_res.model_version == batch_results[i].model_version
            assert single_res.thresholds_applied == batch_results[i].thresholds_applied

    def test_evaluate_dataframe_summary(
        self, evaluator: RiskEvaluator, synthetic_feature_batch: pd.DataFrame
    ) -> None:
        """Verify evaluate_dataframe_summary computes accurate distributions, score statistics, and enforces deep immutability."""
        summary = evaluator.evaluate_dataframe_summary(synthetic_feature_batch)

        # 1. Check counts and percentages
        assert summary.total_transactions == len(synthetic_feature_batch)
        assert sum(summary.action_counts.values()) == summary.total_transactions
        assert sum(summary.risk_tier_counts.values()) == summary.total_transactions
        assert pytest.approx(sum(summary.action_percentages.values()), rel=1e-3) == 100.0
        assert pytest.approx(sum(summary.risk_tier_percentages.values()), rel=1e-3) == 100.0

        # 2. Check stats bounds
        assert 0.0 <= summary.model_score_stats["min"] <= summary.model_score_stats["mean"] <= summary.model_score_stats["max"] <= 1.0
        assert 0 <= summary.risk_score_stats["min"] <= summary.risk_score_stats["mean"] <= summary.risk_score_stats["max"] <= 100

        # 3. Check metadata provenance & additive rule metrics
        assert summary.policy_mode == "TRI_TIER"
        assert summary.thresholds_applied == {"review_threshold": 0.35, "block_threshold": 0.78}
        assert summary.model_version == "1.0.0"
        assert summary.overridden_count == 0
        assert summary.rule_trigger_counts == {}

        # 4. Enforce deep immutability: direct in-place mutation of nested mappings is rejected
        with pytest.raises((TypeError, Exception)):
            summary.action_counts["BLOCK"] = 999  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.action_percentages["BLOCK"] = 99.9  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.risk_tier_counts["CRITICAL"] = 999  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.risk_tier_percentages["CRITICAL"] = 99.9  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.model_score_stats["mean"] = 0.50  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.risk_score_stats["mean"] = 50.0  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.thresholds_applied["block_threshold"] = 0.99  # type: ignore
        with pytest.raises((TypeError, Exception)):
            summary.rule_trigger_counts["RULE_TEST"] = 1  # type: ignore

        # 5. Check to_dict serialization produces independent, mutable dictionaries
        summary_dict = summary.to_dict()
        assert isinstance(summary_dict, dict)
        assert summary_dict["total_transactions"] == len(synthetic_feature_batch)
        assert "action_counts" in summary_dict
        assert "risk_tier_counts" in summary_dict
        assert "model_score_stats" in summary_dict
        assert "risk_score_stats" in summary_dict
        assert summary_dict["overridden_count"] == 0
        assert summary_dict["rule_trigger_counts"] == {}

        # Mutate to_dict() results and ensure source summary is unaffected
        orig_approve_count = summary.action_counts["APPROVE"]
        summary_dict["action_counts"]["APPROVE"] = 9999
        summary_dict["risk_score_stats"]["max"] = 9999
        assert summary.action_counts["APPROVE"] == orig_approve_count
        assert summary.risk_score_stats["max"] != 9999


# =====================================================================
# 6. Increment 5B Hybrid Decision Integration Tests
# =====================================================================

class TestHybridDecisionIntegration:
    """
    Test suite for Phase 6 Increment 5B: Hybrid ML & Rule Decision Integration.

    Validates:
    - Deterministic precedence hierarchy: BLOCK > REVIEW > MONITOR > ML Baseline.
    - Rule provenance metadata (rules_triggered, is_overridden, rule_action).
    - Preservation of exact baseline when RuleEngine is None or empty.
    - Non-downgrading semantics (REVIEW rule cannot downgrade ML BLOCK).
    - Invariance of model_score, risk_score, and policy reason_codes under overrides.
    - Single transaction and DataFrame batch evaluations.
    - Summary metrics aggregation (overridden_count, rule_trigger_counts).
    - Missing rule feature safety.
    """

    @pytest.fixture
    def sample_low_risk_row(self) -> pd.DataFrame:
        """Construct a sample transaction row that generates an ML APPROVE decision."""
        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        return pd.DataFrame([row_dict])

    def test_baseline_preservation_when_rule_engine_is_none(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """Verify exact baseline ML decision when rule_engine is None."""
        evaluator = RiskEvaluator(rule_engine=None)
        res = evaluator.evaluate_transaction(sample_low_risk_row)

        assert res.action == DecisionAction.APPROVE
        assert res.rules_triggered == ()
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.reason_codes == (DecisionReasonCode.BELOW_REVIEW_THRESHOLD,)

    def test_baseline_preservation_when_rule_engine_is_empty(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """Verify exact baseline ML decision when rule_engine has zero rules."""
        evaluator = RiskEvaluator(rule_engine=RuleEngine([]))
        res = evaluator.evaluate_transaction(sample_low_risk_row)

        assert res.action == DecisionAction.APPROVE
        assert res.rules_triggered == ()
        assert res.is_overridden is False
        assert res.rule_action is None

    def test_ml_approve_plus_block_rule_overrides_to_block(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """
        Verify: ML APPROVE + RuleOutcome.BLOCK -> final DecisionAction.BLOCK.
        Asserts is_overridden=True, rule_action=BLOCK, and score invariance.
        """
        rule = RiskRule(
            rule_id="RULE_SANCTIONS_BLOCK",
            description="Sanctioned entity check",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.COMPLIANCE,
            priority=10,
        )
        evaluator = RiskEvaluator(rule_engine=RuleEngine([rule]))

        baseline_evaluator = RiskEvaluator()
        baseline_res = baseline_evaluator.evaluate_transaction(sample_low_risk_row)
        assert baseline_res.action == DecisionAction.APPROVE

        hybrid_res = evaluator.evaluate_transaction(sample_low_risk_row)

        # 1. Action override
        assert hybrid_res.action == DecisionAction.BLOCK
        assert hybrid_res.is_overridden is True
        assert hybrid_res.rule_action == RuleOutcome.BLOCK
        assert hybrid_res.rules_triggered == ("RULE_SANCTIONS_BLOCK",)

        # 2. Score & policy provenance invariance
        assert hybrid_res.model_score == baseline_res.model_score
        assert hybrid_res.risk_score == baseline_res.risk_score
        assert hybrid_res.risk_tier == baseline_res.risk_tier
        assert hybrid_res.reason == baseline_res.reason
        assert hybrid_res.reason_codes == baseline_res.reason_codes
        assert hybrid_res.model_version == baseline_res.model_version
        assert hybrid_res.thresholds_applied == baseline_res.thresholds_applied

    def test_ml_review_plus_block_rule_overrides_to_block(self) -> None:
        """
        Verify: ML REVIEW + RuleOutcome.BLOCK -> final DecisionAction.BLOCK.
        Asserts is_overridden=True, rule_action=BLOCK.
        """
        # Configure policy where threshold forces ML REVIEW
        policy_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.0,  # Forces any score >= 0.0 into REVIEW
            block_threshold=0.99,
        )
        rule = RiskRule(
            rule_id="RULE_VELOCITY_BLOCK",
            description="Extreme velocity block",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.BLOCK,
        )
        evaluator = RiskEvaluator(policy_config=policy_cfg, rule_engine=RuleEngine([rule]))

        row = pd.DataFrame([{
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }])
        res = evaluator.evaluate_transaction(row)

        assert res.action == DecisionAction.BLOCK
        assert res.is_overridden is True
        assert res.rule_action == RuleOutcome.BLOCK
        assert res.rules_triggered == ("RULE_VELOCITY_BLOCK",)

    def test_ml_approve_plus_review_rule_overrides_to_review(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """
        Verify: ML APPROVE + RuleOutcome.REVIEW -> final DecisionAction.REVIEW.
        Asserts is_overridden=True, rule_action=REVIEW.
        """
        rule = RiskRule(
            rule_id="RULE_DEVICE_REVIEW",
            description="Suspicious device requires review",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.DEVICE,
        )
        evaluator = RiskEvaluator(rule_engine=RuleEngine([rule]))
        res = evaluator.evaluate_transaction(sample_low_risk_row)

        assert res.action == DecisionAction.REVIEW
        assert res.is_overridden is True
        assert res.rule_action == RuleOutcome.REVIEW
        assert res.rules_triggered == ("RULE_DEVICE_REVIEW",)

    def test_ml_review_plus_review_rule_keeps_review_no_override(self) -> None:
        """
        Verify: ML REVIEW + RuleOutcome.REVIEW -> final DecisionAction.REVIEW.
        Action was already REVIEW, so is_overridden=False, rule_action=None.
        """
        policy_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.0,  # Baseline is REVIEW
            block_threshold=0.99,
        )
        rule = RiskRule(
            rule_id="RULE_SUSPICIOUS_REVIEW",
            description="Suspicious merchant",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.REVIEW,
        )
        evaluator = RiskEvaluator(policy_config=policy_cfg, rule_engine=RuleEngine([rule]))

        row = pd.DataFrame([{
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }])
        res = evaluator.evaluate_transaction(row)

        assert res.action == DecisionAction.REVIEW
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.rules_triggered == ("RULE_SUSPICIOUS_REVIEW",)

    def test_ml_block_plus_review_rule_never_downgrades(self) -> None:
        """
        Verify: ML BLOCK + RuleOutcome.REVIEW -> final DecisionAction.BLOCK.
        Must NEVER downgrade ML BLOCK. is_overridden=False, rule_action=None.
        """
        policy_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.0,
            block_threshold=0.0,  # Baseline is BLOCK
        )
        rule = RiskRule(
            rule_id="RULE_SUSPICIOUS_REVIEW",
            description="Review rule match",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.REVIEW,
        )
        evaluator = RiskEvaluator(policy_config=policy_cfg, rule_engine=RuleEngine([rule]))

        row = pd.DataFrame([{
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }])
        res = evaluator.evaluate_transaction(row)

        assert res.action == DecisionAction.BLOCK
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.rules_triggered == ("RULE_SUSPICIOUS_REVIEW",)

    def test_ml_block_plus_block_rule_keeps_block_no_override(self) -> None:
        """
        Verify: ML BLOCK + RuleOutcome.BLOCK -> final DecisionAction.BLOCK.
        Action was already BLOCK, so is_overridden=False, rule_action=None.
        """
        policy_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.0,
            block_threshold=0.0,  # Baseline is BLOCK
        )
        rule = RiskRule(
            rule_id="RULE_SANCTIONS_BLOCK",
            description="Sanctions block",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.BLOCK,
        )
        evaluator = RiskEvaluator(policy_config=policy_cfg, rule_engine=RuleEngine([rule]))

        row = pd.DataFrame([{
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }])
        res = evaluator.evaluate_transaction(row)

        assert res.action == DecisionAction.BLOCK
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.rules_triggered == ("RULE_SANCTIONS_BLOCK",)

    def test_ml_action_plus_monitor_rule_leaves_action_unchanged(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """
        Verify: ML APPROVE + RuleOutcome.MONITOR -> final DecisionAction.APPROVE.
        Rule is recorded in rules_triggered, but is_overridden=False and rule_action=None.
        """
        rule = RiskRule(
            rule_id="RULE_NEW_DEVICE_MONITOR",
            description="Log new device observed",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.MONITOR,
            rule_type=RuleType.DEVICE,
        )
        evaluator = RiskEvaluator(rule_engine=RuleEngine([rule]))
        res = evaluator.evaluate_transaction(sample_low_risk_row)

        assert res.action == DecisionAction.APPROVE
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.rules_triggered == ("RULE_NEW_DEVICE_MONITOR",)

    def test_multiple_rules_deterministic_precedence_and_order(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """
        Verify: Multiple matched rules resolve with BLOCK > REVIEW > MONITOR precedence,
        and rules_triggered maintains deterministic sorted priority/id order.
        """
        r_monitor = RiskRule(
            rule_id="R1_MONITOR",
            description="Monitor rule",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.1,
            outcome=RuleOutcome.MONITOR,
            priority=10,
        )
        r_review = RiskRule(
            rule_id="R2_REVIEW",
            description="Review rule",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.1,
            outcome=RuleOutcome.REVIEW,
            priority=20,
        )
        r_block = RiskRule(
            rule_id="R3_BLOCK",
            description="Block rule",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.1,
            outcome=RuleOutcome.BLOCK,
            priority=30,
        )
        # Pass rules in reverse order to engine
        engine = RuleEngine([r_block, r_review, r_monitor])
        evaluator = RiskEvaluator(rule_engine=engine)

        res = evaluator.evaluate_transaction(sample_low_risk_row)

        # Precedence: BLOCK enacted
        assert res.action == DecisionAction.BLOCK
        assert res.is_overridden is True
        assert res.rule_action == RuleOutcome.BLOCK

        # Deterministic order: sorted by priority (10, 20, 30)
        assert res.rules_triggered == ("R1_MONITOR", "R2_REVIEW", "R3_BLOCK")

    def test_single_transaction_hybrid_formats_consistency(
        self, sample_low_risk_row: pd.DataFrame
    ) -> None:
        """Verify 1-row DataFrame, pd.Series, and dict produce identical hybrid results."""
        rule = RiskRule(
            rule_id="R_HIGH_AMT_REVIEW",
            description="Review high amount",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=0.5,
            outcome=RuleOutcome.REVIEW,
        )
        evaluator = RiskEvaluator(rule_engine=RuleEngine([rule]))

        res_df = evaluator.evaluate_transaction(sample_low_risk_row)
        res_series = evaluator.evaluate_transaction(sample_low_risk_row.iloc[0])
        res_dict = evaluator.evaluate_transaction(sample_low_risk_row.iloc[0].to_dict())

        assert res_df.action == res_series.action == res_dict.action == DecisionAction.REVIEW
        assert res_df.is_overridden == res_series.is_overridden == res_dict.is_overridden is True
        assert res_df.rule_action == res_series.rule_action == res_dict.rule_action == RuleOutcome.REVIEW
        assert res_df.rules_triggered == res_series.rules_triggered == res_dict.rules_triggered == ("R_HIGH_AMT_REVIEW",)
        assert res_df.model_score == res_series.model_score == res_dict.model_score

    def test_batch_dataframe_hybrid_evaluation_and_summary_metrics(self) -> None:
        """
        Verify batch DataFrame hybrid evaluation across multiple rows with varied rule triggers,
        and verify accurate summary aggregation (overridden_count, rule_trigger_counts).
        """
        r_block = RiskRule(
            rule_id="RULE_BLOCK_HIGH_AMT",
            description="Block high amount",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=1000.0,
            outcome=RuleOutcome.BLOCK,
            priority=10,
        )
        r_review = RiskRule(
            rule_id="RULE_REVIEW_MID_AMT",
            description="Review mid amount",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=500.0,
            outcome=RuleOutcome.REVIEW,
            priority=20,
        )
        r_monitor = RiskRule(
            rule_id="RULE_MONITOR_UNUSUAL_HOUR",
            description="Monitor unusual hours",
            feature_name="hour_of_day",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=20.0,
            outcome=RuleOutcome.MONITOR,
            priority=30,
        )
        r_dormant = RiskRule(
            rule_id="RULE_NEVER_TRIGGERED",
            description="Unmatched rule",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=999999.0,
            outcome=RuleOutcome.BLOCK,
            priority=40,
        )
        engine = RuleEngine([r_block, r_review, r_monitor, r_dormant])
        evaluator = RiskEvaluator(rule_engine=engine)

        # Construct 4-row batch with low ML scores (all would be ML APPROVE)
        base_row = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }

        row0 = base_row.copy()
        row0["amount"] = 50.0   # No rules triggered -> APPROVE (no override)
        row0["hour_of_day"] = 10.0

        row1 = base_row.copy()
        row1["amount"] = 1500.0 # Triggers RULE_BLOCK_HIGH_AMT & RULE_REVIEW_MID_AMT -> BLOCK (override)
        row1["hour_of_day"] = 10.0

        row2 = base_row.copy()
        row2["amount"] = 600.0  # Triggers RULE_REVIEW_MID_AMT -> REVIEW (override)
        row2["hour_of_day"] = 10.0

        row3 = base_row.copy()
        row3["amount"] = 50.0   # Triggers RULE_MONITOR_UNUSUAL_HOUR -> APPROVE (no override)
        row3["hour_of_day"] = 23.0

        df_batch = pd.DataFrame([row0, row1, row2, row3])
        results = evaluator.evaluate_dataframe(df_batch)

        assert len(results) == 4

        # Row 0: ML APPROVE, no rule
        assert results[0].action == DecisionAction.APPROVE
        assert results[0].is_overridden is False
        assert results[0].rule_action is None
        assert results[0].rules_triggered == ()

        # Row 1: Overridden to BLOCK
        assert results[1].action == DecisionAction.BLOCK
        assert results[1].is_overridden is True
        assert results[1].rule_action == RuleOutcome.BLOCK
        assert results[1].rules_triggered == ("RULE_BLOCK_HIGH_AMT", "RULE_REVIEW_MID_AMT")

        # Row 2: Overridden to REVIEW
        assert results[2].action == DecisionAction.REVIEW
        assert results[2].is_overridden is True
        assert results[2].rule_action == RuleOutcome.REVIEW
        assert results[2].rules_triggered == ("RULE_REVIEW_MID_AMT",)

        # Row 3: APPROVE (monitor triggered)
        assert results[3].action == DecisionAction.APPROVE
        assert results[3].is_overridden is False
        assert results[3].rule_action is None
        assert results[3].rules_triggered == ("RULE_MONITOR_UNUSUAL_HOUR",)

        # Summary verification
        summary = evaluator.evaluate_dataframe_summary(df_batch)
        assert summary.total_transactions == 4
        assert summary.action_counts == {"APPROVE": 2, "REVIEW": 1, "BLOCK": 1}
        assert summary.overridden_count == 2
        assert summary.rule_trigger_counts == {
            "RULE_BLOCK_HIGH_AMT": 1,
            "RULE_REVIEW_MID_AMT": 2,
            "RULE_MONITOR_UNUSUAL_HOUR": 1,
            "RULE_NEVER_TRIGGERED": 0,
        }

        # Verify summary serialization
        s_dict = summary.to_dict()
        assert s_dict["overridden_count"] == 2
        assert s_dict["rule_trigger_counts"]["RULE_REVIEW_MID_AMT"] == 2
        assert s_dict["rule_trigger_counts"]["RULE_NEVER_TRIGGERED"] == 0

    def test_missing_rule_feature_tolerance(self, sample_low_risk_row: pd.DataFrame) -> None:
        """
        Verify: When a rule references a feature NOT in the transaction (e.g. is_vpn_detected),
        the rule evaluates to False, ML scores normally, and no error is raised.
        """
        rule = RiskRule(
            rule_id="RULE_CUSTOM_NONEXISTENT",
            description="Custom missing feature rule",
            feature_name="is_vpn_detected",
            operator=RuleOperator.IS_TRUE,
            comparison_value=True,
            outcome=RuleOutcome.BLOCK,
        )
        evaluator = RiskEvaluator(rule_engine=RuleEngine([rule]))
        res = evaluator.evaluate_transaction(sample_low_risk_row)

        assert res.action == DecisionAction.APPROVE
        assert res.is_overridden is False
        assert res.rule_action is None
        assert res.rules_triggered == ()


# =====================================================================
# 7. Artifact Immutability Verification Test
# =====================================================================

class TestArtifactImmutability:
    """Test suite verifying that no Phase 4 or Phase 5 artifacts were modified."""

    def test_all_artifact_hashes_unchanged(self) -> None:
        """Verify SHA-256 hashes of all 8 Phase 4/5 artifacts remain 100% identical."""
        for rel_path, expected_hash in BASELINE_ARTIFACT_HASHES.items():
            p = Path(rel_path)
            assert p.exists(), f"Artifact missing: {rel_path}"
            current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            assert current_hash == expected_hash, (
                f"CRITICAL: Artifact {rel_path} was modified! Expected {expected_hash}, got {current_hash}"
            )
