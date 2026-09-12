"""
Automated Integration Tests for Phase 6 Increment 2: RiskEvaluator & End-to-End Decision Pipeline.

Validates:
1. RiskEvaluator Initialization: Default and custom paths/configs, FileNotFoundError on missing paths.
2. Synthetic End-to-End Evaluation: Scalar (DataFrame, Series, Dict) and batch DataFrame scoring.
3. Defensive Input Validation: Missing columns, non-finite values, empty inputs, type errors.
4. Input Immutability: Asserts input DataFrames are not mutated during evaluation.
5. Authoritative Phase 5 Policy Consistency: Validates that RiskEvaluator in BINARY_AUTO mode with
   block_threshold=0.78 on Validation data dynamically matches authoritative Phase 5 confusion matrix.
6. Artifact Immutability: Guarantees zero artifact modification across Phase 4 and Phase 5 artifacts.
"""

import json
import hashlib
from pathlib import Path
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
)
from ml.risk_engine.evaluator import (
    RiskEvaluator,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
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
    """Create a module-scoped RiskEvaluator instance loaded from frozen artifacts."""
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
        """Verify default evaluator setup with frozen model and preprocessor."""
        assert evaluator.model is not None
        assert evaluator.preprocessor is not None
        assert evaluator.config.policy_mode == PolicyMode.TRI_TIER
        assert evaluator.config.review_threshold == 0.35
        assert evaluator.config.block_threshold == 0.78

    def test_custom_policy_config(self) -> None:
        """Verify evaluator respects custom policy configuration."""
        custom_cfg = DecisionPolicyConfig(
            policy_mode=PolicyMode.BINARY_AUTO,
            block_threshold=0.65,
        )
        evaluator = RiskEvaluator(policy_config=custom_cfg)
        assert evaluator.config.policy_mode == PolicyMode.BINARY_AUTO
        assert evaluator.config.block_threshold == 0.65

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
# 5. Artifact Immutability Verification Test
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
