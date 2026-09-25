"""
Unit Tests for Phase 14.4 Champion vs. Challenger Comparator.

Validates:
1. Dynamic Champion threshold extraction from metadata (never hard-coded).
2. Dynamic Candidate frozen threshold extraction from manifest.
3. Metadata discrepancy detection (fails loudly on conflicting thresholds).
4. Neutral mathematical metric delta calculations (POSITIVE, NEGATIVE, ZERO).
5. Rate metric percentage-point (pp) delta computations.
6. Absolute and relative change calculations for costs, counts, and latencies.
7. Identical latency benchmarking methodology (same slice, warm-up, timing, percentiles).
8. Feature schema and partition integrity enforcement.
9. Champion immutability enforcement via SHA-256 verification.
10. Candidate lifecycle state invariants (preserves CANDIDATE, is_active_champion=False).
11. Read-only OOT protection (zero leakage / tuning).
12. Comparison artifact JSON serialization and round-trip fidelity.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.evaluation.comparator import (
    ChampionChallengerComparator,
    ChampionImmutabilityViolationError,
    ChampionMetadataDiscrepancyError,
)
from ml.lifecycle.evaluation.comparison_schemas import (
    ChampionChallengerComparisonResult,
    ClassificationComparison,
    DeltaSign,
    LatencyComparison,
    MetricDelta,
    OperationalComparison,
    compute_metric_delta,
)
from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics
from ml.lifecycle.schemas import (
    EvaluationMetricsSummary,
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)

pytestmark = pytest.mark.unit


class MockModel:
    """Mock model returning controlled probabilities."""
    def __init__(self, prob_offset: float = 0.0) -> None:
        self.prob_offset = prob_offset

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.linspace(0.01, 0.99, num=n) + self.prob_offset
        return np.clip(probs, 0.0, 1.0)


class MockPreprocessor:
    """Mock preprocessor performing pass-through transformation to float matrix."""
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones((len(X), 55), dtype=np.float64)


class TestChampionChallengerComparatorUnit:
    """Unit test suite for Phase 14.4 Comparator."""

    @pytest.fixture
    def mock_champion_bundle(self, tmp_path: Path):
        """Create mock Champion directory with model, preprocessor, and metadata."""
        champ_dir = tmp_path / "champion"
        champ_dir.mkdir()

        import joblib
        m_path = champ_dir / "champion_model.joblib"
        p_path = champ_dir / "champion_preprocessor.joblib"
        meta_path = champ_dir / "model_metadata.json"

        joblib.dump(MockModel(0.0), m_path)
        joblib.dump(MockPreprocessor(), p_path)

        metadata = {
            "model_version": "1.0.0",
            "selected_threshold": 0.94,
            "validation_benchmark": {"best_f1_threshold": 0.94},
            "oot_test_metrics_frozen_threshold": {"threshold": 0.94},
            "feature_count": 55,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return champ_dir

    @pytest.fixture
    def mock_candidate_bundle(self, tmp_path: Path):
        """Create mock Candidate bundle with frozen threshold 0.78."""
        cand_dir = tmp_path / "candidate_v1_1_0"
        cand_dir.mkdir()

        import joblib
        m_path = cand_dir / "model.joblib"
        p_path = cand_dir / "preprocessor.joblib"
        manifest_path = cand_dir / "manifest.json"

        joblib.dump(MockModel(0.05), m_path)
        joblib.dump(MockPreprocessor(), p_path)

        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE,
            operating_threshold=0.78,
            hyperparameters={"max_depth": 6, "learning_rate": 0.08},
            training_metadata={"training_dataset_rows": 5000, "positive_fraud_count": 50},
            sha256_checksums={
                "model": calculate_file_sha256(m_path),
                "preprocessor": calculate_file_sha256(p_path),
            },
        )
        manifest.save(manifest_path)
        return cand_dir

    @pytest.fixture
    def mock_datasets(self, tmp_path: Path):
        """Create mock 55-feature validation and OOT parquet files with canonical schema."""
        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, TARGET_COLUMN
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        np.random.seed(42)
        
        # Validation df (200 rows, 10 fraud)
        val_data = {}
        for col in PREDICTIVE_FEATURE_COLUMNS:
            if col in ["merchant_category", "job_category"]:
                val_data[col] = np.random.choice(["grocery_pos", "gas_transport", "shopping_net"], size=200)
            else:
                val_data[col] = np.random.randn(200)
        val_df = pd.DataFrame(val_data)
        val_df[TARGET_COLUMN] = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
        val_path = data_dir / "val.parquet"
        val_df.to_parquet(val_path, index=False)

        # OOT df (200 rows, 10 fraud)
        oot_data = {}
        for col in PREDICTIVE_FEATURE_COLUMNS:
            if col in ["merchant_category", "job_category"]:
                oot_data[col] = np.random.choice(["grocery_pos", "gas_transport", "shopping_net"], size=200)
            else:
                oot_data[col] = np.random.randn(200)
        oot_df = pd.DataFrame(oot_data)
        oot_df[TARGET_COLUMN] = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
        oot_path = data_dir / "oot.parquet"
        oot_df.to_parquet(oot_path, index=False)

        return val_path, oot_path

    def test_dynamic_champion_threshold_loading(self, mock_champion_bundle):
        """Verify Champion threshold is loaded dynamically from metadata (never hard-coded)."""
        comparator = ChampionChallengerComparator()
        _, _, metadata, threshold, paths = comparator.load_authoritative_champion(mock_champion_bundle)

        assert threshold == 0.94
        assert metadata["model_version"] == "1.0.0"
        assert paths["model"].exists()
        assert paths["preprocessor"].exists()
        assert paths["metadata"].exists()

    def test_candidate_frozen_threshold_loading(self, mock_candidate_bundle):
        """Verify Candidate frozen threshold is loaded from manifest."""
        comparator = ChampionChallengerComparator()
        _, _, manifest, threshold, paths = comparator.load_candidate_bundle(mock_candidate_bundle)

        assert threshold == 0.78
        assert manifest.model_version == "1.1.0"
        assert manifest.status == ModelLifecycleStatus.CANDIDATE
        assert paths["model"].exists()

    def test_champion_metadata_discrepancy_raises_error(self, tmp_path: Path):
        """Verify conflicting thresholds in Champion metadata sources raise ChampionMetadataDiscrepancyError."""
        bad_champ_dir = tmp_path / "bad_champion"
        bad_champ_dir.mkdir()

        import joblib
        joblib.dump(MockModel(), bad_champ_dir / "champion_model.joblib")
        joblib.dump(MockPreprocessor(), bad_champ_dir / "champion_preprocessor.joblib")

        # Conflicting thresholds: selected_threshold=0.94 vs validation_benchmark=0.85
        meta = {
            "model_version": "1.0.0",
            "selected_threshold": 0.94,
            "validation_benchmark": {"best_f1_threshold": 0.85},
        }
        with open(bad_champ_dir / "model_metadata.json", "w") as f:
            json.dump(meta, f)

        comparator = ChampionChallengerComparator()
        with pytest.raises(ChampionMetadataDiscrepancyError) as exc_info:
            comparator.load_authoritative_champion(bad_champ_dir)

        assert "Discrepancy detected in Champion metadata threshold sources" in str(exc_info.value)

    def test_delta_calculation_rate_metrics_pp(self):
        """Verify neutral percentage-point and relative change for rates in [0, 1]."""
        # PR-AUC increase: 0.94 -> 0.96
        d = compute_metric_delta("pr_auc", champion_val=0.9400, candidate_val=0.9600, is_rate=True)
        assert d.metric_name == "pr_auc"
        assert d.champion_value == 0.9400
        assert d.candidate_value == 0.9600
        assert d.absolute_delta == 0.0200
        assert d.percentage_point_delta == 2.0  # +2.0 percentage points
        assert round(d.relative_change_pct, 4) == round((0.02 / 0.94) * 100.0, 4)
        assert d.delta_sign == DeltaSign.POSITIVE

        # FPR decrease: 0.0020 -> 0.0010
        d_fpr = compute_metric_delta("fpr", champion_val=0.0020, candidate_val=0.0010, is_rate=True)
        assert d_fpr.absolute_delta == -0.0010
        assert d_fpr.percentage_point_delta == -0.10
        assert d_fpr.delta_sign == DeltaSign.NEGATIVE

        # Unchanged: 0.95 -> 0.95
        d_same = compute_metric_delta("recall", champion_val=0.9500, candidate_val=0.9500, is_rate=True)
        assert d_same.absolute_delta == 0.0
        assert d_same.percentage_point_delta == 0.0
        assert d_same.delta_sign == DeltaSign.ZERO

    def test_delta_calculation_cost_and_counts(self):
        """Verify neutral absolute delta and relative change for financial costs and counts."""
        # Cost reduction: $20,000 -> $15,000
        d_cost = compute_metric_delta("expected_cost", champion_val=20000.0, candidate_val=15000.0, is_rate=False)
        assert d_cost.percentage_point_delta is None
        assert d_cost.absolute_delta == -5000.0
        assert d_cost.relative_change_pct == -25.0
        assert d_cost.delta_sign == DeltaSign.NEGATIVE

        # Count increase: TP 800 -> 870
        d_tp = compute_metric_delta("tp", champion_val=800, candidate_val=870, is_rate=False)
        assert d_tp.absolute_delta == 70.0
        assert d_tp.percentage_point_delta is None
        assert d_tp.delta_sign == DeltaSign.POSITIVE

    def test_delta_sign_neutrality(self):
        """Assert DeltaSign contains strictly POSITIVE, NEGATIVE, ZERO with no evaluative labels."""
        valid_signs = {s.value for s in DeltaSign}
        assert valid_signs == {"POSITIVE", "NEGATIVE", "ZERO"}
        assert "IMPROVED" not in valid_signs
        assert "REGRESSED" not in valid_signs
        assert "BETTER" not in valid_signs
        assert "WORSE" not in valid_signs

    def test_identical_latency_benchmark_harness(self, mock_champion_bundle, mock_candidate_bundle):
        """Verify latency benchmark uses identical slice, warm-up iterations, and timing methodology."""
        comparator = ChampionChallengerComparator()
        champ_m, champ_p, _, _, _ = comparator.load_authoritative_champion(mock_champion_bundle)
        cand_m, cand_p, _, _, _ = comparator.load_candidate_bundle(mock_candidate_bundle)

        df_sample = pd.DataFrame(np.random.randn(100, 55))

        lat_comp = comparator.benchmark_identical_latency(
            champ_model=champ_m,
            champ_prep=champ_p,
            cand_model=cand_m,
            cand_prep=cand_p,
            X_val=df_sample,
            sample_count=50,
            warmup_count=5,
        )

        assert isinstance(lat_comp, LatencyComparison)
        assert lat_comp.warmup_iterations == 5
        assert lat_comp.measured_samples == 50
        assert "mean_ms" in lat_comp.champion_latency
        assert "p50_ms" in lat_comp.champion_latency
        assert "p95_ms" in lat_comp.champion_latency
        assert "p99_ms" in lat_comp.champion_latency
        assert "mean_ms" in lat_comp.deltas
        assert isinstance(lat_comp.deltas["mean_ms"], MetricDelta)

    def test_identical_evaluation_partitions_enforced(
        self, mock_champion_bundle, mock_candidate_bundle, tmp_path: Path
    ):
        """Verify comparator enforces 55 features and rejects mismatched partition schemas."""
        bad_val = tmp_path / "bad_val.parquet"
        # Only 10 features instead of 55
        pd.DataFrame(np.random.randn(50, 10)).to_parquet(bad_val, index=False)

        oot_val = tmp_path / "good_oot.parquet"
        pd.DataFrame(np.random.randn(50, 55)).to_parquet(oot_val, index=False)

        cfg = LifecycleConfig(expected_feature_count=55)
        comparator = ChampionChallengerComparator(lifecycle_config=cfg)

        with pytest.raises((ValueError, AssertionError)):
            comparator.compare(
                candidate_bundle_dir=mock_candidate_bundle,
                champion_source=mock_champion_bundle,
                val_dataset_path=bad_val,
                oot_dataset_path=oot_val,
                benchmark_latency=False,
            )


    def test_champion_artifact_immutability_enforced(
        self, mock_champion_bundle, mock_candidate_bundle, mock_datasets, monkeypatch
    ):
        """Verify tampering with Champion artifact triggers ChampionImmutabilityViolationError."""
        val_path, oot_path = mock_datasets
        comparator = ChampionChallengerComparator(
            lifecycle_config=LifecycleConfig(expected_feature_count=55)
        )

        champ_m_path = mock_champion_bundle / "champion_model.joblib"

        # Tamper with file during evaluation
        original_compare_classification = comparator.compare_classification_metrics
        def tampering_hook(c_m, cand_m):
            # Tamper with Champion model file
            with open(champ_m_path, "ab") as f:
                f.write(b"tamper")
            return original_compare_classification(c_m, cand_m)

        monkeypatch.setattr(comparator, "compare_classification_metrics", tampering_hook)

        with pytest.raises(ChampionImmutabilityViolationError) as exc_info:
            comparator.compare(
                candidate_bundle_dir=mock_candidate_bundle,
                champion_source=mock_champion_bundle,
                val_dataset_path=val_path,
                oot_dataset_path=oot_path,
                benchmark_latency=False,
            )

        assert "Champion immutability violation detected" in str(exc_info.value)

    def test_candidate_state_invariants_and_provenance(
        self, mock_champion_bundle, mock_candidate_bundle, mock_datasets
    ):
        """Verify Candidate lifecycle state remains CANDIDATE and is_active_champion=False."""
        val_path, oot_path = mock_datasets
        comparator = ChampionChallengerComparator(
            lifecycle_config=LifecycleConfig(expected_feature_count=55)
        )

        result = comparator.compare(
            candidate_bundle_dir=mock_candidate_bundle,
            champion_source=mock_champion_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            benchmark_latency=False,
        )

        assert result.candidate_version == "1.1.0"
        assert result.champion_version == "1.0.0"

        # Reload candidate manifest to assert zero lifecycle mutation
        manifest = ModelBundleManifest.load(mock_candidate_bundle / "manifest.json")
        assert manifest.status == ModelLifecycleStatus.CANDIDATE
        assert manifest.operating_threshold == 0.78
        assert manifest.training_metadata["training_dataset_rows"] == 5000

    def test_comparison_artifact_serialization_roundtrip(
        self, mock_champion_bundle, mock_candidate_bundle, mock_datasets, tmp_path: Path
    ):
        """Verify comparison result serializes to valid JSON and reloads losslessly."""
        val_path, oot_path = mock_datasets
        out_file = tmp_path / "test_comparison.json"

        comparator = ChampionChallengerComparator(
            lifecycle_config=LifecycleConfig(expected_feature_count=55)
        )

        result = comparator.compare(
            candidate_bundle_dir=mock_candidate_bundle,
            champion_source=mock_champion_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            output_comparison_file=out_file,
            benchmark_latency=True,
        )

        assert out_file.exists()
        reloaded = ChampionChallengerComparisonResult.load(out_file)

        assert reloaded.champion_version == "1.0.0"
        assert reloaded.candidate_version == "1.1.0"
        assert reloaded.champion_operating_threshold == 0.94
        assert reloaded.candidate_operating_threshold == 0.78
        assert "pr_auc" in reloaded.validation_comparison.deltas
        assert "pr_auc" in reloaded.oot_comparison.deltas
        assert reloaded.latency_comparison.measured_samples == 50

    def test_zero_oot_leakage_invariants(
        self, mock_champion_bundle, mock_candidate_bundle, mock_datasets
    ):
        """Verify protected OOT evaluation is strictly read-only and does not mutate model state."""
        val_path, oot_path = mock_datasets

        cand_m_sha_before = calculate_file_sha256(mock_candidate_bundle / "model.joblib")
        cand_p_sha_before = calculate_file_sha256(mock_candidate_bundle / "preprocessor.joblib")

        comparator = ChampionChallengerComparator(
            lifecycle_config=LifecycleConfig(expected_feature_count=55)
        )

        result = comparator.compare(
            candidate_bundle_dir=mock_candidate_bundle,
            champion_source=mock_champion_bundle,
            val_dataset_path=val_path,
            oot_dataset_path=oot_path,
            benchmark_latency=False,
        )

        # Assert candidate artifacts remain unmodified after OOT evaluation
        assert calculate_file_sha256(mock_candidate_bundle / "model.joblib") == cand_m_sha_before
        assert calculate_file_sha256(mock_candidate_bundle / "preprocessor.joblib") == cand_p_sha_before
        assert result.candidate_operating_threshold == 0.78

