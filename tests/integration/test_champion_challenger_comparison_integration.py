"""
Integration Tests for Phase 14.4 Champion vs. Challenger Comparative Evaluation.

Validates end-to-end integration:
1. Side-by-side comparison across actual 277,859-row validation partition (`val_features.parquet`).
2. Side-by-side comparison across actual 277,860-row protected OOT partition (`test_features.parquet`).
3. Dynamic extraction of Champion operating threshold (0.94) and Candidate frozen threshold (0.78).
4. Neutral mathematical delta calculation (Validation, OOT, Operational, Latency).
5. Champion artifact SHA-256 immutability verification.
6. Candidate state invariant preservation (status = CANDIDATE, is_active_champion = False).
7. Persistence and schema validity of `comparison_v1.0.0_vs_v1.1.0.json`.
"""

import json
from pathlib import Path
import shutil
import pytest

from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.evaluation.comparator import ChampionChallengerComparator
from ml.lifecycle.evaluation.comparison_schemas import ChampionChallengerComparisonResult
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
)
from ml.models.config import TEST_FEATURES_PATH, VAL_FEATURES_PATH

pytestmark = [pytest.mark.integration]


class TestChampionChallengerComparisonIntegration:
    """End-to-end comparative evaluation integration tests on real production artifacts."""

    @pytest.fixture
    def candidate_bundle_dir(self, tmp_path: Path):
        """Prepare a candidate bundle for integration testing."""
        bundle_dir = tmp_path / "candidate_v1_1_0"
        bundle_dir.mkdir()

        m_src = Path("ml/models/artifacts/champion_model.joblib")
        p_src = Path("ml/models/artifacts/champion_preprocessor.joblib")

        m_dest = bundle_dir / "model.joblib"
        p_dest = bundle_dir / "preprocessor.joblib"

        shutil.copyfile(m_src, m_dest)
        shutil.copyfile(p_src, p_dest)

        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE,
            operating_threshold=0.78,
            hyperparameters={"max_depth": 6, "learning_rate": 0.08, "n_estimators": 300},
            training_metadata={
                "training_dataset_rows": 648336,
                "positive_fraud_count": 3120,
                "negative_legit_count": 645216,
                "fraud_prevalence": 0.004812,
            },
            sha256_checksums={
                "model": calculate_file_sha256(m_dest),
                "preprocessor": calculate_file_sha256(p_dest),
            },
        )
        manifest.save(bundle_dir / "manifest.json")
        return bundle_dir

    def test_real_champion_vs_candidate_comparison_validation_and_oot(
        self, candidate_bundle_dir, tmp_path: Path
    ):
        """
        Verify end-to-end comparative evaluation on real 277,859 validation rows
        and real 277,860 protected OOT rows.
        """
        champion_model_p = Path("ml/models/artifacts/champion_model.joblib")
        champion_prep_p = Path("ml/models/artifacts/champion_preprocessor.joblib")
        champion_meta_p = Path("ml/models/artifacts/model_metadata.json")

        champ_m_sha = calculate_file_sha256(champion_model_p)
        champ_p_sha = calculate_file_sha256(champion_prep_p)
        champ_meta_sha = calculate_file_sha256(champion_meta_p)

        out_comparison_file = tmp_path / "comparison_v1.0.0_vs_v1.1.0.json"

        comparator = ChampionChallengerComparator()

        result = comparator.compare(
            candidate_bundle_dir=candidate_bundle_dir,
            val_dataset_path=VAL_FEATURES_PATH,
            oot_dataset_path=TEST_FEATURES_PATH,
            output_comparison_file=out_comparison_file,
            benchmark_latency=True,
        )

        assert isinstance(result, ChampionChallengerComparisonResult)
        assert result.champion_version == "1.0.0"
        assert result.candidate_version == "1.1.0"

        # Assert authoritative thresholds dynamically extracted
        assert result.champion_operating_threshold == 0.94
        assert result.candidate_operating_threshold == 0.78

        # Validate partition sizes in metadata
        assert result.dataset_metadata["validation_sample_count"] == 277859
        assert result.dataset_metadata["validation_fraud_count"] == 1221
        assert result.dataset_metadata["oot_sample_count"] == 277860
        assert result.dataset_metadata["oot_fraud_count"] == 924

        # Validate Validation Comparison
        val_comp = result.validation_comparison
        assert val_comp.champion_metrics.threshold == 0.94
        assert val_comp.candidate_metrics.threshold == 0.78
        assert "pr_auc" in val_comp.deltas
        assert "recall" in val_comp.deltas
        assert "precision" in val_comp.deltas
        assert "expected_cost" in val_comp.deltas

        # Validate OOT Comparison
        oot_comp = result.oot_comparison
        assert oot_comp.champion_metrics.threshold == 0.94
        assert oot_comp.candidate_metrics.threshold == 0.78
        assert "pr_auc" in oot_comp.deltas
        assert "recall" in oot_comp.deltas
        assert "expected_cost" in oot_comp.deltas

        # Validate Operational Routing Comparison
        val_op = result.validation_operational_comparison
        assert "APPROVE" in val_op.action_count_deltas
        assert "REVIEW" in val_op.action_count_deltas
        assert "BLOCK" in val_op.action_count_deltas
        assert "fraud_in_block" in val_op.fraud_routing_deltas

        # Validate Latency Comparison
        lat_comp = result.latency_comparison
        assert lat_comp.measured_samples == 50
        assert lat_comp.warmup_iterations == 5
        assert "mean_ms" in lat_comp.deltas
        assert "p95_ms" in lat_comp.deltas

        # CRITICAL: Champion Immutability Assertion
        assert calculate_file_sha256(champion_model_p) == champ_m_sha
        assert calculate_file_sha256(champion_prep_p) == champ_p_sha
        assert calculate_file_sha256(champion_meta_p) == champ_meta_sha

        # CRITICAL: Candidate State Invariant
        cand_manifest = ModelBundleManifest.load(candidate_bundle_dir / "manifest.json")
        assert cand_manifest.status == ModelLifecycleStatus.CANDIDATE

        # Validate serialized JSON artifact
        assert out_comparison_file.exists()
        reloaded = ChampionChallengerComparisonResult.load(out_comparison_file)
        assert reloaded.champion_version == "1.0.0"
        assert reloaded.candidate_version == "1.1.0"
