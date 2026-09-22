"""
Unit Tests for Phase 13.1 Monitoring Foundation & Baseline Profile Generation.

Verifies:
- 55-feature baseline profile schema, 10-decile bin edges, 5,000 reference sample size, vocabularies.
- Prediction baseline profile 10 score bins, 10 risk score buckets, tier/action distributions.
- Performance baseline profile benchmarks at primary operating threshold 0.78 and comparison threshold 0.50.
- Checksum integrity, deterministic reproducibility, and serialization roundtrips.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from ml.models.config import PREDICTIVE_FEATURE_COLUMNS, CATEGORICAL_PREDICTORS
from ml.monitoring.config import default_monitoring_config, MonitoringConfig
from ml.monitoring.schemas import (
    FeatureBaselineProfile,
    PredictionBaselineProfile,
    PerformanceBaselineProfile,
    NumericalFeatureProfile,
    CategoricalFeatureProfile,
    compute_content_sha256,
)


class TestMonitoringBaselineFoundation:
    """Test suite for Phase 13.1 baseline profile artifacts and schemas."""

    def test_feature_baseline_profile_exists_and_loads(self):
        """Verify baseline_feature_profile_v1.0.0.json exists and loads successfully."""
        path = default_monitoring_config.feature_profile_path
        assert path.exists(), f"Feature baseline profile not found at {path}"

        profile = FeatureBaselineProfile.load(path)
        assert profile.model_version == "1.0.0"
        assert profile.dataset_name == "train_features.parquet"
        assert profile.dataset_row_count == 1296675
        assert profile.num_features == 55
        assert profile.random_seed == 42
        assert profile.sha256_checksum is not None
        assert len(profile.sha256_checksum) == 64

    def test_feature_baseline_profile_55_feature_completeness(self):
        """Verify that exactly the 55 predictive features are present and categorized correctly."""
        profile = FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)
        assert len(profile.features) == 55
        assert set(profile.features.keys()) == set(PREDICTIVE_FEATURE_COLUMNS)

        for col in PREDICTIVE_FEATURE_COLUMNS:
            assert col in profile.features
            feat = profile.features[col]
            if col in CATEGORICAL_PREDICTORS:
                assert isinstance(feat, CategoricalFeatureProfile)
                assert feat.data_type == "categorical"
                assert feat.count == 1296675
                assert feat.missing_count == 0
                assert feat.missing_rate == 0.0
                assert len(feat.vocabulary) > 0
                # Probabilities should sum to approximately 1.0
                prob_sum = sum(feat.probabilities.values())
                assert np.isclose(prob_sum, 1.0, atol=1e-3), f"Probabilities for '{col}' sum to {prob_sum}"
            else:
                assert isinstance(feat, NumericalFeatureProfile)
                assert feat.data_type == "numerical"
                assert feat.count == 1296675
                assert feat.missing_count == 0
                assert feat.missing_rate == 0.0
                assert len(feat.bin_edges) == 11, f"Expected 11 bin edges (10 bins), got {len(feat.bin_edges)}"
                assert len(feat.ref_samples) == 5000, f"Expected 5000 reference samples, got {len(feat.ref_samples)}"
                assert feat.min <= feat.mean <= feat.max
                assert feat.std >= 0.0

    def test_prediction_baseline_profile_exists_and_loads(self):
        """Verify baseline_prediction_profile_v1.0.0.json exists and adheres to schema."""
        path = default_monitoring_config.prediction_profile_path
        assert path.exists(), f"Prediction baseline profile not found at {path}"

        profile = PredictionBaselineProfile.load(path)
        assert profile.model_version == "1.0.0"
        assert profile.dataset_name == "val_features.parquet"
        assert profile.dataset_row_count == 277859
        assert profile.sha256_checksum is not None
        assert len(profile.sha256_checksum) == 64

        # Verify 10 model score bins
        assert len(profile.model_score_bins) == 10
        score_prop_sum = sum(b.proportion for b in profile.model_score_bins)
        assert np.isclose(score_prop_sum, 1.0, atol=1e-3)

        # Verify 10 risk score buckets
        assert len(profile.risk_score_buckets) == 10
        risk_prop_sum = sum(b.proportion for b in profile.risk_score_buckets)
        assert np.isclose(risk_prop_sum, 1.0, atol=1e-3)

        # Verify Tier Proportions
        assert set(profile.tier_proportions.keys()) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        tier_sum = sum(profile.tier_proportions.values())
        assert np.isclose(tier_sum, 1.0, atol=1e-3)

        # Verify Action Proportions
        assert set(profile.action_proportions.keys()) == {"APPROVE", "REVIEW", "BLOCK"}
        action_sum = sum(profile.action_proportions.values())
        assert np.isclose(action_sum, 1.0, atol=1e-3)

        # Verify Summary Statistics
        assert 0.0 <= profile.mean_model_score <= 1.0
        assert 0.0 <= profile.mean_risk_score <= 100.0
        assert 0.0 <= profile.is_overridden_rate <= 1.0

    def test_performance_baseline_profile_exists_and_matches_oot_benchmarks(self):
        """Verify baseline_performance_profile_v1.0.0.json matches verified OOT benchmark metrics."""
        path = default_monitoring_config.performance_profile_path
        assert path.exists(), f"Performance baseline profile not found at {path}"

        profile = PerformanceBaselineProfile.load(path)
        assert profile.model_version == "1.0.0"
        assert profile.dataset_name == "test_features.parquet"
        assert profile.dataset_row_count == 277860
        assert profile.total_frauds == 924
        assert profile.total_legitimate == 276936
        assert profile.default_operating_threshold == 0.78
        assert np.isclose(profile.pr_auc, 0.96054, atol=1e-4)
        assert np.isclose(profile.roc_auc, 0.99905, atol=1e-4)

        # Primary operating metrics at tau* = 0.78
        prim = profile.primary_operating_metrics
        assert prim.threshold == 0.78
        assert np.isclose(prim.precision, 0.78187, atol=1e-4)
        assert np.isclose(prim.recall, 0.94264, atol=1e-4)
        assert np.isclose(prim.f1, 0.85476, atol=1e-4)
        assert np.isclose(prim.fpr, 0.00088, atol=1e-4)
        assert prim.confusion_matrix.tp == 871
        assert prim.confusion_matrix.fp == 243
        assert prim.confusion_matrix.fn == 53
        assert prim.confusion_matrix.tn == 276693
        assert prim.confusion_matrix.total == 277860

        # Operational hybrid decision metrics
        ops = profile.operational_decision_metrics
        assert np.isclose(ops.decision_precision_block, 0.78187, atol=1e-4)
        assert np.isclose(ops.decision_recall_intervention, 0.97727, atol=1e-4)
        assert np.isclose(ops.review_queue_purity, 0.01932, atol=1e-4)
        assert ops.total_blocks_count == 1114
        assert ops.fraud_in_block_count == 871
        assert ops.total_reviews_count == 1656
        assert ops.fraud_in_review_count == 32

        # Comparison metrics at tau = 0.50
        comp = profile.comparison_metrics
        assert comp is not None
        assert comp.threshold == 0.50
        assert np.isclose(comp.precision, 0.60858, atol=1e-4)
        assert np.isclose(comp.recall, 0.96753, atol=1e-4)
        assert np.isclose(comp.f1, 0.74718, atol=1e-4)
        assert np.isclose(comp.fpr, 0.00208, atol=1e-4)
        assert comp.confusion_matrix.tp == 894
        assert comp.confusion_matrix.fp == 575
        assert comp.confusion_matrix.fn == 30
        assert comp.confusion_matrix.tn == 276361

    def test_checksum_verification_and_tamper_detection(self):
        """Verify that modifying profile data invalidates computed SHA-256 digest."""
        profile = FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)
        original_checksum = profile.sha256_checksum
        assert original_checksum is not None

        # Re-compute digest on unmodified data
        d = profile.to_dict()
        d["sha256_checksum"] = None
        raw_json = json.dumps(d, indent=2, sort_keys=True)
        computed_checksum = compute_content_sha256(raw_json)
        assert computed_checksum == original_checksum

        # Tamper with row count
        d["dataset_row_count"] = 999999
        tampered_json = json.dumps(d, indent=2, sort_keys=True)
        tampered_checksum = compute_content_sha256(tampered_json)
        assert tampered_checksum != original_checksum

    def test_serialization_roundtrip_equality(self):
        """Verify to_dict() and from_dict() roundtrip preservation across all 3 profile types."""
        # Feature profile
        feat_p = FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)
        feat_reconstructed = FeatureBaselineProfile.from_dict(feat_p.to_dict())
        assert feat_reconstructed.to_dict() == feat_p.to_dict()

        # Prediction profile
        pred_p = PredictionBaselineProfile.load(default_monitoring_config.prediction_profile_path)
        pred_reconstructed = PredictionBaselineProfile.from_dict(pred_p.to_dict())
        assert pred_reconstructed.to_dict() == pred_p.to_dict()

        # Performance profile
        perf_p = PerformanceBaselineProfile.load(default_monitoring_config.performance_profile_path)
        perf_reconstructed = PerformanceBaselineProfile.from_dict(perf_p.to_dict())
        assert perf_reconstructed.to_dict() == perf_p.to_dict()
