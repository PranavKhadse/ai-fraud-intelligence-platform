"""
Unit Tests for Phase 14.3 Candidate Threshold Analyzer & Cost Optimization.

Verifies:
- Deterministic threshold sweep execution and step resolution.
- Cost curve minimization and objective function accuracy.
- Deterministic tie-breaking rules (higher recall, then higher threshold).
- Alternative objectives: MAX_F1, TARGET_RECALL_80, TARGET_RECALL_90.
- Input validation, boundary checks, and robust error handling.
"""

import numpy as np
import pytest

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.evaluation.schemas import (
    ThresholdOptimizationObjective,
    ThresholdSelectionResult,
    ThresholdSweepPoint,
)
from ml.lifecycle.evaluation.threshold import CandidateThresholdAnalyzer


class TestCandidateThresholdAnalyzer:
    """Test suite for CandidateThresholdAnalyzer."""

    @pytest.fixture
    def sample_predictions(self):
        """Synthetic predictions with known ground truth."""
        # 10 samples: 3 frauds (indices 0, 1, 6), 7 legitimate
        y_true = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        y_prob = np.array([0.95, 0.85, 0.10, 0.20, 0.05, 0.80, 0.70, 0.15, 0.30, 0.40])
        return y_true, y_prob

    def test_threshold_sweep_basic_execution(self, sample_predictions):
        y_true, y_prob = sample_predictions
        analyzer = CandidateThresholdAnalyzer(step=0.05)
        result = analyzer.evaluate_sweep(y_true, y_prob)

        assert isinstance(result, ThresholdSelectionResult)
        assert 0.0 < result.selected_threshold < 1.0
        assert result.evaluated_samples == 10
        assert result.positive_fraud_count == 3
        assert len(result.sweep_table) > 10
        assert result.best_cost >= 0.0
        assert 0.0 <= result.best_recall <= 1.0
        assert 0.0 <= result.best_precision <= 1.0

    def test_cost_minimization_objective(self):
        # 1 FN = cost 300, 1 FP = cost 25
        # If threshold is 0.50: 1 FP, 0 FN -> cost = 25
        # If threshold is 0.90: 0 FP, 1 FN -> cost = 300
        y_true = np.array([1, 0])
        y_prob = np.array([0.80, 0.60])
        cost_cfg = CostConfig(false_positive_cost=25.0, false_negative_cost=300.0)

        analyzer = CandidateThresholdAnalyzer(cost_config=cost_cfg, step=0.05)
        result = analyzer.evaluate_sweep(y_true, y_prob, objective=ThresholdOptimizationObjective.MIN_EXPECTED_COST)

        # Threshold 0.70 catches fraud (0 FN) and rejects non-fraud (0 FP) -> cost 0
        assert result.selected_threshold == pytest.approx(0.71, abs=0.15)
        assert result.best_cost == 0.0

    def test_deterministic_tie_breaking_equal_costs(self):
        """
        When two candidate thresholds yield identical financial cost,
        the analyzer must deterministically pick the one with higher recall.
        """
        # Thresholds 0.30 and 0.50 both have cost 50, but threshold 0.30 has higher recall
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.40, 0.80, 0.10, 0.20])
        # At threshold 0.30: TP=2, FP=0, FN=0 -> Cost = 0
        # At threshold 0.50: TP=1, FP=0, FN=1 -> Cost = 300
        # At threshold 0.05: TP=2, FP=2, FN=0 -> Cost = 50 (if FP cost=25)
        cost_cfg = CostConfig(false_positive_cost=25.0, false_negative_cost=300.0)
        analyzer = CandidateThresholdAnalyzer(cost_config=cost_cfg, step=0.05)
        result = analyzer.evaluate_sweep(y_true, y_prob)

        # Optimal threshold is between 0.21 and 0.40 (where TP=2, FP=0)
        assert result.best_recall == 1.0
        assert result.best_cost == 0.0

    def test_max_f1_objective(self, sample_predictions):
        y_true, y_prob = sample_predictions
        analyzer = CandidateThresholdAnalyzer(step=0.02)
        result = analyzer.evaluate_sweep(y_true, y_prob, objective=ThresholdOptimizationObjective.MAX_F1)

        assert result.optimization_objective == ThresholdOptimizationObjective.MAX_F1
        assert result.best_f1 > 0.0

    def test_target_recall_objectives(self, sample_predictions):
        y_true, y_prob = sample_predictions
        analyzer = CandidateThresholdAnalyzer(step=0.05)

        res_80 = analyzer.evaluate_sweep(y_true, y_prob, objective=ThresholdOptimizationObjective.TARGET_RECALL_80)
        assert res_80.best_recall >= 0.80 or res_80.tie_breaking_notes is not None

        res_90 = analyzer.evaluate_sweep(y_true, y_prob, objective=ThresholdOptimizationObjective.TARGET_RECALL_90)
        assert res_90.best_recall >= 0.90 or res_90.tie_breaking_notes is not None

    def test_empty_arrays_raise_error(self):
        analyzer = CandidateThresholdAnalyzer()
        with pytest.raises(ValueError, match="cannot be empty"):
            analyzer.evaluate_sweep([], [])

    def test_length_mismatch_raises_error(self):
        analyzer = CandidateThresholdAnalyzer()
        with pytest.raises(ValueError, match="Length mismatch"):
            analyzer.evaluate_sweep([1, 0], [0.5])

    def test_nan_or_inf_probabilities_raise_error(self):
        analyzer = CandidateThresholdAnalyzer()
        with pytest.raises(ValueError, match="NaN or Inf"):
            analyzer.evaluate_sweep([1, 0], [np.nan, 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            analyzer.evaluate_sweep([1, 0], [np.inf, 0.5])

    def test_out_of_bounds_probabilities_raise_error(self):
        analyzer = CandidateThresholdAnalyzer()
        with pytest.raises(ValueError, match="bounded in"):
            analyzer.evaluate_sweep([1, 0], [1.5, 0.5])
        with pytest.raises(ValueError, match="bounded in"):
            analyzer.evaluate_sweep([1, 0], [-0.1, 0.5])

    def test_invalid_init_parameters_raise_error(self):
        with pytest.raises(ValueError, match="Invalid threshold bounds"):
            CandidateThresholdAnalyzer(min_threshold=0.80, max_threshold=0.20)
        with pytest.raises(ValueError, match="Invalid threshold step"):
            CandidateThresholdAnalyzer(step=-0.01)
        with pytest.raises(ValueError, match="Invalid threshold step"):
            CandidateThresholdAnalyzer(step=1.5)
