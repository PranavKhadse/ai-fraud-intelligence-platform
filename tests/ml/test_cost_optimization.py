"""
Automated Unit Tests for Phase 5 Cost Optimization Foundation & Cost Engine.

Validates:
1. CostConfig validation: non-negativity, finite numeric constraints, custom values, and to_dict serialization.
2. ConfusionMatrixCounts calculation: exact TP, TN, FP, FN, total consistency, and edge cases.
3. DecisionCostResult calculation: exact mathematical cost formulas, zero defaults, and average cost per transaction.
4. Threshold binarization: model_score >= threshold logic with continuous score arrays.
5. Error handling and input validation:
   - Mismatched array lengths.
   - Empty input arrays.
   - Invalid negative costs.
   - Non-binary ground truth labels.
   - Non-binary predictions.
   - Negative and non-integer review counts.
   - Non-finite numeric values (NaN, Inf).
6. Deterministic repeatability across multiple invocations.
7. Validation Threshold Grid Generation & Optimization:
   - Deterministic grid spanning 0.01 to 0.99 with exact step 0.01.
   - Explicit presence of Phase 4 threshold 0.94.
   - Identification of Cost-Optimal and F1-Optimal thresholds.
   - Policy comparison calculation (monetary cost savings, fraud caught delta).
   - Zero test-set leakage (exclusively evaluates on Validation data).
   - Artifact structure and serialization schemas.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ml.cost_optimization.config import CostConfig
from ml.cost_optimization.cost_engine import (
    ConfusionMatrixCounts,
    DecisionCostResult,
    compute_confusion_matrix_counts,
    compute_fixed_decision_cost,
    compute_threshold_cost,
)
from ml.cost_optimization.optimize import (
    generate_threshold_grid,
    evaluate_threshold_grid_costs,
    compare_threshold_policies,
    run_validation_cost_optimization,
)
from ml.cost_optimization.sensitivity import (
    SensitivityScenario,
    PREDEFINED_SCENARIOS,
    run_scenario_sensitivity_analysis,
    run_cost_ratio_grid_analysis,
    run_validation_sensitivity_workflow,
)


class TestCostConfig:
    """Tests for CostConfig dataclass and validation."""

    def test_default_values_and_types(self):
        """Verify default illustrative values are configured as expected."""
        cfg = CostConfig()
        assert cfg.false_positive_cost == 15.0
        assert cfg.false_negative_cost == 200.0
        assert cfg.manual_review_cost == 5.0
        assert cfg.true_negative_cost == 0.0
        assert cfg.true_positive_cost == 0.0

    def test_custom_values_assignment(self):
        """Verify custom positive cost assignments."""
        cfg = CostConfig(
            false_positive_cost=25.5,
            false_negative_cost=350.0,
            manual_review_cost=10.0,
            true_negative_cost=0.5,
            true_positive_cost=2.0,
        )
        assert cfg.false_positive_cost == 25.5
        assert cfg.false_negative_cost == 350.0
        assert cfg.manual_review_cost == 10.0
        assert cfg.true_negative_cost == 0.5
        assert cfg.true_positive_cost == 2.0

    def test_negative_cost_raises_value_error(self):
        """Verify that negative cost values are rejected with ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CostConfig(false_positive_cost=-1.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            CostConfig(false_negative_cost=-50.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            CostConfig(manual_review_cost=-5.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            CostConfig(true_negative_cost=-0.01)

        with pytest.raises(ValueError, match="must be non-negative"):
            CostConfig(true_positive_cost=-10.0)

    def test_non_finite_cost_raises_value_error(self):
        """Verify that NaN and Inf costs are rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            CostConfig(false_positive_cost=float("nan"))

        with pytest.raises(ValueError, match="must be finite"):
            CostConfig(false_negative_cost=float("inf"))

    def test_non_numeric_cost_raises_type_error(self):
        """Verify that string or non-numeric types are rejected."""
        with pytest.raises(TypeError, match="must be numeric"):
            CostConfig(false_positive_cost="15.0")  # type: ignore

    def test_to_dict_serialization(self):
        """Verify to_dict returns a valid Python dictionary with expected keys."""
        cfg = CostConfig(false_positive_cost=20.0, false_negative_cost=150.0)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["false_positive_cost"] == 20.0
        assert d["false_negative_cost"] == 150.0
        assert d["manual_review_cost"] == 5.0
        assert d["true_negative_cost"] == 0.0
        assert d["true_positive_cost"] == 0.0


class TestConfusionMatrixCounts:
    """Tests for compute_confusion_matrix_counts and ConfusionMatrixCounts."""

    def test_confusion_matrix_counts_exact(self):
        """Verify exact TP, TN, FP, FN counts on a known array."""
        # y_true: [0, 0, 0, 0, 1, 1, 1, 1]
        # y_pred: [0, 0, 1, 1, 0, 0, 1, 1]
        # -> TN=2, FP=2, FN=2, TP=2, Total=8
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        cm = compute_confusion_matrix_counts(y_true, y_pred)
        assert cm.tp == 2
        assert cm.tn == 2
        assert cm.fp == 2
        assert cm.fn == 2
        assert cm.total == 8

    def test_confusion_matrix_perfect_predictions(self):
        """Verify perfect predictions yield zero FP and zero FN."""
        y_true = [0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 1, 1]
        cm = compute_confusion_matrix_counts(y_true, y_pred)
        assert cm.tp == 2
        assert cm.tn == 3
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.total == 5

    def test_confusion_matrix_all_legitimate(self):
        """Verify behavior when dataset contains only legitimate records."""
        y_true = np.zeros(100, dtype=int)
        y_pred = np.zeros(100, dtype=int)
        y_pred[0:5] = 1  # 5 false alarms

        cm = compute_confusion_matrix_counts(y_true, y_pred)
        assert cm.tn == 95
        assert cm.fp == 5
        assert cm.tp == 0
        assert cm.fn == 0
        assert cm.total == 100

    def test_confusion_matrix_handles_pandas_series(self):
        """Verify Pandas Series inputs are processed cleanly."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 0, 1, 1])
        cm = compute_confusion_matrix_counts(y_true, y_pred)
        assert cm.tp == 1
        assert cm.tn == 1
        assert cm.fp == 1
        assert cm.fn == 1
        assert cm.total == 4

    def test_confusion_matrix_mismatched_lengths_raises_error(self):
        """Verify length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_confusion_matrix_counts([0, 1], [0, 1, 0])

    def test_confusion_matrix_empty_input_raises_error(self):
        """Verify empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_confusion_matrix_counts([], [])

    def test_confusion_matrix_non_binary_values_raises_error(self):
        """Verify non-binary values (e.g. 2, -1, float decimals) raise ValueError."""
        with pytest.raises(ValueError, match="strictly binary"):
            compute_confusion_matrix_counts([0, 1, 2], [0, 1, 1])

        with pytest.raises(ValueError, match="strictly binary"):
            compute_confusion_matrix_counts([0, 1, 1], [0, 0.5, 1])

    def test_confusion_matrix_nan_inf_raises_error(self):
        """Verify NaNs or Infs raise ValueError."""
        with pytest.raises(ValueError, match="NaN or infinite"):
            compute_confusion_matrix_counts([0, 1, np.nan], [0, 1, 0])


class TestDecisionCostCalculations:
    """Tests for compute_fixed_decision_cost and DecisionCostResult."""

    def test_decision_cost_exact_formula_calculation(self):
        """
        Verify exact formula calculation:
        total_cost = FP * CFP + FN * CFN + review_count * Creview + TN * CTN + TP * CTP
        """
        # TN=5, FP=2, FN=3, TP=10, Total=20
        y_true = [0]*7 + [1]*13
        y_pred = [0]*5 + [1]*2 + [0]*3 + [1]*10

        cfg = CostConfig(
            false_positive_cost=15.0,
            false_negative_cost=200.0,
            manual_review_cost=5.0,
            true_negative_cost=1.0,
            true_positive_cost=2.0,
        )
        review_count = 4

        res = compute_fixed_decision_cost(y_true, y_pred, cost_config=cfg, review_count=review_count)

        expected_fp_cost = 2 * 15.0      # 30.0
        expected_fn_cost = 3 * 200.0     # 600.0
        expected_review_cost = 4 * 5.0   # 20.0
        expected_tn_cost = 5 * 1.0       # 5.0
        expected_tp_cost = 10 * 2.0      # 20.0
        expected_total_cost = 30.0 + 600.0 + 20.0 + 5.0 + 20.0  # 675.0
        expected_avg_cost = 675.0 / 20.0                         # 33.75

        assert res.fp_cost == expected_fp_cost
        assert res.fn_cost == expected_fn_cost
        assert res.review_cost == expected_review_cost
        assert res.tn_cost == expected_tn_cost
        assert res.tp_cost == expected_tp_cost
        assert res.total_cost == expected_total_cost
        assert res.average_cost_per_transaction == expected_avg_cost
        assert res.review_count == 4

    def test_zero_cost_defaults(self):
        """Verify default TN and TP costs contribute 0 to total cost."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]  # TN=1, FP=1, FN=1, TP=1

        cfg = CostConfig(false_positive_cost=10.0, false_negative_cost=100.0)
        res = compute_fixed_decision_cost(y_true, y_pred, cost_config=cfg, review_count=0)

        assert res.fp_cost == 10.0
        assert res.fn_cost == 100.0
        assert res.review_cost == 0.0
        assert res.tn_cost == 0.0
        assert res.tp_cost == 0.0
        assert res.total_cost == 110.0
        assert res.average_cost_per_transaction == 110.0 / 4.0

    def test_negative_review_count_raises_value_error(self):
        """Verify negative review_count is rejected."""
        cfg = CostConfig()
        with pytest.raises(ValueError, match="must be non-negative"):
            compute_fixed_decision_cost([0, 1], [0, 1], cost_config=cfg, review_count=-1)

    def test_non_integer_review_count_raises_type_error(self):
        """Verify non-integer review count is rejected."""
        cfg = CostConfig()
        with pytest.raises(TypeError, match="must be an integer"):
            compute_fixed_decision_cost([0, 1], [0, 1], cost_config=cfg, review_count=2.5)  # type: ignore

    def test_invalid_cost_config_type_raises_type_error(self):
        """Verify passing a dictionary instead of CostConfig raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of CostConfig"):
            compute_fixed_decision_cost([0, 1], [0, 1], cost_config={"fp": 10.0})  # type: ignore

    def test_result_to_dict_structure(self):
        """Verify DecisionCostResult to_dict structure."""
        y_true = [0, 1]
        y_pred = [0, 1]
        cfg = CostConfig()
        res = compute_fixed_decision_cost(y_true, y_pred, cost_config=cfg)
        d = res.to_dict()

        assert "confusion_matrix" in d
        assert "total_cost" in d
        assert "average_cost_per_transaction" in d
        assert "cost_breakdown" in d
        assert "review_count" in d
        assert "cost_config" in d
        assert d["total_cost"] == 0.0


class TestThresholdCost:
    """Tests for compute_threshold_cost binarizing model scores."""

    def test_threshold_binarization_logic(self):
        """Verify threshold cutoff at 0.5 binarizes continuous model scores."""
        y_true = np.array([0, 0, 1, 1])
        model_scores = np.array([0.1, 0.49, 0.50, 0.95])

        cfg = CostConfig(false_positive_cost=10.0, false_negative_cost=100.0)
        res = compute_threshold_cost(y_true, model_scores, threshold=0.5, cost_config=cfg)

        # score 0.1 -> pred 0 (TN)
        # score 0.49 -> pred 0 (TN)
        # score 0.50 -> pred 1 (TP)
        # score 0.95 -> pred 1 (TP)
        # Result: TN=2, FP=0, FN=0, TP=2
        assert res.confusion_matrix.tn == 2
        assert res.confusion_matrix.fp == 0
        assert res.confusion_matrix.fn == 0
        assert res.confusion_matrix.tp == 2
        assert res.total_cost == 0.0

    def test_threshold_cost_tradeoff(self):
        """Verify that a high threshold produces more FN and lower FP."""
        y_true = np.array([0, 0, 1, 1])
        model_scores = np.array([0.2, 0.6, 0.7, 0.9])
        cfg = CostConfig(false_positive_cost=10.0, false_negative_cost=100.0)

        # Threshold 0.5:
        # y_pred: [0, 1, 1, 1] -> TN=1, FP=1, FN=0, TP=2 -> cost = 1 * 10 = 10.0
        res_low = compute_threshold_cost(y_true, model_scores, threshold=0.5, cost_config=cfg)
        assert res_low.confusion_matrix.fp == 1
        assert res_low.confusion_matrix.fn == 0
        assert res_low.total_cost == 10.0

        # Threshold 0.8:
        # y_pred: [0, 0, 0, 1] -> TN=2, FP=0, FN=1, TP=1 -> cost = 1 * 100 = 100.0
        res_high = compute_threshold_cost(y_true, model_scores, threshold=0.8, cost_config=cfg)
        assert res_high.confusion_matrix.fp == 0
        assert res_high.confusion_matrix.fn == 1
        assert res_high.total_cost == 100.0

    def test_threshold_cost_non_finite_scores_raises_error(self):
        """Verify scores containing NaN/Inf raise ValueError."""
        cfg = CostConfig()
        with pytest.raises(ValueError, match="NaN or infinite"):
            compute_threshold_cost([0, 1], [0.2, np.nan], threshold=0.5, cost_config=cfg)

    def test_threshold_cost_empty_scores_raises_error(self):
        """Verify empty scores raise ValueError."""
        cfg = CostConfig()
        with pytest.raises(ValueError, match="must not be empty"):
            compute_threshold_cost([], [], threshold=0.5, cost_config=cfg)


class TestDeterministicRepeatability:
    """Tests ensuring calculations are 100% deterministic."""

    def test_repeated_calculations_identical(self):
        """Verify identical results over multiple calls."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], size=1000, p=[0.95, 0.05])
        model_scores = rng.uniform(0.0, 1.0, size=1000)
        cfg = CostConfig(false_positive_cost=15.0, false_negative_cost=250.0, manual_review_cost=7.5)

        res1 = compute_threshold_cost(y_true, model_scores, threshold=0.85, cost_config=cfg, review_count=50)
        res2 = compute_threshold_cost(y_true, model_scores, threshold=0.85, cost_config=cfg, review_count=50)

        assert res1.total_cost == res2.total_cost
        assert res1.average_cost_per_transaction == res2.average_cost_per_transaction
        assert res1.confusion_matrix.to_dict() == res2.confusion_matrix.to_dict()
        assert res1.to_dict() == res2.to_dict()


class TestThresholdOptimizationGrid:
    """Tests for threshold grid generation and boundaries."""

    def test_generate_threshold_grid_boundaries_and_step(self):
        """Verify generated grid spans 0.01 to 0.99 with exact 0.01 step."""
        grid = generate_threshold_grid(start=0.01, stop=0.99, step=0.01, required_points=(0.94,))
        assert len(grid) == 99
        assert grid[0] == 0.01
        assert grid[-1] == 0.99
        assert np.all(np.diff(grid) > 0)  # Monotonically increasing

    def test_explicit_presence_of_threshold_094(self):
        """Verify Phase 4 threshold 0.94 is explicitly present in the grid."""
        grid = generate_threshold_grid(start=0.01, stop=0.99, step=0.01, required_points=(0.94,))
        assert 0.94 in np.round(grid, 4)

    def test_custom_required_points_included(self):
        """Verify arbitrary custom points (e.g. 0.9432) are included."""
        grid = generate_threshold_grid(start=0.1, stop=0.9, step=0.1, required_points=(0.9432, 0.5555))
        assert 0.9432 in np.round(grid, 4)
        assert 0.5555 in np.round(grid, 4)


class TestThresholdGridEvaluation:
    """Tests for evaluate_threshold_grid_costs and policy comparison."""

    def test_evaluate_threshold_grid_known_synthetic(self):
        """Verify grid evaluation identifies optimal thresholds on synthetic data."""
        # 10 samples: 2 fraud, 8 legit
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        model_scores = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.70, 0.80, 0.75, 0.95])

        cfg = CostConfig(false_positive_cost=10.0, false_negative_cost=100.0)

        results = evaluate_threshold_grid_costs(
            y_true=y_true,
            model_scores=model_scores,
            cost_config=cfg,
            thresholds=np.arange(0.1, 1.0, 0.1),
        )

        assert "cost_optimal" in results
        assert "f1_optimal" in results
        assert "threshold_094" in results
        assert len(results["sweep_table"]) == 9

        # At threshold 0.7: y_pred=[0,0,0,0,0,0,1,1,1,1] -> FP=2, FN=0 -> Cost = 2*10 = 20.0
        # At threshold 0.9: y_pred=[0,0,0,0,0,0,0,0,0,1] -> FP=0, FN=1 -> Cost = 1*100 = 100.0
        assert results["cost_optimal"]["total_cost"] <= results["f1_optimal"]["total_cost"]

    def test_compare_threshold_policies_keys_and_savings(self):
        """Verify compare_threshold_policies calculates cost differences accurately."""
        y_true = np.array([0]*95 + [1]*5)
        model_scores = np.linspace(0.01, 0.99, 100)
        cfg = CostConfig(false_positive_cost=15.0, false_negative_cost=200.0)

        sweep = evaluate_threshold_grid_costs(y_true, model_scores, cost_config=cfg)
        comp = compare_threshold_policies(sweep)

        assert "phase4_threshold_094" in comp
        assert "f1_optimal_threshold" in comp
        assert "cost_optimal_threshold" in comp
        assert "optimization_impact" in comp
        assert "cost_assumptions" in comp

        impact = comp["optimization_impact"]
        assert "cost_at_094" in impact
        assert "cost_at_cost_opt" in impact
        assert "absolute_cost_savings" in impact
        assert impact["absolute_cost_savings"] == impact["cost_at_094"] - impact["cost_at_cost_opt"]

    def test_evaluate_threshold_grid_length_mismatch(self):
        """Verify length mismatch raises ValueError."""
        cfg = CostConfig()
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_threshold_grid_costs([0, 1], [0.5], cost_config=cfg)


class TestZeroLeakageAndArtifacts:
    """Tests guaranteeing zero test-set leakage and artifact validity."""

    def test_no_oot_test_set_path_used_by_default(self):
        """Verify default optimization path points strictly to val_features.parquet."""
        import inspect
        sig = inspect.signature(run_validation_cost_optimization)
        default_val_path = sig.parameters["val_path"].default
        assert "val_features.parquet" in str(default_val_path)
        assert "test_features.parquet" not in str(default_val_path)

    def test_persisted_artifacts_schema_and_keys(self):
        """Verify that persisted JSON artifacts exist and have valid structure."""
        sweep_path = Path("ml/cost_optimization/artifacts/threshold_optimization_results.json")
        comp_path = Path("ml/cost_optimization/artifacts/threshold_comparison.json")
        plot_path = Path("docs/figures/cost_vs_threshold.png")
        report_path = Path("docs/cost_optimization_report.md")

        assert sweep_path.exists(), f"Missing artifact: {sweep_path}"
        assert comp_path.exists(), f"Missing artifact: {comp_path}"
        assert plot_path.exists(), f"Missing plot: {plot_path}"
        assert report_path.exists(), f"Missing report: {report_path}"

        with open(sweep_path, "r", encoding="utf-8") as f:
            sweep_data = json.load(f)
        assert "cost_optimal" in sweep_data
        assert "f1_optimal" in sweep_data
        assert "threshold_094" in sweep_data
        assert "sweep_table" in sweep_data
        assert len(sweep_data["sweep_table"]) == 99

        with open(comp_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        assert comp_data["phase4_threshold_094"]["threshold"] == 0.94
        assert comp_data["cost_optimal_threshold"]["threshold"] == 0.78
        assert comp_data["optimization_impact"]["absolute_cost_savings"] == 10585.0


class TestSensitivityAnalysis:
    """Tests for sensitivity scenario evaluation and cost ratio grids."""

    def test_predefined_scenarios_presence_and_costs(self):
        """Verify standard illustrative scenarios A, B, and C exist with correct non-negative costs."""
        scenario_names = [s.name for s in PREDEFINED_SCENARIOS]
        assert "Scenario A (Customer-friendly)" in scenario_names
        assert "Scenario B (Balanced)" in scenario_names
        assert "Scenario C (Fraud-loss-sensitive)" in scenario_names

        for s in PREDEFINED_SCENARIOS:
            assert s.cost_config.false_positive_cost >= 0.0
            assert s.cost_config.false_negative_cost >= 0.0
            assert s.cost_config.manual_review_cost >= 0.0
            assert np.isfinite(s.cost_config.false_positive_cost)
            assert np.isfinite(s.cost_config.false_negative_cost)

    def test_run_scenario_sensitivity_analysis_shifts_thresholds(self):
        """Verify that varying cost ratios shift the optimal threshold monotonically."""
        # 100 samples with 10 frauds
        rng = np.random.default_rng(42)
        y_true = np.array([0]*90 + [1]*10)
        # Give higher scores to fraud but with overlap
        model_scores = np.concatenate([rng.uniform(0.0, 0.7, size=90), rng.uniform(0.3, 0.99, size=10)])

        results = run_scenario_sensitivity_analysis(y_true, model_scores)

        assert "Scenario A (Customer-friendly)" in results
        assert "Scenario B (Balanced)" in results
        assert "Scenario C (Fraud-loss-sensitive)" in results

        tau_a = results["Scenario A (Customer-friendly)"]["cost_optimal"]["threshold"]
        tau_b = results["Scenario B (Balanced)"]["cost_optimal"]["threshold"]
        tau_c = results["Scenario C (Fraud-loss-sensitive)"]["cost_optimal"]["threshold"]

        # Higher penalty on FN (Scenario C) forces lower threshold to catch more fraud
        assert tau_c <= tau_b <= tau_a

    def test_scenario_sensitivity_preserves_094_comparison(self):
        """Verify each scenario contains explicit comparison against Phase 4 threshold 0.94."""
        y_true = np.array([0]*90 + [1]*10)
        model_scores = np.linspace(0.01, 0.99, 100)

        results = run_scenario_sensitivity_analysis(y_true, model_scores)
        for sc_name, sc_data in results.items():
            assert "threshold_094" in sc_data
            assert "comparison" in sc_data
            assert sc_data["threshold_094"]["threshold"] == 0.94
            assert "optimization_impact" in sc_data["comparison"]

    def test_run_cost_ratio_grid_analysis_dimensions(self):
        """Verify 2D grid produces correct matrix dimensions and non-empty records."""
        y_true = np.array([0]*90 + [1]*10)
        model_scores = np.linspace(0.01, 0.99, 100)
        fp_costs = [10.0, 20.0]
        fn_costs = [100.0, 200.0, 500.0]

        grid_res = run_cost_ratio_grid_analysis(
            y_true=y_true,
            model_scores=model_scores,
            fp_costs=fp_costs,
            fn_costs=fn_costs,
        )

        assert len(grid_res["grid_records"]) == 6
        assert len(grid_res["matrix_optimal_threshold"]) == 2
        assert len(grid_res["matrix_optimal_threshold"][0]) == 3

    def test_sensitivity_workflow_no_oot_test_access(self):
        """Verify sensitivity workflow defaults strictly to validation partition."""
        import inspect
        sig = inspect.signature(run_validation_sensitivity_workflow)
        default_val = sig.parameters["val_path"].default
        assert "val_features.parquet" in str(default_val)
        assert "test_features.parquet" not in str(default_val)

    def test_sensitivity_artifacts_schema(self):
        """Verify generated sensitivity JSON and heatmap plot exist and conform to schema."""
        sens_path = Path("ml/cost_optimization/artifacts/sensitivity_analysis.json")
        heatmap_path = Path("docs/figures/cost_sensitivity_heatmap.png")

        assert sens_path.exists(), f"Missing artifact: {sens_path}"
        assert heatmap_path.exists(), f"Missing plot: {heatmap_path}"

        with open(sens_path, "r", encoding="utf-8") as f:
            sens_data = json.load(f)

        assert "scenarios" in sens_data
        assert "cost_ratio_grid" in sens_data
        assert "Scenario A (Customer-friendly)" in sens_data["scenarios"]
        assert "Scenario B (Balanced)" in sens_data["scenarios"]
        assert "Scenario C (Fraud-loss-sensitive)" in sens_data["scenarios"]
        assert len(sens_data["cost_ratio_grid"]["grid_records"]) == 49
