"""
Phase 5: Imbalance Handling & Cost Optimization Package.

Provides a configurable, cost-sensitive fraud decision optimization framework.
Calculates expected financial and operational costs from model scores and binary decisions.
"""

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
from ml.cost_optimization.oot_evaluation import (
    compute_policy_oot_metrics,
    evaluate_all_approve_baseline,
    compare_oot_policies,
    run_frozen_oot_evaluation,
)

__all__ = [
    "CostConfig",
    "ConfusionMatrixCounts",
    "DecisionCostResult",
    "compute_confusion_matrix_counts",
    "compute_fixed_decision_cost",
    "compute_threshold_cost",
    "generate_threshold_grid",
    "evaluate_threshold_grid_costs",
    "compare_threshold_policies",
    "run_validation_cost_optimization",
    "SensitivityScenario",
    "PREDEFINED_SCENARIOS",
    "run_scenario_sensitivity_analysis",
    "run_cost_ratio_grid_analysis",
    "run_validation_sensitivity_workflow",
    "compute_policy_oot_metrics",
    "evaluate_all_approve_baseline",
    "compare_oot_policies",
    "run_frozen_oot_evaluation",
]
