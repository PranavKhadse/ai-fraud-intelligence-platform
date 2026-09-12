"""
Cost Calculation Engine for Phase 5 Imbalance Handling & Cost Optimization.

Calculates confusion matrix counts, component costs, total expected decision cost,
and average cost per transaction under configurable fixed-cost parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, List, Sequence
import numpy as np
import pandas as pd

from ml.cost_optimization.config import CostConfig


@dataclass(frozen=True)
class ConfusionMatrixCounts:
    """
    Decomposed confusion matrix classification counts.

    Attributes:
        tp: True Positives (actual fraud correctly predicted as fraud).
        tn: True Negatives (actual legitimate correctly predicted as legitimate).
        fp: False Positives (actual legitimate incorrectly predicted as fraud).
        fn: False Negatives (actual fraud incorrectly predicted as legitimate).
        total: Total number of evaluated transactions.
    """
    tp: int
    tn: int
    fp: int
    fn: int
    total: int

    def __post_init__(self) -> None:
        """Validate non-negativity and total consistency."""
        for name, count in [("tp", self.tp), ("tn", self.tn), ("fp", self.fp), ("fn", self.fn)]:
            if not isinstance(count, (int, np.integer)):
                raise TypeError(f"Count '{name}' must be an integer, got {type(count).__name__}.")
            if count < 0:
                raise ValueError(f"Count '{name}' must be non-negative (>= 0), got {count}.")
        expected_total = self.tp + self.tn + self.fp + self.fn
        if self.total != expected_total:
            raise ValueError(
                f"Total mismatch: total={self.total} but tp+tn+fp+fn={expected_total}."
            )

    def to_dict(self) -> Dict[str, int]:
        """Convert confusion matrix counts to a dictionary."""
        return {
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "total": int(self.total),
        }


@dataclass(frozen=True)
class DecisionCostResult:
    """
    Complete cost evaluation result for a decision policy.

    Attributes:
        confusion_matrix: Decomposed confusion matrix counts (TP, TN, FP, FN, Total).
        total_cost: Total expected monetary/operational decision cost.
        average_cost_per_transaction: Total cost divided by total transaction count.
        fp_cost: False positive cost component (FP * false_positive_cost).
        fn_cost: False negative cost component (FN * false_negative_cost).
        review_cost: Manual review cost component (review_count * manual_review_cost).
        tn_cost: True negative cost component (TN * true_negative_cost).
        tp_cost: True positive cost component (TP * true_positive_cost).
        review_count: Number of transactions routed to manual investigation.
        cost_config: Configuration used for this evaluation.
    """
    confusion_matrix: ConfusionMatrixCounts
    total_cost: float
    average_cost_per_transaction: float
    fp_cost: float
    fn_cost: float
    review_cost: float
    tn_cost: float
    tp_cost: float
    review_count: int
    cost_config: CostConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a structured serializable dictionary."""
        return {
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "total_cost": round(float(self.total_cost), 4),
            "average_cost_per_transaction": round(float(self.average_cost_per_transaction), 6),
            "cost_breakdown": {
                "fp_cost": round(float(self.fp_cost), 4),
                "fn_cost": round(float(self.fn_cost), 4),
                "review_cost": round(float(self.review_cost), 4),
                "tn_cost": round(float(self.tn_cost), 4),
                "tp_cost": round(float(self.tp_cost), 4),
            },
            "review_count": int(self.review_count),
            "cost_config": self.cost_config.to_dict(),
        }


def _validate_and_convert_binary_array(
    arr: Union[np.ndarray, Sequence[Any], pd.Series],
    param_name: str,
) -> np.ndarray:
    """Validate that an input sequence is non-empty, finite, and strictly binary (0 or 1)."""
    if hasattr(arr, "values"):
        arr_np = np.asarray(arr.values)
    else:
        arr_np = np.asarray(arr)

    if arr_np.size == 0:
        raise ValueError(f"Input '{param_name}' must not be empty.")

    if not np.issubdtype(arr_np.dtype, np.number) and not np.issubdtype(arr_np.dtype, np.bool_):
        raise TypeError(f"Input '{param_name}' must contain numeric binary values (0 or 1), got dtype {arr_np.dtype}.")

    if not np.all(np.isfinite(arr_np)):
        raise ValueError(f"Input '{param_name}' contains NaN or infinite values.")

    # Check that values are strictly binary 0 or 1
    unique_vals = np.unique(arr_np)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"Input '{param_name}' must contain strictly binary values (0 or 1), found distinct values: {unique_vals.tolist()}."
        )

    return arr_np.astype(np.int64)


def compute_confusion_matrix_counts(
    y_true: Union[np.ndarray, Sequence[int], pd.Series],
    y_pred: Union[np.ndarray, Sequence[int], pd.Series],
) -> ConfusionMatrixCounts:
    """
    Compute decomposed confusion matrix counts (TP, TN, FP, FN, Total).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Binary model predictions (0 or 1).

    Returns:
        ConfusionMatrixCounts object with verified counts.

    Raises:
        ValueError: If array lengths mismatch, inputs are empty, contain non-binary values, or NaNs.
    """
    y_t = _validate_and_convert_binary_array(y_true, "y_true")
    y_p = _validate_and_convert_binary_array(y_pred, "y_pred")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch between y_true ({len(y_t)}) and y_pred ({len(y_p)})."
        )

    # Vectorized boolean arithmetic
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    total = len(y_t)

    return ConfusionMatrixCounts(tp=tp, tn=tn, fp=fp, fn=fn, total=total)


def compute_fixed_decision_cost(
    y_true: Union[np.ndarray, Sequence[int], pd.Series],
    y_pred: Union[np.ndarray, Sequence[int], pd.Series],
    cost_config: CostConfig,
    review_count: int = 0,
) -> DecisionCostResult:
    """
    Calculate the total and average decision cost for binary predictions.

    Formula:
        total_cost = (
            FP * false_positive_cost
            + FN * false_negative_cost
            + review_count * manual_review_cost
            + TN * true_negative_cost
            + TP * true_positive_cost
        )
        average_cost_per_transaction = total_cost / total_transactions

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Binary predictions (0 or 1).
        cost_config: Configurable decision cost parameters.
        review_count: Number of manual investigations performed (default: 0).
                      Must be explicitly provided; not inferred from y_pred.

    Returns:
        DecisionCostResult containing full cost decomposition.

    Raises:
        ValueError: If review_count is negative or invalid, or if inputs are invalid.
    """
    if not isinstance(cost_config, CostConfig):
        raise TypeError(f"'cost_config' must be an instance of CostConfig, got {type(cost_config).__name__}.")

    if not isinstance(review_count, (int, np.integer)):
        raise TypeError(f"'review_count' must be an integer, got {type(review_count).__name__}.")

    if review_count < 0:
        raise ValueError(f"'review_count' must be non-negative (>= 0), got {review_count}.")

    cm = compute_confusion_matrix_counts(y_true, y_pred)

    fp_cost = float(cm.fp * cost_config.false_positive_cost)
    fn_cost = float(cm.fn * cost_config.false_negative_cost)
    review_cost = float(review_count * cost_config.manual_review_cost)
    tn_cost = float(cm.tn * cost_config.true_negative_cost)
    tp_cost = float(cm.tp * cost_config.true_positive_cost)

    total_cost = fp_cost + fn_cost + review_cost + tn_cost + tp_cost
    avg_cost = total_cost / cm.total if cm.total > 0 else 0.0

    return DecisionCostResult(
        confusion_matrix=cm,
        total_cost=total_cost,
        average_cost_per_transaction=avg_cost,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        review_cost=review_cost,
        tn_cost=tn_cost,
        tp_cost=tp_cost,
        review_count=int(review_count),
        cost_config=cost_config,
    )


def compute_threshold_cost(
    y_true: Union[np.ndarray, Sequence[int], pd.Series],
    model_scores: Union[np.ndarray, Sequence[float], pd.Series],
    threshold: float,
    cost_config: CostConfig,
    review_count: int = 0,
) -> DecisionCostResult:
    """
    Evaluate the decision cost for continuous model scores binarized at a decision threshold.

    Decision Rule:
        y_pred = 1 if model_score >= threshold else 0

    Args:
        y_true: Ground truth binary labels (0 or 1).
        model_scores: Continuous model output scores (e.g., raw XGBoost prediction scores).
        threshold: Decision cutoff threshold.
        cost_config: Configurable decision cost parameters.
        review_count: Explicit manual review count (default: 0).

    Returns:
        DecisionCostResult for the specified threshold.
    """
    if hasattr(model_scores, "values"):
        scores_np = np.asarray(model_scores.values, dtype=np.float64)
    else:
        scores_np = np.asarray(model_scores, dtype=np.float64)

    if scores_np.size == 0:
        raise ValueError("Input 'model_scores' must not be empty.")

    if not np.all(np.isfinite(scores_np)):
        raise ValueError("Input 'model_scores' contains NaN or infinite values.")

    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError(f"'threshold' must be a finite numeric float, got {threshold}.")

    y_pred = (scores_np >= threshold).astype(np.int64)

    return compute_fixed_decision_cost(
        y_true=y_true,
        y_pred=y_pred,
        cost_config=cost_config,
        review_count=review_count,
    )
