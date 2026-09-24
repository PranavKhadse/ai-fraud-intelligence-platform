"""
Standardized Evaluation Metrics Engine for Phase 14 Candidate Evaluation.

Calculates comprehensive performance, financial cost, and diagnostic metrics
for severe class-imbalanced fraud detection (~0.4% - 0.5% fraud rate).
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.schemas import EvaluationMetricsSummary


def calculate_evaluation_metrics(
    y_true: Union[np.ndarray, Sequence[int]],
    y_prob: Union[np.ndarray, Sequence[float]],
    threshold: float,
    cost_config: Optional[CostConfig] = None,
    latencies_ms: Optional[Sequence[float]] = None,
) -> EvaluationMetricsSummary:
    """
    Compute strongly typed EvaluationMetricsSummary for model predictions against ground truth.

    Args:
        y_true: Binary ground truth array (0 for legitimate, 1 for fraud).
        y_prob: Model predicted probabilities P(Fraud | X) in [0.0, 1.0].
        threshold: Decision threshold for classification (e.g. 0.78, 0.94).
        cost_config: Optional financial cost parameters (defaults to CostConfig()).
        latencies_ms: Optional measured inference latencies in milliseconds.

    Returns:
        EvaluationMetricsSummary instance populated with all standard metrics.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)

    if len(y_t) == 0:
        raise ValueError("y_true and y_prob cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} samples, y_prob has {len(y_p)} samples."
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Operating threshold must be in (0.0, 1.0), got {threshold}.")
    if np.any(np.isnan(y_p)) or np.any(np.isinf(y_p)):
        raise ValueError("Predicted probabilities contain NaN or Inf values.")
    if np.any(y_p < 0.0) or np.any(y_p > 1.0):
        raise ValueError("Predicted probabilities must be bounded in [0.0, 1.0].")

    # Binarize predictions
    y_pred = (y_p >= threshold).astype(np.int64)

    # Primary rank metrics
    # Handle single-class edge cases gracefully if present
    unique_classes = np.unique(y_t)
    if len(unique_classes) > 1:
        pr_auc = float(average_precision_score(y_t, y_p))
        roc_auc = float(roc_auc_score(y_t, y_p))
    else:
        pr_auc = 1.0 if unique_classes[0] == 1 else 0.0
        roc_auc = 1.0

    # Threshold-dependent metrics
    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_t, y_pred))

    # Confusion matrix breakdown
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total_samples = len(y_t)

    # False Positive Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Financial cost calculation
    cfg = cost_config or CostConfig()
    expected_cost = float(
        fp * cfg.false_positive_cost
        + fn * cfg.false_negative_cost
        + tp * cfg.true_positive_cost
        + tn * cfg.true_negative_cost
    )

    # Operational routing counts
    fraud_in_block = tp
    review_queue_purity = precision

    return EvaluationMetricsSummary(
        pr_auc=round(pr_auc, 5),
        roc_auc=round(roc_auc, 5),
        precision=round(precision, 5),
        recall=round(recall, 5),
        f1=round(f1, 5),
        fpr=round(fpr, 5),
        accuracy=round(accuracy, 5),
        threshold=round(float(threshold), 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total_samples=total_samples,
        expected_cost=round(expected_cost, 2),
        fraud_in_block_count=fraud_in_block,
        fraud_in_review_count=0,
        review_queue_purity=round(review_queue_purity, 5),
    )


def calculate_latency_distribution(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """
    Compute standard summary statistics for inference latencies.
    """
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }
