"""
Standardized Evaluation Metrics Module for Imbalanced Fraud Detection.

Prioritizes PR-AUC (Average Precision) as the primary metric under severe class imbalance (~0.5% fraud rate),
along with ROC-AUC, Precision, Recall, F1-Score, and complete confusion matrix decomposition.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud classification metrics for a given prediction probability vector.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted fraud probabilities P(Fraud | X) in [0, 1].
        threshold: Decision threshold for binarizing probabilities (default: 0.5).
        
    Returns:
        Dict containing PR-AUC, ROC-AUC, Precision, Recall, F1, Accuracy, and Confusion Matrix.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    
    # Assert probability bounds
    assert np.all(y_p >= 0.0) and np.all(y_p <= 1.0), "Predicted probabilities must be bounded within [0, 1]!"
    assert len(y_t) == len(y_p), "Length mismatch between ground truth and predicted probabilities!"
    
    # Binarize predictions at threshold
    y_pred = (y_p >= threshold).astype(np.int64)
    
    # Primary and secondary threshold-independent rank metrics
    pr_auc = float(average_precision_score(y_t, y_p))
    roc_auc = float(roc_auc_score(y_t, y_p))
    
    # Threshold-dependent metrics
    precision = float(precision_score(y_t, y_pred, zero_division=0))
    recall = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_t, y_pred))
    
    # Confusion matrix breakdown
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    fraud_support = int(np.sum(y_t == 1))
    legit_support = int(np.sum(y_t == 0))
    predicted_fraud_count = int(np.sum(y_pred == 1))
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "accuracy": round(accuracy, 5),
        "threshold": round(threshold, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "support": {
            "fraud": fraud_support,
            "legitimate": legit_support,
            "total": len(y_t),
            "predicted_fraud": predicted_fraud_count,
        },
        "rates": {
            "tpr": round(tpr, 5),
            "fpr": round(fpr, 5),
        },
    }
