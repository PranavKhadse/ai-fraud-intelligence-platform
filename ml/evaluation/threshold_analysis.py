"""
Validation-Set Threshold Analysis Module for Fraud Probability Calibration.

Evaluates candidate decision thresholds exclusively on the Validation partition
to identify the optimal decision boundary for operational fraud review.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_classification_metrics


def evaluate_threshold_sweep(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    step: float = 0.01,
) -> Dict[str, Any]:
    """
    Perform a complete threshold sweep from 0.01 to 0.99 on Validation set probabilities.
    
    Args:
        y_val: Validation ground truth binary labels.
        y_val_prob: Validation predicted probabilities P(Fraud | X).
        step: Threshold step resolution (default: 0.01).
        
    Returns:
        Dict containing threshold grid evaluations, best F1 threshold, and operating points.
    """
    thresholds = np.arange(0.01, 1.00, step)
    records = []
    
    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics = {}
    
    for t in thresholds:
        t_val = round(float(t), 4)
        m = compute_classification_metrics(y_val, y_val_prob, threshold=t_val)
        
        row = {
            "threshold": t_val,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "predicted_fraud_count": m["support"]["predicted_fraud"],
            "tp": m["confusion_matrix"]["tp"],
            "fp": m["confusion_matrix"]["fp"],
            "fn": m["confusion_matrix"]["fn"],
            "tn": m["confusion_matrix"]["tn"],
            "fpr": m["rates"]["fpr"],
            "tpr": m["rates"]["tpr"],
        }
        records.append(row)
        
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = t_val
            best_metrics = m

    df_sweep = pd.DataFrame(records)
    
    # Operating point: High Recall (>= 80% recall with highest precision)
    rec_80_candidates = df_sweep[df_sweep["recall"] >= 0.80]
    rec_80_point = rec_80_candidates.sort_values(by="precision", ascending=False).iloc[0].to_dict() if len(rec_80_candidates) > 0 else {}
    
    # Operating point: High Recall (>= 90% recall with highest precision)
    rec_90_candidates = df_sweep[df_sweep["recall"] >= 0.90]
    rec_90_point = rec_90_candidates.sort_values(by="precision", ascending=False).iloc[0].to_dict() if len(rec_90_candidates) > 0 else {}
    
    return {
        "best_f1_threshold": best_threshold,
        "best_f1_value": best_f1,
        "best_f1_metrics": best_metrics,
        "operating_points": {
            "max_f1": best_metrics,
            "recall_80_target": rec_80_point,
            "recall_90_target": rec_90_point,
        },
        "sweep_table": records,
    }
