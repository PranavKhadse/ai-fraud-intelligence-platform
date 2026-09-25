"""
Candidate Threshold Analyzer for Phase 14.3.

Performs deterministic threshold sweeps on validation probabilities, computes financial cost curves,
and selects optimal operating threshold tau* under configurable objectives and deterministic tie-breaking.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.evaluation.schemas import (
    ThresholdOptimizationObjective,
    ThresholdSelectionResult,
    ThresholdSweepPoint,
)


class CandidateThresholdAnalyzer:
    """
    Validation-set threshold analyzer and cost optimization engine for candidate models.
    """

    def __init__(
        self,
        cost_config: Optional[CostConfig] = None,
        step: float = 0.01,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
    ) -> None:
        """
        Initialize the candidate threshold analyzer.

        Args:
            cost_config: Decision cost parameters for calculating expected financial cost.
            step: Grid resolution step (default: 0.01).
            min_threshold: Minimum threshold in sweep grid (default: 0.01).
            max_threshold: Maximum threshold in sweep grid (default: 0.99).
        """
        self.cost_config = cost_config or CostConfig()
        self.step = float(step)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)

        if not (0.0 < self.min_threshold < self.max_threshold < 1.0):
            raise ValueError(
                f"Invalid threshold bounds: min={self.min_threshold}, max={self.max_threshold}. Must satisfy 0 < min < max < 1."
            )
        if self.step <= 0.0 or self.step >= (self.max_threshold - self.min_threshold):
            raise ValueError(f"Invalid threshold step {self.step}.")

    def evaluate_sweep(
        self,
        y_true: Union[np.ndarray, Sequence[int]],
        y_prob: Union[np.ndarray, Sequence[float]],
        objective: ThresholdOptimizationObjective = ThresholdOptimizationObjective.MIN_EXPECTED_COST,
    ) -> ThresholdSelectionResult:
        """
        Perform a full threshold sweep on validation ground truth and predicted probabilities,
        and select the optimal operating threshold tau* with deterministic tie-breaking.

        Args:
            y_true: Ground truth binary labels (0 for legitimate, 1 for fraud).
            y_prob: Model predicted probabilities P(Fraud | X) in [0.0, 1.0].
            objective: Optimization criterion for threshold selection.

        Returns:
            ThresholdSelectionResult containing selected tau*, metrics at tau*, and full sweep table.
        """
        y_t = np.asarray(y_true, dtype=np.int64)
        y_p = np.asarray(y_prob, dtype=np.float64)

        if len(y_t) == 0 or len(y_p) == 0:
            raise ValueError("Evaluation arrays y_true and y_prob cannot be empty.")
        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_prob)={len(y_p)}.")
        if np.any(np.isnan(y_p)) or np.any(np.isinf(y_p)):
            raise ValueError("Predicted probabilities contain NaN or Inf values.")
        if np.any(y_p < 0.0) or np.any(y_p > 1.0):
            raise ValueError("Predicted probabilities must be bounded in [0.0, 1.0].")

        # Generate deterministic threshold grid strictly bounded in [min_threshold, max_threshold]
        grid = np.arange(self.min_threshold, self.max_threshold + 1e-9, self.step)
        grid = np.array(
            sorted(
                set(
                    round(float(t), 4)
                    for t in grid
                    if self.min_threshold <= round(float(t), 4) <= self.max_threshold
                )
            ),
            dtype=np.float64,
        )

        sweep_points: List[ThresholdSweepPoint] = []
        sweep_records: List[Dict[str, Any]] = []

        cfg = self.cost_config

        for t in grid:
            t_val = round(float(t), 4)
            y_pred = (y_p >= t_val).astype(np.int64)

            cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            prec = float(precision_score(y_t, y_pred, zero_division=0))
            rec = float(recall_score(y_t, y_pred, zero_division=0))
            f1 = float(f1_score(y_t, y_pred, zero_division=0))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

            expected_cost = float(
                fp * cfg.false_positive_cost
                + fn * cfg.false_negative_cost
                + tp * cfg.true_positive_cost
                + tn * cfg.true_negative_cost
            )

            pt = ThresholdSweepPoint(
                threshold=t_val,
                expected_cost=round(expected_cost, 2),
                precision=round(prec, 5),
                recall=round(rec, 5),
                f1=round(f1, 5),
                fpr=round(fpr, 5),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
            sweep_points.append(pt)
            sweep_records.append(pt.model_dump())

        # Select tau* according to objective with deterministic tie-breaking
        selected_pt, tie_notes = self._select_optimal_point(sweep_points, objective)

        return ThresholdSelectionResult(
            selected_threshold=selected_pt.threshold,
            optimization_objective=objective,
            best_cost=selected_pt.expected_cost,
            best_f1=selected_pt.f1,
            best_recall=selected_pt.recall,
            best_precision=selected_pt.precision,
            sweep_table=sweep_records,
            tie_breaking_notes=tie_notes,
            evaluated_samples=len(y_t),
            positive_fraud_count=int(np.sum(y_t == 1)),
        )

    def _select_optimal_point(
        self,
        points: List[ThresholdSweepPoint],
        objective: ThresholdOptimizationObjective,
    ) -> tuple[ThresholdSweepPoint, Optional[str]]:
        """
        Select best ThresholdSweepPoint under the chosen objective with deterministic tie-breaking.
        """
        if not points:
            raise ValueError("No sweep points available to select optimal threshold.")

        tie_notes: Optional[str] = None

        if objective == ThresholdOptimizationObjective.MIN_EXPECTED_COST:
            min_cost = min(p.expected_cost for p in points)
            # Find all candidates within epsilon = 1e-4 of min cost
            candidates = [p for p in points if abs(p.expected_cost - min_cost) <= 1e-4]
            if len(candidates) > 1:
                # Tie-breaker 1: Highest recall (catch more fraud for equal cost)
                max_rec = max(p.recall for p in candidates)
                rec_candidates = [p for p in candidates if abs(p.recall - max_rec) <= 1e-5]
                # Tie-breaker 2: Higher threshold (more conservative against FPs)
                selected = max(rec_candidates, key=lambda p: p.threshold)
                tie_notes = (
                    f"Tie-breaking applied across {len(candidates)} points with cost ${min_cost:,.2f}. "
                    f"Selected tau={selected.threshold} (recall={selected.recall:.4f})."
                )
            else:
                selected = candidates[0]

        elif objective == ThresholdOptimizationObjective.MAX_F1:
            max_f1 = max(p.f1 for p in points)
            candidates = [p for p in points if abs(p.f1 - max_f1) <= 1e-5]
            if len(candidates) > 1:
                # Tie-breaker: Highest recall, then higher threshold
                max_rec = max(p.recall for p in candidates)
                rec_candidates = [p for p in candidates if abs(p.recall - max_rec) <= 1e-5]
                selected = max(rec_candidates, key=lambda p: p.threshold)
                tie_notes = f"Tie-breaking applied across {len(candidates)} points with F1 {max_f1:.4f}."
            else:
                selected = candidates[0]

        elif objective == ThresholdOptimizationObjective.TARGET_RECALL_80:
            rec_candidates = [p for p in points if p.recall >= 0.80]
            if rec_candidates:
                # Minimize cost among candidates reaching target recall
                min_cost = min(p.expected_cost for p in rec_candidates)
                cost_candidates = [p for p in rec_candidates if abs(p.expected_cost - min_cost) <= 1e-4]
                selected = max(cost_candidates, key=lambda p: p.threshold)
                tie_notes = f"Target Recall >= 0.80 met. Selected lowest cost point tau={selected.threshold}."
            else:
                # Fallback to max recall point
                selected = max(points, key=lambda p: (p.recall, -p.expected_cost))
                tie_notes = "Target Recall >= 0.80 not achievable; selected highest achievable recall."

        elif objective == ThresholdOptimizationObjective.TARGET_RECALL_90:
            rec_candidates = [p for p in points if p.recall >= 0.90]
            if rec_candidates:
                min_cost = min(p.expected_cost for p in rec_candidates)
                cost_candidates = [p for p in rec_candidates if abs(p.expected_cost - min_cost) <= 1e-4]
                selected = max(cost_candidates, key=lambda p: p.threshold)
                tie_notes = f"Target Recall >= 0.90 met. Selected lowest cost point tau={selected.threshold}."
            else:
                selected = max(points, key=lambda p: (p.recall, -p.expected_cost))
                tie_notes = "Target Recall >= 0.90 not achievable; selected highest achievable recall."

        else:
            selected = min(points, key=lambda p: p.expected_cost)

        return selected, tie_notes
