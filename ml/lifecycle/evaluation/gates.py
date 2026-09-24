"""
Objective Quality Gates for Phase 14 Candidate Model Evaluation.

Evaluates candidate models against explicit, configurable quality thresholds
across ranking power (PR-AUC, ROC-AUC), detection effectiveness (Recall, Precision, F1),
operational safety (FPR, Max Cost), and production latency.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from ml.lifecycle.evaluation.schemas import GateResult
from ml.lifecycle.schemas import EvaluationMetricsSummary


class CandidateGateConfig(BaseModel):
    """
    Configurable objective gate criteria for candidate model evaluation.
    All defaults reflect the project's established performance baseline standards.
    """
    model_config = ConfigDict(frozen=True)

    min_pr_auc: Optional[float] = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable PR-AUC (Average Precision) under class imbalance",
    )
    min_roc_auc: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable ROC-AUC discrimination power",
    )
    min_recall: Optional[float] = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable fraud catch rate (Recall) at operating threshold",
    )
    min_precision: Optional[float] = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable precision at operating threshold",
    )
    min_f1: Optional[float] = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable harmonic mean F1-Score",
    )
    max_fpr: Optional[float] = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable False Positive Rate at operating threshold",
    )
    max_expected_cost: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional maximum total financial decision cost",
    )
    max_p95_latency_ms: Optional[float] = Field(
        default=50.0,
        ge=0.0,
        description="Maximum acceptable p95 inference latency in milliseconds",
    )


default_candidate_gate_config = CandidateGateConfig()


def evaluate_candidate_gates(
    metrics: EvaluationMetricsSummary,
    gate_config: Optional[CandidateGateConfig] = None,
    latency_stats: Optional[Dict[str, float]] = None,
) -> Tuple[bool, List[GateResult]]:
    """
    Evaluate candidate model metrics against configured objective quality gates.

    Args:
        metrics: Computed EvaluationMetricsSummary for the candidate.
        gate_config: Optional gate configuration (defaults to default_candidate_gate_config).
        latency_stats: Optional latency distribution metrics (e.g. {"p95_ms": 12.4}).

    Returns:
        Tuple[bool, List[GateResult]]: (overall_passed, list of individual gate results).
    """
    cfg = gate_config or default_candidate_gate_config
    gate_results: List[GateResult] = []

    # 1. PR-AUC Gate
    if cfg.min_pr_auc is not None:
        passed = metrics.pr_auc >= cfg.min_pr_auc
        gate_results.append(
            GateResult(
                gate_name="gate_min_pr_auc",
                metric_name="pr_auc",
                operator=">=",
                target_value=cfg.min_pr_auc,
                actual_value=metrics.pr_auc,
                passed=passed,
                description=f"PR-AUC must be >= {cfg.min_pr_auc:.4f} (achieved {metrics.pr_auc:.4f})",
            )
        )

    # 2. ROC-AUC Gate
    if cfg.min_roc_auc is not None:
        passed = metrics.roc_auc >= cfg.min_roc_auc
        gate_results.append(
            GateResult(
                gate_name="gate_min_roc_auc",
                metric_name="roc_auc",
                operator=">=",
                target_value=cfg.min_roc_auc,
                actual_value=metrics.roc_auc,
                passed=passed,
                description=f"ROC-AUC must be >= {cfg.min_roc_auc:.4f} (achieved {metrics.roc_auc:.4f})",
            )
        )

    # 3. Recall Gate
    if cfg.min_recall is not None:
        passed = metrics.recall >= cfg.min_recall
        gate_results.append(
            GateResult(
                gate_name="gate_min_recall",
                metric_name="recall",
                operator=">=",
                target_value=cfg.min_recall,
                actual_value=metrics.recall,
                passed=passed,
                description=f"Recall (Fraud Catch Rate) must be >= {cfg.min_recall:.4f} (achieved {metrics.recall:.4f})",
            )
        )

    # 4. Precision Gate
    if cfg.min_precision is not None:
        passed = metrics.precision >= cfg.min_precision
        gate_results.append(
            GateResult(
                gate_name="gate_min_precision",
                metric_name="precision",
                operator=">=",
                target_value=cfg.min_precision,
                actual_value=metrics.precision,
                passed=passed,
                description=f"Precision must be >= {cfg.min_precision:.4f} (achieved {metrics.precision:.4f})",
            )
        )

    # 5. F1-Score Gate
    if cfg.min_f1 is not None:
        passed = metrics.f1 >= cfg.min_f1
        gate_results.append(
            GateResult(
                gate_name="gate_min_f1",
                metric_name="f1",
                operator=">=",
                target_value=cfg.min_f1,
                actual_value=metrics.f1,
                passed=passed,
                description=f"F1-Score must be >= {cfg.min_f1:.4f} (achieved {metrics.f1:.4f})",
            )
        )

    # 6. Maximum False Positive Rate Gate
    if cfg.max_fpr is not None:
        passed = metrics.fpr <= cfg.max_fpr
        gate_results.append(
            GateResult(
                gate_name="gate_max_fpr",
                metric_name="fpr",
                operator="<=",
                target_value=cfg.max_fpr,
                actual_value=metrics.fpr,
                passed=passed,
                description=f"False Positive Rate must be <= {cfg.max_fpr:.4f} (achieved {metrics.fpr:.4f})",
            )
        )

    # 7. Maximum Expected Cost Gate (Optional)
    if cfg.max_expected_cost is not None and metrics.expected_cost is not None:
        passed = metrics.expected_cost <= cfg.max_expected_cost
        gate_results.append(
            GateResult(
                gate_name="gate_max_expected_cost",
                metric_name="expected_cost",
                operator="<=",
                target_value=cfg.max_expected_cost,
                actual_value=metrics.expected_cost,
                passed=passed,
                description=f"Expected cost must be <= ${cfg.max_expected_cost:,.2f} (achieved ${metrics.expected_cost:,.2f})",
            )
        )

    # 8. Maximum Latency Gate
    if cfg.max_p95_latency_ms is not None and latency_stats is not None:
        actual_p95 = latency_stats.get("p95_ms", 0.0)
        passed = actual_p95 <= cfg.max_p95_latency_ms
        gate_results.append(
            GateResult(
                gate_name="gate_max_p95_latency",
                metric_name="p95_latency_ms",
                operator="<=",
                target_value=cfg.max_p95_latency_ms,
                actual_value=actual_p95,
                passed=passed,
                description=f"p95 inference latency must be <= {cfg.max_p95_latency_ms:.2f}ms (achieved {actual_p95:.2f}ms)",
            )
        )

    overall_passed = len(gate_results) > 0 and all(g.passed for g in gate_results)
    return overall_passed, gate_results
