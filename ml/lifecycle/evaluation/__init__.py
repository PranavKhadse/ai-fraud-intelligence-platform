"""
Candidate Evaluation Module for Phase 14 Model Lifecycle & Controlled Promotion.
"""

from ml.lifecycle.evaluation.evaluator import CandidateModelEvaluator
from ml.lifecycle.evaluation.gates import (
    CandidateGateConfig,
    default_candidate_gate_config,
    evaluate_candidate_gates,
)
from ml.lifecycle.evaluation.metrics import (
    calculate_evaluation_metrics,
    calculate_latency_distribution,
)
from ml.lifecycle.evaluation.schemas import (
    CandidateEvaluationResult,
    EvaluationScope,
    GateResult,
)

__all__ = [
    "CandidateModelEvaluator",
    "CandidateGateConfig",
    "default_candidate_gate_config",
    "evaluate_candidate_gates",
    "calculate_evaluation_metrics",
    "calculate_latency_distribution",
    "CandidateEvaluationResult",
    "EvaluationScope",
    "GateResult",
]
