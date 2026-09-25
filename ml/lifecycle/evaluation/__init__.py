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
from ml.lifecycle.evaluation.operational import OperationalDecisionEvaluator
from ml.lifecycle.evaluation.pipeline import FullCandidateEvaluationPipeline
from ml.lifecycle.evaluation.schemas import (
    CandidateEvaluationResult,
    EvaluationScope,
    FullCandidateEvaluationResult,
    GateResult,
    OperationalDecisionMetrics,
    ThresholdOptimizationObjective,
    ThresholdSelectionResult,
    ThresholdSweepPoint,
)
from ml.lifecycle.evaluation.threshold import CandidateThresholdAnalyzer
from ml.lifecycle.evaluation.comparator import (
    ChampionChallengerComparator,
    ChampionImmutabilityViolationError,
    ChampionMetadataDiscrepancyError,
)
from ml.lifecycle.evaluation.comparison_schemas import (
    ChampionChallengerComparisonResult,
    ClassificationComparison,
    DeltaSign,
    LatencyComparison,
    MetricDelta,
    OperationalComparison,
    compute_metric_delta,
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
    # Phase 14.3 Full Evaluation & Threshold Analysis additions
    "CandidateThresholdAnalyzer",
    "OperationalDecisionEvaluator",
    "FullCandidateEvaluationPipeline",
    "ThresholdOptimizationObjective",
    "ThresholdSweepPoint",
    "ThresholdSelectionResult",
    "OperationalDecisionMetrics",
    "FullCandidateEvaluationResult",
    # Phase 14.4 Champion vs. Challenger Comparator additions
    "ChampionChallengerComparator",
    "ChampionImmutabilityViolationError",
    "ChampionMetadataDiscrepancyError",
    "ChampionChallengerComparisonResult",
    "ClassificationComparison",
    "DeltaSign",
    "LatencyComparison",
    "MetricDelta",
    "OperationalComparison",
    "compute_metric_delta",
]

