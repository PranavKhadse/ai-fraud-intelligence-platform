"""
Promotion Governance and Human Sign-Off Module for Phase 14.5.

Exposes:
- PromotionGateSpecification: Typed governance policy parameters.
- PromotionGateEvaluator: Deterministic 11-dimensional gate evaluator.
- HumanSignOffEngine: RBAC-enforced human sign-off engine.
- Governance Schemas: PromotionGateResult, GateStatus, PromotionEligibilityAssessment,
  PromotionSignOffRecord, SignOffDecision.
- Exceptions: UnauthorizedSignOffActorError, PromotionGateBlockedError, SignOffConflictError.
"""

from ml.lifecycle.governance.config import (
    PromotionGateSpecification,
    default_promotion_gate_specification,
)
from ml.lifecycle.governance.evaluator import (
    EvidenceCorruptedError,
    PromotionGateEvaluator,
)
from ml.lifecycle.governance.schemas import (
    GateEvaluationRule,
    GateStatus,
    PromotionEligibilityAssessment,
    PromotionGateResult,
    PromotionSignOffRecord,
    SignOffDecision,
)
from ml.lifecycle.governance.signoff import (
    HumanSignOffEngine,
    PromotionGateBlockedError,
    SignOffConflictError,
    UnauthorizedSignOffActorError,
    compute_assessment_sha256,
)

__all__ = [
    # Config
    "PromotionGateSpecification",
    "default_promotion_gate_specification",
    # Evaluator
    "PromotionGateEvaluator",
    "EvidenceCorruptedError",
    # Sign-Off Engine
    "HumanSignOffEngine",
    "UnauthorizedSignOffActorError",
    "PromotionGateBlockedError",
    "SignOffConflictError",
    "compute_assessment_sha256",
    # Schemas & Enums
    "GateEvaluationRule",
    "GateStatus",
    "PromotionEligibilityAssessment",
    "PromotionGateResult",
    "PromotionSignOffRecord",
    "SignOffDecision",
]
