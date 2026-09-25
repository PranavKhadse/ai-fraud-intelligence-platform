"""
Governance Schemas, Enums, and Audit Data Models for Phase 14.5.

Defines typed Pydantic models for multi-dimensional promotion gate criteria,
individual evaluation gate outcomes, promotion eligibility assessments, and
immutable human sign-off audit records.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class GateStatus(str, Enum):
    """Evaluation status of an individual promotion governance gate."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class SignOffDecision(str, Enum):
    """Explicit human sign-off decision on candidate promotion eligibility."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class GateEvaluationRule(BaseModel):
    """Definition of an individual governance evaluation rule."""
    model_config = ConfigDict(frozen=True)

    gate_id: str = Field(..., description="Unique machine-readable gate identifier")
    category: str = Field(..., description="Governance dimension category")
    description: str = Field(..., description="Human-readable rule description")
    is_policy_configuration: bool = Field(..., description="True if defined policy constraint, False if existing invariant")
    threshold_value: Union[float, int, str, bool] = Field(..., description="Acceptance threshold value")
    operator: str = Field(..., description="Comparison operator ('>=', '<=', '==', '>', '<')")
    tolerance: Optional[float] = Field(None, description="Optional parity tolerance margin")


class PromotionGateResult(BaseModel):
    """
    Outcome of evaluating a single promotion governance criterion.

    Captures the required threshold, the actual candidate measurement, the
    champion measurement (if comparative), the computed delta, the pass/fail status,
    and whether the rule represents a governance policy or an existing system invariant.
    """
    model_config = ConfigDict(frozen=True)

    gate_id: str = Field(..., description="Unique machine-readable gate identifier")
    category: str = Field(..., description="Governance category (e.g., Statistical Ranking, Operational Safety)")
    metric_name: str = Field(..., description="Target metric name")
    operator: str = Field(..., description="Comparison operator ('>=', '<=', '==', '>', '<')")
    required_value: Union[float, int, str, bool] = Field(..., description="Required threshold or condition value")
    actual_value: Union[float, int, str, bool] = Field(..., description="Actual measured value from candidate evidence")
    champion_value: Optional[Union[float, int, str, bool]] = Field(None, description="Champion reference value if comparative")
    delta_value: Optional[Union[float, int]] = Field(None, description="Computed delta (candidate - champion) if numeric")
    status: GateStatus = Field(..., description="Evaluation outcome: PASS, FAIL, or BLOCKED")
    is_policy_configuration: bool = Field(
        ...,
        description="True if threshold is a defined governance policy constraint; False if an existing repository requirement"
    )
    description: str = Field(..., description="Human-readable description of the governance requirement")
    failure_reason: Optional[str] = Field(None, description="Detailed explanation if gate failed or was blocked")


class PromotionEligibilityAssessment(BaseModel):
    """
    Synthesized outcome of the multi-dimensional promotion gate evaluation.

    Contains individual gate outcomes, summary statistics, cryptographic evidence
    provenance, and the overall binary eligibility decision.
    """
    model_config = ConfigDict(frozen=True)

    candidate_version: str = Field(..., description="Candidate model version evaluated")
    champion_version: str = Field(..., description="Champion model version used as baseline reference")
    is_eligible: bool = Field(..., description="True if and only if all mandatory gates PASS; False otherwise")
    total_gates_evaluated: int = Field(..., ge=0, description="Total number of gates evaluated")
    passed_gates_count: int = Field(..., ge=0, description="Count of gates with status PASS")
    failed_gates_count: int = Field(..., ge=0, description="Count of gates with status FAIL")
    blocked_gates_count: int = Field(..., ge=0, description="Count of gates with status BLOCKED")
    gate_results: List[PromotionGateResult] = Field(..., description="List of individual gate results")
    evidence_artifact_path: str = Field(..., description="Filesystem path to the primary comparison artifact")
    evidence_artifact_sha256: str = Field(..., description="Cryptographic SHA-256 digest of the comparison artifact")
    evaluated_at: str = Field(..., description="ISO 8601 UTC timestamp of governance evaluation")
    assessed_by: str = Field(
        default="promotion_gate_evaluator",
        description="Identifier of the evaluation subsystem executing the assessment"
    )


class PromotionSignOffRecord(BaseModel):
    """
    Immutable audit record capturing an explicit, authorized human sign-off decision.

    Cryptographically links the human decision and mandatory business rationale to
    the exact promotion eligibility assessment and Phase 14.4 comparison evidence.
    """
    model_config = ConfigDict(frozen=True)

    signoff_id: str = Field(..., description="Unique UUID string identifying this sign-off event")
    candidate_version: str = Field(..., description="Candidate model version signed off")
    champion_version: str = Field(..., description="Champion model version referenced")
    actor_id: str = Field(..., min_length=1, description="Authenticated actor identifier authoring the decision")
    actor_role: str = Field(..., min_length=1, description="Role of the authenticated actor (ADMIN, ANALYST)")
    decision: SignOffDecision = Field(..., description="Formal sign-off outcome: APPROVED or REJECTED")
    signoff_rationale: str = Field(
        ...,
        min_length=15,
        description="Mandatory explanatory business rationale (minimum 15 characters)"
    )
    is_eligible_at_signoff: bool = Field(
        ...,
        description="Eligibility status of the candidate at the point of sign-off"
    )
    eligibility_assessment_sha256: str = Field(
        ...,
        description="Cryptographic SHA-256 digest of the PromotionEligibilityAssessment JSON"
    )
    comparison_artifact_sha256: str = Field(
        ...,
        description="Cryptographic SHA-256 digest of the primary Phase 14.4 comparison artifact"
    )
    signed_at: str = Field(..., description="ISO 8601 UTC timestamp of sign-off")
