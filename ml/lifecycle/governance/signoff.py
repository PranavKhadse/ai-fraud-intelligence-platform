"""
Explicit Human Sign-Off Engine for Phase 14.5.

Enforces role-based authorization (RBAC), eligibility gate preconditions, mandatory
governance rationales, and cryptographic audit linkage to record formal human sign-off
decisions without executing model promotion.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from backend.app.core.security import ActorContext
from backend.app.db.models.enums import AuditActorType
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.governance.schemas import (
    GateStatus,
    PromotionEligibilityAssessment,
    PromotionSignOffRecord,
    SignOffDecision,
)


class UnauthorizedSignOffActorError(PermissionError):
    """Raised when the initiating actor does not possess authorized governance credentials."""
    pass


class PromotionGateBlockedError(ValueError):
    """Raised when attempting an APPROVED sign-off on an ineligible candidate."""
    pass


class SignOffConflictError(ValueError):
    """Raised when an attempt is made to overwrite an existing, immutable sign-off record."""
    pass


# Permitted governance roles for human sign-off
PERMITTED_SIGNOFF_ROLES = {
    AuditActorType.ADMIN.value,
    AuditActorType.ANALYST.value,
}


def compute_assessment_sha256(assessment: PromotionEligibilityAssessment) -> str:
    """Compute deterministic SHA-256 digest of a PromotionEligibilityAssessment."""
    canonical_json = json.dumps(assessment.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


class HumanSignOffEngine:
    """
    Governance sign-off engine for recording formal human approval or rejection.

    Guarantees:
    - Role-based access control (strictly ADMIN and ANALYST).
    - Hard precondition enforcement (APPROVED blocked if candidate is ineligible).
    - Immutable sign-off persistence with conflict prevention.
    - Zero promotion or model state side-effects (preserves CANDIDATE status).
    """

    def __init__(
        self,
        lifecycle_config: Optional[LifecycleConfig] = None,
    ) -> None:
        """
        Initialize the Human Sign-Off Engine.

        Args:
            lifecycle_config: Model lifecycle directory and path configuration.
        """
        self.lifecycle_cfg = lifecycle_config or default_lifecycle_config

    def execute_signoff(
        self,
        assessment: PromotionEligibilityAssessment,
        actor: ActorContext,
        decision: Union[str, SignOffDecision],
        rationale: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> PromotionSignOffRecord:
        """
        Validate and record an explicit human governance sign-off.

        Args:
            assessment: Evaluated PromotionEligibilityAssessment.
            actor: Validated ActorContext of the human operator.
            decision: Formal decision (SignOffDecision.APPROVED or SignOffDecision.REJECTED).
            rationale: Mandatory governance explanation (min 15 characters).
            output_path: Optional destination path for the sign-off artifact JSON.

        Returns:
            PromotionSignOffRecord containing full cryptographic audit details.

        Raises:
            UnauthorizedSignOffActorError: If actor lacks required permissions.
            PromotionGateBlockedError: If approval is attempted on ineligible candidate.
            ValueError: If rationale is invalid or missing.
            SignOffConflictError: If a conflicting sign-off already exists at target path.
        """
        # 1. Authorize Actor Role
        actor_role_str = (
            actor.actor_role.value
            if isinstance(actor.actor_role, AuditActorType)
            else str(actor.actor_role).upper()
        )
        if actor_role_str not in PERMITTED_SIGNOFF_ROLES:
            raise UnauthorizedSignOffActorError(
                f"Actor '{actor.actor_id}' with role '{actor_role_str}' is not authorized to sign off on model promotion. "
                f"Required roles: {sorted(list(PERMITTED_SIGNOFF_ROLES))}"
            )

        if not actor.actor_id or not actor.actor_id.strip():
            raise UnauthorizedSignOffActorError("actor_id cannot be empty.")

        # 2. Parse Decision
        norm_decision = (
            decision
            if isinstance(decision, SignOffDecision)
            else SignOffDecision(str(decision).upper())
        )

        # 3. Enforce Eligibility Preconditions
        if norm_decision == SignOffDecision.APPROVED and not assessment.is_eligible:
            failing_gates = [
                g.gate_id for g in assessment.gate_results if g.status in (GateStatus.FAIL, GateStatus.BLOCKED)
            ]
            raise PromotionGateBlockedError(
                f"Cannot execute APPROVED sign-off for candidate '{assessment.candidate_version}'. "
                f"Candidate failed {len(failing_gates)} governance criteria: {failing_gates}."
            )

        # 4. Validate Rationale
        if not rationale or len(rationale.strip()) < 15:
            raise ValueError(
                "signoff_rationale must contain at least 15 non-whitespace characters explaining the governance decision."
            )

        # 5. Compute Hashes
        assessment_sha = compute_assessment_sha256(assessment)
        comp_sha = assessment.evidence_artifact_sha256

        # 6. Generate Immutable Record
        now_utc = datetime.now(timezone.utc).isoformat()
        record = PromotionSignOffRecord(
            signoff_id=str(uuid.uuid4()),
            candidate_version=assessment.candidate_version,
            champion_version=assessment.champion_version,
            actor_id=actor.actor_id.strip(),
            actor_role=actor_role_str,
            decision=norm_decision,
            signoff_rationale=rationale.strip(),
            is_eligible_at_signoff=assessment.is_eligible,
            eligibility_assessment_sha256=assessment_sha,
            comparison_artifact_sha256=comp_sha,
            signed_at=now_utc,
        )

        # 7. Persist with Conflict Prevention
        target_dir = self.lifecycle_cfg.registry_root / "signoffs"
        target_path_p: Path
        if output_path is not None:
            target_path_p = Path(output_path).resolve()
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path_p = target_dir / f"signoff_v{assessment.candidate_version}.json"

        target_path_p.parent.mkdir(parents=True, exist_ok=True)

        if target_path_p.exists():
            # Check for identical duplicate vs conflict
            try:
                with open(target_path_p, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                existing_record = PromotionSignOffRecord.model_validate(existing_data)
                if (
                    existing_record.candidate_version == record.candidate_version
                    and existing_record.decision == record.decision
                    and existing_record.actor_id == record.actor_id
                    and existing_record.eligibility_assessment_sha256 == record.eligibility_assessment_sha256
                ):
                    # Idempotent re-execution
                    return existing_record
                else:
                    raise SignOffConflictError(
                        f"A conflicting sign-off record already exists at '{target_path_p}' for candidate "
                        f"'{assessment.candidate_version}' with decision '{existing_record.decision.value}' "
                        f"by '{existing_record.actor_id}'. Finalized sign-off records are immutable."
                    )
            except (json.JSONDecodeError, ValueError) as e:
                if isinstance(e, SignOffConflictError):
                    raise
                raise SignOffConflictError(
                    f"Target sign-off file '{target_path_p}' already exists and contains invalid or conflicting data: {e}"
                )

        # Atomic Write
        temp_file = target_path_p.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2)
            temp_file.replace(target_path_p)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        return record
