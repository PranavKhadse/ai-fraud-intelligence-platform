"""
Promotion & Rollback Schemas, Enums, and State Models for Phase 14.6.

Defines typed Pydantic models for the durable promotion operation lifecycle,
journal states, execution records, results, and custom domain exceptions.
"""

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class PromotionOperationState(str, Enum):
    """
    Durable lifecycle states for model promotion and rollback operations.
    Enables crash detection, idempotent resumption, and safe compensation.
    """
    REQUESTED = "REQUESTED"                      # Preconditions evaluated, row locks requested
    VALIDATED = "VALIDATED"                      # Sign-off, hashes, and preconditions verified
    STAGED = "STAGED"                            # Bundle copied to staging & isolated inference verified
    FILESYSTEM_SWAPPED = "FILESYSTEM_SWAPPED"    # Active files swapped, pending DB commit
    DB_COMMITTED = "DB_COMMITTED"                # PostgreSQL transaction committed
    FINALIZED = "FINALIZED"                      # Post-commit verification passed, execution record written
    FAILED_BEFORE_COMMIT = "FAILED_BEFORE_COMMIT"# Failed before commit, compensated safely
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"      # Post-commit inconsistency; requires explicit recovery
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"    # Successfully reconciled after crash or failure


# Pre-commit states where filesystem rollback/compensation is safe and appropriate
PRE_COMMIT_STATES = {
    PromotionOperationState.REQUESTED,
    PromotionOperationState.VALIDATED,
    PromotionOperationState.STAGED,
    PromotionOperationState.FILESYSTEM_SWAPPED,
    PromotionOperationState.FAILED_BEFORE_COMMIT,
}

# Post-commit states where database state is authoritative
POST_COMMIT_STATES = {
    PromotionOperationState.DB_COMMITTED,
    PromotionOperationState.FINALIZED,
    PromotionOperationState.RECOVERY_REQUIRED,
    PromotionOperationState.RECOVERY_COMPLETED,
}


class PromotionOperationJournal(BaseModel):
    """
    Durable on-disk journal tracking the exact progression of a promotion or rollback operation.
    Persisted to ml/models/registry/operations/{operation_id}.json
    """
    model_config = ConfigDict(frozen=False)

    operation_id: str = Field(..., description="Unique operation identifier (e.g. promo_1.1.0_a1b2c3d4)")
    operation_type: str = Field(..., description="Type of operation: PROMOTION or ROLLBACK")
    candidate_version: str = Field(..., description="Target model version being activated")
    champion_version: str = Field(..., description="Current/previous active Champion version")
    current_state: PromotionOperationState = Field(..., description="Current operational lifecycle state")
    state_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Chronological record of state transitions with ISO timestamps"
    )
    backup_path: Optional[str] = Field(None, description="Path to active artifact backup directory")
    staging_path: Optional[str] = Field(None, description="Path to isolated staging directory")
    actor_id: str = Field(..., description="Actor ID executing the promotion/rollback")
    actor_role: str = Field(..., description="Role of the executing actor (ADMIN, ANALYST)")
    rationale: str = Field(..., description="Explanatory business rationale")
    sign_off_id: Optional[str] = Field(None, description="Linked human sign-off UUID")
    sign_off_sha256: Optional[str] = Field(None, description="SHA-256 digest of the sign-off record")
    assessment_sha256: Optional[str] = Field(None, description="SHA-256 digest of the eligibility assessment")
    comparison_sha256: Optional[str] = Field(None, description="SHA-256 digest of the Phase 14.4 comparison artifact")
    candidate_model_sha256: Optional[str] = Field(None, description="SHA-256 of candidate model.joblib")
    candidate_preprocessor_sha256: Optional[str] = Field(None, description="SHA-256 of candidate preprocessor.joblib")
    candidate_manifest_sha256: Optional[str] = Field(None, description="SHA-256 of candidate manifest.json")
    previous_champion_hashes: Optional[Dict[str, str]] = Field(None, description="Hashes of previous active champion")
    new_champion_hashes: Optional[Dict[str, str]] = Field(None, description="Hashes of newly activated champion")
    operating_threshold: Optional[float] = Field(None, description="Calibrated operating decision threshold")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of creation")
    updated_at: str = Field(..., description="ISO 8601 UTC timestamp of last state change")
    error_message: Optional[str] = Field(None, description="Details of any error encountered")

    def transition_to(self, new_state: PromotionOperationState, error: Optional[str] = None) -> None:
        """Record a state transition in the journal history."""
        now_utc = datetime.now(timezone.utc).isoformat()
        self.current_state = new_state
        self.updated_at = now_utc
        if error:
            self.error_message = error
        self.state_history.append({
            "state": new_state.value,
            "timestamp": now_utc,
            "error": error or "",
        })

    def save(self, filepath: Path) -> None:
        """Atomically persist journal to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_path = filepath.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
        temp_path.replace(filepath)


class PromotionExecutionRecord(BaseModel):
    """
    Immutable audit record capturing a successfully finalized model promotion.
    Persisted to ml/models/registry/promotions/promotion_{version}_{timestamp}.json
    """
    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(..., description="Unique promotion operation identifier")
    promoted_version: str = Field(..., description="Model version promoted to Champion")
    demoted_champion_version: str = Field(..., description="Previous Champion version archived")
    promoted_at: str = Field(..., description="ISO 8601 UTC timestamp of promotion commit")
    promoted_by: str = Field(..., description="Identifier of actor executing promotion")
    actor_role: str = Field(..., description="Role of the actor (ADMIN, ANALYST)")
    promotion_rationale: str = Field(..., description="Mandatory governance rationale")
    sign_off_id: str = Field(..., description="Linked human sign-off UUID")
    sign_off_sha256: str = Field(..., description="Cryptographic SHA-256 of sign-off JSON")
    assessment_sha256: str = Field(..., description="Cryptographic SHA-256 of eligibility assessment JSON")
    comparison_sha256: str = Field(..., description="Cryptographic SHA-256 of comparison artifact JSON")
    model_sha256: str = Field(..., description="SHA-256 of promoted model binary")
    preprocessor_sha256: str = Field(..., description="SHA-256 of promoted preprocessor binary")
    manifest_sha256: str = Field(..., description="SHA-256 of promoted bundle manifest")
    previous_champion_hashes: Dict[str, str] = Field(..., description="Cryptographic hashes of archived Champion")
    new_champion_hashes: Dict[str, str] = Field(..., description="Cryptographic hashes of active Champion")
    operating_threshold: float = Field(..., description="Operating decision threshold (tau*)")
    execution_state: str = Field(default="FINALIZED", description="Final state of the promotion")
    audit_log_id: Optional[str] = Field(None, description="UUID of PostgreSQL AuditLog entry")


class RollbackExecutionRecord(BaseModel):
    """
    Immutable audit record capturing a successfully finalized model rollback.
    Persisted to ml/models/registry/rollbacks/rollback_{version}_{timestamp}.json
    """
    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(..., description="Unique rollback operation identifier")
    rolled_back_from_version: str = Field(..., description="Champion version demoted to ROLLED_BACK")
    restored_champion_version: str = Field(..., description="Historical version restored to Champion")
    rolled_back_at: str = Field(..., description="ISO 8601 UTC timestamp of rollback commit")
    rolled_back_by: str = Field(..., description="Identifier of actor executing rollback")
    actor_role: str = Field(..., description="Role of the actor (ADMIN, ANALYST)")
    rollback_rationale: str = Field(..., description="Explanatory business rationale for rollback")
    model_sha256: str = Field(..., description="SHA-256 of restored model binary")
    preprocessor_sha256: str = Field(..., description="SHA-256 of restored preprocessor binary")
    manifest_sha256: str = Field(..., description="SHA-256 of restored bundle manifest")
    previous_champion_hashes: Dict[str, str] = Field(..., description="Cryptographic hashes of demoted Champion")
    new_champion_hashes: Dict[str, str] = Field(..., description="Cryptographic hashes of restored Champion")
    operating_threshold: float = Field(..., description="Restored operating decision threshold")
    execution_state: str = Field(default="FINALIZED", description="Final state of the rollback")
    audit_log_id: Optional[str] = Field(None, description="UUID of PostgreSQL AuditLog entry")


class PromotionResult(BaseModel):
    """Summary result returned upon executing model promotion."""
    model_config = ConfigDict(frozen=True)

    success: bool
    operation_id: str
    promoted_version: str
    demoted_version: str
    state: PromotionOperationState
    execution_record_path: Optional[str] = None
    journal_path: Optional[str] = None
    message: str


class RollbackResult(BaseModel):
    """Summary result returned upon executing model rollback."""
    model_config = ConfigDict(frozen=True)

    success: bool
    operation_id: str
    restored_version: str
    rolled_back_version: str
    state: PromotionOperationState
    execution_record_path: Optional[str] = None
    journal_path: Optional[str] = None
    message: str


# ==============================================================================
# Custom Domain Exceptions
# ==============================================================================

class PromotionPreconditionError(ValueError):
    """Raised when any mandatory promotion precondition check fails."""
    pass


class PromotionStagingError(RuntimeError):
    """Raised when staging directory preparation or isolated validation fails."""
    pass


class PromotionIntegrityError(ValueError):
    """Raised when cryptographic checksums or artifact hashes do not match expectations."""
    pass


class PromotionConcurrencyError(RuntimeError):
    """Raised when concurrent modification or stale state is detected during locking."""
    pass


class PromotionPostCommitVerificationError(RuntimeError):
    """Raised when post-commit DB vs filesystem consistency check fails."""
    pass


class RollbackPreconditionError(ValueError):
    """Raised when rollback preconditions fail (e.g. missing historical bundle)."""
    pass


class RollbackIntegrityError(ValueError):
    """Raised when rollback bundle checksums or validation fails."""
    pass


class RecoveryRequiredError(RuntimeError):
    """Raised when a system inconsistency requires manual/explicit recovery intervention."""
    pass
