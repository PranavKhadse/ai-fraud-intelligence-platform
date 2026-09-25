"""
Model Promotion Engine for Phase 14.6 Staged Model Promotion.

Implements the explicit, deterministic, cryptographically verified, auditable
Staged Model Promotion subsystem:
- 16-point precondition verification.
- Concurrency control with PostgreSQL row locks (SELECT FOR UPDATE).
- Durable operation journaling (REQUESTED -> VALIDATED -> STAGED -> FILESYSTEM_SWAPPED -> DB_COMMITTED -> FINALIZED).
- Isolated staging validation on candidate bundle (synthetic 55-feature matrix inference).
- Historical archiving of current Champion v1.0.0 (if not already archived).
- Safe active artifact backup & compensating restoration on pre-commit failures.
- Atomic PostgreSQL transaction with AuditLog integration.
- Post-commit consistency verification (DB active Champion == filesystem artifacts == hashes).
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import ActorContext
from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.db.models.model_registry import ModelRegistryEntry
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.governance.schemas import (
    GateStatus,
    PromotionEligibilityAssessment,
    PromotionSignOffRecord,
    SignOffDecision,
)
from ml.lifecycle.governance.signoff import PERMITTED_SIGNOFF_ROLES, compute_assessment_sha256
from ml.lifecycle.promotion.schemas import (
    PromotionExecutionRecord,
    PromotionIntegrityError,
    PromotionOperationJournal,
    PromotionOperationState,
    PromotionPostCommitVerificationError,
    PromotionPreconditionError,
    PromotionResult,
    PromotionStagingError,
    RecoveryRequiredError,
)
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
    validate_bundle_manifest,
    validate_semantic_version,
)
from ml.models.config import (
    CATEGORICAL_PREDICTORS,
    PREDICTIVE_FEATURE_COLUMNS,
)

logger = logging.getLogger("ml.lifecycle.promotion")


def create_synthetic_55_feature_dataframe(num_rows: int = 5) -> pd.DataFrame:
    """
    Construct a deterministic, valid synthetic DataFrame containing all 55 canonical features.
    Used for isolated staging inference validation.
    """
    merch_cats = ["shopping_net", "grocery_pos", "misc_net", "entertainment", "gas_transport"]
    job_cats = ["engineer", "technician", "manager", "clerk", "teacher"]

    data: Dict[str, List[Any]] = {}
    for col in PREDICTIVE_FEATURE_COLUMNS:
        if col in CATEGORICAL_PREDICTORS:
            if col == "merchant_category":
                data[col] = [merch_cats[i % len(merch_cats)] for i in range(num_rows)]
            else:
                data[col] = [job_cats[i % len(job_cats)] for i in range(num_rows)]
        else:
            # Deterministic non-NaN numeric values
            data[col] = [float(10.0 + i * 2.5) for i in range(num_rows)]
            
    return pd.DataFrame(data, columns=PREDICTIVE_FEATURE_COLUMNS)


class ModelPromotionEngine:
    """
    Autonomous engine coordinating staged model promotion from Candidate/Challenger to active Champion.
    """

    def __init__(
        self,
        config: Optional[LifecycleConfig] = None,
    ) -> None:
        """
        Initialize the promotion engine.

        Args:
            config: Lifecycle directory and file path configuration.
        """
        self.config = config or default_lifecycle_config

    async def promote_candidate(
        self,
        candidate_version: str,
        sign_off: PromotionSignOffRecord,
        actor: ActorContext,
        session: AsyncSession,
        assessment: Optional[PromotionEligibilityAssessment] = None,
        comparison_path: Optional[Union[str, Path]] = None,
    ) -> PromotionResult:
        """
        Execute deterministic, staged model promotion.

        Sequence:
        Step 1: Validate promotion preconditions.
        Step 2: Acquire PostgreSQL row locks (SELECT FOR UPDATE).
        Step 3: Re-read and revalidate Candidate + current Champion.
        Step 4: Create durable promotion-operation record.
        Step 5: Create isolated staging directory.
        Step 6: Copy candidate bundle into staging.
        Step 7: Verify candidate checksums inside staging.
        Step 8: Perform isolated staging validation.
        Step 9: Ensure previous Champion has an immutable versioned historical bundle.
        Step 10: Create and verify active-artifact backup.
        Step 11: Prepare and swap active filesystem artifacts.
        Step 12: Verify prepared active artifacts before committing DB state.
        Step 13: Transition DB state atomically (Previous Champion -> ARCHIVED, Candidate -> CHAMPION).
        Step 14: Write AuditLog entry inside transaction.
        Step 15: Commit transaction.
        Step 16: Post-commit consistency verification.
        Step 17: Mark promotion operation FINALIZED.
        Step 18: Persist immutable promotion execution record.
        Step 19: Clean temporary staging/backup files according to policy.
        """
        candidate_version = candidate_version.strip()
        if not validate_semantic_version(candidate_version):
            raise PromotionPreconditionError(f"Candidate version '{candidate_version}' does not conform to semver.")

        # --- Step 1 & 2: Concurrency control & row locks ---
        # Fetch current active champion and target candidate under row lock
        stmt = (
            select(ModelRegistryEntry)
            .where(
                (ModelRegistryEntry.is_active_champion == True)  # noqa: E712
                | (ModelRegistryEntry.model_version == candidate_version)
            )
            .with_for_update()
        )
        res = await session.execute(stmt)
        entries = res.scalars().all()

        current_champions = [e for e in entries if e.is_active_champion]
        if len(current_champions) != 1:
            raise PromotionPreconditionError(
                f"Expected exactly 1 active Champion in DB, found {len(current_champions)}."
            )
        db_champion = current_champions[0]
        champion_version = db_champion.model_version

        candidate_entries = [e for e in entries if e.model_version == candidate_version]
        if not candidate_entries:
            raise PromotionPreconditionError(
                f"Candidate version '{candidate_version}' not found in PostgreSQL model registry."
            )
        db_candidate = candidate_entries[0]

        # --- Step 3: Precondition verification ---
        self._verify_preconditions(
            db_candidate=db_candidate,
            db_champion=db_champion,
            candidate_version=candidate_version,
            sign_off=sign_off,
            actor=actor,
            assessment=assessment,
            comparison_path=comparison_path,
        )

        # Candidate bundle on disk
        candidate_dir = self._locate_candidate_bundle_dir(candidate_version)
        candidate_model_path = candidate_dir / "model.joblib"
        candidate_prep_path = candidate_dir / "preprocessor.joblib"
        candidate_manifest_path = candidate_dir / "manifest.json"

        if not candidate_model_path.exists() or not candidate_prep_path.exists() or not candidate_manifest_path.exists():
            raise PromotionPreconditionError(
                f"Candidate bundle at '{candidate_dir}' is missing mandatory files (model, preprocessor, manifest)."
            )

        # Verify candidate on-disk checksums match DB
        cand_model_sha = calculate_file_sha256(candidate_model_path)
        cand_prep_sha = calculate_file_sha256(candidate_prep_path)
        cand_manifest_sha = calculate_file_sha256(candidate_manifest_path)

        if cand_model_sha.lower() != db_candidate.sha256_model.lower():
            raise PromotionIntegrityError(
                f"Candidate model SHA-256 on disk ({cand_model_sha}) does not match DB ({db_candidate.sha256_model})."
            )
        if cand_prep_sha.lower() != db_candidate.sha256_preprocessor.lower():
            raise PromotionIntegrityError(
                f"Candidate preprocessor SHA-256 on disk ({cand_prep_sha}) does not match DB ({db_candidate.sha256_preprocessor})."
            )
        if db_candidate.sha256_manifest and cand_manifest_sha.lower() != db_candidate.sha256_manifest.lower():
            raise PromotionIntegrityError(
                f"Candidate manifest SHA-256 on disk ({cand_manifest_sha}) does not match DB ({db_candidate.sha256_manifest})."
            )

        # Verify on-disk active champion hashes match DB
        champ_model_sha = calculate_file_sha256(self.config.champion_model_path)
        champ_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)
        if champ_model_sha.lower() != db_champion.sha256_model.lower():
            raise PromotionIntegrityError(
                f"Current Champion model SHA-256 on disk ({champ_model_sha}) does not match DB ({db_champion.sha256_model})."
            )
        if champ_prep_sha.lower() != db_champion.sha256_preprocessor.lower():
            raise PromotionIntegrityError(
                f"Current Champion preprocessor SHA-256 on disk ({champ_prep_sha}) does not match DB ({db_champion.sha256_preprocessor})."
            )

        # Load candidate manifest
        with open(candidate_manifest_path, "r", encoding="utf-8") as f:
            candidate_manifest_dict = json.load(f)

        # --- Step 4: Create durable operation journal ---
        operation_id = f"promo_{candidate_version}_{uuid.uuid4().hex[:8]}"
        journal_path = self.config.operations_dir / f"{operation_id}.json"
        
        journal = PromotionOperationJournal(
            operation_id=operation_id,
            operation_type="PROMOTION",
            candidate_version=candidate_version,
            champion_version=champion_version,
            current_state=PromotionOperationState.REQUESTED,
            actor_id=actor.actor_id,
            actor_role=actor.actor_role,
            rationale=sign_off.signoff_rationale,
            sign_off_id=sign_off.signoff_id,
            sign_off_sha256=sign_off.eligibility_assessment_sha256,
            assessment_sha256=sign_off.eligibility_assessment_sha256,
            comparison_sha256=sign_off.comparison_artifact_sha256,
            candidate_model_sha256=cand_model_sha,
            candidate_preprocessor_sha256=cand_prep_sha,
            candidate_manifest_sha256=cand_manifest_sha,
            previous_champion_hashes={
                "model": champ_model_sha,
                "preprocessor": champ_prep_sha,
            },
            operating_threshold=float(db_candidate.operating_threshold),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        journal.transition_to(PromotionOperationState.VALIDATED)
        journal.save(journal_path)

        # Staging & Backup Directories
        staging_dir = self.config.staging_dir / operation_id
        backup_dir = self.config.staging_dir / f"{self.config.active_backup_prefix}{operation_id}"

        try:
            # --- Step 5, 6, 7 & 8: Isolated Staging & Validation ---
            staging_dir.mkdir(parents=True, exist_ok=True)
            journal.staging_path = str(staging_dir)

            staged_model = staging_dir / "model.joblib"
            staged_prep = staging_dir / "preprocessor.joblib"
            staged_manifest = staging_dir / "manifest.json"

            shutil.copy2(candidate_model_path, staged_model)
            shutil.copy2(candidate_prep_path, staged_prep)
            shutil.copy2(candidate_manifest_path, staged_manifest)

            # Check checksums inside staging
            if calculate_file_sha256(staged_model) != cand_model_sha:
                raise PromotionStagingError("Checksum mismatch on staged model.joblib.")
            if calculate_file_sha256(staged_prep) != cand_prep_sha:
                raise PromotionStagingError("Checksum mismatch on staged preprocessor.joblib.")

            # Perform isolated staging validation
            self._perform_isolated_staging_validation(
                staged_model_path=staged_model,
                staged_prep_path=staged_prep,
                candidate_manifest_dict=candidate_manifest_dict,
                staging_dir=staging_dir,
                operation_id=operation_id,
            )

            journal.transition_to(PromotionOperationState.STAGED)
            journal.save(journal_path)

            # --- Step 9: Archive Previous Champion if historical bundle missing ---
            self._ensure_historical_champion_bundle(
                champion_version=champion_version,
                db_champion=db_champion,
            )

            # --- Step 10: Create Active Artifact Backup ---
            backup_dir.mkdir(parents=True, exist_ok=True)
            journal.backup_path = str(backup_dir)

            backup_model = backup_dir / "champion_model.joblib"
            backup_prep = backup_dir / "champion_preprocessor.joblib"
            backup_meta = backup_dir / "model_metadata.json"

            shutil.copy2(self.config.champion_model_path, backup_model)
            shutil.copy2(self.config.champion_preprocessor_path, backup_prep)
            if self.config.champion_metadata_path.exists():
                shutil.copy2(self.config.champion_metadata_path, backup_meta)

            # Verify backup hashes
            if calculate_file_sha256(backup_model) != champ_model_sha:
                raise PromotionStagingError("Backup model checksum verification failed.")
            if calculate_file_sha256(backup_prep) != champ_prep_sha:
                raise PromotionStagingError("Backup preprocessor checksum verification failed.")

            # --- Step 11 & 12: Swap Active Filesystem Artifacts ---
            shutil.copy2(staged_model, self.config.champion_model_path)
            shutil.copy2(staged_prep, self.config.champion_preprocessor_path)
            shutil.copy2(staging_dir / "model_metadata.json", self.config.champion_metadata_path)

            # Verify active files
            new_active_model_sha = calculate_file_sha256(self.config.champion_model_path)
            new_active_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)

            if new_active_model_sha != cand_model_sha or new_active_prep_sha != cand_prep_sha:
                raise PromotionIntegrityError(
                    f"Active artifact swap failed hash verification: model={new_active_model_sha}, prep={new_active_prep_sha}"
                )

            journal.new_champion_hashes = {
                "model": new_active_model_sha,
                "preprocessor": new_active_prep_sha,
            }
            journal.transition_to(PromotionOperationState.FILESYSTEM_SWAPPED)
            journal.save(journal_path)

            # --- Step 13 & 14: Atomic DB Transition & Audit Log ---
            now_utc = datetime.now(timezone.utc)

            # Demote Champion
            db_champion.status = ModelLifecycleStatus.ARCHIVED.value
            db_champion.is_active_champion = False

            # Promote Candidate
            db_candidate.status = ModelLifecycleStatus.CHAMPION.value
            db_candidate.is_active_champion = True
            db_candidate.promoted_at = now_utc
            db_candidate.promoted_by = actor.actor_id
            db_candidate.promotion_rationale = sign_off.signoff_rationale

            # Write DB AuditLog
            audit_entry = AuditLog(
                id=uuid.uuid4(),
                event_type="MODEL_PROMOTION_EXECUTED",
                entity_type=AuditEntityType.SYSTEM,
                entity_id=db_candidate.id,
                action="MODEL_PROMOTION",
                actor_type=AuditActorType(actor.actor_role) if actor.actor_role in AuditActorType.__members__ else AuditActorType.ADMIN,
                actor_id=actor.actor_id,
                payload={
                    "operation_id": operation_id,
                    "promoted_version": candidate_version,
                    "demoted_version": champion_version,
                    "sign_off_id": sign_off.signoff_id,
                    "operating_threshold": float(db_candidate.operating_threshold),
                    "model_sha256": cand_model_sha,
                    "preprocessor_sha256": cand_prep_sha,
                },
            )
            session.add(audit_entry)

            # --- Step 15: Commit Transaction ---
            await session.commit()

            journal.transition_to(PromotionOperationState.DB_COMMITTED)
            journal.save(journal_path)

        except Exception as e:
            # Pre-commit compensation
            logger.error(f"Error during promotion prior to DB commit: {e}. Executing compensation.")
            if journal.current_state in (PromotionOperationState.FILESYSTEM_SWAPPED, PromotionOperationState.STAGED):
                self._compensate_filesystem_rollback(backup_dir=backup_dir)
            
            await session.rollback()
            journal.transition_to(PromotionOperationState.FAILED_BEFORE_COMMIT, error=str(e))
            journal.save(journal_path)

            # Clean staging
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            raise

        # --- Step 16: Post-Commit Consistency Verification ---
        try:
            # Verify DB active champion
            verify_stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.is_active_champion == True)  # noqa: E712
            verify_res = await session.execute(verify_stmt)
            active_entries = [e for e in verify_res.scalars().all() if e.is_active_champion]
            if len(active_entries) != 1 or active_entries[0].model_version != candidate_version:
                raise PromotionPostCommitVerificationError(
                    f"Post-commit DB verification failed: expected active champion '{candidate_version}', found {[e.model_version for e in active_entries]}"
                )

            # Verify active filesystem artifacts
            final_model_sha = calculate_file_sha256(self.config.champion_model_path)
            final_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)

            if final_model_sha != cand_model_sha or final_prep_sha != cand_prep_sha:
                raise PromotionPostCommitVerificationError(
                    f"Post-commit filesystem hash mismatch: model={final_model_sha} (expected {cand_model_sha}), prep={final_prep_sha} (expected {cand_prep_sha})"
                )
        except Exception as post_err:
            logger.critical(f"CRITICAL: Post-commit consistency check failed: {post_err}")
            journal.transition_to(PromotionOperationState.RECOVERY_REQUIRED, error=str(post_err))
            journal.save(journal_path)
            raise PromotionPostCommitVerificationError(
                f"Post-commit consistency failure: {post_err}. Manual recovery required."
            ) from post_err

        # --- Step 17: Finalize Operation ---
        journal.transition_to(PromotionOperationState.FINALIZED)
        journal.save(journal_path)

        # --- Step 18: Persist Immutable Promotion Execution Record ---
        self.config.promotions_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        exec_record_path = self.config.promotions_dir / f"promotion_{candidate_version}_{timestamp_slug}.json"

        exec_record = PromotionExecutionRecord(
            operation_id=operation_id,
            promoted_version=candidate_version,
            demoted_champion_version=champion_version,
            promoted_at=now_utc.isoformat(),
            promoted_by=actor.actor_id,
            actor_role=actor.actor_role,
            promotion_rationale=sign_off.signoff_rationale,
            sign_off_id=sign_off.signoff_id,
            sign_off_sha256=sign_off.eligibility_assessment_sha256,
            assessment_sha256=sign_off.eligibility_assessment_sha256,
            comparison_sha256=sign_off.comparison_artifact_sha256,
            model_sha256=cand_model_sha,
            preprocessor_sha256=cand_prep_sha,
            manifest_sha256=cand_manifest_sha,
            previous_champion_hashes={
                "model": champ_model_sha,
                "preprocessor": champ_prep_sha,
            },
            new_champion_hashes={
                "model": cand_model_sha,
                "preprocessor": cand_prep_sha,
            },
            operating_threshold=float(db_candidate.operating_threshold),
            execution_state="FINALIZED",
            audit_log_id=str(audit_entry.id),
        )

        with open(exec_record_path, "w", encoding="utf-8") as f:
            json.dump(exec_record.model_dump(mode="json"), f, indent=2)

        # --- Step 19: Clean staging / backup ---
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        logger.info(
            f"Promotion finalized successfully: Candidate {candidate_version} is now active Champion. "
            f"Previous Champion {champion_version} archived."
        )

        return PromotionResult(
            success=True,
            operation_id=operation_id,
            promoted_version=candidate_version,
            demoted_version=champion_version,
            state=PromotionOperationState.FINALIZED,
            execution_record_path=str(exec_record_path),
            journal_path=str(journal_path),
            message=f"Model v{candidate_version} promoted to active Champion successfully.",
        )

    def _verify_preconditions(
        self,
        db_candidate: ModelRegistryEntry,
        db_champion: ModelRegistryEntry,
        candidate_version: str,
        sign_off: PromotionSignOffRecord,
        actor: ActorContext,
        assessment: Optional[PromotionEligibilityAssessment],
        comparison_path: Optional[Union[str, Path]],
    ) -> None:
        """Verify all 16 governance and precondition rules."""
        # Candidate status
        if db_candidate.status not in (ModelLifecycleStatus.CANDIDATE.value, ModelLifecycleStatus.CHALLENGER.value):
            raise PromotionPreconditionError(
                f"Candidate status must be CANDIDATE or CHALLENGER, got '{db_candidate.status}'."
            )

        if db_candidate.is_active_champion:
            raise PromotionPreconditionError(
                f"Candidate '{candidate_version}' is already marked as active Champion."
            )

        # Champion status
        if not db_champion.is_active_champion or db_champion.status != ModelLifecycleStatus.CHAMPION.value:
            raise PromotionPreconditionError(
                f"Champion '{db_champion.model_version}' is not in active CHAMPION status."
            )

        # Sign-off decision
        if sign_off.decision != SignOffDecision.APPROVED:
            raise PromotionPreconditionError(
                f"Sign-off decision must be APPROVED, got '{sign_off.decision}'."
            )

        # Sign-off RBAC
        if sign_off.actor_role not in PERMITTED_SIGNOFF_ROLES:
            raise PromotionPreconditionError(
                f"Sign-off actor role '{sign_off.actor_role}' is not permitted. Must be one of: {PERMITTED_SIGNOFF_ROLES}."
            )

        # Executing actor RBAC
        if actor.actor_role not in PERMITTED_SIGNOFF_ROLES:
            raise PromotionPreconditionError(
                f"Executing actor role '{actor.actor_role}' is not authorized to promote models."
            )

        # Sign-off rationale length
        if not sign_off.signoff_rationale or len(sign_off.signoff_rationale.strip()) < 15:
            raise PromotionPreconditionError("Sign-off rationale must be at least 15 characters long.")

        # Candidate and Champion version linkage
        if sign_off.candidate_version != candidate_version:
            raise PromotionPreconditionError(
                f"Sign-off candidate version '{sign_off.candidate_version}' does not match target '{candidate_version}'."
            )

        if sign_off.champion_version != db_champion.model_version:
            raise PromotionPreconditionError(
                f"Sign-off champion version '{sign_off.champion_version}' does not match current active champion '{db_champion.model_version}'."
            )

        # Candidate eligibility flag in sign-off
        if not sign_off.is_eligible_at_signoff:
            raise PromotionPreconditionError("Candidate was not eligible at time of sign-off.")

        # If assessment provided, verify cryptographic linkage
        if assessment is not None:
            calc_assess_sha = compute_assessment_sha256(assessment)
            if calc_assess_sha != sign_off.eligibility_assessment_sha256:
                raise PromotionIntegrityError(
                    f"Eligibility assessment SHA-256 ({calc_assess_sha}) does not match sign-off record ({sign_off.eligibility_assessment_sha256})."
                )
            if not assessment.is_eligible:
                raise PromotionPreconditionError("Provided assessment indicates candidate is not eligible.")
            if assessment.candidate_version != candidate_version:
                raise PromotionPreconditionError("Assessment candidate version mismatch.")
            if assessment.champion_version != db_champion.model_version:
                raise PromotionPreconditionError("Assessment champion version mismatch.")
            if assessment.evidence_artifact_sha256 != sign_off.comparison_artifact_sha256:
                raise PromotionIntegrityError("Assessment comparison artifact SHA-256 does not match sign-off record.")

        # If comparison artifact path provided, verify checksum
        if comparison_path is not None:
            p = Path(comparison_path)
            if not p.exists():
                raise PromotionPreconditionError(f"Comparison evidence artifact not found at: {p}")
            calc_comp_sha = calculate_file_sha256(p)
            if calc_comp_sha != sign_off.comparison_artifact_sha256:
                raise PromotionIntegrityError(
                    f"Comparison artifact SHA-256 on disk ({calc_comp_sha}) does not match sign-off record ({sign_off.comparison_artifact_sha256})."
                )

        # Check operating threshold
        if db_candidate.operating_threshold <= Decimal("0.0") or db_candidate.operating_threshold >= Decimal("1.0"):
            raise PromotionPreconditionError(
                f"Invalid candidate operating threshold: {db_candidate.operating_threshold}"
            )

    def _locate_candidate_bundle_dir(self, version: str) -> Path:
        """Find candidate bundle directory under candidates_dir or bundles_dir."""
        cand_path = self.config.candidates_dir / version
        if cand_path.exists():
            return cand_path
        bundle_path = self.config.bundles_dir / version
        if bundle_path.exists():
            return bundle_path
        raise PromotionPreconditionError(
            f"Candidate bundle for version '{version}' not found on disk at {cand_path} or {bundle_path}."
        )

    def _perform_isolated_staging_validation(
        self,
        staged_model_path: Path,
        staged_prep_path: Path,
        candidate_manifest_dict: Dict[str, Any],
        staging_dir: Path,
        operation_id: str,
    ) -> None:
        """
        Execute isolated deserialization, 55-feature synthetic inference, and metadata contract creation.
        """
        try:
            model = joblib.load(staged_model_path)
            preprocessor = joblib.load(staged_prep_path)
        except Exception as e:
            raise PromotionStagingError(f"Failed to deserialize staged candidate artifacts: {e}") from e

        # Construct synthetic 55-feature DataFrame
        df_synthetic = create_synthetic_55_feature_dataframe(num_rows=5)

        try:
            # Transform features
            X_trans = preprocessor.transform(df_synthetic)
            # Run inference
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_trans)
                if probs is None or len(probs) != 5:
                    raise ValueError("Unexpected predict_proba output length.")
                if isinstance(probs, np.ndarray):
                    if np.any(np.isnan(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
                        raise ValueError("Model probabilities out of [0.0, 1.0] bounds.")
            else:
                preds = model.predict(X_trans)
                if preds is None or len(preds) != 5:
                    raise ValueError(f"Unexpected predict shape: {getattr(preds, 'shape', None)}")
        except Exception as inf_err:
            raise PromotionStagingError(f"Isolated staging inference test failed: {inf_err}") from inf_err

        # Validate operating threshold
        threshold = float(candidate_manifest_dict.get("operating_threshold", 0.78))
        if threshold <= 0.0 or threshold >= 1.0:
            raise PromotionStagingError(f"Invalid operating threshold in manifest: {threshold}")

        # Build runtime model_metadata.json conforming to RiskEvaluator contract
        val_metrics = candidate_manifest_dict.get("validation_metrics", {})
        oot_metrics = candidate_manifest_dict.get("oot_holdout_metrics", {})
        hyperparams = candidate_manifest_dict.get("hyperparameters", {})
        training_meta = candidate_manifest_dict.get("training_metadata", {})

        runtime_metadata = {
            "model_version": candidate_manifest_dict.get("model_version", "1.1.0"),
            "champion_model": candidate_manifest_dict.get("model_family", "xgboost"),
            "champion_hyperparameters": hyperparams,
            "selected_threshold": threshold,
            "partition_statistics": {
                "train": training_meta,
            },
            "validation_benchmark": val_metrics,
            "oot_test_metrics_frozen_threshold": oot_metrics,
            "feature_count": len(PREDICTIVE_FEATURE_COLUMNS),
            "feature_names": PREDICTIVE_FEATURE_COLUMNS,
            "promotion_metadata": {
                "operation_id": operation_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        metadata_path = staging_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(runtime_metadata, f, indent=2)

    def _ensure_historical_champion_bundle(
        self,
        champion_version: str,
        db_champion: ModelRegistryEntry,
    ) -> None:
        """
        Ensure an immutable historical bundle exists for the Champion being replaced.
        If missing from bundles_dir/{champion_version}, archives active artifacts before swap.
        """
        target_bundle_dir = self.config.bundles_dir / champion_version
        if target_bundle_dir.exists() and (target_bundle_dir / "model.joblib").exists():
            return

        logger.info(f"Archiving Champion v{champion_version} to historical bundle: {target_bundle_dir}")
        target_bundle_dir.mkdir(parents=True, exist_ok=True)

        bundle_model = target_bundle_dir / "model.joblib"
        bundle_prep = target_bundle_dir / "preprocessor.joblib"
        bundle_manifest = target_bundle_dir / "manifest.json"

        shutil.copy2(self.config.champion_model_path, bundle_model)
        shutil.copy2(self.config.champion_preprocessor_path, bundle_prep)

        # Create or copy manifest
        manifest_payload = {
            "model_version": champion_version,
            "model_family": db_champion.model_family,
            "created_at": db_champion.created_at.isoformat() if db_champion.created_at else datetime.now(timezone.utc).isoformat(),
            "status": "ARCHIVED",
            "operating_threshold": float(db_champion.operating_threshold),
            "hyperparameters": db_champion.hyperparameters or {},
            "training_metadata": db_champion.training_metadata or {},
            "validation_metrics": db_champion.validation_metrics or {},
            "oot_holdout_metrics": db_champion.oot_metrics or {},
            "sha256_checksums": {
                "model": calculate_file_sha256(bundle_model),
                "preprocessor": calculate_file_sha256(bundle_prep),
            },
        }

        with open(bundle_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

    def _compensate_filesystem_rollback(self, backup_dir: Path) -> None:
        """Restore active artifacts from backup in case of pre-commit failure."""
        logger.warning(f"Executing compensating filesystem restoration from backup: {backup_dir}")
        backup_model = backup_dir / "champion_model.joblib"
        backup_prep = backup_dir / "champion_preprocessor.joblib"
        backup_meta = backup_dir / "model_metadata.json"

        if backup_model.exists():
            shutil.copy2(backup_model, self.config.champion_model_path)
        if backup_prep.exists():
            shutil.copy2(backup_prep, self.config.champion_preprocessor_path)
        if backup_meta.exists():
            shutil.copy2(backup_meta, self.config.champion_metadata_path)
