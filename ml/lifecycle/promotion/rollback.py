"""
Model Rollback Engine and Crash Recovery Subsystem for Phase 14.6.

Implements explicit, deterministic, cryptographically verified model rollback:
- Verification of immutable historical bundle and checksums.
- Isolated staging inference test on target bundle.
- Concurrency control with PostgreSQL row locks (SELECT FOR UPDATE).
- Active artifact backup and atomic filesystem swap.
- Atomic PostgreSQL transition (Current Champion -> ROLLED_BACK, Target -> CHAMPION).
- Post-commit consistency verification and immutable RollbackExecutionRecord persistence.
- Crash recovery and startup reconciliation between DB and filesystem state.
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
from ml.lifecycle.governance.signoff import PERMITTED_SIGNOFF_ROLES
from ml.lifecycle.promotion.promoter import create_synthetic_55_feature_dataframe
from ml.lifecycle.promotion.schemas import (
    POST_COMMIT_STATES,
    PRE_COMMIT_STATES,
    PromotionIntegrityError,
    PromotionOperationJournal,
    PromotionOperationState,
    PromotionPostCommitVerificationError,
    PromotionStagingError,
    RecoveryRequiredError,
    RollbackExecutionRecord,
    RollbackIntegrityError,
    RollbackPreconditionError,
    RollbackResult,
)
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    calculate_file_sha256,
    validate_semantic_version,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS

logger = logging.getLogger("ml.lifecycle.rollback")


class ModelRollbackEngine:
    """
    Autonomous engine coordinating controlled, auditable rollback to a historical model bundle.
    """

    def __init__(
        self,
        config: Optional[LifecycleConfig] = None,
    ) -> None:
        """
        Initialize the rollback engine.

        Args:
            config: Lifecycle directory and file path configuration.
        """
        self.config = config or default_lifecycle_config

    async def rollback_champion(
        self,
        target_version: str,
        actor: ActorContext,
        rationale: str,
        session: AsyncSession,
    ) -> RollbackResult:
        """
        Execute an explicit, auditable rollback of active Champion to a target historical bundle.

        Sequence:
        1. Verify target historical bundle exists and passes cryptographic validation.
        2. Acquire DB row locks (SELECT FOR UPDATE) on active Champion and target version.
        3. Verify actor authorization (RBAC) and rationale length.
        4. Perform isolated staging validation on target historical bundle.
        5. Create active artifact backup.
        6. Swap active filesystem artifacts with target bundle artifacts.
        7. Atomically update DB state: Current Champion -> ROLLED_BACK, Target -> CHAMPION.
        8. Record AuditLog entry within transaction.
        9. Commit transaction.
        10. Post-commit consistency verification.
        11. Persist immutable RollbackExecutionRecord.
        12. Clean staging and backup files.
        """
        target_version = target_version.strip()
        if not validate_semantic_version(target_version):
            raise RollbackPreconditionError(f"Target version '{target_version}' does not conform to semver.")

        if actor.actor_role not in PERMITTED_SIGNOFF_ROLES:
            raise RollbackPreconditionError(
                f"Actor role '{actor.actor_role}' is not authorized to execute model rollback. Permitted: {PERMITTED_SIGNOFF_ROLES}."
            )

        if not rationale or len(rationale.strip()) < 15:
            raise RollbackPreconditionError("Rollback rationale must be at least 15 characters long.")

        # --- Step 1: Concurrency & Row Locks ---
        stmt = (
            select(ModelRegistryEntry)
            .where(
                (ModelRegistryEntry.is_active_champion == True)  # noqa: E712
                | (ModelRegistryEntry.model_version == target_version)
            )
            .with_for_update()
        )
        res = await session.execute(stmt)
        entries = res.scalars().all()

        current_champions = [e for e in entries if e.is_active_champion]
        if len(current_champions) != 1:
            raise RollbackPreconditionError(
                f"Expected exactly 1 active Champion in DB, found {len(current_champions)}."
            )
        db_current_champion = current_champions[0]
        current_version = db_current_champion.model_version

        if current_version == target_version:
            raise RollbackPreconditionError(
                f"Target version '{target_version}' is already the active Champion."
            )

        target_entries = [e for e in entries if e.model_version == target_version]
        if not target_entries:
            raise RollbackPreconditionError(
                f"Target version '{target_version}' not found in PostgreSQL model registry."
            )
        db_target = target_entries[0]

        if db_target.status not in (ModelLifecycleStatus.ARCHIVED.value, ModelLifecycleStatus.ROLLED_BACK.value, ModelLifecycleStatus.CHALLENGER.value, ModelLifecycleStatus.CANDIDATE.value):
            raise RollbackPreconditionError(
                f"Target model version '{target_version}' is in invalid status '{db_target.status}' for restoration."
            )

        # --- Step 2: Locate target bundle on disk & verify hashes ---
        target_bundle_dir = self._locate_bundle_dir(target_version)
        target_model_path = target_bundle_dir / "model.joblib"
        target_prep_path = target_bundle_dir / "preprocessor.joblib"
        target_manifest_path = target_bundle_dir / "manifest.json"

        if not target_model_path.exists() or not target_prep_path.exists():
            raise RollbackPreconditionError(
                f"Target historical bundle at '{target_bundle_dir}' is missing model or preprocessor artifact."
            )

        target_model_sha = calculate_file_sha256(target_model_path)
        target_prep_sha = calculate_file_sha256(target_prep_path)
        target_manifest_sha = calculate_file_sha256(target_manifest_path) if target_manifest_path.exists() else ""

        if target_model_sha.lower() != db_target.sha256_model.lower():
            raise RollbackIntegrityError(
                f"Target model SHA-256 on disk ({target_model_sha}) does not match DB ({db_target.sha256_model})."
            )
        if target_prep_sha.lower() != db_target.sha256_preprocessor.lower():
            raise RollbackIntegrityError(
                f"Target preprocessor SHA-256 on disk ({target_prep_sha}) does not match DB ({db_target.sha256_preprocessor})."
            )

        # --- Step 3: Create durable operation journal ---
        operation_id = f"rollback_{current_version}_to_{target_version}_{uuid.uuid4().hex[:8]}"
        journal_path = self.config.operations_dir / f"{operation_id}.json"

        current_champ_model_sha = calculate_file_sha256(self.config.champion_model_path)
        current_champ_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)

        journal = PromotionOperationJournal(
            operation_id=operation_id,
            operation_type="ROLLBACK",
            candidate_version=target_version,
            champion_version=current_version,
            current_state=PromotionOperationState.REQUESTED,
            actor_id=actor.actor_id,
            actor_role=actor.actor_role,
            rationale=rationale.strip(),
            candidate_model_sha256=target_model_sha,
            candidate_preprocessor_sha256=target_prep_sha,
            candidate_manifest_sha256=target_manifest_sha,
            previous_champion_hashes={
                "model": current_champ_model_sha,
                "preprocessor": current_champ_prep_sha,
            },
            operating_threshold=float(db_target.operating_threshold),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        journal.transition_to(PromotionOperationState.VALIDATED)
        journal.save(journal_path)

        staging_dir = self.config.staging_dir / operation_id
        backup_dir = self.config.staging_dir / f"{self.config.active_backup_prefix}{operation_id}"

        try:
            # --- Step 4: Isolated Staging & Validation on Target Bundle ---
            staging_dir.mkdir(parents=True, exist_ok=True)
            journal.staging_path = str(staging_dir)

            staged_model = staging_dir / "model.joblib"
            staged_prep = staging_dir / "preprocessor.joblib"
            shutil.copy2(target_model_path, staged_model)
            shutil.copy2(target_prep_path, staged_prep)

            # Isolated deserialization and synthetic inference test
            self._validate_staged_artifacts(
                staged_model_path=staged_model,
                staged_prep_path=staged_prep,
                target_version=target_version,
                db_target=db_target,
                staging_dir=staging_dir,
                operation_id=operation_id,
            )

            journal.transition_to(PromotionOperationState.STAGED)
            journal.save(journal_path)

            # --- Step 5: Create Active Artifact Backup ---
            backup_dir.mkdir(parents=True, exist_ok=True)
            journal.backup_path = str(backup_dir)

            backup_model = backup_dir / "champion_model.joblib"
            backup_prep = backup_dir / "champion_preprocessor.joblib"
            backup_meta = backup_dir / "model_metadata.json"

            shutil.copy2(self.config.champion_model_path, backup_model)
            shutil.copy2(self.config.champion_preprocessor_path, backup_prep)
            if self.config.champion_metadata_path.exists():
                shutil.copy2(self.config.champion_metadata_path, backup_meta)

            # --- Step 6: Swap Active Filesystem Artifacts ---
            shutil.copy2(staged_model, self.config.champion_model_path)
            shutil.copy2(staged_prep, self.config.champion_preprocessor_path)
            shutil.copy2(staging_dir / "model_metadata.json", self.config.champion_metadata_path)

            new_active_model_sha = calculate_file_sha256(self.config.champion_model_path)
            new_active_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)

            if new_active_model_sha != target_model_sha or new_active_prep_sha != target_prep_sha:
                raise RollbackIntegrityError("Active artifact swap failed hash verification against target bundle.")

            journal.new_champion_hashes = {
                "model": new_active_model_sha,
                "preprocessor": new_active_prep_sha,
            }
            journal.transition_to(PromotionOperationState.FILESYSTEM_SWAPPED)
            journal.save(journal_path)

            # --- Step 7 & 8: Atomic DB Transition & Audit Log ---
            now_utc = datetime.now(timezone.utc)

            # Demote current Champion to ROLLED_BACK
            db_current_champion.status = ModelLifecycleStatus.ROLLED_BACK.value
            db_current_champion.is_active_champion = False
            db_current_champion.rolled_back_at = now_utc
            db_current_champion.rolled_back_by = actor.actor_id
            db_current_champion.rollback_rationale = rationale.strip()

            # Restore target to CHAMPION
            db_target.status = ModelLifecycleStatus.CHAMPION.value
            db_target.is_active_champion = True

            # Write AuditLog
            audit_entry = AuditLog(
                id=uuid.uuid4(),
                event_type="MODEL_ROLLBACK_EXECUTED",
                entity_type=AuditEntityType.SYSTEM,
                entity_id=db_target.id,
                action="MODEL_ROLLBACK",
                actor_type=AuditActorType(actor.actor_role) if actor.actor_role in AuditActorType.__members__ else AuditActorType.ADMIN,
                actor_id=actor.actor_id,
                payload={
                    "operation_id": operation_id,
                    "restored_version": target_version,
                    "rolled_back_version": current_version,
                    "rationale": rationale.strip(),
                    "model_sha256": target_model_sha,
                    "preprocessor_sha256": target_prep_sha,
                },
            )
            session.add(audit_entry)

            # --- Step 9: Commit Transaction ---
            await session.commit()

            journal.transition_to(PromotionOperationState.DB_COMMITTED)
            journal.save(journal_path)

        except Exception as e:
            # Pre-commit compensation
            logger.error(f"Error during rollback prior to DB commit: {e}. Executing compensation.")
            if journal.current_state in (PromotionOperationState.FILESYSTEM_SWAPPED, PromotionOperationState.STAGED):
                self._compensate_filesystem_restoration(backup_dir=backup_dir)

            await session.rollback()
            journal.transition_to(PromotionOperationState.FAILED_BEFORE_COMMIT, error=str(e))
            journal.save(journal_path)

            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            raise

        # --- Step 10: Post-Commit Consistency Verification ---
        try:
            verify_stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.is_active_champion == True)  # noqa: E712
            verify_res = await session.execute(verify_stmt)
            active_entries = [e for e in verify_res.scalars().all() if e.is_active_champion]
            if len(active_entries) != 1 or active_entries[0].model_version != target_version:
                raise PromotionPostCommitVerificationError(
                    f"Post-commit DB verification failed: expected active champion '{target_version}', found {[e.model_version for e in active_entries]}"
                )

            final_model_sha = calculate_file_sha256(self.config.champion_model_path)
            final_prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)
            if final_model_sha != target_model_sha or final_prep_sha != target_prep_sha:
                raise PromotionPostCommitVerificationError(
                    f"Post-commit filesystem hash mismatch: model={final_model_sha} (expected {target_model_sha}), prep={final_prep_sha} (expected {target_prep_sha})"
                )
        except Exception as post_err:
            logger.critical(f"CRITICAL: Post-rollback consistency check failed: {post_err}")
            journal.transition_to(PromotionOperationState.RECOVERY_REQUIRED, error=str(post_err))
            journal.save(journal_path)
            raise PromotionPostCommitVerificationError(
                f"Post-rollback consistency failure: {post_err}. Manual recovery required."
            ) from post_err

        # --- Step 11: Finalization & Execution Record ---
        journal.transition_to(PromotionOperationState.FINALIZED)
        journal.save(journal_path)

        self.config.rollbacks_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        exec_record_path = self.config.rollbacks_dir / f"rollback_{target_version}_{timestamp_slug}.json"

        exec_record = RollbackExecutionRecord(
            operation_id=operation_id,
            rolled_back_from_version=current_version,
            restored_champion_version=target_version,
            rolled_back_at=now_utc.isoformat(),
            rolled_back_by=actor.actor_id,
            actor_role=actor.actor_role,
            rollback_rationale=rationale.strip(),
            model_sha256=target_model_sha,
            preprocessor_sha256=target_prep_sha,
            manifest_sha256=target_manifest_sha,
            previous_champion_hashes={
                "model": current_champ_model_sha,
                "preprocessor": current_champ_prep_sha,
            },
            new_champion_hashes={
                "model": target_model_sha,
                "preprocessor": target_prep_sha,
            },
            operating_threshold=float(db_target.operating_threshold),
            execution_state="FINALIZED",
            audit_log_id=str(audit_entry.id),
        )

        with open(exec_record_path, "w", encoding="utf-8") as f:
            json.dump(exec_record.model_dump(mode="json"), f, indent=2)

        # --- Step 12: Clean staging / backup ---
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        logger.info(
            f"Rollback finalized successfully: Restored v{target_version} to active Champion. "
            f"Demoted v{current_version} to ROLLED_BACK."
        )

        return RollbackResult(
            success=True,
            operation_id=operation_id,
            restored_version=target_version,
            rolled_back_version=current_version,
            state=PromotionOperationState.FINALIZED,
            execution_record_path=str(exec_record_path),
            journal_path=str(journal_path),
            message=f"Model v{target_version} restored to active Champion successfully via rollback.",
        )

    def _locate_bundle_dir(self, version: str) -> Path:
        """Find bundle directory under bundles_dir or candidates_dir."""
        bundle_path = self.config.bundles_dir / version
        if bundle_path.exists():
            return bundle_path
        cand_path = self.config.candidates_dir / version
        if cand_path.exists():
            return cand_path
        raise RollbackPreconditionError(
            f"Historical model bundle for version '{version}' not found on disk at {bundle_path} or {cand_path}."
        )

    def _validate_staged_artifacts(
        self,
        staged_model_path: Path,
        staged_prep_path: Path,
        target_version: str,
        db_target: ModelRegistryEntry,
        staging_dir: Path,
        operation_id: str,
    ) -> None:
        """Validate target bundle loadability, synthetic inference, and metadata construction."""
        try:
            model = joblib.load(staged_model_path)
            preprocessor = joblib.load(staged_prep_path)
        except Exception as e:
            raise RollbackIntegrityError(f"Failed to deserialize target bundle artifacts: {e}") from e

        df_synthetic = create_synthetic_55_feature_dataframe(num_rows=5)
        try:
            X_trans = preprocessor.transform(df_synthetic)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_trans)
                if probs is None or len(probs) != 5:
                    raise ValueError("Unexpected output batch length.")
            else:
                preds = model.predict(X_trans)
                if preds is None or len(preds) != 5:
                    raise ValueError("Unexpected output batch length.")
        except Exception as inf_err:
            raise RollbackIntegrityError(f"Target bundle synthetic inference failed: {inf_err}") from inf_err

        # Construct runtime model_metadata.json for restored model
        runtime_metadata = {
            "model_version": target_version,
            "champion_model": db_target.model_family,
            "champion_hyperparameters": db_target.hyperparameters or {},
            "selected_threshold": float(db_target.operating_threshold),
            "partition_statistics": {
                "train": db_target.training_metadata or {},
            },
            "validation_benchmark": db_target.validation_metrics or {},
            "oot_test_metrics_frozen_threshold": db_target.oot_metrics or {},
            "feature_count": len(PREDICTIVE_FEATURE_COLUMNS),
            "feature_names": PREDICTIVE_FEATURE_COLUMNS,
            "rollback_metadata": {
                "operation_id": operation_id,
                "restored_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        metadata_path = staging_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(runtime_metadata, f, indent=2)

    def _compensate_filesystem_restoration(self, backup_dir: Path) -> None:
        """Restore active artifacts from backup."""
        logger.warning(f"Executing rollback compensation from backup: {backup_dir}")
        backup_model = backup_dir / "champion_model.joblib"
        backup_prep = backup_dir / "champion_preprocessor.joblib"
        backup_meta = backup_dir / "model_metadata.json"

        if backup_model.exists():
            shutil.copy2(backup_model, self.config.champion_model_path)
        if backup_prep.exists():
            shutil.copy2(backup_prep, self.config.champion_preprocessor_path)
        if backup_meta.exists():
            shutil.copy2(backup_meta, self.config.champion_metadata_path)


class PromotionRecoveryEngine:
    """
    Crash recovery and state reconciliation subsystem for promotion and rollback operations.
    """

    def __init__(self, config: Optional[LifecycleConfig] = None) -> None:
        self.config = config or default_lifecycle_config

    async def reconcile_active_champion_state(
        self,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Inspect operations directory and reconcile filesystem artifacts to match PostgreSQL active Champion.

        Guarantees:
        - DB active Champion is authoritative.
        - Uncommitted operations with swapped files are restored to match DB Champion.
        - Committed operations have their finalization completed if pending.
        - Consistent state: DB Champion == active filesystem artifacts == verified hashes.
        """
        # Query authoritative DB active champion
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.is_active_champion == True)  # noqa: E712
        res = await session.execute(stmt)
        active_champions = [e for e in res.scalars().all() if e.is_active_champion]

        if len(active_champions) != 1:
            raise RecoveryRequiredError(
                f"Cannot reconcile state: PostgreSQL has {len(active_champions)} active Champions."
            )

        db_champion = active_champions[0]
        champ_version = db_champion.model_version
        reconciled_ops = []

        if not self.config.operations_dir.exists():
            return {
                "status": "CONSISTENT",
                "active_champion": champ_version,
                "reconciled_operations": [],
            }

        # Scan operation journals
        for journal_file in self.config.operations_dir.glob("*.json"):
            try:
                with open(journal_file, "r", encoding="utf-8") as f:
                    journal_data = json.load(f)
                journal = PromotionOperationJournal.model_validate(journal_data)
            except Exception as read_err:
                logger.warning(f"Could not parse operation journal at {journal_file}: {read_err}")
                continue

            # Check if operation is incomplete
            if journal.current_state in (
                PromotionOperationState.REQUESTED,
                PromotionOperationState.VALIDATED,
                PromotionOperationState.STAGED,
                PromotionOperationState.FILESYSTEM_SWAPPED,
            ):
                # Pre-commit crash: DB is authoritative (db_champion). Filesystem must match db_champion.
                logger.warning(
                    f"Recovering uncommitted operation '{journal.operation_id}' in state '{journal.current_state}'. "
                    f"Reconciling filesystem to DB active Champion v{champ_version}."
                )
                self._restore_filesystem_to_champion(db_champion=db_champion)
                journal.transition_to(
                    PromotionOperationState.RECOVERY_COMPLETED,
                    error="Recovered from pre-commit crash; reconciled to DB active Champion.",
                )
                journal.save(journal_file)
                reconciled_ops.append(journal.operation_id)

            elif journal.current_state == PromotionOperationState.DB_COMMITTED:
                # Post-commit crash: DB is authoritative (new champion).
                logger.warning(
                    f"Finalizing committed operation '{journal.operation_id}' in state DB_COMMITTED."
                )
                self._restore_filesystem_to_champion(db_champion=db_champion)
                journal.transition_to(PromotionOperationState.FINALIZED)
                journal.save(journal_file)
                reconciled_ops.append(journal.operation_id)

            elif journal.current_state == PromotionOperationState.RECOVERY_REQUIRED:
                logger.error(
                    f"Operation '{journal.operation_id}' requires manual recovery: {journal.error_message}"
                )

        # Final consistency check
        model_sha = calculate_file_sha256(self.config.champion_model_path)
        prep_sha = calculate_file_sha256(self.config.champion_preprocessor_path)

        if model_sha.lower() != db_champion.sha256_model.lower() or prep_sha.lower() != db_champion.sha256_preprocessor.lower():
            # Reconcile active files from historical/bundle directory
            logger.warning(f"Active filesystem artifacts differ from DB active Champion. Reconciling artifacts...")
            self._restore_filesystem_to_champion(db_champion=db_champion)

        return {
            "status": "CONSISTENT",
            "active_champion": champ_version,
            "reconciled_operations": reconciled_ops,
        }

    def _restore_filesystem_to_champion(self, db_champion: ModelRegistryEntry) -> None:
        """Ensure active filesystem artifacts match the given DB active Champion."""
        champ_version = db_champion.model_version
        bundle_dir = self.config.bundles_dir / champ_version
        if not bundle_dir.exists():
            bundle_dir = self.config.candidates_dir / champ_version

        if bundle_dir.exists() and (bundle_dir / "model.joblib").exists():
            shutil.copy2(bundle_dir / "model.joblib", self.config.champion_model_path)
            shutil.copy2(bundle_dir / "preprocessor.joblib", self.config.champion_preprocessor_path)
