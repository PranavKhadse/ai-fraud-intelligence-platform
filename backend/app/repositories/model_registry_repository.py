"""
Model Registry Repository for Phase 14 Model Lifecycle & Controlled Promotion.

Provides asynchronous database access for `model_registry_entries`:
- Querying model entries by ID, semantic version, and lifecycle status.
- Retrieving the unique active production champion model.
- Enforcing lifecycle state transition rules.
- Atomic promotion and rollback of champion model pointers.
- Safe, idempotent initial registration of legacy Champion v1.0.0.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.model_registry import ModelRegistryEntry
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    ModelBundleManifest,
    VALID_LIFECYCLE_TRANSITIONS,
    calculate_file_sha256,
    validate_semantic_version,
)


class ModelRegistryRepository:
    """
    Asynchronous repository managing ModelRegistryEntry persistence and lifecycle operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize repository with an active AsyncSession.
        """
        self._session = session

    async def get_by_id(self, entry_id: uuid.UUID) -> Optional[ModelRegistryEntry]:
        """Retrieve a model registry record by UUID primary key."""
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.id == entry_id)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_by_version(self, model_version: str) -> Optional[ModelRegistryEntry]:
        """Retrieve a model registry record by semantic version string."""
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.model_version == model_version.strip())
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_active_champion(self) -> Optional[ModelRegistryEntry]:
        """Retrieve the single active production champion model entry."""
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.is_active_champion == True)  # noqa: E712
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def list_entries(
        self,
        status: Optional[Union[str, ModelLifecycleStatus]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ModelRegistryEntry]:
        """
        List model registry entries ordered by creation date descending.
        """
        stmt = select(ModelRegistryEntry)
        if status is not None:
            status_str = status.value if isinstance(status, ModelLifecycleStatus) else str(status)
            stmt = stmt.where(ModelRegistryEntry.status == status_str)
            
        stmt = stmt.order_by(ModelRegistryEntry.created_at.desc()).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def create_entry(self, entry: ModelRegistryEntry) -> ModelRegistryEntry:
        """
        Persist a new model registry record.
        """
        if not validate_semantic_version(entry.model_version):
            raise ValueError(f"Invalid model_version '{entry.model_version}'. Must follow semver MAJOR.MINOR.PATCH.")

        existing = await self.get_by_version(entry.model_version)
        if existing is not None:
            raise ValueError(f"Model version '{entry.model_version}' already exists in registry.")

        if entry.is_active_champion:
            active = await self.get_active_champion()
            if active is not None and active.id != entry.id:
                raise ValueError(
                    f"Cannot create new active champion directly while '{active.model_version}' is active. "
                    "Use set_active_champion() to coordinate atomic promotion."
                )

        self._session.add(entry)
        await self._session.flush()
        return entry

    async def update_status(
        self,
        model_version: str,
        new_status: Union[str, ModelLifecycleStatus],
        rolled_back_by: Optional[str] = None,
        rollback_rationale: Optional[str] = None,
    ) -> ModelRegistryEntry:
        """
        Update the lifecycle status of a model entry with state machine validation.
        """
        entry = await self.get_by_version(model_version)
        if entry is None:
            raise ValueError(f"Model version '{model_version}' not found in registry.")

        target_status = new_status if isinstance(new_status, ModelLifecycleStatus) else ModelLifecycleStatus(new_status)
        current_status = ModelLifecycleStatus(entry.status)

        allowed_next = VALID_LIFECYCLE_TRANSITIONS.get(current_status, set())
        if target_status not in allowed_next and target_status != current_status:
            raise ValueError(
                f"Invalid lifecycle transition from '{current_status.value}' to '{target_status.value}'. "
                f"Allowed transitions: {[s.value for s in allowed_next]}"
            )

        entry.status = target_status.value

        if target_status == ModelLifecycleStatus.ROLLED_BACK:
            entry.is_active_champion = False
            entry.rolled_back_at = datetime.now(timezone.utc)
            entry.rolled_back_by = rolled_back_by
            entry.rollback_rationale = rollback_rationale

        await self._session.flush()
        return entry

    async def set_active_champion(
        self,
        model_version: str,
        promoted_by: str,
        promotion_rationale: str,
    ) -> ModelRegistryEntry:
        """
        Atomically promote a model version to active Champion, archiving the previous Champion.
        """
        if not promoted_by or not promoted_by.strip():
            raise ValueError("promoted_by is mandatory for champion promotion.")
        if not promotion_rationale or len(promotion_rationale.strip()) < 15:
            raise ValueError("promotion_rationale must be at least 15 characters long.")

        target_entry = await self.get_by_version(model_version)
        if target_entry is None:
            raise ValueError(f"Target model version '{model_version}' not found in registry.")

        current_status = ModelLifecycleStatus(target_entry.status)
        if current_status not in (ModelLifecycleStatus.CHALLENGER, ModelLifecycleStatus.ARCHIVED, ModelLifecycleStatus.ROLLED_BACK, ModelLifecycleStatus.CANDIDATE):
            raise ValueError(f"Model version '{model_version}' in status '{current_status.value}' cannot be promoted to CHAMPION.")

        now_utc = datetime.now(timezone.utc)

        # Deactivate any current champion
        current_champion = await self.get_active_champion()
        if current_champion is not None and current_champion.id != target_entry.id:
            current_champion.is_active_champion = False
            current_champion.status = ModelLifecycleStatus.ARCHIVED.value

        # Activate target
        target_entry.is_active_champion = True
        target_entry.status = ModelLifecycleStatus.CHAMPION.value
        target_entry.promoted_at = now_utc
        target_entry.promoted_by = promoted_by.strip()
        target_entry.promotion_rationale = promotion_rationale.strip()

        await self._session.flush()
        return target_entry

    async def register_initial_champion_if_empty(
        self,
        config: Optional[LifecycleConfig] = None,
    ) -> Tuple[ModelRegistryEntry, bool]:
        """
        Idempotently register the existing Champion v1.0.0 in the registry.

        Guarantees:
        - Reads existing champion metadata and calculates actual SHA-256 artifact checksums.
        - Does NOT copy, overwrite, retrain, or modify champion artifacts.
        - Idempotent: returns existing v1.0.0 record if already registered and consistent.
        - Fails safely if an inconsistent record exists.

        Returns:
            Tuple[ModelRegistryEntry, bool]: (registry_entry, was_newly_created)
        """
        cfg = config or default_lifecycle_config

        existing = await self.get_by_version("1.0.0")
        if existing is not None:
            # Verify consistency
            actual_model_sha = calculate_file_sha256(cfg.champion_model_path)
            actual_prep_sha = calculate_file_sha256(cfg.champion_preprocessor_path)

            if existing.sha256_model.lower() != actual_model_sha.lower():
                raise ValueError(
                    f"Inconsistent registry state for v1.0.0: model checksum in DB ({existing.sha256_model}) "
                    f"does not match disk artifact ({actual_model_sha})."
                )
            if existing.sha256_preprocessor.lower() != actual_prep_sha.lower():
                raise ValueError(
                    f"Inconsistent registry state for v1.0.0: preprocessor checksum in DB ({existing.sha256_preprocessor}) "
                    f"does not match disk artifact ({actual_prep_sha})."
                )
            return existing, False

        # If files do not exist, raise error
        if not cfg.champion_model_path.exists():
            raise FileNotFoundError(f"Champion model artifact not found at: {cfg.champion_model_path}")
        if not cfg.champion_preprocessor_path.exists():
            raise FileNotFoundError(f"Champion preprocessor artifact not found at: {cfg.champion_preprocessor_path}")

        # Compute checksums
        model_sha = calculate_file_sha256(cfg.champion_model_path)
        prep_sha = calculate_file_sha256(cfg.champion_preprocessor_path)

        # Read model_metadata.json if present
        meta_data: Dict[str, Any] = {}
        if cfg.champion_metadata_path.exists():
            try:
                with open(cfg.champion_metadata_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                meta_data = {}

        val_metrics = meta_data.get("validation_benchmark", {})
        oot_metrics = meta_data.get("oot_test_metrics_frozen_threshold", {})
        hyperparams = meta_data.get("champion_hyperparameters", {})
        training_meta = meta_data.get("partition_statistics", {}).get("train", {})

        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal(str(cfg.default_operating_threshold)),
            bundle_directory=str(cfg.active_artifacts_dir),
            model_artifact_path=str(cfg.champion_model_path),
            preprocessor_artifact_path=str(cfg.champion_preprocessor_path),
            manifest_path=str(cfg.champion_metadata_path) if cfg.champion_metadata_path.exists() else None,
            sha256_model=model_sha,
            sha256_preprocessor=prep_sha,
            sha256_manifest=calculate_file_sha256(cfg.champion_metadata_path) if cfg.champion_metadata_path.exists() else None,
            validation_metrics=val_metrics,
            oot_metrics=oot_metrics,
            hyperparameters=hyperparams,
            training_metadata=training_meta,
            policy_configuration={
                "policy_mode": "TRI_TIER",
                "review_threshold": 0.35,
                "block_threshold": cfg.default_operating_threshold,
            },
            promoted_at=datetime.now(timezone.utc),
            promoted_by="system_initialization",
            promotion_rationale="Initial baseline champion v1.0.0 established in Phase 4/5 model selection.",
        )

        self._session.add(entry)
        await self._session.flush()
        return entry, True
