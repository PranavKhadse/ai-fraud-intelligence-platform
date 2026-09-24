"""
Challenger Model Training Pipeline for Phase 14.2.

Coordinates process-isolated training of candidate XGBoost models, validates artifact integrity,
enforces champion immutability, and generates strongly typed candidate bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union
import pandas as pd

from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.dataset_builder import ChallengerDataset
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    ModelBundleManifest,
    calculate_file_sha256,
    verify_bundle_integrity,
)
from ml.lifecycle.training.config import ChallengerTrainingConfig

if TYPE_CHECKING:
    from backend.app.db.models.model_registry import ModelRegistryEntry
    from backend.app.repositories.model_registry_repository import ModelRegistryRepository

logger = logging.getLogger("ChallengerTrainingPipeline")


class TrainingExecutionError(Exception):
    """Raised when isolated training worker fails during model execution."""
    pass


class ChampionImmutabilityViolationError(Exception):
    """Raised if active Champion model artifacts are modified during candidate training."""
    pass


@dataclass(frozen=True)
class CandidateBundle:
    """Strongly typed representation of a newly trained candidate model bundle."""
    model_version: str
    bundle_dir: Path
    model_path: Path
    preprocessor_path: Path
    manifest_path: Path
    checksums_path: Path
    manifest: ModelBundleManifest
    fit_duration_seconds: float


class ChallengerTrainingPipeline:
    """
    Orchestrates challenger model training with process isolation and immutability guarantees.
    """

    def __init__(self, lifecycle_config: Optional[LifecycleConfig] = None) -> None:
        self.config = lifecycle_config or default_lifecycle_config

    def _snapshot_champion_checksums(self) -> Dict[str, Optional[str]]:
        """Compute current checksums of active champion artifacts to detect any tampering."""
        checksums: Dict[str, Optional[str]] = {}
        for name, path in [
            ("model", self.config.champion_model_path),
            ("preprocessor", self.config.champion_preprocessor_path),
            ("metadata", self.config.champion_metadata_path),
        ]:
            if path.exists() and path.is_file():
                checksums[name] = calculate_file_sha256(path)
            else:
                checksums[name] = None
        return checksums

    def _verify_champion_immutability(
        self, initial_checksums: Dict[str, Optional[str]]
    ) -> None:
        """Verify that champion artifacts have not been modified or corrupted."""
        current_checksums = self._snapshot_champion_checksums()
        for name, initial_sha in initial_checksums.items():
            current_sha = current_checksums.get(name)
            if initial_sha != current_sha:
                raise ChampionImmutabilityViolationError(
                    f"Champion immutability violation detected! Artifact '{name}' SHA-256 changed "
                    f"from {initial_sha} to {current_sha} during candidate training."
                )

    def train_challenger(
        self,
        dataset: Union[ChallengerDataset, Path, str],
        training_config: Optional[ChallengerTrainingConfig] = None,
    ) -> CandidateBundle:
        """
        Train a new candidate XGBoost model in a process-isolated worker.

        Args:
            dataset: In-memory ChallengerDataset or Path to training parquet dataset.
            training_config: Training configuration (hyperparameters, version, output directory).

        Returns:
            CandidateBundle containing verified artifacts, manifest, and checksums.
        """
        cfg = training_config or ChallengerTrainingConfig()
        output_bundle_dir = cfg.bundle_dir
        output_bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Snapshot champion artifact hashes before training
        initial_champion_hashes = self._snapshot_champion_checksums()

        # 2. Prepare dataset path and metadata
        temp_dataset_file: Optional[Path] = None
        if isinstance(dataset, ChallengerDataset):
            temp_dir = Path(tempfile.mkdtemp(prefix="challenger_dataset_"))
            temp_dataset_file = temp_dir / "challenger_train.parquet"
            dataset.df.to_parquet(temp_dataset_file, index=False)
            dataset_parquet_path = temp_dataset_file
            training_metadata = dataset.metadata.model_dump()
        elif isinstance(dataset, (str, Path)):
            dataset_parquet_path = Path(dataset)
            if not dataset_parquet_path.exists():
                raise FileNotFoundError(f"Training dataset parquet not found: {dataset_parquet_path}")
            training_metadata = {"source_dataset_path": str(dataset_parquet_path)}
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        # 3. Create job configuration JSON
        temp_job_cfg_dir = Path(tempfile.mkdtemp(prefix="training_job_"))
        job_cfg_path = temp_job_cfg_dir / "job_config.json"
        job_payload = {
            "dataset_parquet_path": str(dataset_parquet_path),
            "output_dir": str(output_bundle_dir),
            "model_version": cfg.model_version,
            "operating_threshold": cfg.operating_threshold,
            "hyperparameters": cfg.hyperparameters,
            "training_metadata": training_metadata,
        }
        with open(job_cfg_path, "w", encoding="utf-8") as f:
            json.dump(job_payload, f, indent=2)

        start_time = time.perf_counter()

        try:
            if cfg.is_isolated_process:
                # 4. Launch worker in dedicated subprocess
                cmd = [
                    sys.executable,
                    "-m",
                    "ml.lifecycle.training.worker",
                    "--job-config",
                    str(job_cfg_path),
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=cfg.timeout_seconds,
                )
                if proc.returncode != 0:
                    raise TrainingExecutionError(
                        f"Isolated training worker failed with exit code {proc.returncode}.\n"
                        f"Stderr:\n{proc.stderr}\nStdout:\n{proc.stdout}"
                    )
            else:
                from ml.lifecycle.training.worker import run_training_worker
                run_training_worker(job_cfg_path)

        finally:
            # Clean up temporary dataset if created
            if temp_dataset_file and temp_dataset_file.exists():
                try:
                    temp_dataset_file.unlink()
                except Exception:
                    pass

        fit_duration = round(time.perf_counter() - start_time, 3)

        # 5. Verify champion immutability after training
        self._verify_champion_immutability(initial_champion_hashes)

        # 6. Verify generated candidate bundle
        manifest_path = output_bundle_dir / "manifest.json"
        model_path = output_bundle_dir / "model.joblib"
        prep_path = output_bundle_dir / "preprocessor.joblib"
        checksums_path = output_bundle_dir / "checksums.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not generated at: {manifest_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not generated at: {model_path}")
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not generated at: {prep_path}")
        if not checksums_path.exists():
            raise FileNotFoundError(f"Checksums not generated at: {checksums_path}")

        # Validate physical bundle integrity
        is_valid, error_reason = verify_bundle_integrity(output_bundle_dir)
        if not is_valid:
            raise ValueError(f"Candidate bundle failed physical integrity check: {error_reason}")

        manifest = ModelBundleManifest.load(manifest_path)

        return CandidateBundle(
            model_version=cfg.model_version,
            bundle_dir=output_bundle_dir,
            model_path=model_path,
            preprocessor_path=prep_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            manifest=manifest,
            fit_duration_seconds=fit_duration,
        )

    async def register_candidate(
        self,
        bundle: CandidateBundle,
        repository: ModelRegistryRepository,
    ) -> ModelRegistryEntry:
        """
        Persist a candidate bundle into the platform model registry.

        Guarantees:
        - Registers strictly as ModelLifecycleStatus.CANDIDATE.
        - Sets is_active_champion = False.
        - Preserves existing Champion v1.0.0.
        """
        import uuid
        from decimal import Decimal
        from backend.app.db.models.model_registry import ModelRegistryEntry

        # Check if version already exists
        existing = await repository.get_by_version(bundle.model_version)
        if existing is not None:
            raise ValueError(
                f"Model version '{bundle.model_version}' is already registered in registry."
            )

        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version=bundle.model_version,
            model_family=bundle.manifest.model_family,
            status=ModelLifecycleStatus.CANDIDATE.value,
            is_active_champion=False,
            operating_threshold=Decimal(str(bundle.manifest.operating_threshold)),
            bundle_directory=str(bundle.bundle_dir),
            model_artifact_path=str(bundle.model_path),
            preprocessor_artifact_path=str(bundle.preprocessor_path),
            manifest_path=str(bundle.manifest_path),
            sha256_model=bundle.manifest.sha256_checksums.get("model.joblib", ""),
            sha256_preprocessor=bundle.manifest.sha256_checksums.get("preprocessor.joblib", ""),
            sha256_manifest=calculate_file_sha256(bundle.manifest_path),
            validation_metrics=bundle.manifest.validation_metrics or {},
            oot_metrics=bundle.manifest.oot_holdout_metrics or {},
            hyperparameters=bundle.manifest.hyperparameters,
            training_metadata=bundle.manifest.training_metadata,
            policy_configuration=bundle.manifest.policy_configuration,
        )

        return await repository.create_entry(entry)
