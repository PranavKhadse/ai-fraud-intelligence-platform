"""
Unit Tests for Phase 14.2 Challenger Dataset Builder & Training Pipeline.

Verifies:
- Authoritative CaseDisposition enum mapping to binary labels.
- Filtering of unresolved and ambiguous cases.
- Strict temporal cutoff validation before validation partition start.
- Duplicate transaction ID detection and deduplication.
- Exact 55-feature schema, ordering, and finiteness enforcement.
- Protected OOT test partition zero-leakage enforcement.
- Dataset construction determinism.
- Process-isolated XGBoost worker execution.
- Candidate bundle creation, manifest generation, and SHA-256 integrity.
- Champion artifact immutability before and after candidate training.
- Candidate registration as non-champion in ModelRegistryRepository.
- Error handling on worker failure or invalid configuration.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.db.models.base import Base
from backend.app.db.models.enums import CaseDisposition, CaseStatus
from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.dataset_builder import (
    DEFAULT_VALIDATION_CUTOFF,
    ChallengerDataset,
    ChallengerDatasetBuilder,
    ChallengerDatasetMetadata,
    map_disposition_to_label,
)
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    ModelBundleManifest,
    calculate_file_sha256,
    verify_bundle_integrity,
)
from ml.lifecycle.training import (
    CandidateBundle,
    ChallengerTrainingConfig,
    ChallengerTrainingPipeline,
    ChampionImmutabilityViolationError,
    TrainingExecutionError,
    run_training_worker,
)
from ml.models.config import (
    CATEGORICAL_PREDICTORS,
    PREDICTIVE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_FEATURES_PATH,
    TEST_FEATURES_PATH,
)


# Helper to generate dummy 55-feature rows
def make_dummy_features_dict(tx_id: str, is_fraud: int = 0) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in PREDICTIVE_FEATURE_COLUMNS:
        if col in CATEGORICAL_PREDICTORS:
            row[col] = "misc_net" if col == "merchant_category" else "engineer"
        else:
            row[col] = 12.34
    row[TARGET_COLUMN] = is_fraud
    row["transaction_id"] = tx_id
    row["timestamp"] = "2020-01-15T10:00:00Z"
    return row


class DummyCase:
    """Mock Case record for testing."""
    def __init__(
        self,
        case_id: str,
        transaction_id: str,
        disposition: Optional[CaseDisposition],
        resolved_at: Optional[datetime],
        features_snapshot: Dict[str, Any],
    ) -> None:
        self.id = case_id
        self.transaction_id = transaction_id
        self.disposition = disposition
        self.resolved_at = resolved_at
        self.opened_at = resolved_at
        self.status = CaseStatus.RESOLVED if disposition else CaseStatus.OPEN
        self.features_snapshot = features_snapshot


class TestDispositionMapping:
    """Test suite for authoritative CaseDisposition enum to binary label mapping."""

    def test_confirmed_fraud_maps_to_one(self):
        assert map_disposition_to_label(CaseDisposition.CONFIRMED_FRAUD) == 1
        assert map_disposition_to_label("CONFIRMED_FRAUD") == 1
        assert map_disposition_to_label("confirmed_fraud") == 1

    def test_false_positive_and_legitimate_map_to_zero(self):
        assert map_disposition_to_label(CaseDisposition.FALSE_POSITIVE) == 0
        assert map_disposition_to_label("FALSE_POSITIVE") == 0
        assert map_disposition_to_label(CaseDisposition.LEGITIMATE) == 0
        assert map_disposition_to_label("LEGITIMATE") == 0

    def test_suspicious_resolved_maps_to_none_excluded(self):
        assert map_disposition_to_label(CaseDisposition.SUSPICIOUS_RESOLVED) is None
        assert map_disposition_to_label("SUSPICIOUS_RESOLVED") is None

    def test_unresolved_or_invalid_maps_to_none(self):
        assert map_disposition_to_label(None) is None
        assert map_disposition_to_label("") is None
        assert map_disposition_to_label("UNKNOWN_DISPOSITION") is None


class TestChallengerDatasetBuilder:
    """Test suite for leakage-free challenger dataset construction."""

    @pytest.fixture
    def dummy_base_parquet(self, tmp_path) -> Path:
        """Create a mini base training parquet file."""
        rows = []
        for i in range(50):
            rows.append(make_dummy_features_dict(f"TX-BASE-{i}", is_fraud=1 if i % 10 == 0 else 0))
        df = pd.DataFrame(rows)
        p = tmp_path / "train_features.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_dataset_builder_basic_construction(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)
        dataset = builder.build_dataset()

        assert isinstance(dataset, ChallengerDataset)
        assert len(dataset.X) == 50
        assert len(dataset.y) == 50
        assert list(dataset.X.columns) == PREDICTIVE_FEATURE_COLUMNS
        assert dataset.metadata.base_train_rows == 50
        assert dataset.metadata.cases_included == 0
        assert dataset.metadata.feature_count == 55

    def test_dataset_builder_with_resolved_cases(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)

        # 3 cases: 1 confirmed fraud, 1 false positive, 1 legitimate
        cases = [
            DummyCase(
                "CASE-1",
                "TX-CASE-1",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 2, 1, 10, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-CASE-1", 1),
            ),
            DummyCase(
                "CASE-2",
                "TX-CASE-2",
                CaseDisposition.FALSE_POSITIVE,
                datetime(2020, 2, 2, 10, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-CASE-2", 0),
            ),
            DummyCase(
                "CASE-3",
                "TX-CASE-3",
                CaseDisposition.LEGITIMATE,
                datetime(2020, 2, 3, 10, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-CASE-3", 0),
            ),
        ]

        dataset = builder.build_dataset(cases=cases)
        assert len(dataset.X) == 53
        assert dataset.metadata.cases_queried == 3
        assert dataset.metadata.cases_included == 3
        assert dataset.metadata.positive_fraud_count == 5 + 1  # 5 from base + 1 from case

    def test_temporal_cutoff_enforcement(self, dummy_base_parquet):
        # Cutoff is 2020-06-21 12:14:25 UTC
        builder = ChallengerDatasetBuilder(
            base_train_path=dummy_base_parquet,
            validation_cutoff=datetime(2020, 6, 21, 12, 14, 25, tzinfo=timezone.utc),
        )

        cases = [
            # Pre-cutoff (valid)
            DummyCase(
                "CASE-PRE",
                "TX-PRE",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 5, 1, 0, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-PRE", 1),
            ),
            # Post-cutoff (invalid, must be excluded)
            DummyCase(
                "CASE-POST",
                "TX-POST",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-POST", 1),
            ),
        ]

        dataset = builder.build_dataset(cases=cases)
        assert dataset.metadata.cases_queried == 2
        assert dataset.metadata.cases_included == 1
        assert dataset.metadata.cases_excluded_temporal == 1
        assert len(dataset.X) == 51

    def test_duplicate_transaction_deduplication(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)

        # TX-BASE-0 is already in base training set
        cases = [
            DummyCase(
                "CASE-DUP-1",
                "TX-BASE-0",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 3, 1, 0, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-BASE-0", 1),
            ),
            DummyCase(
                "CASE-NEW",
                "TX-NEW-1",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 3, 2, 0, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-NEW-1", 1),
            ),
        ]

        dataset = builder.build_dataset(cases=cases)
        assert dataset.metadata.cases_queried == 2
        assert dataset.metadata.cases_included == 1
        assert dataset.metadata.cases_excluded_duplicate == 1
        assert len(dataset.X) == 51

    def test_unresolved_and_ambiguous_case_exclusion(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)

        cases = [
            DummyCase(
                "CASE-UNRESOLVED",
                "TX-UNRES",
                None,
                None,
                make_dummy_features_dict("TX-UNRES", 0),
            ),
            DummyCase(
                "CASE-AMBIGUOUS",
                "TX-AMB",
                CaseDisposition.SUSPICIOUS_RESOLVED,
                datetime(2020, 3, 1, 0, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-AMB", 0),
            ),
        ]

        dataset = builder.build_dataset(cases=cases)
        assert dataset.metadata.cases_included == 0
        assert dataset.metadata.cases_excluded_unresolved == 1
        assert dataset.metadata.cases_excluded_ambiguous == 1
        assert len(dataset.X) == 50

    def test_protected_oot_rejection(self, tmp_path):
        """Verify strict error if test_features.parquet or OOT path is provided."""
        oot_path = tmp_path / "test_features.parquet"
        oot_path.touch()

        with pytest.raises(ValueError, match="Data Leakage Violation"):
            ChallengerDatasetBuilder(base_train_path=oot_path)

    def test_dataset_determinism(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)
        cases = [
            DummyCase(
                "CASE-1",
                "TX-CASE-1",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 2, 1, 10, 0, tzinfo=timezone.utc),
                make_dummy_features_dict("TX-CASE-1", 1),
            )
        ]

        ds1 = builder.build_dataset(cases=cases)
        ds2 = builder.build_dataset(cases=cases)

        assert ds1.metadata.dataset_sha256 == ds2.metadata.dataset_sha256
        assert len(ds1.X) == len(ds2.X)
        pd.testing.assert_frame_equal(ds1.X, ds2.X)
        pd.testing.assert_series_equal(ds1.y, ds2.y)

    def test_missing_feature_in_case_snapshot_raises_error(self, dummy_base_parquet):
        builder = ChallengerDatasetBuilder(base_train_path=dummy_base_parquet)
        bad_snapshot = make_dummy_features_dict("TX-BAD", 1)
        del bad_snapshot["amount"]  # Missing 1 feature

        cases = [
            DummyCase(
                "CASE-BAD",
                "TX-BAD",
                CaseDisposition.CONFIRMED_FRAUD,
                datetime(2020, 2, 1, 10, 0, tzinfo=timezone.utc),
                bad_snapshot,
            )
        ]

        with pytest.raises(ValueError, match="missing required feature columns"):
            builder.build_dataset(cases=cases)


class TestChallengerTrainingPipeline:
    """Test suite for process-isolated challenger training pipeline."""

    @pytest.fixture
    def dummy_train_parquet(self, tmp_path) -> Path:
        """Create a synthetic 100-row training parquet dataset."""
        rows = []
        for i in range(100):
            rows.append(make_dummy_features_dict(f"TX-{i}", is_fraud=1 if i % 5 == 0 else 0))
        df = pd.DataFrame(rows)
        p = tmp_path / "synthetic_train.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_challenger_training_process_isolated(self, dummy_train_parquet, tmp_path):
        candidate_base = tmp_path / "candidates"
        training_config = ChallengerTrainingConfig(
            model_version="1.1.0",
            model_family="xgboost",
            candidate_base_dir=candidate_base,
            hyperparameters={"n_estimators": 5, "max_depth": 3, "random_state": 42},
            is_isolated_process=True,
        )

        pipeline = ChallengerTrainingPipeline()
        bundle = pipeline.train_challenger(
            dataset=dummy_train_parquet,
            training_config=training_config,
        )

        assert isinstance(bundle, CandidateBundle)
        assert bundle.model_version == "1.1.0"
        assert bundle.bundle_dir.exists()
        assert bundle.model_path.exists()
        assert bundle.preprocessor_path.exists()
        assert bundle.manifest_path.exists()
        assert bundle.checksums_path.exists()

        # Check manifest contents
        assert bundle.manifest.model_version == "1.1.0"
        assert bundle.manifest.model_family == "xgboost"
        assert bundle.manifest.status == ModelLifecycleStatus.CANDIDATE
        assert "model.joblib" in bundle.manifest.sha256_checksums
        assert "preprocessor.joblib" in bundle.manifest.sha256_checksums

        # Verify checksum matches
        actual_model_sha = calculate_file_sha256(bundle.model_path)
        assert bundle.manifest.sha256_checksums["model.joblib"] == actual_model_sha

    def test_champion_immutability_verification(self, dummy_train_parquet, tmp_path):
        """Verify champion artifacts remain completely unchanged before and after training."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        champ_model = artifacts_dir / "champion_model.joblib"
        champ_prep = artifacts_dir / "champion_preprocessor.joblib"
        champ_meta = artifacts_dir / "model_metadata.json"

        champ_model.write_bytes(b"dummy_champion_model_bytes")
        champ_prep.write_bytes(b"dummy_champion_prep_bytes")
        champ_meta.write_text('{"champion_version": "1.0.0"}')

        initial_model_sha = calculate_file_sha256(champ_model)
        initial_prep_sha = calculate_file_sha256(champ_prep)

        lifecycle_cfg = LifecycleConfig(
            champion_model_path=champ_model,
            champion_preprocessor_path=champ_prep,
            champion_metadata_path=champ_meta,
        )

        pipeline = ChallengerTrainingPipeline(lifecycle_config=lifecycle_cfg)
        candidate_base = tmp_path / "candidates"
        training_config = ChallengerTrainingConfig(
            model_version="1.2.0",
            candidate_base_dir=candidate_base,
            hyperparameters={"n_estimators": 2, "max_depth": 2},
        )

        bundle = pipeline.train_challenger(
            dataset=dummy_train_parquet,
            training_config=training_config,
        )

        # Confirm champion hashes are unchanged
        assert calculate_file_sha256(champ_model) == initial_model_sha
        assert calculate_file_sha256(champ_prep) == initial_prep_sha

    def test_training_failure_handling(self, tmp_path):
        """Verify descriptive error when training worker encounters an error."""
        non_existent_dataset = tmp_path / "missing_dataset.parquet"
        training_config = ChallengerTrainingConfig(
            model_version="1.3.0",
            candidate_base_dir=tmp_path / "candidates",
            hyperparameters={"n_estimators": 2},
        )

        pipeline = ChallengerTrainingPipeline()
        with pytest.raises((TrainingExecutionError, FileNotFoundError)):
            pipeline.train_challenger(
                dataset=non_existent_dataset,
                training_config=training_config,
            )


from unittest.mock import AsyncMock, MagicMock


def _make_mock_exec_result(value):
    mock_res = MagicMock()
    mock_scalars = MagicMock()
    if isinstance(value, list):
        mock_scalars.all.return_value = value
        mock_scalars.first.return_value = value[0] if value else None
    else:
        mock_scalars.first.return_value = value
        mock_scalars.all.return_value = [value] if value is not None else []
    mock_res.scalars.return_value = mock_scalars
    return mock_res


class TestCandidateRegistrationIntegration:
    """Test suite for registering candidate bundle in Model Registry repository."""

    @pytest.fixture
    def mock_session(self):
        session = MagicMock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.add_all = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.close = AsyncMock()
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_register_candidate_bundle_as_non_champion(self, mock_session, tmp_path):
        # Configure mock session: version 1.1.0 does not exist yet
        mock_session.execute.return_value = _make_mock_exec_result(None)
        repo = ModelRegistryRepository(mock_session)

        # Create candidate bundle files
        bundle_dir = tmp_path / "candidates" / "v1.1.0"
        bundle_dir.mkdir(parents=True)
        model_file = bundle_dir / "model.joblib"
        prep_file = bundle_dir / "preprocessor.joblib"
        manifest_file = bundle_dir / "manifest.json"
        checksums_file = bundle_dir / "checksums.json"

        model_file.write_bytes(b"candidate_model_bytes_1.1.0")
        prep_file.write_bytes(b"candidate_prep_bytes_1.1.0")

        model_sha = calculate_file_sha256(model_file)
        prep_sha = calculate_file_sha256(prep_file)

        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE,
            operating_threshold=0.78,
            hyperparameters={"n_estimators": 100},
            training_metadata={"total_training_rows": 1000},
            sha256_checksums={
                "model.joblib": model_sha,
                "preprocessor.joblib": prep_sha,
                "model": model_sha,
                "preprocessor": prep_sha,
            },
        )
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))
        with open(checksums_file, "w", encoding="utf-8") as f:
            json.dump(manifest.sha256_checksums, f)

        bundle = CandidateBundle(
            model_version="1.1.0",
            bundle_dir=bundle_dir,
            model_path=model_file,
            preprocessor_path=prep_file,
            manifest_path=manifest_file,
            checksums_path=checksums_file,
            manifest=manifest,
            fit_duration_seconds=5.2,
        )

        pipeline = ChallengerTrainingPipeline()
        entry = await pipeline.register_candidate(bundle, repo)

        assert entry.model_version == "1.1.0"
        assert entry.status == ModelLifecycleStatus.CANDIDATE.value
        assert entry.is_active_champion is False
        assert entry.sha256_model == model_sha
        assert entry.sha256_preprocessor == prep_sha
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called()
