"""
Unit Tests for Phase 14.1 Model Lifecycle Foundation & Model Registry.

Verifies:
- Semantic version validation (MAJOR.MINOR.PATCH).
- Model bundle manifest validation, serialization, and deserialization.
- Cryptographic SHA-256 calculation and tamper detection.
- Bundle directory physical integrity verification.
- Lifecycle state transition rules and terminal states.
- ModelRegistryRepository CRUD operations and active champion exclusivity.
- Safe, idempotent initial registration of legacy Champion v1.0.0.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.db.models.base import Base
from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    ModelBundleManifest,
    EvaluationMetricsSummary,
    PromotionRecord,
    RollbackRecord,
    VALID_LIFECYCLE_TRANSITIONS,
    calculate_file_sha256,
    validate_bundle_manifest,
    validate_semantic_version,
    verify_bundle_integrity,
)


class TestSemanticVersionValidation:
    """Test suite for semantic version string validation."""

    def test_valid_semver_strings(self):
        valid_versions = ["1.0.0", "0.1.0", "2.14.3", "10.0.15", "0.0.1", "1.0.0"]
        for v in valid_versions:
            assert validate_semantic_version(v) is True, f"Expected '{v}' to be valid semver"

    def test_invalid_semver_strings(self):
        invalid_versions = [
            "v1.0.0",        # Prefix 'v' not allowed
            "1.0",           # Missing patch
            "1.0.0.0",       # Four components
            "1.0.0-beta",    # Pre-release tag not allowed in strict semver
            "01.0.0",        # Leading zero in major
            "1.02.0",        # Leading zero in minor
            "1.0.03",        # Leading zero in patch
            "latest",        # Non-numeric
            "",              # Empty string
            "   ",           # Whitespace
            None,            # None
            123,             # Non-string
        ]
        for v in invalid_versions:
            assert validate_semantic_version(v) is False, f"Expected '{v}' to be rejected as invalid semver"


class TestChecksumAndIntegrity:
    """Test suite for cryptographic SHA-256 calculation and bundle integrity checks."""

    def test_calculate_file_sha256(self, tmp_path: Path):
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Fraud Detection Model Test Content 2026", encoding="utf-8")
        
        sha = calculate_file_sha256(test_file)
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert sha == sha.lower()
        
        # Verify deterministic output
        assert calculate_file_sha256(test_file) == sha

    def test_calculate_file_sha256_missing_file_raises(self, tmp_path: Path):
        non_existent = tmp_path / "missing_file.joblib"
        with pytest.raises(FileNotFoundError, match="File not found"):
            calculate_file_sha256(non_existent)

    def test_verify_bundle_integrity_valid(self, tmp_path: Path):
        bundle_dir = tmp_path / "bundle_v1_0_0"
        bundle_dir.mkdir()
        
        model_file = bundle_dir / "model.joblib"
        prep_file = bundle_dir / "preprocessor.joblib"
        manifest_file = bundle_dir / "manifest.json"
        
        model_file.write_bytes(b"MODEL_BINARY_MOCK_DATA")
        prep_file.write_bytes(b"PREPROCESSOR_BINARY_MOCK_DATA")
        
        model_sha = calculate_file_sha256(model_file)
        prep_sha = calculate_file_sha256(prep_file)
        
        manifest = ModelBundleManifest(
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION,
            operating_threshold=0.78,
            sha256_checksums={
                "model": model_sha,
                "preprocessor": prep_sha,
            },
        )
        manifest.save(manifest_file)
        
        is_valid, errors = verify_bundle_integrity(bundle_dir, manifest)
        assert is_valid is True
        assert len(errors) == 0

    def test_verify_bundle_integrity_missing_files(self, tmp_path: Path):
        bundle_dir = tmp_path / "incomplete_bundle"
        bundle_dir.mkdir()
        
        # Missing model.joblib and preprocessor.joblib
        is_valid, errors = verify_bundle_integrity(bundle_dir)
        assert is_valid is False
        assert any("Missing required model artifact" in e for e in errors)
        assert any("Missing required preprocessor artifact" in e for e in errors)

    def test_verify_bundle_integrity_tampered_checksum(self, tmp_path: Path):
        bundle_dir = tmp_path / "tampered_bundle"
        bundle_dir.mkdir()
        
        model_file = bundle_dir / "model.joblib"
        prep_file = bundle_dir / "preprocessor.joblib"
        manifest_file = bundle_dir / "manifest.json"
        
        model_file.write_bytes(b"AUTHENTIC_DATA")
        prep_file.write_bytes(b"PREPROCESSOR_DATA")
        
        manifest = ModelBundleManifest(
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHALLENGER,
            operating_threshold=0.78,
            sha256_checksums={
                "model": "0000000000000000000000000000000000000000000000000000000000000000",
                "preprocessor": calculate_file_sha256(prep_file),
            },
        )
        manifest.save(manifest_file)
        
        is_valid, errors = verify_bundle_integrity(bundle_dir, manifest)
        assert is_valid is False
        assert any("Model artifact SHA-256 checksum mismatch" in e for e in errors)


class TestModelBundleManifestValidation:
    """Test suite for ModelBundleManifest validation rules."""

    def test_valid_manifest_creation_and_roundtrip(self):
        manifest = ModelBundleManifest(
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHALLENGER,
            operating_threshold=0.76,
            hyperparameters={"max_depth": 6, "learning_rate": 0.08},
            training_metadata={"rows": 100000},
            sha256_checksums={
                "model": "a" * 64,
                "preprocessor": "b" * 64,
            },
        )
        
        validate_bundle_manifest(manifest)
        json_str = manifest.to_json()
        restored = ModelBundleManifest.from_json(json_str)
        
        assert restored.model_version == "1.1.0"
        assert restored.model_family == "xgboost"
        assert restored.status == ModelLifecycleStatus.CHALLENGER
        assert restored.operating_threshold == 0.76
        assert restored.hyperparameters["max_depth"] == 6

    def test_invalid_model_family_rejected(self):
        with pytest.raises(ValueError, match="only 'xgboost' is permitted"):
            ModelBundleManifest(
                model_version="1.1.0",
                model_family="lightgbm",
                sha256_checksums={"model": "a" * 64, "preprocessor": "b" * 64},
            )

    def test_invalid_threshold_bounds_rejected(self):
        with pytest.raises(ValueError):
            ModelBundleManifest(
                model_version="1.1.0",
                operating_threshold=1.5,
                sha256_checksums={"model": "a" * 64, "preprocessor": "b" * 64},
            )
        with pytest.raises(ValueError):
            ModelBundleManifest(
                model_version="1.1.0",
                operating_threshold=0.0,
                sha256_checksums={"model": "a" * 64, "preprocessor": "b" * 64},
            )

    def test_invalid_checksum_format_rejected(self):
        with pytest.raises(ValueError, match="Must be a 64-character lowercase hex string"):
            ModelBundleManifest(
                model_version="1.1.0",
                sha256_checksums={"model": "invalid_hex", "preprocessor": "b" * 64},
            )


class TestLifecycleStateTransitions:
    """Test suite for lifecycle state transition rules."""

    def test_allowed_state_transitions(self):
        # CANDIDATE -> CHALLENGER
        assert ModelLifecycleStatus.CHALLENGER in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.CANDIDATE]
        # CANDIDATE -> REJECTED
        assert ModelLifecycleStatus.REJECTED in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.CANDIDATE]
        # CHALLENGER -> CHAMPION
        assert ModelLifecycleStatus.CHAMPION in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.CHALLENGER]
        # CHAMPION -> ARCHIVED
        assert ModelLifecycleStatus.ARCHIVED in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.CHAMPION]
        # CHAMPION -> ROLLED_BACK
        assert ModelLifecycleStatus.ROLLED_BACK in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.CHAMPION]
        # ARCHIVED -> CHAMPION (via rollback)
        assert ModelLifecycleStatus.CHAMPION in VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.ARCHIVED]

    def test_rejected_is_terminal_state(self):
        assert len(VALID_LIFECYCLE_TRANSITIONS[ModelLifecycleStatus.REJECTED]) == 0


from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_session():
    """Mock AsyncSession for verifying transaction-boundary neutrality and query structure."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.flush = AsyncMock()
    return session


def _make_mock_exec_result(value):
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = value
    mock_scalars = MagicMock()
    if isinstance(value, list):
        mock_scalars.all.return_value = value
        mock_scalars.first.return_value = value[0] if value else None
    else:
        mock_scalars.first.return_value = value
        mock_scalars.all.return_value = [value] if value is not None else []
    mock_res.scalars.return_value = mock_scalars
    return mock_res


class TestModelRegistryRepository:
    """Test suite for ModelRegistryRepository database operations."""

    @pytest.fixture
    def sample_entry(self):
        return ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_entry):
        mock_session.execute.return_value = _make_mock_exec_result(sample_entry)

        repo = ModelRegistryRepository(mock_session)
        result = await repo.get_by_id(sample_entry.id)
        assert result is sample_entry
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_session.execute.return_value = _make_mock_exec_result(None)

        repo = ModelRegistryRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_version_found(self, mock_session, sample_entry):
        mock_session.execute.return_value = _make_mock_exec_result(sample_entry)

        repo = ModelRegistryRepository(mock_session)
        result = await repo.get_by_version("1.0.0")
        assert result is sample_entry

    @pytest.mark.asyncio
    async def test_get_active_champion(self, mock_session, sample_entry):
        mock_session.execute.return_value = _make_mock_exec_result(sample_entry)

        repo = ModelRegistryRepository(mock_session)
        result = await repo.get_active_champion()
        assert result is sample_entry
        assert result.is_active_champion is True

    @pytest.mark.asyncio
    async def test_list_entries(self, mock_session, sample_entry):
        mock_session.execute.return_value = _make_mock_exec_result([sample_entry])

        repo = ModelRegistryRepository(mock_session)
        results = await repo.list_entries(status=ModelLifecycleStatus.CHAMPION, limit=10)
        assert len(results) == 1
        assert results[0] is sample_entry

    @pytest.mark.asyncio
    async def test_create_entry_success(self, mock_session):
        mock_session.execute.return_value = _make_mock_exec_result(None)

        repo = ModelRegistryRepository(mock_session)
        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        created = await repo.create_entry(entry)
        assert created.model_version == "1.1.0"
        mock_session.add.assert_called_once_with(entry)
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_duplicate_version_raises(self, mock_session, sample_entry):
        mock_session.execute.return_value = _make_mock_exec_result(sample_entry)

        repo = ModelRegistryRepository(mock_session)
        entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        with pytest.raises(ValueError, match="already exists"):
            await repo.create_entry(entry)

    @pytest.mark.asyncio
    async def test_update_lifecycle_status_valid_and_invalid(self, mock_session):
        candidate_entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CANDIDATE.value,
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )
        mock_session.execute.return_value = _make_mock_exec_result(candidate_entry)

        repo = ModelRegistryRepository(mock_session)
        # Valid: CANDIDATE -> CHALLENGER
        updated = await repo.update_status("1.1.0", ModelLifecycleStatus.CHALLENGER)
        assert updated.status == ModelLifecycleStatus.CHALLENGER.value
        mock_session.flush.assert_awaited()

        # Invalid: CHALLENGER -> CANDIDATE
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            await repo.update_status("1.1.0", ModelLifecycleStatus.CANDIDATE)

    @pytest.mark.asyncio
    async def test_update_lifecycle_status_not_found(self, mock_session):
        mock_session.execute.return_value = _make_mock_exec_result(None)

        repo = ModelRegistryRepository(mock_session)
        with pytest.raises(ValueError, match="not found"):
            await repo.update_status("9.9.9", ModelLifecycleStatus.CHALLENGER)

    @pytest.mark.asyncio
    async def test_set_active_champion_exclusivity(self, mock_session):
        challenger_entry = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.1.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHALLENGER.value,
            is_active_champion=False,
            operating_threshold=Decimal("0.7600"),
            sha256_model="c" * 64,
            sha256_preprocessor="d" * 64,
        )
        existing_champion = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model="a" * 64,
            sha256_preprocessor="b" * 64,
        )

        def mock_execute_side_effect(stmt):
            stmt_str = str(stmt)
            if "model_version = :model_version_1" in stmt_str or "model_version = :model_version" in stmt_str:
                return _make_mock_exec_result(challenger_entry)
            elif "is_active_champion = true" in stmt_str or "is_active_champion = :is_active_champion_1" in stmt_str:
                return _make_mock_exec_result([existing_champion])
            else:
                return _make_mock_exec_result(challenger_entry)

        mock_session.execute.side_effect = mock_execute_side_effect

        repo = ModelRegistryRepository(mock_session)
        promoted = await repo.set_active_champion(
            model_version="1.1.0",
            promoted_by="lead_fraud_analyst",
            promotion_rationale="Passed all gates with superior PR-AUC",
        )

        assert promoted.is_active_champion is True
        assert promoted.status == ModelLifecycleStatus.CHAMPION.value
        assert promoted.promoted_by == "lead_fraud_analyst"
        assert existing_champion.is_active_champion is False
        assert existing_champion.status == ModelLifecycleStatus.ARCHIVED.value

    @pytest.mark.asyncio
    async def test_register_initial_champion_if_empty_creates_new(self, mock_session, tmp_path: Path):
        mock_session.execute.return_value = _make_mock_exec_result(None)

        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        m_file = art_dir / "champion_model.joblib"
        p_file = art_dir / "champion_preprocessor.joblib"
        meta_file = art_dir / "model_metadata.json"

        m_file.write_bytes(b"MOCK_MODEL_BYTES")
        p_file.write_bytes(b"MOCK_PREPROCESSOR_BYTES")
        meta_file.write_text(
            json.dumps({
                "model_version": "1.0.0",
                "champion_hyperparameters": {"max_depth": 6},
                "validation_benchmark": {"pr_auc": 0.9619},
            }),
            encoding="utf-8",
        )

        cfg = LifecycleConfig(
            champion_model_path=m_file,
            champion_preprocessor_path=p_file,
            champion_metadata_path=meta_file,
            active_artifacts_dir=art_dir,
        )

        repo = ModelRegistryRepository(mock_session)
        entry, newly_created = await repo.register_initial_champion_if_empty(cfg)

        assert newly_created is True
        assert entry.model_version == "1.0.0"
        assert entry.is_active_champion is True
        assert entry.status == ModelLifecycleStatus.CHAMPION.value
        assert entry.sha256_model == calculate_file_sha256(m_file)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_register_initial_champion_if_empty_idempotent(self, mock_session, tmp_path: Path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        m_file = art_dir / "champion_model.joblib"
        p_file = art_dir / "champion_preprocessor.joblib"
        meta_file = art_dir / "model_metadata.json"

        m_file.write_bytes(b"MOCK_MODEL_BYTES")
        p_file.write_bytes(b"MOCK_PREPROCESSOR_BYTES")
        meta_file.write_text(
            json.dumps({
                "model_version": "1.0.0",
                "champion_hyperparameters": {"max_depth": 6},
                "validation_benchmark": {"pr_auc": 0.9619},
            }),
            encoding="utf-8",
        )

        cfg = LifecycleConfig(
            champion_model_path=m_file,
            champion_preprocessor_path=p_file,
            champion_metadata_path=meta_file,
            active_artifacts_dir=art_dir,
        )

        existing = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model=calculate_file_sha256(m_file),
            sha256_preprocessor=calculate_file_sha256(p_file),
        )

        mock_session.execute.return_value = _make_mock_exec_result(existing)

        repo = ModelRegistryRepository(mock_session)
        entry, newly_created = await repo.register_initial_champion_if_empty(cfg)

        assert newly_created is False
        assert entry is existing
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_initial_champion_tampered_raises(self, mock_session, tmp_path: Path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        m_file = art_dir / "champion_model.joblib"
        p_file = art_dir / "champion_preprocessor.joblib"
        meta_file = art_dir / "model_metadata.json"

        m_file.write_bytes(b"MOCK_MODEL_BYTES")
        p_file.write_bytes(b"MOCK_PREPROCESSOR_BYTES")
        meta_file.write_text(json.dumps({"model_version": "1.0.0"}), encoding="utf-8")

        cfg = LifecycleConfig(
            champion_model_path=m_file,
            champion_preprocessor_path=p_file,
            champion_metadata_path=meta_file,
            active_artifacts_dir=art_dir,
        )

        existing = ModelRegistryEntry(
            id=uuid.uuid4(),
            model_version="1.0.0",
            model_family="xgboost",
            status=ModelLifecycleStatus.CHAMPION.value,
            is_active_champion=True,
            operating_threshold=Decimal("0.7800"),
            sha256_model="different_checksum_" + "a" * 45,
            sha256_preprocessor=calculate_file_sha256(p_file),
        )

        mock_session.execute.return_value = _make_mock_exec_result(existing)

        repo = ModelRegistryRepository(mock_session)
        with pytest.raises(ValueError, match="Inconsistent registry state"):
            await repo.register_initial_champion_if_empty(cfg)
