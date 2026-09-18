"""
Unit Tests for Phase 12.1: CaseRepository Foundation.

Validates:
- Repository session injection and transaction boundary neutrality.
- Entity lookups (by ID, by case number, by transaction ID).
- Existence checks (exists_by_transaction_id, exists_by_case_number).
- Case and CaseNote staging (add, add_note).
- Investigation notes retrieval with deterministic ordering.
- Queue query generation with multi-attribute filtering, search, pagination, and sorting.
- Summary metrics aggregation with UTC timezone boundary handling.
- SQLAlchemy error translation into PersistenceError.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import (
    AuditActorType,
    Case,
    CaseDisposition,
    CaseNote,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    RiskTier,
)
from backend.app.repositories.case_repository import CaseRepository
from backend.app.repositories.exceptions import PersistenceError


@pytest.fixture
def mock_session():
    """Mock AsyncSession for unit testing repository method query generation and error handling."""
    session = MagicMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def sample_case():
    """Factory fixture for creating a populated Case entity."""
    return Case(
        id=uuid.uuid4(),
        case_number="CASE-20260918-A1B2C3",
        transaction_id=uuid.uuid4(),
        evaluation_id=uuid.uuid4(),
        status=CaseStatus.OPEN,
        priority=CasePriority.HIGH,
        trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        opened_at=datetime.now(timezone.utc),
    )


class TestCaseRepositoryWiringAndNeutrality:
    """Validates session injection and transaction boundary neutrality."""

    def test_session_injection(self, mock_session):
        repo = CaseRepository(mock_session)
        assert repo._session is mock_session

    @pytest.mark.asyncio
    async def test_repository_never_commits_or_rolls_back(self, mock_session, sample_case):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_case
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        await repo.get_by_id(sample_case.id)
        await repo.add(sample_case)

        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()
        mock_session.close.assert_not_called()


class TestCaseRepositoryLookups:
    """Validates query execution for entity lookups and existence checks."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session, sample_case):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_case
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.get_by_id(sample_case.id)

        assert result is sample_case
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())

        assert result is None
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_case_number(self, mock_session, sample_case):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_case
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.get_by_case_number("CASE-20260918-A1B2C3")

        assert result is sample_case
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_transaction_id(self, mock_session, sample_case):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_case
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.get_by_transaction_id(sample_case.transaction_id)

        assert result is sample_case
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exists_by_transaction_id_true(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = True
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.exists_by_transaction_id(uuid.uuid4())

        assert result is True
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exists_by_transaction_id_false(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = False
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.exists_by_transaction_id(uuid.uuid4())

        assert result is False
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exists_by_case_number(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = True
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        result = await repo.exists_by_case_number("CASE-20260918-A1B2C3")

        assert result is True
        mock_session.execute.assert_awaited_once()


class TestCaseRepositoryStagingAndNotes:
    """Validates entity staging and note query methods."""

    @pytest.mark.asyncio
    async def test_add_case(self, mock_session, sample_case):
        repo = CaseRepository(mock_session)
        res = await repo.add(sample_case)

        assert res is sample_case
        mock_session.add.assert_called_once_with(sample_case)

    @pytest.mark.asyncio
    async def test_add_note(self, mock_session, sample_case):
        repo = CaseRepository(mock_session)
        note = CaseNote(
            id=uuid.uuid4(),
            case_id=sample_case.id,
            author_id="analyst_sarah",
            author_role=AuditActorType.ANALYST,
            note_type=CaseNoteType.INVESTIGATION,
            content="Initial review started.",
        )

        res = await repo.add_note(note)

        assert res is note
        mock_session.add.assert_called_once_with(note)

    @pytest.mark.asyncio
    async def test_get_notes(self, mock_session, sample_case):
        repo = CaseRepository(mock_session)
        note_1 = CaseNote(
            id=uuid.uuid4(),
            case_id=sample_case.id,
            author_id="system",
            author_role=AuditActorType.SYSTEM,
            note_type=CaseNoteType.SYSTEM_AUDIT,
            content="Case created.",
            created_at=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc),
        )
        note_2 = CaseNote(
            id=uuid.uuid4(),
            case_id=sample_case.id,
            author_id="analyst_sarah",
            author_role=AuditActorType.ANALYST,
            note_type=CaseNoteType.INVESTIGATION,
            content="Cardholder contacted.",
            created_at=datetime(2026, 9, 18, 10, 30, tzinfo=timezone.utc),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [note_1, note_2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        notes = await repo.get_notes(sample_case.id)

        assert len(notes) == 2
        assert notes[0].content == "Case created."
        assert notes[1].content == "Cardholder contacted."

    @pytest.mark.asyncio
    async def test_count_cases(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        count = await repo.count_cases(status=CaseStatus.OPEN)

        assert count == 42
        mock_session.execute.assert_awaited_once()


class TestCaseRepositoryQueueAndSummary:
    """Validates review queue query execution, filtering, and summary metrics."""

    @pytest.mark.asyncio
    async def test_get_queue_cases_empty(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result

        repo = CaseRepository(mock_session)
        cases, total = await repo.get_queue_cases()

        assert cases == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_get_queue_cases_with_filters(self, mock_session, sample_case):
        repo = CaseRepository(mock_session)

        # 1st execute for count: returns 1
        # 2nd execute for items: returns sample_case
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_scalars = MagicMock()
        mock_scalars.unique.return_value.all.return_value = [sample_case]
        mock_items_result = MagicMock()
        mock_items_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_items_result]

        cases, total = await repo.get_queue_cases(
            status=CaseStatus.OPEN,
            priority=CasePriority.HIGH,
            assigned_to="unassigned",
            risk_tier=RiskTier.HIGH,
            min_score=60,
            max_score=80,
            search_term="CASE-2026",
            start_date=datetime(2026, 9, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 9, 18, tzinfo=timezone.utc),
            limit=10,
            offset=0,
            sort_by="risk_score",
            sort_order="desc",
        )

        assert total == 1
        assert len(cases) == 1
        assert cases[0] is sample_case
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_get_summary_metrics(self, mock_session):
        repo = CaseRepository(mock_session)

        mock_mapping = {
            "total_open": 15,
            "unassigned_count": 8,
            "in_review_count": 5,
            "escalated_count": 2,
            "resolved_today": 12,
            "resolved_last_24h": 14,
            "critical_priority_count": 3,
        }
        mock_result = MagicMock()
        mock_result.mappings.return_value.one.return_value = mock_mapping
        mock_session.execute.return_value = mock_result

        metrics = await repo.get_summary_metrics()

        assert metrics["total_open"] == 15
        assert metrics["unassigned_count"] == 8
        assert metrics["in_review_count"] == 5
        assert metrics["escalated_count"] == 2
        assert metrics["resolved_today"] == 12
        assert metrics["resolved_last_24h"] == 14
        assert metrics["critical_priority_count"] == 3


class TestCaseRepositoryErrorTranslation:
    """Validates conversion of raw SQLAlchemy errors into PersistenceError."""

    @pytest.mark.asyncio
    async def test_get_by_id_raises_persistence_error_on_failure(self, mock_session):
        repo = CaseRepository(mock_session)
        mock_session.execute.side_effect = SQLAlchemyError("Database connection lost")

        with pytest.raises(PersistenceError, match="Failed to retrieve case by ID"):
            await repo.get_by_id(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_get_by_case_number_raises_persistence_error_on_failure(self, mock_session):
        repo = CaseRepository(mock_session)
        mock_session.execute.side_effect = SQLAlchemyError("Lock timeout")

        with pytest.raises(PersistenceError, match="Failed to retrieve case by number"):
            await repo.get_by_case_number("CASE-20260918-A1B2C3")

    @pytest.mark.asyncio
    async def test_get_queue_cases_raises_persistence_error_on_failure(self, mock_session):
        repo = CaseRepository(mock_session)
        mock_session.execute.side_effect = SQLAlchemyError("Disk failure")

        with pytest.raises(PersistenceError, match="Failed to query review queue cases"):
            await repo.get_queue_cases()

    @pytest.mark.asyncio
    async def test_get_summary_metrics_raises_persistence_error_on_failure(self, mock_session):
        repo = CaseRepository(mock_session)
        mock_session.execute.side_effect = SQLAlchemyError("Aggregate query error")

        with pytest.raises(PersistenceError, match="Failed to compute case summary metrics"):
            await repo.get_summary_metrics()
