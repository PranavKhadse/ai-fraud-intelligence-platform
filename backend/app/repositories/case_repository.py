"""
Case Repository for Human Review & Case Management Persistence.

Provides asynchronous CRUD access, queue filtering, summary metrics aggregation,
and investigation context queries for `Case` and `CaseNote` ORM entities.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.case import Case, CaseNote
from backend.app.db.models.enums import AuditEntityType, CasePriority, CaseStatus, RiskTier
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.repositories.exceptions import PersistenceError


class CaseRepository:
    """
    Asynchronous repository for querying and staging `Case` and `CaseNote` ORM entities.

    Design Principles:
    - Session Injection: Receives an active `AsyncSession` externally; never creates sessions.
    - Transaction Boundary Neutrality: Never commits or rolls back implicitly.
    - N+1 Protection: Uses `joinedload` for 1-to-1 parents and `selectinload` for 1-to-many child collections.
    - Exception Wrapping: Converts raw `SQLAlchemyError` exceptions into `PersistenceError`.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the Case repository with an active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(
        self,
        case_id: uuid.UUID,
    ) -> Optional[Case]:
        """
        Query a Case by its primary key UUID with its transaction, evaluation, and notes loaded.

        Args:
            case_id: Internal UUID primary key of the case.

        Returns:
            The Case ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(Case)
                .where(Case.id == case_id)
                .options(
                    joinedload(Case.transaction),
                    joinedload(Case.evaluation),
                    selectinload(Case.notes),
                )
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve case by ID '{case_id}': {exc}"
            ) from exc

    async def get_by_id_for_update(
        self,
        case_id: uuid.UUID,
    ) -> Optional[Case]:
        """
        Query a Case by its primary key UUID with an exclusive row lock (SELECT FOR UPDATE).

        Serializes concurrent mutations of the same locked Case row within the transaction/UoW.

        Args:
            case_id: Internal UUID primary key of the case.

        Returns:
            The locked Case ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(Case)
                .where(Case.id == case_id)
                .with_for_update()
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to acquire row lock for case '{case_id}': {exc}"
            ) from exc

    async def get_case_detail(
        self,
        case_id: uuid.UUID,
    ) -> Optional[Case]:
        """
        Retrieve deep Case investigation detail, eagerly loading Transaction,
        RiskEvaluation, RuleMatches, ReasonCodes, FeatureAttributions, and CaseNotes.

        Args:
            case_id: Internal UUID primary key of the case.

        Returns:
            Deeply loaded Case ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(Case)
                .where(Case.id == case_id)
                .options(
                    joinedload(Case.transaction),
                    joinedload(Case.evaluation).selectinload(RiskEvaluation.rule_matches),
                    joinedload(Case.evaluation).selectinload(RiskEvaluation.reason_codes),
                    joinedload(Case.evaluation).selectinload(RiskEvaluation.feature_attributions),
                    selectinload(Case.notes),
                )
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve detailed case '{case_id}': {exc}"
            ) from exc

    async def get_timeline_events(
        self,
        case_id: uuid.UUID,
        limit: int = 100,
    ) -> List[AuditLog]:
        """
        Retrieve chronological audit events for a case (AuditEntityType.CASE).
        Ordered deterministically by event_timestamp ASC, id ASC.

        Args:
            case_id: Internal UUID primary key of the case.
            limit: Maximum number of audit events to return.

        Returns:
            List of matching AuditLog entities.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(AuditLog)
                .where(
                    and_(
                        AuditLog.entity_type == AuditEntityType.CASE,
                        AuditLog.entity_id == case_id,
                    )
                )
                .order_by(AuditLog.event_timestamp.asc(), AuditLog.id.asc())
                .limit(limit)
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve timeline events for case '{case_id}': {exc}"
            ) from exc

    async def get_by_case_number(
        self,
        case_number: str,
    ) -> Optional[Case]:
        """
        Query a Case by its human-readable case number identifier.

        Args:
            case_number: Unique case reference string (e.g. 'CASE-20260918-A1B2C3').

        Returns:
            The Case ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(Case)
                .where(Case.case_number == case_number)
                .options(
                    joinedload(Case.transaction),
                    joinedload(Case.evaluation),
                    selectinload(Case.notes),
                )
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve case by number '{case_number}': {exc}"
            ) from exc

    async def get_by_transaction_id(
        self,
        transaction_id: uuid.UUID,
    ) -> Optional[Case]:
        """
        Query a Case associated with a specific Transaction ID.

        Args:
            transaction_id: Internal UUID of the evaluated transaction.

        Returns:
            The Case ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(Case)
                .where(Case.transaction_id == transaction_id)
                .options(
                    joinedload(Case.transaction),
                    joinedload(Case.evaluation),
                    selectinload(Case.notes),
                )
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve case for transaction '{transaction_id}': {exc}"
            ) from exc

    async def exists_by_transaction_id(
        self,
        transaction_id: uuid.UUID,
    ) -> bool:
        """
        Efficiently check whether a Case already exists for a Transaction.

        Args:
            transaction_id: Internal UUID of the transaction.

        Returns:
            True if a case exists, otherwise False.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(exists().where(Case.transaction_id == transaction_id))
            result = await self._session.execute(stmt)
            return bool(result.scalar())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to check case existence for transaction '{transaction_id}': {exc}"
            ) from exc

    async def exists_by_case_number(
        self,
        case_number: str,
    ) -> bool:
        """
        Check whether a Case with the specified case number exists.

        Args:
            case_number: Unique case reference string.

        Returns:
            True if a matching case exists, otherwise False.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(exists().where(Case.case_number == case_number))
            result = await self._session.execute(stmt)
            return bool(result.scalar())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to check case existence for number '{case_number}': {exc}"
            ) from exc

    async def add(self, case: Case) -> Case:
        """
        Stage a new Case entity for persistence in the active session.

        Args:
            case: Transient Case entity to attach to the session.

        Returns:
            The same Case instance attached to the session.

        Raises:
            PersistenceError: If staging fails.
        """
        try:
            self._session.add(case)
            return case
        except Exception as exc:
            raise PersistenceError(f"Failed to stage case entity: {exc}") from exc

    async def add_note(self, note: CaseNote) -> CaseNote:
        """
        Stage a new CaseNote entity for persistence in the active session.

        Args:
            note: Transient CaseNote entity to attach to the session.

        Returns:
            The same CaseNote instance attached to the session.

        Raises:
            PersistenceError: If staging fails.
        """
        try:
            self._session.add(note)
            return note
        except Exception as exc:
            raise PersistenceError(f"Failed to stage case note entity: {exc}") from exc

    async def get_notes(
        self,
        case_id: uuid.UUID,
    ) -> List[CaseNote]:
        """
        Retrieve all chronological investigation notes for a case.

        Args:
            case_id: Internal UUID of the parent case.

        Returns:
            List of CaseNote entities ordered by created_at ASC.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(CaseNote)
                .where(CaseNote.case_id == case_id)
                .order_by(CaseNote.created_at.asc())
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve notes for case '{case_id}': {exc}"
            ) from exc

    async def count_cases(
        self,
        status: Optional[CaseStatus] = None,
    ) -> int:
        """
        Count total cases matching an optional status filter.

        Args:
            status: Optional CaseStatus filter.

        Returns:
            Integer total count.

        Raises:
            PersistenceError: If query fails.
        """
        try:
            stmt = select(func.count(Case.id))
            if status is not None:
                stmt = stmt.where(Case.status == status)
            result = await self._session.execute(stmt)
            return int(result.scalar() or 0)
        except SQLAlchemyError as exc:
            raise PersistenceError(f"Failed to count cases: {exc}") from exc

    async def get_queue_cases(
        self,
        status: Optional[CaseStatus] = None,
        priority: Optional[CasePriority] = None,
        assigned_to: Optional[str] = None,
        risk_tier: Optional[RiskTier] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        search_term: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "opened_at",
        sort_order: str = "desc",
    ) -> Tuple[List[Case], int]:
        """
        Query a paginated collection of review queue cases with database-level filtering.

        Joins with `Transaction` and `RiskEvaluation` for filtering and eager loading.

        Args:
            status: Optional CaseStatus filter.
            priority: Optional CasePriority filter.
            assigned_to: Optional analyst ID filter (special value 'unassigned' matches unassigned cases).
            risk_tier: Optional RiskTier filter.
            min_score: Optional minimum risk score (0–100).
            max_score: Optional maximum risk score (0–100).
            search_term: Substring search matching case_number, account_id, merchant_category.
            start_date: Earliest opened_at timestamp.
            end_date: Latest opened_at timestamp.
            limit: Maximum items to return.
            offset: Number of items to skip.
            sort_by: Sort column ('opened_at', 'priority', 'risk_score').
            sort_order: 'asc' or 'desc'.

        Returns:
            Tuple of (list of matching Case ORM entities, total matching count).

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            filters = []

            # 1. Status filter
            if status is not None:
                filters.append(Case.status == status)

            # 2. Priority filter
            if priority is not None:
                filters.append(Case.priority == priority)

            # 3. Assigned to filter (supporting 'unassigned')
            if assigned_to is not None:
                cleaned_assignee = assigned_to.strip()
                if cleaned_assignee.lower() == "unassigned":
                    filters.append(Case.assigned_to.is_(None))
                elif cleaned_assignee:
                    filters.append(Case.assigned_to == cleaned_assignee)

            # 4. Evaluation-based filters (risk_tier, risk_score)
            needs_eval_join = False
            if risk_tier is not None:
                filters.append(RiskEvaluation.risk_tier == risk_tier)
                needs_eval_join = True
            if min_score is not None:
                filters.append(RiskEvaluation.risk_score >= min_score)
                needs_eval_join = True
            if max_score is not None:
                filters.append(RiskEvaluation.risk_score <= max_score)
                needs_eval_join = True

            # 5. Search term (case_number, account_id, merchant_category, external_transaction_id)
            needs_tx_join = False
            if search_term and search_term.strip():
                term = f"%{search_term.strip()}%"
                search_clauses = [
                    Case.case_number.ilike(term),
                    Transaction.account_id.ilike(term),
                    Transaction.merchant_category.ilike(term),
                ]
                filters.append(or_(*search_clauses))
                needs_tx_join = True

            # 6. Date boundaries
            if start_date is not None:
                filters.append(Case.opened_at >= start_date)
            if end_date is not None:
                filters.append(Case.opened_at <= end_date)

            # Count Query
            count_stmt = select(func.count(Case.id))
            if needs_eval_join or sort_by == "risk_score":
                count_stmt = count_stmt.join(Case.evaluation)
            if needs_tx_join:
                count_stmt = count_stmt.join(Case.transaction)
            if filters:
                count_stmt = count_stmt.where(and_(*filters))

            count_res = await self._session.execute(count_stmt)
            total_count = int(count_res.scalar() or 0)

            if total_count == 0:
                return [], 0

            # Items Query
            items_stmt = (
                select(Case)
                .options(
                    joinedload(Case.transaction),
                    joinedload(Case.evaluation),
                )
            )
            if needs_eval_join or sort_by == "risk_score":
                items_stmt = items_stmt.join(Case.evaluation)
            if needs_tx_join:
                items_stmt = items_stmt.join(Case.transaction)
            if filters:
                items_stmt = items_stmt.where(and_(*filters))

            # Sorting
            is_desc = sort_order.lower() == "desc"
            if sort_by == "risk_score":
                sort_col = RiskEvaluation.risk_score.desc() if is_desc else RiskEvaluation.risk_score.asc()
            elif sort_by == "priority":
                sort_col = Case.priority.desc() if is_desc else Case.priority.asc()
            else:
                sort_col = Case.opened_at.desc() if is_desc else Case.opened_at.asc()

            items_stmt = items_stmt.order_by(sort_col).limit(limit).offset(offset)

            items_res = await self._session.execute(items_stmt)
            cases = list(items_res.scalars().unique().all())

            return cases, total_count

        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to query review queue cases: {exc}"
            ) from exc

    async def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Compute operational summary metrics across all review cases using UTC timestamps.

        Returns:
            Dictionary with aggregated KPI counts:
            - total_open: In-flight cases (OPEN, IN_REVIEW, ESCALATED)
            - unassigned_count: OPEN cases without assigned reviewer
            - in_review_count: IN_REVIEW cases
            - escalated_count: ESCALATED cases
            - resolved_today: Resolved in current UTC calendar day (since 00:00:00Z)
            - resolved_last_24h: Resolved in rolling past 24 hours
            - critical_priority_count: Open/In-Review/Escalated cases with CRITICAL priority
        """
        try:
            now_utc = datetime.now(timezone.utc)
            start_of_utc_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            past_24h = now_utc - timedelta(hours=24)

            open_statuses = (CaseStatus.OPEN, CaseStatus.IN_REVIEW, CaseStatus.ESCALATED)

            stmt = select(
                func.count(Case.id).filter(Case.status.in_(open_statuses)).label("total_open"),
                func.count(Case.id).filter(
                    and_(Case.status == CaseStatus.OPEN, Case.assigned_to.is_(None))
                ).label("unassigned_count"),
                func.count(Case.id).filter(Case.status == CaseStatus.IN_REVIEW).label("in_review_count"),
                func.count(Case.id).filter(Case.status == CaseStatus.ESCALATED).label("escalated_count"),
                func.count(Case.id).filter(
                    and_(
                        Case.status.in_((CaseStatus.RESOLVED, CaseStatus.CLOSED)),
                        Case.resolved_at >= start_of_utc_day,
                    )
                ).label("resolved_today"),
                func.count(Case.id).filter(
                    and_(
                        Case.status.in_((CaseStatus.RESOLVED, CaseStatus.CLOSED)),
                        Case.resolved_at >= past_24h,
                    )
                ).label("resolved_last_24h"),
                func.count(Case.id).filter(
                    and_(
                        Case.status.in_(open_statuses),
                        Case.priority == CasePriority.CRITICAL,
                    )
                ).label("critical_priority_count"),
            )

            result = await self._session.execute(stmt)
            row = result.mappings().one()

            return {
                "total_open": int(row["total_open"] or 0),
                "unassigned_count": int(row["unassigned_count"] or 0),
                "in_review_count": int(row["in_review_count"] or 0),
                "escalated_count": int(row["escalated_count"] or 0),
                "resolved_today": int(row["resolved_today"] or 0),
                "resolved_last_24h": int(row["resolved_last_24h"] or 0),
                "critical_priority_count": int(row["critical_priority_count"] or 0),
            }

        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to compute case summary metrics: {exc}"
            ) from exc
