"""
Unit of Work (UoW) Pattern Implementation for Asynchronous Persistence Coordination.

Provides transactional boundary ownership and repository orchestration for the
fraud intelligence platform using SQLAlchemy 2.0 AsyncSession.
"""

from types import TracebackType
from typing import Optional, Type
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.repositories.audit_log_repository import AuditLogRepository
from backend.app.repositories.feature_attribution_repository import FeatureAttributionRepository
from backend.app.repositories.reason_code_repository import ReasonCodeRepository
from backend.app.repositories.risk_evaluation_repository import RiskEvaluationRepository
from backend.app.repositories.rule_match_repository import RuleMatchRepository
from backend.app.repositories.transaction_repository import TransactionRepository


class FraudPersistenceUnitOfWork:
    """
    Asynchronous Unit of Work coordinating persistence repositories under a single transaction.

    Design Principles:
    - Session Injection: Operates strictly on an externally managed `AsyncSession`.
      Never creates engines or sessions independently.
    - Repository Coordination: Exposes all repository instances bound to the same injected session.
    - Transaction Boundary Ownership: Exposes explicit `commit()`, `rollback()`, and `flush()` methods.
    - Context Manager Safety: When used as an async context manager (`async with ...`), any unhandled
      exception triggers an automatic defensive rollback before re-raising. Normal context exit does
      NOT commit automatically, enforcing explicit caller intent for persistence.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the Unit of Work with an existing active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.

        Raises:
            ValueError: If session is None.
        """
        if session is None:
            raise ValueError("AsyncSession must not be None.")
        self._session = session

        # Instantiate repositories bound to the same session
        self._transactions = TransactionRepository(self._session)
        self._risk_evaluations = RiskEvaluationRepository(self._session)
        self._rule_matches = RuleMatchRepository(self._session)
        self._reason_codes = ReasonCodeRepository(self._session)
        self._feature_attributions = FeatureAttributionRepository(self._session)
        self._audit_logs = AuditLogRepository(self._session)

        # Context state tracking
        self._in_context: bool = False

    @property
    def session(self) -> AsyncSession:
        """Return the underlying AsyncSession."""
        return self._session

    @property
    def transactions(self) -> TransactionRepository:
        """Transaction persistence repository."""
        return self._transactions

    @property
    def risk_evaluations(self) -> RiskEvaluationRepository:
        """RiskEvaluation persistence repository."""
        return self._risk_evaluations

    @property
    def rule_matches(self) -> RuleMatchRepository:
        """EvaluationRuleMatch persistence repository."""
        return self._rule_matches

    @property
    def reason_codes(self) -> ReasonCodeRepository:
        """EvaluationReasonCode persistence repository."""
        return self._reason_codes

    @property
    def feature_attributions(self) -> FeatureAttributionRepository:
        """EvaluationFeatureAttribution persistence repository."""
        return self._feature_attributions

    @property
    def audit_logs(self) -> AuditLogRepository:
        """AuditLog persistence repository."""
        return self._audit_logs

    @property
    def in_context(self) -> bool:
        """Whether the Unit of Work is currently inside an async with context block."""
        return self._in_context

    async def commit(self) -> None:
        """
        Explicitly commit pending database changes in the current session.

        Does not catch or swallow database exceptions, allowing caller or context
        to handle and observe persistence failures.
        """
        await self._session.commit()

    async def rollback(self) -> None:
        """
        Explicitly rollback pending uncommitted database changes in the session.
        """
        await self._session.rollback()

    async def flush(self) -> None:
        """
        Flush pending changes to the database without committing the transaction.

        Useful when generated primary keys or database defaults must be materialized
        before subsequent repository operations within the same unit of work.
        """
        await self._session.flush()

    async def __aenter__(self) -> "FraudPersistenceUnitOfWork":
        """
        Enter the async context manager.

        Returns:
            The FraudPersistenceUnitOfWork instance.
        """
        self._in_context = True
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the async context manager.

        If an exception was raised within the context, performs an automatic defensive
        rollback before letting the exception propagate.
        If no exception occurred, does NOT commit automatically (explicit commit is required).
        """
        self._in_context = False
        if exc_val is not None:
            await self.rollback()
        # Returns None / False so any exception is re-raised automatically.


# Convenient alias
UnitOfWork = FraudPersistenceUnitOfWork

__all__ = [
    "FraudPersistenceUnitOfWork",
    "UnitOfWork",
]
