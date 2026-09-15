"""
Transaction Repository for Canonical Financial Transaction Persistence.

Provides asynchronous CRUD access for the `Transaction` ORM model, maintaining
strict transaction-boundary neutrality by delegating commit/rollback ownership
to the caller.
"""

from typing import Optional
import uuid

from sqlalchemy import select, exists
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.transaction import Transaction
from backend.app.repositories.exceptions import PersistenceError


class TransactionRepository:
    """
    Asynchronous repository for querying and staging `Transaction` ORM entities.

    Design Principles:
    - Session Injection: Receives an already-created AsyncSession; never creates its own.
    - Transaction-Boundary Neutrality: Never commits, rolls back, or flushes implicitly.
    - Clean Abstraction: Converts raw SQLAlchemy errors into PersistenceError with exception chaining.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository with an active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(
        self,
        transaction_id: uuid.UUID,
    ) -> Optional[Transaction]:
        """
        Query a Transaction by its primary key UUID.

        Args:
            transaction_id: Internal UUID primary key of the transaction.

        Returns:
            The Transaction ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(Transaction).where(Transaction.id == transaction_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve transaction by ID '{transaction_id}': {exc}"
            ) from exc

    async def get_by_external_id(
        self,
        external_transaction_id: str,
    ) -> Optional[Transaction]:
        """
        Query a Transaction by its client-supplied external transaction identifier.

        Args:
            external_transaction_id: External transaction reference string.

        Returns:
            The Transaction ORM entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(Transaction).where(
                Transaction.external_transaction_id == external_transaction_id
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve transaction by external ID '{external_transaction_id}': {exc}"
            ) from exc

    async def exists_by_external_id(
        self,
        external_transaction_id: str,
    ) -> bool:
        """
        Efficiently check if a Transaction with the specified external ID exists.

        Executes an EXISTS subquery without loading full ORM entity state.

        Args:
            external_transaction_id: External transaction reference string.

        Returns:
            True if a matching transaction exists in the database, otherwise False.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(
                exists().where(Transaction.external_transaction_id == external_transaction_id)
            )
            result = await self._session.execute(stmt)
            return bool(result.scalar())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to check existence for external ID '{external_transaction_id}': {exc}"
            ) from exc

    async def add(
        self,
        transaction: Transaction,
    ) -> Transaction:
        """
        Attach a new Transaction entity to the session for persistence.

        Does not commit or flush; transaction lifecycle is owned by the caller.

        Args:
            transaction: Transaction ORM instance to stage.

        Returns:
            The same Transaction instance attached to the session.
        """
        self._session.add(transaction)
        return transaction


__all__ = ["TransactionRepository"]
