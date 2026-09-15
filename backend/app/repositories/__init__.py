"""
Repositories Package for Data Persistence Abstractions.

Exports:
- `TransactionRepository`: Asynchronous repository for `Transaction` persistence.
- `PersistenceError`: Base domain persistence exception.
- `PersistenceConflictError`: Constraint and conflict exception.
- `PersistenceNotFoundError`: Entity not found exception.
"""

from backend.app.repositories.exceptions import (
    PersistenceError,
    PersistenceConflictError,
    PersistenceNotFoundError,
)
from backend.app.repositories.transaction_repository import TransactionRepository

__all__ = [
    "TransactionRepository",
    "PersistenceError",
    "PersistenceConflictError",
    "PersistenceNotFoundError",
]
