"""
Persistence and Repository Exception Hierarchy.

Provides domain-level exception types for repository and persistence operations,
insulating callers from raw database driver exceptions while preserving diagnostic
context through exception chaining.
"""

from typing import Optional


class PersistenceError(Exception):
    """
    Base exception for all persistence and repository errors.

    Attributes:
        message: Human-readable explanation of the error.
        details: Optional dictionary containing context metadata for debugging.
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PersistenceConflictError(PersistenceError):
    """
    Raised when a persistence operation violates a unique constraint or conflict rule.
    """
    pass


class PersistenceNotFoundError(PersistenceError):
    """
    Raised when an expected persistence entity or record cannot be found.
    """
    pass


__all__ = [
    "PersistenceError",
    "PersistenceConflictError",
    "PersistenceNotFoundError",
]
