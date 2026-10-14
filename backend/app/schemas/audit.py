"""
Pydantic Schemas for Audit Log Verification & Integrity Telemetry.
"""

from datetime import datetime
from typing import List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class AuditIntegrityErrorSchema(BaseModel):
    """Schema representing an individual detected audit integrity anomaly."""
    model_config = ConfigDict(from_attributes=True)

    error_type: str = Field(..., description="Categorized anomaly code (e.g. INVALID_HASH, BROKEN_LINK).")
    message: str = Field(..., description="Human-readable explanation of the detected anomaly.")
    record_id: Optional[uuid.UUID] = Field(None, description="Primary key UUID of the affected audit log record.")
    record_index: int = Field(0, description="0-indexed position within the chronological scan.")
    expected: Optional[str] = Field(None, description="Expected cryptographic hash or constant value.")
    actual: Optional[str] = Field(None, description="Actual stored value found on the record.")


class AuditVerificationResponse(BaseModel):
    """Schema representing the authoritative outcome of an audit verification execution."""
    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(..., description="True if all chained records are cryptographically valid and continuous.")
    total_records: int = Field(..., description="Total audit records scanned.")
    verified_records: int = Field(..., description="Count of successfully validated chained records.")
    legacy_records: int = Field(..., description="Count of pre-chain legacy unchained records acknowledged.")
    errors: List[AuditIntegrityErrorSchema] = Field(default_factory=list, description="List of detected anomalies.")
    first_chained_id: Optional[uuid.UUID] = Field(None, description="Primary key UUID of the first chained record.")
    last_verified_id: Optional[uuid.UUID] = Field(None, description="Primary key UUID of the latest verified record.")
    last_verified_hash: Optional[str] = Field(None, description="SHA-256 digest of the chain head record.")
    verified_at: datetime = Field(..., description="UTC timestamp when the verification was executed.")
    limitations: List[str] = Field(default_factory=list, description="Documented verification scope boundaries.")
    summary: str = Field(..., description="Human-readable outcome summary.")
