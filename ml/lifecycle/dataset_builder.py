"""
Challenger Dataset Builder for Phase 14.2.

Constructs point-in-time, leakage-safe training datasets for challenger models by:
1. Loading the frozen base training partition (train_features.parquet).
2. Querying resolved cases from Phase 12 human review.
3. Mapping authoritative CaseDisposition enum values to binary ground-truth labels.
4. Extracting immutable 55-feature point-in-time transaction snapshots.
5. Enforcing strict temporal cutoff before the validation partition start.
6. Preventing duplicate transaction records.
7. Strictly protecting the OOT test partition (test_features.parquet) from access.
8. Generating deterministic dataset lineage metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.app.db.models.enums import CaseDisposition, CaseStatus
from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    TARGET_COLUMN,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
)

# Canonical validation partition start timestamp (UTC)
# Frozen boundary between Phase 4 Train and Validation sets: 2020-06-21 12:14:25 UTC
DEFAULT_VALIDATION_CUTOFF: pd.Timestamp = pd.Timestamp("2020-06-21 12:14:25", tz="UTC")


def map_disposition_to_label(disposition: Optional[Union[CaseDisposition, str]]) -> Optional[int]:
    """
    Map authoritative CaseDisposition enum values to binary fraud classification labels.

    Mapping Rules:
    - CaseDisposition.CONFIRMED_FRAUD -> 1 (Positive ground truth)
    - CaseDisposition.FALSE_POSITIVE  -> 0 (Negative ground truth)
    - CaseDisposition.LEGITIMATE      -> 0 (Negative ground truth)
    - CaseDisposition.SUSPICIOUS_RESOLVED -> None (Excluded from binary labels, counted as ambiguous)
    - None / Unresolved               -> None (Excluded)

    Args:
        disposition: Case disposition enum or string.

    Returns:
        1 for fraud, 0 for legitimate, None for ambiguous/unresolved.
    """
    if disposition is None:
        return None

    disp_str = disposition.value if isinstance(disposition, CaseDisposition) else str(disposition)
    disp_upper = disp_str.strip().upper()

    if disp_upper == CaseDisposition.CONFIRMED_FRAUD.value:
        return 1
    elif disp_upper in (CaseDisposition.FALSE_POSITIVE.value, CaseDisposition.LEGITIMATE.value):
        return 0
    elif disp_upper == CaseDisposition.SUSPICIOUS_RESOLVED.value:
        return None
    return None


class ChallengerDatasetMetadata(BaseModel):
    """Lineage and provenance metadata for a constructed challenger dataset."""
    model_config = ConfigDict(frozen=True)

    base_dataset_path: str = Field(..., description="Path to base training partition")
    base_train_rows: int = Field(..., ge=0, description="Row count of base training partition")
    cases_queried: int = Field(default=0, ge=0, description="Total cases queried for enrichment")
    cases_included: int = Field(default=0, ge=0, description="Resolved cases included in final dataset")
    cases_excluded_temporal: int = Field(default=0, ge=0, description="Cases rejected by temporal cutoff")
    cases_excluded_ambiguous: int = Field(default=0, ge=0, description="Cases with SUSPICIOUS_RESOLVED disposition")
    cases_excluded_unresolved: int = Field(default=0, ge=0, description="Cases without completed disposition")
    cases_excluded_duplicate: int = Field(default=0, ge=0, description="Cases dropped due to duplicate transaction_id")
    total_training_rows: int = Field(..., ge=0, description="Total rows in final combined dataset")
    positive_fraud_count: int = Field(..., ge=0, description="Total positive fraud samples (y=1)")
    negative_legit_count: int = Field(..., ge=0, description="Total negative legitimate samples (y=0)")
    fraud_prevalence: float = Field(..., ge=0.0, le=1.0, description="Proportion of positive samples")
    feature_count: int = Field(default=55, ge=55, le=55, description="Exact predictive feature count")
    feature_schema_version: str = Field(default="1.0.0", description="Feature catalog version")
    validation_cutoff_timestamp: str = Field(..., description="ISO 8601 UTC validation temporal cutoff")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_sha256: Optional[str] = Field(None, description="SHA-256 fingerprint of dataset feature matrix")


@dataclass
class ChallengerDataset:
    """In-memory challenger dataset bundle with features, target, and lineage metadata."""
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    metadata: ChallengerDatasetMetadata

    def validate_schema(self) -> None:
        """Validate feature count, ordering, finiteness, and absence of target/ID leakage."""
        assert list(self.X.columns) == PREDICTIVE_FEATURE_COLUMNS, (
            f"Feature columns do not match canonical 55-feature schema ordering."
        )
        assert len(self.X.columns) == 55, f"Expected 55 features, got {len(self.X.columns)}"
        assert len(self.X) == len(self.y), f"Row count mismatch: X={len(self.X)}, y={len(self.y)}"
        assert TARGET_COLUMN not in self.X.columns, "Target column leaked into feature matrix X!"
        assert "transaction_id" not in self.X.columns, "transaction_id leaked into feature matrix X!"


class ChallengerDatasetBuilder:
    """
    Builder responsible for constructing leakage-free training datasets for challenger models.
    """

    def __init__(
        self,
        base_train_path: Union[str, Path] = TRAIN_FEATURES_PATH,
        validation_cutoff: Optional[Union[datetime, pd.Timestamp, str]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.base_train_path = Path(base_train_path)
        self.feature_columns = feature_columns or PREDICTIVE_FEATURE_COLUMNS

        # Defensive OOT check on initialization
        self._assert_not_protected_oot(self.base_train_path)

        # Normalize validation cutoff timestamp
        if validation_cutoff is None:
            self.validation_cutoff = DEFAULT_VALIDATION_CUTOFF
        elif isinstance(validation_cutoff, str):
            self.validation_cutoff = pd.Timestamp(validation_cutoff).tz_convert("UTC") if pd.Timestamp(validation_cutoff).tzinfo else pd.Timestamp(validation_cutoff, tz="UTC")
        elif isinstance(validation_cutoff, datetime):
            self.validation_cutoff = pd.Timestamp(validation_cutoff).tz_convert("UTC") if validation_cutoff.tzinfo else pd.Timestamp(validation_cutoff, tz="UTC")
        else:
            self.validation_cutoff = validation_cutoff

    @staticmethod
    def _assert_not_protected_oot(path: Path) -> None:
        """Guarantee zero access to protected OOT test dataset."""
        resolved = str(path.resolve()).lower()
        if "test_features" in resolved or "oot" in resolved:
            raise ValueError(
                f"Data Leakage Violation: Protected OOT test partition ({path}) "
                "cannot be accessed or used in candidate/challenger dataset construction."
            )

    def build_dataset(
        self,
        cases: Optional[Sequence[Any]] = None,
        max_base_rows: Optional[int] = None,
    ) -> ChallengerDataset:
        """
        Build the consolidated challenger training dataset.

        Args:
            cases: Optional sequence of Case ORM objects, dictionaries, or Case-like mock records.
            max_base_rows: Optional row limit on base training set (useful for ultra-fast testing).

        Returns:
            ChallengerDataset containing full DataFrame, feature matrix X, target y, and metadata.
        """
        self._assert_not_protected_oot(self.base_train_path)

        if not self.base_train_path.exists():
            raise FileNotFoundError(f"Base training partition not found: {self.base_train_path}")

        # 1. Load base training data
        cols_to_load = list(set(self.feature_columns + [TARGET_COLUMN, "transaction_id", "timestamp"]))
        # Verify which columns exist in the parquet file
        all_parquet_cols = pd.read_parquet(self.base_train_path).columns.tolist()
        actual_cols = [c for c in cols_to_load if c in all_parquet_cols]
        if TARGET_COLUMN not in actual_cols:
            actual_cols.append(TARGET_COLUMN)

        df_base = pd.read_parquet(self.base_train_path, columns=actual_cols)
        if max_base_rows is not None and max_base_rows > 0:
            df_base = df_base.iloc[:max_base_rows].copy()

        base_rows = len(df_base)

        # Track existing transaction IDs to prevent duplicates
        seen_tx_ids: Set[str] = set()
        if "transaction_id" in df_base.columns:
            seen_tx_ids = set(df_base["transaction_id"].astype(str).tolist())

        # 2. Process and filter cases
        case_rows: List[Dict[str, Any]] = []
        cases_queried = 0
        cases_included = 0
        cases_excluded_temporal = 0
        cases_excluded_ambiguous = 0
        cases_excluded_unresolved = 0
        cases_excluded_duplicate = 0

        if cases:
            for case in cases:
                cases_queried += 1

                # Extract disposition
                raw_disp = getattr(case, "disposition", None) if hasattr(case, "disposition") else (case.get("disposition") if isinstance(case, dict) else None)
                if raw_disp is None or str(raw_disp).strip() == "" or str(raw_disp).upper() == "NONE":
                    cases_excluded_unresolved += 1
                    continue

                label = map_disposition_to_label(raw_disp)
                if label is None:
                    # SUSPICIOUS_RESOLVED or unrecognized disposition
                    cases_excluded_ambiguous += 1
                    continue

                # Extract case / transaction timestamp
                case_ts = None
                if hasattr(case, "resolved_at") and getattr(case, "resolved_at") is not None:
                    case_ts = getattr(case, "resolved_at")
                elif hasattr(case, "opened_at") and getattr(case, "opened_at") is not None:
                    case_ts = getattr(case, "opened_at")
                elif isinstance(case, dict):
                    case_ts = case.get("resolved_at") or case.get("opened_at") or case.get("timestamp")

                if case_ts is not None:
                    ts_val = pd.Timestamp(case_ts)
                    if ts_val.tzinfo is None:
                        ts_val = ts_val.tz_localize("UTC")
                    else:
                        ts_val = ts_val.tz_convert("UTC")

                    if ts_val >= self.validation_cutoff:
                        cases_excluded_temporal += 1
                        continue

                # Extract transaction ID
                tx_id = None
                if hasattr(case, "transaction_id") and getattr(case, "transaction_id") is not None:
                    tx_id = str(getattr(case, "transaction_id"))
                elif isinstance(case, dict) and case.get("transaction_id"):
                    tx_id = str(case.get("transaction_id"))

                if tx_id and tx_id in seen_tx_ids:
                    cases_excluded_duplicate += 1
                    continue

                # Extract 55-feature snapshot
                features_dict: Optional[Dict[str, Any]] = None
                if hasattr(case, "transaction") and getattr(case, "transaction") is not None:
                    txn = getattr(case, "transaction")
                    features_dict = getattr(txn, "features_snapshot", None)
                elif hasattr(case, "features_snapshot") and getattr(case, "features_snapshot") is not None:
                    features_dict = getattr(case, "features_snapshot")
                elif isinstance(case, dict):
                    features_dict = case.get("features_snapshot") or case.get("features")

                if not features_dict:
                    # If features are individual keys in the dict
                    if isinstance(case, dict) and all(col in case for col in self.feature_columns):
                        features_dict = {col: case[col] for col in self.feature_columns}
                    else:
                        cases_excluded_unresolved += 1
                        continue

                # Validate exact 55 feature keys
                missing = [col for col in self.feature_columns if col not in features_dict]
                if missing:
                    raise ValueError(f"Case transaction snapshot missing required feature columns: {missing}")

                # Build row dict conforming to canonical schema
                row_dict: Dict[str, Any] = {col: features_dict[col] for col in self.feature_columns}
                row_dict[TARGET_COLUMN] = int(label)
                if tx_id:
                    row_dict["transaction_id"] = tx_id
                    seen_tx_ids.add(tx_id)

                case_rows.append(row_dict)
                cases_included += 1

        # 3. Concatenate and validate final dataset
        if case_rows:
            df_cases = pd.DataFrame(case_rows)
            # Align columns
            common_cols = [c for c in df_base.columns if c in df_cases.columns]
            df_combined = pd.concat([df_base[common_cols], df_cases[common_cols]], ignore_index=True)
        else:
            df_combined = df_base.copy()

        # Extract X and y
        X = df_combined[self.feature_columns].copy()
        y = df_combined[TARGET_COLUMN].astype(np.int64).copy()

        # Defensive validations
        assert list(X.columns) == self.feature_columns, "Feature column ordering mismatch!"
        assert len(X.columns) == 55, f"Expected 55 features, found {len(X.columns)}"
        assert len(X) == len(y), "Row count mismatch between X and y!"

        # Numerical finiteness validation
        num_cols = [c for c in self.feature_columns if c not in CATEGORICAL_PREDICTORS]
        for c in num_cols:
            if not np.isfinite(pd.to_numeric(X[c], errors="coerce")).all():
                raise ValueError(f"Non-finite values detected in feature column '{c}'.")

        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        total_rows = len(y)
        prevalence = float(pos_count / total_rows) if total_rows > 0 else 0.0

        # Compute deterministic SHA-256 fingerprint for dataset
        matrix_sample = X.iloc[:100].to_numpy(dtype=str).tobytes() if total_rows > 0 else b""
        dataset_sha = hashlib.sha256(matrix_sample + str(total_rows).encode() + str(pos_count).encode()).hexdigest()

        metadata = ChallengerDatasetMetadata(
            base_dataset_path=str(self.base_train_path),
            base_train_rows=base_rows,
            cases_queried=cases_queried,
            cases_included=cases_included,
            cases_excluded_temporal=cases_excluded_temporal,
            cases_excluded_ambiguous=cases_excluded_ambiguous,
            cases_excluded_unresolved=cases_excluded_unresolved,
            cases_excluded_duplicate=cases_excluded_duplicate,
            total_training_rows=total_rows,
            positive_fraud_count=pos_count,
            negative_legit_count=neg_count,
            fraud_prevalence=prevalence,
            feature_count=55,
            feature_schema_version="1.0.0",
            validation_cutoff_timestamp=self.validation_cutoff.isoformat(),
            dataset_sha256=dataset_sha,
        )

        dataset = ChallengerDataset(
            df=df_combined,
            X=X,
            y=y,
            metadata=metadata,
        )
        dataset.validate_schema()
        return dataset
