"""
Pydantic Schemas, Enums, and Integrity Validators for Phase 14 Model Lifecycle.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLifecycleStatus(str, Enum):
    """Lifecycle status states for a versioned model bundle."""
    CANDIDATE = "CANDIDATE"        # Initial trained model awaiting validation
    CHALLENGER = "CHALLENGER"      # Validated candidate actively benchmarked against Champion
    CHAMPION = "CHAMPION"          # Active production champion model serving live inference
    REJECTED = "REJECTED"          # Candidate or Challenger that failed gate criteria
    ARCHIVED = "ARCHIVED"          # Historical Champion retired after promotion of a new Champion
    ROLLED_BACK = "ROLLED_BACK"    # Demoted Champion following an emergency rollback operation


# Strict state machine transition rules for model lifecycle
VALID_LIFECYCLE_TRANSITIONS: Dict[ModelLifecycleStatus, Set[ModelLifecycleStatus]] = {
    ModelLifecycleStatus.CANDIDATE: {
        ModelLifecycleStatus.CHALLENGER,
        ModelLifecycleStatus.REJECTED,
    },
    ModelLifecycleStatus.CHALLENGER: {
        ModelLifecycleStatus.CHAMPION,
        ModelLifecycleStatus.REJECTED,
        ModelLifecycleStatus.ARCHIVED,
    },
    ModelLifecycleStatus.CHAMPION: {
        ModelLifecycleStatus.ARCHIVED,
        ModelLifecycleStatus.ROLLED_BACK,
    },
    ModelLifecycleStatus.ARCHIVED: {
        ModelLifecycleStatus.CHAMPION,  # Allowed when restoring via rollback
    },
    ModelLifecycleStatus.ROLLED_BACK: {
        ModelLifecycleStatus.CHAMPION,  # Allowed if subsequently re-promoted
        ModelLifecycleStatus.ARCHIVED,
    },
    ModelLifecycleStatus.REJECTED: set(),  # Terminal state
}


# Strict semantic versioning regex: MAJOR.MINOR.PATCH (non-negative integers without leading zeroes)
SEMVER_REGEX = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_REGEX = re.compile(r"^[a-f0-9]{64}$")


def validate_semantic_version(version: str) -> bool:
    """
    Validate that a version string strictly adheres to Semantic Versioning MAJOR.MINOR.PATCH.
    """
    if not isinstance(version, str) or not version.strip():
        return False
    return bool(SEMVER_REGEX.match(version.strip()))


def calculate_file_sha256(filepath: Union[str, Path]) -> str:
    """
    Compute cryptographic SHA-256 hex digest for a file.
    
    Args:
        filepath: Path to the target file.
        
    Returns:
        64-character lowercase hexadecimal SHA-256 checksum string.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    p = Path(filepath).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {p}")
        
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class EvaluationMetricsSummary(BaseModel):
    """Summary of model performance and operational decision metrics."""
    model_config = ConfigDict(frozen=True)

    pr_auc: float = Field(..., ge=0.0, le=1.0, description="Precision-Recall Area Under Curve")
    roc_auc: float = Field(..., ge=0.0, le=1.0, description="Receiver Operating Characteristic AUC")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision at operating threshold")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall / Fraud Catch Rate at operating threshold")
    f1: float = Field(..., ge=0.0, le=1.0, description="F1-Score at operating threshold")
    fpr: float = Field(..., ge=0.0, le=1.0, description="False Positive Rate at operating threshold")
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification accuracy")
    threshold: float = Field(..., gt=0.0, lt=1.0, description="Operating probability decision threshold")
    
    # Confusion Matrix Counts
    tp: Optional[int] = Field(None, ge=0, description="True Positive count")
    fp: Optional[int] = Field(None, ge=0, description="False Positive count")
    tn: Optional[int] = Field(None, ge=0, description="True Negative count")
    fn: Optional[int] = Field(None, ge=0, description="False Negative count")
    total_samples: Optional[int] = Field(None, ge=0, description="Total evaluation sample size")
    
    # Financial & Operational Metrics
    expected_cost: Optional[float] = Field(None, ge=0.0, description="Total expected financial cost under cost matrix")
    fraud_in_block_count: Optional[int] = Field(None, ge=0, description="Frauds routed to hard BLOCK tier")
    fraud_in_review_count: Optional[int] = Field(None, ge=0, description="Frauds routed to manual REVIEW queue")
    review_queue_purity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision of the review queue")


class PromotionRecord(BaseModel):
    """Audit record capturing formal promotion sign-off."""
    model_config = ConfigDict(frozen=True)

    promoted_at: str = Field(..., description="ISO 8601 UTC timestamp of promotion")
    promoted_by: str = Field(..., min_length=3, description="User/analyst identifier authoring promotion")
    promotion_rationale: str = Field(..., min_length=15, description="Mandatory governance business rationale")
    previous_champion_version: Optional[str] = Field(None, description="Model version of demoted Champion")


class RollbackRecord(BaseModel):
    """Audit record capturing emergency model rollback."""
    model_config = ConfigDict(frozen=True)

    rolled_back_at: str = Field(..., description="ISO 8601 UTC timestamp of rollback")
    rolled_back_by: str = Field(..., min_length=3, description="User/analyst identifier authoring rollback")
    rollback_rationale: str = Field(..., min_length=15, description="Mandatory explanatory rationale")
    restored_version: str = Field(..., description="Target model version restored to Champion")
    demoted_version: str = Field(..., description="Model version demoted from Champion")


class ModelBundleManifest(BaseModel):
    """
    Canonical metadata manifest for a versioned Model Bundle.
    Couples model binary, preprocessor, operating threshold, and configuration into a unified entity.
    """
    model_config = ConfigDict(extra="ignore")

    model_version: str = Field(..., description="Semantic version string MAJOR.MINOR.PATCH")
    model_family: str = Field(default="xgboost", description="Model architecture family (strictly 'xgboost')")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: ModelLifecycleStatus = Field(default=ModelLifecycleStatus.CANDIDATE)
    operating_threshold: float = Field(default=0.78, gt=0.0, lt=1.0, description="Calibrated decision threshold")
    
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model training hyperparameters")
    training_metadata: Dict[str, Any] = Field(default_factory=dict, description="Training dataset metadata & duration")
    evaluation_configuration: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation & threshold sweep configuration")
    policy_configuration: Dict[str, Any] = Field(default_factory=dict, description="Coupled decision policy configuration")
    
    validation_metrics: Optional[Dict[str, Any]] = Field(None, description="Benchmark metrics on Validation split")
    oot_holdout_metrics: Optional[Dict[str, Any]] = Field(None, description="Protected OOT holdout benchmark metrics")
    
    sha256_checksums: Dict[str, str] = Field(default_factory=dict, description="SHA-256 hashes of bundle files")
    promotion_record: Optional[PromotionRecord] = Field(None, description="Promotion audit record if promoted")
    rollback_records: List[RollbackRecord] = Field(default_factory=list, description="Historical rollback records")

    @field_validator("model_version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        if not validate_semantic_version(v):
            raise ValueError(f"Invalid model_version '{v}'. Must adhere to semantic versioning format MAJOR.MINOR.PATCH (e.g. '1.1.0').")
        return v.strip()

    @field_validator("model_family")
    @classmethod
    def validate_family(cls, v: str) -> str:
        if v.lower() != "xgboost":
            raise ValueError(f"Invalid model_family '{v}'. In Phase 14, only 'xgboost' is permitted as a promotable model family.")
        return "xgboost"

    @field_validator("sha256_checksums")
    @classmethod
    def validate_checksums_dict(cls, v: Dict[str, str]) -> Dict[str, str]:
        for key, val in v.items():
            if not isinstance(val, str) or not SHA256_REGEX.match(val.lower()):
                raise ValueError(f"Invalid SHA-256 checksum for '{key}': '{val}'. Must be a 64-character lowercase hex string.")
        return {k: val.lower() for k, val in v.items()}

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelBundleManifest":
        """Deserialize manifest from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> "ModelBundleManifest":
        """Load manifest from file path."""
        p = Path(manifest_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, destination_path: Union[str, Path]) -> Path:
        """Save manifest to file path."""
        p = Path(destination_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))
        return p


def validate_bundle_manifest(manifest: ModelBundleManifest) -> None:
    """
    Perform deep sanity and constraint assertions on a model bundle manifest.
    
    Raises:
        ValueError: If constraints are breached.
    """
    if not validate_semantic_version(manifest.model_version):
        raise ValueError(f"Model version '{manifest.model_version}' does not adhere to semantic versioning.")
    if manifest.model_family != "xgboost":
        raise ValueError(f"Model family '{manifest.model_family}' is not supported. Must be 'xgboost'.")
    if not (0.0 < manifest.operating_threshold < 1.0):
        raise ValueError(f"Operating threshold {manifest.operating_threshold} out of valid bounds (0.0, 1.0).")
    
    # Required checksum keys
    if "model" not in manifest.sha256_checksums:
        raise ValueError("Missing 'model' checksum in manifest sha256_checksums.")
    if "preprocessor" not in manifest.sha256_checksums:
        raise ValueError("Missing 'preprocessor' checksum in manifest sha256_checksums.")


def verify_bundle_integrity(
    bundle_dir: Union[str, Path],
    manifest: Optional[ModelBundleManifest] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify the physical file integrity and SHA-256 checksums of a model bundle directory.
    
    Args:
        bundle_dir: Path to the bundle directory containing model.joblib, preprocessor.joblib, manifest.json.
        manifest: Optional pre-loaded manifest. If None, loaded from bundle_dir / manifest.json.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
    """
    errors: List[str] = []
    b_path = Path(bundle_dir).resolve()
    
    if not b_path.exists() or not b_path.is_dir():
        return False, [f"Bundle directory does not exist or is not a directory: {b_path}"]
        
    model_file = b_path / "model.joblib"
    prep_file = b_path / "preprocessor.joblib"
    manifest_file = b_path / "manifest.json"
    
    if not model_file.exists():
        errors.append(f"Missing required model artifact: {model_file}")
    if not prep_file.exists():
        errors.append(f"Missing required preprocessor artifact: {prep_file}")
    if not manifest_file.exists():
        errors.append(f"Missing required manifest file: {manifest_file}")
        
    if errors:
        return False, errors
        
    try:
        if manifest is None:
            manifest = ModelBundleManifest.load(manifest_file)
    except Exception as e:
        return False, [f"Failed to parse manifest.json: {str(e)}"]
        
    # Check model artifact SHA-256
    try:
        actual_model_sha = calculate_file_sha256(model_file)
        expected_model_sha = manifest.sha256_checksums.get("model") or manifest.sha256_checksums.get("model.joblib")
        if not expected_model_sha or actual_model_sha.lower() != expected_model_sha.lower():
            errors.append(
                f"Model artifact SHA-256 checksum mismatch. Expected '{expected_model_sha}', got '{actual_model_sha}'."
            )
    except Exception as e:
        errors.append(f"Error checking model artifact: {str(e)}")
        
    # Check preprocessor artifact SHA-256
    try:
        actual_prep_sha = calculate_file_sha256(prep_file)
        expected_prep_sha = manifest.sha256_checksums.get("preprocessor") or manifest.sha256_checksums.get("preprocessor.joblib")
        if not expected_prep_sha or actual_prep_sha.lower() != expected_prep_sha.lower():
            errors.append(
                f"Preprocessor artifact SHA-256 checksum mismatch. Expected '{expected_prep_sha}', got '{actual_prep_sha}'."
            )
    except Exception as e:
        errors.append(f"Error checking preprocessor artifact: {str(e)}")
        
    return len(errors) == 0, errors
