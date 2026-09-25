"""
Model Lifecycle and Configuration for Phase 14 Model Registry & Controlled Promotion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set


@dataclass(frozen=True)
class LifecycleConfig:
    """
    Typed, immutable configuration for the Phase 14 Model Lifecycle and Registry.
    """
    registry_dir: Path = Path("ml/models/registry")
    bundles_dir: Path = Path("ml/models/registry/bundles")
    candidates_dir: Path = Path("ml/models/registry/candidates")
    comparisons_dir: Path = Path("ml/models/registry/comparisons")
    signoffs_dir: Path = Path("ml/models/registry/signoffs")
    staging_dir: Path = Path("ml/models/registry/staging")
    operations_dir: Path = Path("ml/models/registry/operations")
    promotions_dir: Path = Path("ml/models/registry/promotions")
    rollbacks_dir: Path = Path("ml/models/registry/rollbacks")
    active_backup_prefix: str = ".active_champion_backup_"
    active_artifacts_dir: Path = Path("ml/models/artifacts")

    @property
    def registry_root(self) -> Path:
        return self.registry_dir
    
    # Active champion files
    champion_model_path: Path = Path("ml/models/artifacts/champion_model.joblib")
    champion_preprocessor_path: Path = Path("ml/models/artifacts/champion_preprocessor.joblib")
    champion_metadata_path: Path = Path("ml/models/artifacts/model_metadata.json")

    # Algorithm & Schema Constraints
    default_model_family: str = "xgboost"
    supported_model_families: Tuple[str, ...] = ("xgboost",)
    expected_feature_count: int = 55
    default_operating_threshold: float = 0.78
    
    # Checksum Algorithm
    hash_algorithm: str = "sha256"


default_lifecycle_config = LifecycleConfig()
