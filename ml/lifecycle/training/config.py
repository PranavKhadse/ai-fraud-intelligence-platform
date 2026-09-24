"""
Configuration for Phase 14.2 Challenger Training Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ml.lifecycle.config import default_lifecycle_config
from ml.lifecycle.schemas import validate_semantic_version
from ml.models.config import MODEL_CONFIGS, RANDOM_SEED


@dataclass(frozen=True)
class ChallengerTrainingConfig:
    """
    Typed, immutable configuration for training a challenger model.
    """
    model_version: str = "1.1.0"
    model_family: str = "xgboost"
    candidate_base_dir: Path = Path("ml/models/registry/candidates")
    operating_threshold: float = 0.78
    random_state: int = RANDOM_SEED
    hyperparameters: Dict[str, Any] = field(
        default_factory=lambda: MODEL_CONFIGS["xgboost"].copy()
    )
    is_isolated_process: bool = True
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not validate_semantic_version(self.model_version):
            raise ValueError(
                f"Invalid model_version '{self.model_version}'. Must follow semantic versioning MAJOR.MINOR.PATCH."
            )
        if self.model_family.lower() != "xgboost":
            raise ValueError(
                f"Invalid model_family '{self.model_family}'. Phase 14 strictly mandates 'xgboost'."
            )
        if not (0.0 < self.operating_threshold < 1.0):
            raise ValueError(
                f"operating_threshold must be in (0, 1), got {self.operating_threshold}"
            )

    @property
    def bundle_dir(self) -> Path:
        """Destination directory for this candidate bundle."""
        return self.candidate_base_dir / f"v{self.model_version}"
