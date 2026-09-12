"""
Configuration, Enums, and Threshold Constants for the Risk Engine.

Defines decision actions, risk tiers, policy modes, and validated policy configuration.

IMPORTANT:
- Thresholds are specified in raw model ranking score space ([0.0, 1.0]), not 0-100 risk score space.
- The raw model scores from the champion XGBoost model reflect class imbalance penalization
  (scale_pos_weight = 171.75) and represent continuous ranking scores, not calibrated probabilities.
- Default block_threshold=0.78 corresponds to the Phase 5 cost-optimal operating point under
  illustrative cost parameters (CFP=$15.00, CFN=$200.00). It is fully configurable.
"""

from enum import Enum
from dataclasses import dataclass
import math
from typing import Any


class DecisionAction(str, Enum):
    """Enumeration of automated and manual transaction decision actions."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class RiskTier(str, Enum):
    """
    Semantic risk tiers mapped from normalized integer risk scores (0-100).

    Bands:
    - LOW:      0 <= score < 35
    - MEDIUM:  35 <= score < 60
    - HIGH:    60 <= score < 78
    - CRITICAL: 78 <= score <= 100
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyMode(str, Enum):
    """
    Operating policy modes for the decision engine:
    - BINARY_AUTO: Strict binary decision (APPROVE or BLOCK only).
    - TRI_TIER: Tri-state triage (APPROVE, REVIEW for manual analyst queue, BLOCK).
    """
    BINARY_AUTO = "BINARY_AUTO"
    TRI_TIER = "TRI_TIER"


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """
    Immutable configuration for decision policy thresholds and evaluation mode.

    Attributes:
        policy_mode: Operating mode (TRI_TIER or BINARY_AUTO).
        review_threshold: Raw model score cutoff for routing transactions to manual review (TRI_TIER only).
        block_threshold: Raw model score cutoff for automatically declining/blocking transactions.

    Note on Threshold Units:
        Both review_threshold and block_threshold operate on raw continuous model scores in [0.0, 1.0].
        They are NOT in 0-100 normalized risk score space.
    """
    policy_mode: PolicyMode = PolicyMode.TRI_TIER
    review_threshold: float = 0.35
    block_threshold: float = 0.78

    def __post_init__(self) -> None:
        """Validate configuration types, finiteness, bounds, and ordering."""
        # 1. Validate policy_mode
        if not isinstance(self.policy_mode, PolicyMode):
            raise TypeError(
                f"policy_mode must be an instance of PolicyMode enum, got {type(self.policy_mode).__name__} ({self.policy_mode!r})"
            )

        # 2. Reject boolean values for numeric thresholds (since bool is a subclass of int in Python)
        if isinstance(self.review_threshold, bool):
            raise TypeError("review_threshold cannot be a boolean value.")
        if isinstance(self.block_threshold, bool):
            raise TypeError("block_threshold cannot be a boolean value.")

        # 3. Type check numeric thresholds
        if not isinstance(self.review_threshold, (int, float)):
            raise TypeError(
                f"review_threshold must be a float or int, got {type(self.review_threshold).__name__}"
            )
        if not isinstance(self.block_threshold, (int, float)):
            raise TypeError(
                f"block_threshold must be a float or int, got {type(self.block_threshold).__name__}"
            )

        # 4. Finiteness check
        r_val = float(self.review_threshold)
        b_val = float(self.block_threshold)
        if not math.isfinite(r_val):
            raise ValueError(f"review_threshold must be finite, got {r_val}")
        if not math.isfinite(b_val):
            raise ValueError(f"block_threshold must be finite, got {b_val}")

        # 5. Range validation [0.0, 1.0]
        if r_val < 0.0 or r_val > 1.0:
            raise ValueError(f"review_threshold must be in [0.0, 1.0], got {r_val}")
        if b_val < 0.0 or b_val > 1.0:
            raise ValueError(f"block_threshold must be in [0.0, 1.0], got {b_val}")

        # 6. Threshold ordering validation
        if r_val > b_val:
            raise ValueError(
                f"review_threshold ({r_val}) cannot exceed block_threshold ({b_val})."
            )
