"""
Configurable Fraud Decision Cost Parameters for Phase 5 Cost Optimization.

Defines the financial and operational cost matrix parameters for evaluating fraud decisions.
All default parameters represent illustrative business assumptions and must be explicitly
configured to reflect specific institutional economics and risk tolerance.
"""

from dataclasses import dataclass
import math
from typing import Dict, Any


@dataclass(frozen=True)
class CostConfig:
    """
    Typed and validated configuration for fraud decision costs.

    Attributes:
        false_positive_cost: Cost of declining/flagging a legitimate transaction
                             (illustrative: customer friction, support load, interchange loss).
        false_negative_cost: Cost of missing a fraudulent transaction
                             (illustrative: chargeback loss, direct fraud loss, network fees).
        manual_review_cost: Operational labor cost per case routed to manual review queue.
        true_negative_cost: Cost of correctly approving a legitimate transaction (defaults to 0.0).
        true_positive_cost: Cost of correctly blocking a fraudulent transaction (defaults to 0.0).

    Note:
        These default values are illustrative baselines. They do NOT represent universal industry constants.
    """
    false_positive_cost: float = 15.0
    false_negative_cost: float = 200.0
    manual_review_cost: float = 5.0
    true_negative_cost: float = 0.0
    true_positive_cost: float = 0.0

    def __post_init__(self) -> None:
        """Validate that all specified costs are finite, numeric, and non-negative."""
        cost_fields = {
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "manual_review_cost": self.manual_review_cost,
            "true_negative_cost": self.true_negative_cost,
            "true_positive_cost": self.true_positive_cost,
        }
        for name, val in cost_fields.items():
            if not isinstance(val, (int, float)):
                raise TypeError(f"Cost parameter '{name}' must be numeric (int or float), got {type(val).__name__}.")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Cost parameter '{name}' must be finite, got {val}.")
            if val < 0.0:
                raise ValueError(f"Cost parameter '{name}' must be non-negative (>= 0.0), got {val}.")

    def to_dict(self) -> Dict[str, float]:
        """Convert cost configuration to a serializable dictionary."""
        return {
            "false_positive_cost": float(self.false_positive_cost),
            "false_negative_cost": float(self.false_negative_cost),
            "manual_review_cost": float(self.manual_review_cost),
            "true_negative_cost": float(self.true_negative_cost),
            "true_positive_cost": float(self.true_positive_cost),
        }
