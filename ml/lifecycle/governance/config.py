"""
Promotion Gate Configuration & Governance Policy Specification for Phase 14.5.

Defines the multi-dimensional governance acceptance criteria applied to Candidate
evaluation evidence. Each gate explicitly distinguishes between existing system
invariants and newly introduced governance policy constraints.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PromotionGateSpecification(BaseModel):
    """
    Typed, immutable specification of multi-dimensional promotion gate criteria.

    All thresholds represent institutional governance policy constraints or existing
    system requirements. Evaluators must treat them deterministically.
    """
    model_config = ConfigDict(frozen=True)

    # 1. Statistical Ranking Parity Gate (Governance Policy)
    max_allowed_oot_pr_auc_degradation: float = Field(
        default=0.0050,
        ge=0.0,
        le=0.10,
        description="Permissible OOT PR-AUC parity tolerance margin below Champion baseline (Governance Policy)"
    )

    # 2. Validation Statistical Parity Gate (Governance Policy)
    max_allowed_val_pr_auc_degradation: float = Field(
        default=0.0050,
        ge=0.0,
        le=0.10,
        description="Permissible Validation PR-AUC parity tolerance margin below Champion baseline (Governance Policy)"
    )

    # 3. Fraud Catch Rate Non-Regression Gate (Governance Policy)
    require_oot_recall_non_regression: bool = Field(
        default=True,
        description="Mandate that Candidate OOT Recall must be greater than or equal to Champion OOT Recall (Governance Policy)"
    )

    # 4. Missed Fraud Non-Increase Gate (Governance Policy)
    require_oot_fn_non_increase: bool = Field(
        default=True,
        description="Mandate that Candidate OOT False Negatives must not exceed Champion OOT False Negatives (Governance Policy)"
    )

    # 5. Financial Cost Non-Increase Gate (Authoritative Baseline + Governance Policy)
    require_oot_cost_non_increase: bool = Field(
        default=True,
        description="Mandate that Candidate total OOT Expected Cost under authoritative CostConfig must not exceed Champion Cost (Governance Policy)"
    )

    # 6. Operational False Positive Rate Ceiling Gate (Governance Policy)
    max_oot_fpr_ceiling: float = Field(
        default=0.0020,
        ge=0.0,
        le=0.05,
        description="Maximum permissible False Positive Rate on OOT holdout (Governance Policy risk appetite limit)"
    )

    # 7. Automated Block Precision Floor Gate (Governance Policy)
    min_oot_block_precision: float = Field(
        default=0.7000,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable precision for automated hard BLOCK decisions on OOT holdout (Governance Policy)"
    )

    # 8. Production Inference Latency SLA Gate (Governance Policy)
    max_p95_latency_ms: float = Field(
        default=5.00,
        gt=0.0,
        description="Maximum absolute p95 inference latency ceiling in milliseconds (Governance Policy SLA)"
    )
    max_latency_ratio_vs_champion: float = Field(
        default=1.50,
        gt=1.0,
        description="Maximum permissible p95 latency growth ratio relative to Champion (Governance Policy SLA)"
    )

    # 9. Dynamic Champion Integrity Verification (Existing Requirement)
    require_dynamic_champion_integrity: bool = Field(
        default=True,
        description="Require dynamic SHA-256 calculation of on-disk Champion files matching registered metadata (Existing Requirement)"
    )

    # 10A. Predictive Feature Schema Compatibility (Existing Requirement)
    expected_feature_count: int = Field(
        default=55,
        ge=1,
        description="Mandatory 55-feature canonical contract count (Existing Requirement)"
    )

    # 10B. Evaluation Dataset Partition Integrity (Existing Requirement)
    expected_validation_sample_count: int = Field(
        default=277859,
        ge=1,
        description="Expected row count in frozen validation partition (Existing Requirement)"
    )
    expected_oot_sample_count: int = Field(
        default=277860,
        ge=1,
        description="Expected row count in protected OOT holdout partition (Existing Requirement)"
    )


default_promotion_gate_specification = PromotionGateSpecification()
