"""
Dashboard Pydantic Schemas for Phase 11: Fraud Intelligence Dashboard.

Defines strongly typed, validated request and response contracts for dashboard metrics,
KPI overviews, and operational analytics.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from backend.app.db.models.enums import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
    RuleType,
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
    AuditActorType,
)


class DashboardOverviewResponse(BaseModel):
    """
    High-level operational KPI summary response for the fraud dashboard overview grid.
    """
    model_config = ConfigDict(extra="forbid")

    total_transactions: int = Field(
        ...,
        ge=0,
        description="Total count of evaluated transactions persisted in the database.",
        json_schema_extra={"example": 10540},
    )
    total_amount: float = Field(
        ...,
        ge=0.0,
        description="Total monetary volume of evaluated transactions in USD.",
        json_schema_extra={"example": 542980.50},
    )
    approval_count: int = Field(
        ...,
        ge=0,
        description="Total count of approved transactions (decision_action = 'APPROVE').",
        json_schema_extra={"example": 9820},
    )
    approval_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of evaluated transactions approved (0.0% to 100.0%).",
        json_schema_extra={"example": 93.17},
    )
    review_count: int = Field(
        ...,
        ge=0,
        description="Total count of transactions escalated to manual review (decision_action = 'REVIEW').",
        json_schema_extra={"example": 450},
    )
    review_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of evaluated transactions escalated to review (0.0% to 100.0%).",
        json_schema_extra={"example": 4.27},
    )
    block_count: int = Field(
        ...,
        ge=0,
        description="Total count of automated fraud blocks (decision_action = 'BLOCK').",
        json_schema_extra={"example": 270},
    )
    block_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of evaluated transactions blocked as fraud (0.0% to 100.0%).",
        json_schema_extra={"example": 2.56},
    )
    average_risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean calibrated 0–100 risk score across all evaluated transactions.",
        json_schema_extra={"example": 18.45},
    )
    average_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Mean end-to-end evaluation latency in milliseconds.",
        json_schema_extra={"example": 24.85},
    )

    @classmethod
    def empty(cls) -> "DashboardOverviewResponse":
        """Generate safe zero-value defaults for an unseeded / empty database."""
        return cls(
            total_transactions=0,
            total_amount=0.0,
            approval_count=0,
            approval_rate=0.0,
            review_count=0,
            review_rate=0.0,
            block_count=0,
            block_rate=0.0,
            average_risk_score=0.0,
            average_latency_ms=0.0,
        )


class TransactionListItem(BaseModel):
    """
    Compact transaction summary record optimized for real-time dashboard feed listings.
    Omits bulky feature snapshots, TreeSHAP values, and sensitive attributes.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Internal primary key UUID of the transaction record.",
        json_schema_extra={"example": "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f"},
    )
    external_transaction_id: Optional[str] = Field(
        None,
        description="Optional client-supplied transaction identifier.",
        json_schema_extra={"example": "TX_987654"},
    )
    transaction_timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the transaction occurred.",
        json_schema_extra={"example": "2026-09-17T14:30:00Z"},
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction monetary amount.",
        json_schema_extra={"example": 249.99},
    )
    currency: str = Field(
        default="USD",
        description="ISO 4217 three-letter currency code.",
        json_schema_extra={"example": "USD"},
    )
    merchant_category: str = Field(
        ...,
        description="Industry category of the merchant terminal.",
        json_schema_extra={"example": "electronics"},
    )
    risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Calibrated integer risk score in [0, 100].",
        json_schema_extra={"example": 72},
    )
    risk_tier: RiskTier = Field(
        ...,
        description="Categorical risk tier (LOW, MEDIUM, HIGH, CRITICAL).",
        json_schema_extra={"example": "HIGH"},
    )
    decision_action: DecisionAction = Field(
        ...,
        description="Final operational decision action (APPROVE, REVIEW, BLOCK).",
        json_schema_extra={"example": "REVIEW"},
    )
    is_overridden: bool = Field(
        default=False,
        description="Whether a deterministic business rule overrode the baseline ML policy action.",
        json_schema_extra={"example": False},
    )
    evaluation_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Evaluation execution latency in milliseconds.",
        json_schema_extra={"example": 14.5},
    )


class TransactionListResponse(BaseModel):
    """
    Paginated response payload containing a list of compact transaction records and pagination metadata.
    """
    model_config = ConfigDict(extra="forbid")

    items: List[TransactionListItem] = Field(
        ...,
        description="List of compact transaction items matching query filters.",
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total count of matching transactions across all pages.",
        json_schema_extra={"example": 450},
    )
    limit: int = Field(
        ...,
        ge=1,
        le=100,
        description="Maximum number of items returned per page.",
        json_schema_extra={"example": 20},
    )
    offset: int = Field(
        ...,
        ge=0,
        description="Number of items skipped from the start of the result set.",
        json_schema_extra={"example": 0},
    )


class TransactionMetadataDetail(BaseModel):
    """
    Detailed canonical transaction metadata and demographic features for analyst investigation.
    Excludes sensitive card numbers, CVVs, and credentials.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Internal primary key UUID of the transaction record.",
        json_schema_extra={"example": "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f"},
    )
    external_transaction_id: Optional[str] = Field(
        None,
        description="Optional client-supplied transaction identifier.",
        json_schema_extra={"example": "TX_987654"},
    )
    account_id: str = Field(
        ...,
        description="Cardholder or account identifier token.",
        json_schema_extra={"example": "ACC_102938"},
    )
    merchant_id: Optional[str] = Field(
        None,
        description="Optional merchant identifier.",
        json_schema_extra={"example": "MERCH_5544"},
    )
    merchant_category: str = Field(
        ...,
        description="Merchant industry category classification.",
        json_schema_extra={"example": "electronics"},
    )
    job_category: str = Field(
        ...,
        description="Cardholder employment category.",
        json_schema_extra={"example": "engineering"},
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction monetary amount in specified currency.",
        json_schema_extra={"example": 499.99},
    )
    currency: str = Field(
        default="USD",
        description="ISO 4217 three-letter currency code.",
        json_schema_extra={"example": "USD"},
    )
    cardholder_lat: float = Field(
        ...,
        description="Cardholder home latitude coordinate.",
        json_schema_extra={"example": 37.7749},
    )
    cardholder_long: float = Field(
        ...,
        description="Cardholder home longitude coordinate.",
        json_schema_extra={"example": -122.4194},
    )
    merchant_lat: float = Field(
        ...,
        description="Merchant terminal latitude coordinate.",
        json_schema_extra={"example": 37.7833},
    )
    merchant_long: float = Field(
        ...,
        description="Merchant terminal longitude coordinate.",
        json_schema_extra={"example": -122.4167},
    )
    city_pop: int = Field(
        ...,
        ge=0,
        description="Cardholder city population.",
        json_schema_extra={"example": 873965},
    )
    transaction_timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the transaction occurred.",
        json_schema_extra={"example": "2026-09-17T14:30:00Z"},
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the record was ingested into the database.",
        json_schema_extra={"example": "2026-09-17T14:30:01Z"},
    )


class EvaluationDetail(BaseModel):
    """
    Persisted risk evaluation decision, calibrated risk metrics, policy boundaries, and latency telemetry.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Internal primary key UUID of the risk evaluation record.",
        json_schema_extra={"example": "e4f5a6b7-c8d9-0e1f-2a3b-4c5d6e7f8a9b"},
    )
    model_version: str = Field(
        ...,
        description="Champion model version identifier active during evaluation.",
        json_schema_extra={"example": "1.0.0"},
    )
    policy_mode: PolicyMode = Field(
        ...,
        description="Operating decision policy mode (TRI_TIER or BINARY_AUTO).",
        json_schema_extra={"example": "TRI_TIER"},
    )
    model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous model ranking probability in [0.0, 1.0].",
        json_schema_extra={"example": 0.8452},
    )
    risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Calibrated integer risk score in [0, 100].",
        json_schema_extra={"example": 85},
    )
    risk_tier: RiskTier = Field(
        ...,
        description="Categorical risk tier (LOW, MEDIUM, HIGH, CRITICAL).",
        json_schema_extra={"example": "HIGH"},
    )
    decision_action: DecisionAction = Field(
        ...,
        description="Final operational business decision (APPROVE, REVIEW, BLOCK).",
        json_schema_extra={"example": "BLOCK"},
    )
    baseline_action: DecisionAction = Field(
        ...,
        description="Baseline ML policy action before deterministic business rule overrides.",
        json_schema_extra={"example": "REVIEW"},
    )
    is_overridden: bool = Field(
        default=False,
        description="True if a business rule escalated or overrode the baseline ML action.",
        json_schema_extra={"example": True},
    )
    rule_action: Optional[RuleOutcome] = Field(
        None,
        description="Highest precedence rule outcome enacting an override if applicable.",
        json_schema_extra={"example": "BLOCK"},
    )
    decision_reason: str = Field(
        ...,
        description="Human-readable decision boundary explanation.",
        json_schema_extra={"example": "Risk score 85 meets BLOCK threshold (>= 80) and triggered RULE_VELOCITY_BURST."},
    )
    output_margin: Optional[float] = Field(
        None,
        description="Raw model log-odds margin (logit of probability).",
        json_schema_extra={"example": 1.725},
    )
    base_value: Optional[float] = Field(
        None,
        description="Global TreeSHAP baseline expected margin.",
        json_schema_extra={"example": -3.542},
    )
    evaluation_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Evaluation execution latency in milliseconds.",
        json_schema_extra={"example": 14.8},
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Distributed request correlation tracing identifier.",
        json_schema_extra={"example": "corr_987654321"},
    )
    evaluated_at: datetime = Field(
        ...,
        description="UTC timestamp when the risk evaluation was executed.",
        json_schema_extra={"example": "2026-09-17T14:30:00.500Z"},
    )


class FeatureAttributionDetail(BaseModel):
    """
    Persisted local TreeSHAP feature attribution explaining model output log-odds contribution.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Primary key UUID of the feature attribution record.",
        json_schema_extra={"example": "f1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d"},
    )
    feature_name: str = Field(
        ...,
        description="Exact predictive feature name matching model contract.",
        json_schema_extra={"example": "txn_count_1h"},
    )
    display_name: str = Field(
        ...,
        description="Human-friendly feature label for UI and investigation reports.",
        json_schema_extra={"example": "Transactions in Past 1 Hour"},
    )
    raw_value: Optional[Any] = Field(
        None,
        description="Unencoded raw value of the feature from the transaction payload.",
        json_schema_extra={"example": 6},
    )
    shap_value: float = Field(
        ...,
        description="Additive TreeSHAP attribution value in log-odds margin space.",
        json_schema_extra={"example": 1.452},
    )
    direction: AttributionDirection = Field(
        ...,
        description="Direction of risk influence ('RISK_INCREASING' or 'MITIGATING').",
        json_schema_extra={"example": "RISK_INCREASING"},
    )
    relative_contribution_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage contribution within its directional group.",
        json_schema_extra={"example": 42.5},
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-indexed presentation priority rank within its directional group.",
        json_schema_extra={"example": 1},
    )


class ReasonCodeDetail(BaseModel):
    """
    Persisted plain-English reason code explaining the primary risk drivers.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Primary key UUID of the reason code record.",
        json_schema_extra={"example": "r1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d"},
    )
    code: str = Field(
        ...,
        description="Standard uppercase reason identifier.",
        json_schema_extra={"example": "VELOCITY_BURST_1H"},
    )
    headline: str = Field(
        ...,
        description="Concise summary headline for analyst review.",
        json_schema_extra={"example": "Rapid Transaction Velocity"},
    )
    description: str = Field(
        ...,
        description="Interpolated plain-English explanation of the risk driver.",
        json_schema_extra={"example": "6 transactions attempted within the last hour, exceeding normal pattern."},
    )
    category: str = Field(
        ...,
        description="Domain category (AMOUNT, VELOCITY, GEOGRAPHY, ACCOUNT_HISTORY, etc.).",
        json_schema_extra={"example": "VELOCITY"},
    )
    source: ReasonSource = Field(
        ...,
        description="Origin of reason code ('MODEL' for TreeSHAP or 'RULE' for rule triggers).",
        json_schema_extra={"example": "MODEL"},
    )
    severity: ReasonSeverity = Field(
        ...,
        description="Operational severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO).",
        json_schema_extra={"example": "HIGH"},
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-indexed presentation priority rank.",
        json_schema_extra={"example": 1},
    )


class RuleMatchDetail(BaseModel):
    """
    Persisted triggered business rule metadata and threshold condition.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Primary key UUID of the rule match record.",
        json_schema_extra={"example": "m1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d"},
    )
    rule_id: str = Field(
        ...,
        description="Canonical rule identifier string.",
        json_schema_extra={"example": "RULE_VELOCITY_BURST_BLOCK"},
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of rule intent.",
        json_schema_extra={"example": "Block cardholder when hourly transaction count exceeds 5."},
    )
    feature_name: str = Field(
        ...,
        description="Exact predictive feature evaluated against the rule threshold.",
        json_schema_extra={"example": "txn_count_1h"},
    )
    operator: str = Field(
        ...,
        description="Comparison operator ('>', '<', 'in', 'is_true').",
        json_schema_extra={"example": ">"},
    )
    comparison_value: str = Field(
        ...,
        description="Configured threshold comparison value.",
        json_schema_extra={"example": "5"},
    )
    outcome: RuleOutcome = Field(
        ...,
        description="Target outcome tier (BLOCK, REVIEW, MONITOR).",
        json_schema_extra={"example": "BLOCK"},
    )
    rule_type: RuleType = Field(
        ...,
        description="Domain category (VELOCITY, AMOUNT, GEOGRAPHY, COMPLIANCE, etc.).",
        json_schema_extra={"example": "VELOCITY"},
    )
    priority: int = Field(
        ...,
        ge=1,
        description="Evaluation priority integer (lower numbers evaluate earlier).",
        json_schema_extra={"example": 10},
    )


class AuditLogDetail(BaseModel):
    """
    Sanitized audit log event representing operational decisions and lifecycle actions.
    Excludes internal passwords, secrets, credentials, and connection information.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Primary key UUID of the audit log record.",
        json_schema_extra={"example": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"},
    )
    event_type: str = Field(
        ...,
        description="Standard event identifier (e.g. 'RISK_EVALUATION_PERSISTED').",
        json_schema_extra={"example": "RISK_EVALUATION_PERSISTED"},
    )
    action: str = Field(
        ...,
        description="Action executed on the entity (e.g. 'PERSIST_EVALUATION').",
        json_schema_extra={"example": "PERSIST_EVALUATION"},
    )
    actor_type: str = Field(
        ...,
        description="Category of the originating actor (SYSTEM, ANALYST, ADMIN, API_CLIENT).",
        json_schema_extra={"example": "SYSTEM"},
    )
    actor_id: Optional[str] = Field(
        None,
        description="Identifier of the executing actor or API client key.",
        json_schema_extra={"example": "fraud_engine_worker_01"},
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Distributed request correlation ID for end-to-end tracing.",
        json_schema_extra={"example": "corr_987654321"},
    )
    client_ip: Optional[str] = Field(
        None,
        description="Originating client IP address.",
        json_schema_extra={"example": "192.168.1.100"},
    )
    event_timestamp: datetime = Field(
        ...,
        description="UTC timestamp when the event occurred.",
        json_schema_extra={"example": "2026-09-17T14:30:00.600Z"},
    )


class TransactionDetailResponse(BaseModel):
    """
    Complete transaction deep inspection response payload for analyst investigation.
    Combines transaction metadata, evaluation decision, 55-feature snapshot, TreeSHAP
    attributions, reason codes, triggered business rules, and audit trail events.
    """
    model_config = ConfigDict(extra="forbid")

    transaction: TransactionMetadataDetail = Field(
        ...,
        description="Core canonical transaction metadata and demographics.",
    )
    evaluation: Optional[EvaluationDetail] = Field(
        None,
        description="Persisted risk evaluation decision and telemetry if evaluation exists.",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete 55-feature engineered behavioral vector at evaluation time.",
    )
    feature_attributions: List[FeatureAttributionDetail] = Field(
        default_factory=list,
        description="Ranked TreeSHAP local feature attributions in margin space.",
    )
    reason_codes: List[ReasonCodeDetail] = Field(
        default_factory=list,
        description="Ranked plain-English reason codes explaining primary risk drivers.",
    )
    rule_matches: List[RuleMatchDetail] = Field(
        default_factory=list,
        description="Deterministic business rules triggered during evaluation.",
    )
    audit_trail: List[AuditLogDetail] = Field(
        default_factory=list,
        description="Chronological audit trail events associated with this transaction/evaluation.",
    )


# =============================================================================
# Increment 11.4: Explainability Visualizer & Trend Analytics Schemas
# =============================================================================

class DistributionBucket(BaseModel):
    """
    Histogram bin for score distributions with explicit boundary labels, counts, and percentages.
    """
    model_config = ConfigDict(extra="forbid")

    bucket_label: str = Field(
        ...,
        description="Formatted human-readable boundary label (e.g. '0–9', '90–100', '0.0–0.1').",
        json_schema_extra={"example": "80–89"},
    )
    lower_bound: float = Field(
        ...,
        description="Inclusive lower numerical boundary of the bucket.",
        json_schema_extra={"example": 80.0},
    )
    upper_bound: float = Field(
        ...,
        description="Upper numerical boundary of the bucket.",
        json_schema_extra={"example": 89.0},
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of transactions falling within this bucket range.",
        json_schema_extra={"example": 142},
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of total evaluated transactions in this bucket (0.0% to 100.0%).",
        json_schema_extra={"example": 14.2},
    )


class CategoryCount(BaseModel):
    """
    Categorical frequency bucket for risk tiers and decision actions.
    """
    model_config = ConfigDict(extra="forbid")

    category: str = Field(
        ...,
        description="Categorical label (e.g. 'APPROVE', 'REVIEW', 'BLOCK', 'LOW', 'CRITICAL').",
        json_schema_extra={"example": "BLOCK"},
    )
    count: int = Field(
        ...,
        ge=0,
        description="Total occurrences of this category.",
        json_schema_extra={"example": 250},
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of total evaluations represented by this category.",
        json_schema_extra={"example": 2.5},
    )


class AnalyticsDistributionsResponse(BaseModel):
    """
    Comprehensive score and decision distribution metrics aggregated across evaluated transactions.
    """
    model_config = ConfigDict(extra="forbid")

    total_evaluated: int = Field(
        ...,
        ge=0,
        description="Total evaluated transactions matching query filters.",
        json_schema_extra={"example": 10000},
    )
    risk_score_distribution: List[DistributionBucket] = Field(
        ...,
        description="10-bucket histogram across calibrated risk scores [0–9, 10–19, ..., 90–100].",
    )
    model_score_distribution: List[DistributionBucket] = Field(
        ...,
        description="10-bucket histogram across continuous model probability margins [0.0–0.1, ..., 0.9–1.0].",
    )
    risk_tier_distribution: List[CategoryCount] = Field(
        ...,
        description="Breakdown by risk tier (LOW, MEDIUM, HIGH, CRITICAL).",
    )
    decision_distribution: List[CategoryCount] = Field(
        ...,
        description="Breakdown by operational decision action (APPROVE, REVIEW, BLOCK).",
    )


class TrendInterval(str, Enum):
    """
    Time aggregation bucketing interval for trend analytics.
    """
    HOURLY = "hourly"
    DAILY = "daily"


class TrendDataPoint(BaseModel):
    """
    Aggregated operational and risk telemetry metrics for an individual time bucket.
    """
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        ...,
        description="Start UTC timestamp of the time bucket.",
        json_schema_extra={"example": "2026-09-17T12:00:00Z"},
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total transaction volume during this time window.",
        json_schema_extra={"example": 120},
    )
    total_amount: float = Field(
        ...,
        ge=0.0,
        description="Total transaction monetary value in USD during this time window.",
        json_schema_extra={"example": 45200.50},
    )
    average_risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean calibrated risk score of transactions evaluated during this window.",
        json_schema_extra={"example": 22.4},
    )
    approval_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions approved during this window.",
        json_schema_extra={"example": 105},
    )
    review_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions escalated to review during this window.",
        json_schema_extra={"example": 10},
    )
    block_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions blocked during this window.",
        json_schema_extra={"example": 5},
    )
    high_critical_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions scoring in HIGH or CRITICAL risk tiers.",
        json_schema_extra={"example": 8},
    )


class AnalyticsTrendsResponse(BaseModel):
    """
    Time-series trend analytics payload covering transaction volumes, scores, and decisions.
    """
    model_config = ConfigDict(extra="forbid")

    interval: TrendInterval = Field(
        ...,
        description="Active time-series bucketing interval ('hourly' or 'daily').",
        json_schema_extra={"example": "hourly"},
    )
    start_date: datetime = Field(
        ...,
        description="Effective start UTC datetime boundary of the query.",
        json_schema_extra={"example": "2026-09-16T00:00:00Z"},
    )
    end_date: datetime = Field(
        ...,
        description="Effective end UTC datetime boundary of the query.",
        json_schema_extra={"example": "2026-09-17T00:00:00Z"},
    )
    data_points: List[TrendDataPoint] = Field(
        ...,
        description="Chronologically ordered collection of contiguous time-series points.",
    )


class RuleOutcomeBreakdown(BaseModel):
    """
    Outcome frequency breakdown for an individual business rule.
    """
    model_config = ConfigDict(extra="forbid")

    outcome: RuleOutcome = Field(
        ...,
        description="Rule outcome action (BLOCK, REVIEW, APPROVE, MONITOR).",
        json_schema_extra={"example": "BLOCK"},
    )
    count: int = Field(
        ...,
        ge=0,
        description="Number of rule triggers resulting in this outcome.",
        json_schema_extra={"example": 45},
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of this rule's triggers with this outcome.",
        json_schema_extra={"example": 100.0},
    )


class RuleAnalyticsItem(BaseModel):
    """
    Operational analytics and trigger telemetry for an individual business rule.
    """
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(
        ...,
        description="Canonical rule identifier string.",
        json_schema_extra={"example": "RULE_VELOCITY_BURST_BLOCK"},
    )
    description: str = Field(
        ...,
        description="Human-readable rule intent description.",
        json_schema_extra={"example": "Block cardholder when hourly transaction count exceeds 5."},
    )
    rule_type: RuleType = Field(
        ...,
        description="Domain rule category (VELOCITY, AMOUNT, GEOGRAPHY, etc.).",
        json_schema_extra={"example": "VELOCITY"},
    )
    priority: int = Field(
        ...,
        ge=1,
        description="Configured evaluation priority rank.",
        json_schema_extra={"example": 10},
    )
    trigger_count: int = Field(
        ...,
        ge=0,
        description="Total number of persisted rule match records for this rule.",
        json_schema_extra={"example": 45},
    )
    affected_transactions: int = Field(
        ...,
        ge=0,
        description="Distinct count of transactions/evaluations where this rule was triggered.",
        json_schema_extra={"example": 45},
    )
    trigger_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of total analyzed evaluations that triggered this rule.",
        json_schema_extra={"example": 4.5},
    )
    override_count: int = Field(
        ...,
        ge=0,
        description="Distinct evaluations associated with this rule where an override was enacted (is_overridden=True).",
        json_schema_extra={"example": 38},
    )
    outcomes: List[RuleOutcomeBreakdown] = Field(
        ...,
        description="Breakdown of outcomes produced by this rule.",
    )


class AnalyticsRulesResponse(BaseModel):
    """
    Aggregated business rule performance and trigger frequency report.
    """
    model_config = ConfigDict(extra="forbid")

    total_rules_active: int = Field(
        ...,
        ge=0,
        description="Total distinct business rules triggered in the analyzed time window.",
        json_schema_extra={"example": 6},
    )
    total_evaluations_analyzed: int = Field(
        ...,
        ge=0,
        description="Total distinct risk evaluations processed in the analyzed time window.",
        json_schema_extra={"example": 1000},
    )
    rules: List[RuleAnalyticsItem] = Field(
        ...,
        description="Ranked list of business rules ordered by trigger count descending.",
    )


class SimulationRequest(BaseModel):
    """
    Request payload for executing an in-memory what-if fraud risk scenario evaluation.
    """
    model_config = ConfigDict(extra="forbid")

    baseline_transaction_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Optional UUID or external ID of an existing persisted transaction to compare against.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    simulated_features: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing candidate feature values adhering to the canonical 55-feature contract.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of top risk-increasing TreeSHAP features to return.",
        json_schema_extra={"example": 5},
    )
    top_mitigating: int = Field(
        3,
        ge=1,
        le=20,
        description="Number of top mitigating (risk-reducing) TreeSHAP features to return.",
        json_schema_extra={"example": 3},
    )
    max_reasons: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum count of synthesized plain-English reason codes to return.",
        json_schema_extra={"example": 5},
    )


class FeatureDiffItem(BaseModel):
    """
    Comparative delta for an individual transaction feature between baseline and simulated scenario.
    """
    model_config = ConfigDict(extra="forbid")

    feature_name: str = Field(..., description="Exact predictive feature column identifier.")
    display_name: str = Field(..., description="Human-friendly feature label.")
    category: str = Field(..., description="Feature domain categorization.")
    baseline_value: Optional[Any] = Field(None, description="Original feature value from baseline transaction.")
    simulated_value: Any = Field(..., description="Modified simulated feature value.")
    is_modified: bool = Field(..., description="Flag indicating if the simulated value differs from baseline.")
    delta: Optional[float] = Field(None, description="Numerical difference (simulated - baseline) for numeric features.")


class RuleDiffStatus(str, Enum):
    """
    Status of a business rule execution in simulation comparison.
    """
    NEWLY_TRIGGERED = "NEWLY_TRIGGERED"
    RESOLVED = "RESOLVED"
    PERSISTENT = "PERSISTENT"
    NEITHER = "NEITHER"


class RuleDiffItem(BaseModel):
    """
    Comparative diff for a business rule's execution state between baseline and simulated scenario.
    """
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., description="Unique rule identifier.")
    description: str = Field(..., description="Human-readable rule intent description.")
    rule_type: str = Field(..., description="Domain category of the rule.")
    priority: int = Field(..., description="Evaluation priority ordering.")
    outcome: str = Field(..., description="Configured rule action outcome (BLOCK, REVIEW, MONITOR).")
    baseline_triggered: bool = Field(..., description="Whether this rule triggered on the baseline transaction.")
    simulated_triggered: bool = Field(..., description="Whether this rule triggered on the simulated feature set.")
    diff_status: RuleDiffStatus = Field(
        ...,
        description="Rule state transition status ('NEWLY_TRIGGERED', 'RESOLVED', 'PERSISTENT', 'NEITHER').",
    )


class BaselineEvaluationSummary(BaseModel):
    """
    Persisted baseline transaction evaluation summary.
    """
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., description="Primary UUID of baseline transaction.")
    external_transaction_id: str = Field(..., description="External business transaction identifier.")
    risk_score: int = Field(..., ge=0, le=100, description="Baseline calibrated risk score.")
    risk_tier: RiskTier = Field(..., description="Baseline risk tier classification.")
    decision_action: DecisionAction = Field(..., description="Baseline final decision policy action.")
    model_score: float = Field(..., ge=0.0, le=1.0, description="Baseline raw model fraud probability.")
    is_overridden: bool = Field(..., description="Whether baseline ML decision was overridden by rules.")
    rules_triggered_count: int = Field(..., ge=0, description="Count of rules triggered by baseline transaction.")


class SimulatedEvaluationSummary(BaseModel):
    """
    Simulated scenario risk evaluation summary.
    """
    model_config = ConfigDict(extra="forbid")

    risk_score: int = Field(..., ge=0, le=100, description="Simulated calibrated risk score (0-100).")
    risk_tier: RiskTier = Field(..., description="Simulated risk tier classification.")
    decision_action: DecisionAction = Field(..., description="Simulated final policy action.")
    model_score: float = Field(..., ge=0.0, le=1.0, description="Simulated raw ML fraud probability.")
    base_value: Optional[float] = Field(None, description="Global TreeSHAP baseline expected margin (log-odds).")
    output_margin: Optional[float] = Field(None, description="Simulated model output margin (log-odds).")
    is_overridden: bool = Field(..., description="Whether simulated ML action was overridden by rules.")
    decision_reason: str = Field(..., description="Explanation of simulated policy evaluation.")
    rule_matches: List[RuleMatchDetail] = Field(..., description="List of business rules triggered by simulated features.")
    reason_codes: List[ReasonCodeDetail] = Field(..., description="Synthesized plain-English reason codes.")
    feature_attributions: List[FeatureAttributionDetail] = Field(..., description="Local TreeSHAP feature attributions.")


class SimulationComparisonSummary(BaseModel):
    """
    Comparative summary measuring shifts between baseline and simulated scenario.
    """
    model_config = ConfigDict(extra="forbid")

    risk_score_delta: int = Field(..., description="Signed risk score delta (simulated - baseline).")
    model_score_delta: float = Field(..., description="Signed model probability delta (simulated - baseline).")
    tier_changed: bool = Field(..., description="Whether risk tier changed between baseline and simulation.")
    action_changed: bool = Field(..., description="Whether decision action changed between baseline and simulation.")
    modified_features_count: int = Field(..., ge=0, description="Total count of features modified from baseline.")
    feature_diffs: List[FeatureDiffItem] = Field(..., description="Itemized feature comparison diffs.")
    rule_diffs: List[RuleDiffItem] = Field(..., description="Deduplicated business rule comparison diffs.")


class SimulationResponse(BaseModel):
    """
    Unified response payload for in-memory what-if simulation results.
    """
    model_config = ConfigDict(extra="forbid")

    is_simulation: bool = Field(True, description="Strict constant flag confirming simulated evaluation.")
    simulated_at: datetime = Field(..., description="Timestamp when scenario was simulated.")
    evaluation_latency_ms: float = Field(..., ge=0.0, description="In-memory simulation latency in milliseconds.")
    baseline_transaction_id: Optional[str] = Field(None, description="Baseline transaction ID if compared against baseline.")
    baseline: Optional[BaselineEvaluationSummary] = Field(None, description="Summary of baseline evaluation if provided.")
    simulated: SimulatedEvaluationSummary = Field(..., description="Comprehensive simulated evaluation results.")
    comparison: Optional[SimulationComparisonSummary] = Field(None, description="Comparison diffs if baseline was provided.")


