"""
Pydantic v2 Request and Response Schemas for Fraud Detection Prediction and Explainability.

Defines:
- TransactionPredictRequest: Validates exact 55-feature predictive vector with metadata tolerance.
- RuleMatchResponse: Serialized business rule outcome.
- ReasonCodeResponse: Human-readable reason code details.
- FeatureAttributionResponse: Local TreeSHAP risk and mitigating factors.
- PredictionResponse: Unified prediction, policy action, reason codes, and rule telemetry.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class RuleMatchResponse(BaseModel):
    """
    Structured detail of a triggered business rule.
    """
    rule_id: str = Field(..., description="Unique business rule identifier")
    description: str = Field(..., description="Human-readable rule intent description")
    feature_name: str = Field(..., description="Evaluated feature column name")
    operator: str = Field(..., description="Rule comparison operator (e.g. '>', '<', 'is_true')")
    comparison_value: Any = Field(..., description="Configured rule threshold value")
    outcome: str = Field(..., description="Rule outcome tier (BLOCK, REVIEW, MONITOR, APPROVE)")
    rule_type: str = Field(..., description="Rule domain categorization (VELOCITY, AMOUNT, GEOGRAPHY, etc.)")
    priority: int = Field(..., description="Evaluation priority rank (lower numbers evaluate earlier)")


class ReasonCodeResponse(BaseModel):
    """
    Standardized, plain-English reason code explaining transaction risk or mitigation.
    """
    code: str = Field(..., description="Standard uppercase reason code (e.g. 'ELEVATED_24H_AVG_AMOUNT')")
    headline: str = Field(..., description="Concise, professional headline")
    description: str = Field(..., description="Interpolated plain-English explanation")
    category: str = Field(..., description="Domain category (AMOUNT, VELOCITY, GEOGRAPHY, etc.)")
    source: str = Field(..., description="Origin of reason code ('MODEL' for TreeSHAP or 'RULE' for rules)")
    severity: str = Field(..., description="Operational severity level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')")
    rank: int = Field(..., description="1-indexed presentation priority rank")


class FeatureAttributionResponse(BaseModel):
    """
    Local TreeSHAP feature contribution record.
    """
    feature_name: str = Field(..., description="Exact predictive feature column name")
    display_name: str = Field(..., description="Human-friendly display name")
    raw_value: Any = Field(..., description="Unencoded raw value from the transaction")
    shap_value: float = Field(..., description="Additive TreeSHAP attribution in log-odds margin space")
    direction: str = Field(..., description="Attribution direction ('RISK_INCREASING' or 'MITIGATING')")
    relative_contribution_pct: float = Field(..., description="Relative percentage contribution in direction group")
    rank: int = Field(..., description="1-indexed importance rank")


class TransactionPredictRequest(BaseModel):
    """
    Pydantic v2 Request schema representing a financial transaction with all 55 predictive features.
    
    Supports optional metadata fields (transaction_id, account_id, timestamp, etc.)
    and extra keys via ConfigDict(extra='allow').
    """
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Optional metadata
    transaction_id: Optional[str] = Field(None, max_length=128, description="Optional external transaction identifier")
    account_id: Optional[str] = Field(None, max_length=128, description="Optional account / cardholder identifier")
    merchant_id: Optional[str] = Field(None, max_length=128, description="Optional merchant identifier")
    timestamp: Optional[str] = Field(None, description="Optional raw transaction timestamp")

    # Canonical Numeric Predictors (6 features)
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD (>= 0)")
    cardholder_lat: float = Field(..., ge=-90.0, le=90.0, description="Cardholder latitude (-90 to 90)")
    cardholder_long: float = Field(..., ge=-180.0, le=180.0, description="Cardholder longitude (-180 to 180)")
    merchant_lat: float = Field(..., ge=-90.0, le=90.0, description="Merchant latitude (-90 to 90)")
    merchant_long: float = Field(..., ge=-180.0, le=180.0, description="Merchant longitude (-180 to 180)")
    city_pop: float = Field(..., ge=0.0, description="Cardholder city population (>= 0)")

    # Categorical Predictors (2 features)
    merchant_category: str = Field(..., min_length=1, description="Merchant industry category")
    job_category: str = Field(..., min_length=1, description="Cardholder job classification category")

    # Group 1: Temporal Features (11 features)
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    week_of_year: int = Field(..., ge=1, le=53, description="ISO week of year (1-53)")
    is_weekend: int = Field(..., ge=0, le=1, description="Binary flag indicating weekend (0 or 1)")
    is_night: int = Field(..., ge=0, le=1, description="Binary flag indicating overnight hour (0 or 1)")
    hour_sin: float = Field(..., description="Cyclical sine transformation of hour")
    hour_cos: float = Field(..., description="Cyclical cosine transformation of hour")
    day_of_week_sin: float = Field(..., description="Cyclical sine transformation of day of week")
    day_of_week_cos: float = Field(..., description="Cyclical cosine transformation of day of week")

    # Group 2: Velocity Features (7 features)
    txn_count_1h: float = Field(..., ge=0.0, description="Number of account transactions in past 1 hour")
    txn_count_6h: float = Field(..., ge=0.0, description="Number of account transactions in past 6 hours")
    txn_count_24h: float = Field(..., ge=0.0, description="Number of account transactions in past 24 hours")
    txn_count_7d: float = Field(..., ge=0.0, description="Number of account transactions in past 7 days")
    txn_count_30d: float = Field(..., ge=0.0, description="Number of account transactions in past 30 days")
    time_since_prev_txn_seconds: float = Field(..., description="Elapsed seconds since previous transaction on account")
    is_first_account_txn: int = Field(..., ge=0, le=1, description="Binary flag indicating first seen account transaction")

    # Group 3: Spending Features (8 features)
    amt_sum_1h: float = Field(..., description="Cumulative transaction dollar amount in past 1 hour")
    amt_sum_24h: float = Field(..., description="Cumulative transaction dollar amount in past 24 hours")
    amt_sum_7d: float = Field(..., description="Cumulative transaction dollar amount in past 7 days")
    amt_sum_30d: float = Field(..., description="Cumulative transaction dollar amount in past 30 days")
    amt_mean_24h: float = Field(..., description="Mean transaction amount in past 24 hours")
    amt_mean_7d: float = Field(..., description="Mean transaction amount in past 7 days")
    amt_max_24h: float = Field(..., description="Maximum single transaction amount in past 24 hours")
    amt_median_30d: float = Field(..., description="Median transaction amount in past 30 days")

    # Group 4: Spending Deviation Features (5 features)
    historical_amount_mean: float = Field(..., description="Historical baseline mean transaction amount")
    historical_amount_std: float = Field(..., description="Historical baseline standard deviation")
    historical_amount_median: float = Field(..., description="Historical baseline median transaction amount")
    amount_zscore: float = Field(..., description="Standard deviation deviation (Z-score) of current amount")
    amount_ratio_to_historical_mean: float = Field(..., description="Ratio of current amount to historical mean")

    # Group 5: Account History Features (6 features)
    account_txn_count_before: float = Field(..., ge=0.0, description="Total prior transaction count for account")
    account_total_spend_before: float = Field(..., description="Total cumulative historical dollar spend")
    account_avg_amount_before: float = Field(..., description="Average historical transaction amount")
    account_max_amount_before: float = Field(..., description="Maximum historical transaction amount")
    account_unique_merchant_count_before: float = Field(..., ge=0.0, description="Distinct merchants visited historically")
    account_unique_category_count_before: float = Field(..., ge=0.0, description="Distinct merchant categories visited historically")

    # Group 6: Merchant Interaction Features (6 features)
    account_merchant_txn_count_before: float = Field(..., ge=0.0, description="Account transactions at this merchant")
    account_category_txn_count_before: float = Field(..., ge=0.0, description="Account transactions in this category")
    account_merchant_spend_before: float = Field(..., description="Account cumulative spend at this merchant")
    account_category_spend_before: float = Field(..., description="Account cumulative spend in this category")
    merchant_txn_count_before: float = Field(..., ge=0.0, description="Global transaction count for this merchant")
    category_txn_count_before: float = Field(..., ge=0.0, description="Global transaction count for this category")

    # Group 7: Geographic Features (4 features)
    cardholder_merchant_distance_km: float = Field(..., ge=0.0, description="Great-circle distance (km) between cardholder and merchant")
    distance_from_prev_merchant_km: float = Field(..., ge=0.0, description="Distance (km) from previous merchant location")
    implied_travel_speed_kmh: float = Field(..., ge=0.0, description="Implied travel speed (km/h) between consecutive transactions")
    is_impossible_travel_speed: int = Field(..., ge=0, le=1, description="Flag for physically impossible travel speed (>800 km/h)")


class PredictionResponse(BaseModel):
    """
    Standardized fraud prediction and risk intelligence evaluation response payload.
    """
    transaction_id: Optional[str] = Field(None, description="Transaction identifier if supplied")
    model_score: float = Field(..., description="Continuous ML fraud probability in [0.0, 1.0]")
    risk_score: int = Field(..., description="Calibrated integer risk score in [0, 100]")
    risk_tier: str = Field(..., description="Categorical risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    decision_action: str = Field(..., description="Operational decision: APPROVE, REVIEW, BLOCK")
    policy_mode: str = Field(..., description="Decision policy mode (e.g. 'TRI_TIER')")
    reason: str = Field(..., description="Human-readable policy decision boundary explanation")
    is_overridden: bool = Field(..., description="True if a business rule escalated the baseline ML decision")
    rule_action: Optional[str] = Field(None, description="Outcome of triggering business rule if overridden")
    rules_triggered: List[str] = Field(default_factory=list, description="List of triggered business rule IDs")
    rule_matches: List[RuleMatchResponse] = Field(default_factory=list, description="Structured details of triggered rules")
    reason_codes: List[ReasonCodeResponse] = Field(default_factory=list, description="Ordered plain-English reason codes")
    top_risk_factors: List[FeatureAttributionResponse] = Field(default_factory=list, description="Top positive TreeSHAP risk factors")
    top_mitigating_factors: List[FeatureAttributionResponse] = Field(default_factory=list, description="Top negative TreeSHAP mitigating factors")
    model_version: Optional[str] = Field(None, description="Model artifact provenance version string")
    evaluated_at: str = Field(..., description="ISO 8601 evaluation timestamp")
