import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.enums import DecisionAction, RiskTier
from backend.app.db.session import get_db_session
from backend.app.repositories.dashboard_repository import DashboardRepository
from backend.app.repositories.exceptions import PersistenceError
from backend.app.schemas.predict import TransactionPredictRequest
from backend.app.services.risk_service import RiskService, get_risk_service
from ml.risk_engine.catalog import get_standard_rule_catalog
from backend.app.schemas.dashboard import (
    AnalyticsDistributionsResponse,
    AnalyticsRulesResponse,
    AnalyticsTrendsResponse,
    BaselineEvaluationSummary,
    DashboardOverviewResponse,
    FeatureAttributionDetail,
    FeatureDiffItem,
    ReasonCodeDetail,
    RuleDiffItem,
    RuleDiffStatus,
    RuleMatchDetail,
    SimulatedEvaluationSummary,
    SimulationComparisonSummary,
    SimulationRequest,
    SimulationResponse,
    TransactionDetailResponse,
    TransactionListResponse,
    TrendInterval,
)

logger = logging.getLogger("fraud_api.dashboard")

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def get_dashboard_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DashboardRepository:
    """Dependency provider for DashboardRepository."""
    return DashboardRepository(session)


@router.get(
    "/overview",
    response_model=DashboardOverviewResponse,
    summary="Get Fraud Intelligence Overview Metrics",
    description=(
        "Retrieve high-level operational KPIs including total transactions, total amount, "
        "approval/review/block counts and rates, average risk score, and average evaluation latency."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_dashboard_overview(
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> DashboardOverviewResponse:
    """
    Compute and return operational overview KPIs across all persisted transactions.
    """
    try:
        return await repo.get_overview_metrics()
    except PersistenceError as exc:
        logger.error(f"Failed to fetch dashboard overview metrics: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard overview metrics.",
        ) from exc


@router.get(
    "/transactions",
    response_model=TransactionListResponse,
    summary="Get Paginated Transaction Feed",
    description=(
        "Retrieve a paginated collection of compact transaction records with optional "
        "filtering by decision, risk tier, score boundaries, search term, and date range. "
        "Sensitive information and bulky feature vectors are omitted."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_dashboard_transactions(
    limit: int = Query(20, ge=1, le=100, description="Max transactions per page (1 to 100)."),
    offset: int = Query(0, ge=0, description="Pagination offset index (>= 0)."),
    decision_action: Optional[DecisionAction] = Query(None, description="Filter by decision action."),
    risk_tier: Optional[RiskTier] = Query(None, description="Filter by risk tier."),
    search_term: Optional[str] = Query(None, description="Substring search matching ID or merchant category."),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum calibrated risk score [0, 100]."),
    max_score: Optional[int] = Query(None, ge=0, le=100, description="Maximum calibrated risk score [0, 100]."),
    start_date: Optional[datetime] = Query(None, description="Earliest timestamp boundary (ISO 8601)."),
    end_date: Optional[datetime] = Query(None, description="Latest timestamp boundary (ISO 8601)."),
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> TransactionListResponse:
    """
    Query paginated transaction records with database-level filtering.
    """
    if min_score is not None and max_score is not None and min_score > max_score:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_score cannot exceed max_score.",
        )

    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        return await repo.get_transactions(
            limit=limit,
            offset=offset,
            decision_action=decision_action,
            risk_tier=risk_tier,
            search_term=search_term,
            min_score=min_score,
            max_score=max_score,
            start_date=start_date,
            end_date=end_date,
        )
    except PersistenceError as exc:
        logger.error(f"Failed to fetch dashboard transactions: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard transactions.",
        ) from exc


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionDetailResponse,
    summary="Get Detailed Transaction Investigation Context",
    description=(
        "Retrieve complete fraud evaluation context for a transaction including canonical metadata, "
        "persisted risk evaluation, 55-feature snapshot, TreeSHAP attributions, reason codes, "
        "triggered business rules, and audit trail. This is a read-only investigation endpoint."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Transaction not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Transaction 'c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f' not found."}
                }
            },
        }
    },
)
async def get_dashboard_transaction_detail(
    transaction_id: str,
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> TransactionDetailResponse:
    """
    Retrieve full transaction investigation details by UUID or client external ID.
    """
    if not transaction_id or not transaction_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction identifier must not be empty.",
        )

    try:
        detail = await repo.get_transaction_detail(transaction_id.strip())
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction '{transaction_id}' not found.",
            )
        return detail
    except HTTPException:
        raise
    except PersistenceError as exc:
        logger.error(f"Failed to retrieve transaction detail for '{transaction_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction details.",
        ) from exc


@router.get(
    "/analytics/distributions",
    response_model=AnalyticsDistributionsResponse,
    summary="Get Fraud Risk & Score Distributions",
    description=(
        "Retrieve aggregated risk score (10-bucket histogram), model probability margin (10 buckets), "
        "risk tier breakdown, and operational decision action distributions using database-side SQL aggregation."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_dashboard_analytics_distributions(
    start_date: Optional[datetime] = Query(None, description="Earliest evaluated timestamp (ISO 8601)."),
    end_date: Optional[datetime] = Query(None, description="Latest evaluated timestamp (ISO 8601)."),
    decision_action: Optional[DecisionAction] = Query(None, description="Filter by decision action."),
    risk_tier: Optional[RiskTier] = Query(None, description="Filter by risk tier."),
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> AnalyticsDistributionsResponse:
    """
    Compute and return score histograms, probability distributions, and tier/decision breakdowns.
    """
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        return await repo.get_analytics_distributions(
            start_date=start_date,
            end_date=end_date,
            decision_action=decision_action,
            risk_tier=risk_tier,
        )
    except PersistenceError as exc:
        logger.error(f"Failed to fetch analytics distributions: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics distributions.",
        ) from exc


@router.get(
    "/analytics/trends",
    response_model=AnalyticsTrendsResponse,
    summary="Get Time-Series Fraud Risk Trends",
    description=(
        "Retrieve time-series aggregated transaction volumes, monetary values, average risk scores, "
        "and decision action breakdowns bucketed by hour or day with contiguous zero-filling."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_dashboard_analytics_trends(
    start_date: Optional[datetime] = Query(None, description="Earliest timestamp boundary (ISO 8601)."),
    end_date: Optional[datetime] = Query(None, description="Latest timestamp boundary (ISO 8601)."),
    interval: Optional[TrendInterval] = Query(None, description="Time bucketing interval ('hourly' or 'daily')."),
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> AnalyticsTrendsResponse:
    """
    Compute and return contiguous time-series telemetry trends for volume, risk scores, and decisions.
    """
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        return await repo.get_analytics_trends(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
    except PersistenceError as exc:
        logger.error(f"Failed to fetch analytics trends: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics trends.",
        ) from exc


@router.get(
    "/analytics/rules",
    response_model=AnalyticsRulesResponse,
    summary="Get Business Rule Trigger & Performance Analytics",
    description=(
        "Retrieve operational analytics for deterministic business rules including trigger frequencies, "
        "distinct affected transactions, trigger rates, override counts, and outcome distributions."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_dashboard_analytics_rules(
    start_date: Optional[datetime] = Query(None, description="Earliest evaluated timestamp (ISO 8601)."),
    end_date: Optional[datetime] = Query(None, description="Latest evaluated timestamp (ISO 8601)."),
    limit: int = Query(20, ge=1, le=100, description="Max ranked rule items to return (1 to 100)."),
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> AnalyticsRulesResponse:
    """
    Compute and return ranked business rule performance metrics ordered by trigger count.
    """
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        return await repo.get_analytics_rules(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except PersistenceError as exc:
        logger.error(f"Failed to fetch analytics rules: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics rules.",
        ) from exc


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    summary="Execute Read-Only What-If Transaction Simulation",
    description=(
        "Simulates a candidate fraud risk scenario in-memory using the production RiskEvaluator and RuleEngine. "
        "Strictly read-only; no database records or audit logs are persisted."
    ),
    status_code=status.HTTP_200_OK,
)
async def simulate_transaction(
    request: SimulationRequest,
    risk_service: RiskService = Depends(get_risk_service),
    repo: DashboardRepository = Depends(get_dashboard_repository),
) -> SimulationResponse:
    """
    Evaluate candidate transaction features in-memory and return risk score, tier, decision action,
    rule evaluations, TreeSHAP attributions, and baseline comparison deltas without persisting state.
    """
    # 1. Baseline Lookup (if baseline_transaction_id is provided)
    baseline_summary: Optional[BaselineEvaluationSummary] = None
    baseline_features: Dict[str, Any] = {}
    baseline_rule_ids: set[str] = set()

    if request.baseline_transaction_id is not None:
        raw_id = request.baseline_transaction_id.strip()
        if not raw_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Baseline transaction identifier cannot be empty.",
            )
        try:
            baseline_tx = await repo.get_transaction_detail(raw_id)
        except PersistenceError as exc:
            logger.error(f"Persistence error looking up baseline transaction '{raw_id}': {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve baseline transaction details.",
            ) from exc

        if baseline_tx is None or baseline_tx.evaluation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Baseline transaction '{raw_id}' not found.",
            )

        b_tx = baseline_tx.transaction
        b_ev = baseline_tx.evaluation
        baseline_summary = BaselineEvaluationSummary(
            transaction_id=b_tx.id,
            external_transaction_id=b_tx.external_transaction_id,
            risk_score=b_ev.risk_score,
            risk_tier=b_ev.risk_tier,
            decision_action=b_ev.decision_action,
            model_score=b_ev.model_score,
            is_overridden=b_ev.is_overridden,
            rules_triggered_count=len(baseline_tx.rule_matches),
        )
        if baseline_tx.features:
            if isinstance(baseline_tx.features, dict):
                baseline_features = dict(baseline_tx.features)
            elif hasattr(baseline_tx.features, "features_snapshot"):
                baseline_features = dict(baseline_tx.features.features_snapshot)
        baseline_rule_ids = {rm.rule_id for rm in baseline_tx.rule_matches}

    # 2. Validate simulated_features against the canonical 55-feature contract
    if not isinstance(request.simulated_features, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="simulated_features must be a dictionary of 55 features.",
        )

    # Check for unknown / extraneous features
    from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
    canonical_keys = set(PREDICTIVE_FEATURE_COLUMNS)
    unknown_keys = set(request.simulated_features.keys()) - canonical_keys
    if unknown_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown or unregistered features detected: {sorted(list(unknown_keys))}",
        )

    # Check for missing features
    missing_keys = canonical_keys - set(request.simulated_features.keys())
    if missing_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required features in simulation payload: {sorted(list(missing_keys))}",
        )

    # Reject NaN and Infinity across all numerical values
    for k, v in request.simulated_features.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Feature '{k}' contains invalid float value (NaN or Infinity).",
            )

    try:
        validated_predict_req = TransactionPredictRequest(**request.simulated_features)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        )

    # 3. Execute in-memory ML inference & TreeSHAP explainability
    eval_start_time = time.perf_counter()
    try:
        explanation = risk_service.evaluator.explain_transaction(
            df_or_series=validated_predict_req.model_dump(),
            top_k=request.top_k,
            top_mitigating=request.top_mitigating,
            max_reasons=request.max_reasons,
        )
    except Exception as exc:
        logger.error(f"In-memory simulation evaluation failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation evaluation failed: {exc}",
        ) from exc
    eval_latency_ms = round((time.perf_counter() - eval_start_time) * 1000.0, 2)

    # 4. Map simulated evaluation details
    sim_rule_matches: List[RuleMatchDetail] = [
        RuleMatchDetail(
            id=f"sim-rule-{idx+1}",
            rule_id=m.rule_id,
            description=m.description,
            feature_name=m.feature_name,
            operator=m.operator.value if hasattr(m.operator, "value") else str(m.operator),
            comparison_value=str(m.comparison_value),
            outcome=m.outcome.value if hasattr(m.outcome, "value") else str(m.outcome),
            rule_type=m.rule_type.value if hasattr(m.rule_type, "value") else str(m.rule_type),
            priority=m.priority,
        )
        for idx, m in enumerate(explanation.rule_matches)
    ]

    sim_reason_codes: List[ReasonCodeDetail] = [
        ReasonCodeDetail(
            id=f"sim-rc-{idx+1}",
            code=r.code,
            headline=r.headline,
            description=r.description,
            category=r.category,
            source=r.source.value if hasattr(r.source, "value") else str(r.source),
            severity=r.severity.value if hasattr(r.severity, "value") else str(r.severity),
            rank=r.rank,
        )
        for idx, r in enumerate(explanation.reason_codes)
    ]

    # Combine top risk and mitigating feature attributions
    combined_attributions = list(explanation.top_risk_factors) + list(explanation.top_mitigating_factors)
    sim_feature_attributions: List[FeatureAttributionDetail] = [
        FeatureAttributionDetail(
            id=f"sim-fa-{idx+1}",
            feature_name=fa.feature_name,
            display_name=fa.display_name,
            raw_value=str(fa.raw_value),
            shap_value=round(float(fa.shap_value), 6),
            direction=fa.direction.value if hasattr(fa.direction, "value") else str(fa.direction),
            relative_contribution_pct=round(float(fa.relative_contribution_pct), 2),
            rank=fa.rank,
        )
        for idx, fa in enumerate(combined_attributions)
    ]

    if explanation.is_overridden:
        rules_str = ", ".join(explanation.rules_triggered)
        decision_reason = (
            f"Baseline ML action overridden to {explanation.action.value} by rule(s): {rules_str}."
        )
    else:
        decision_reason = (
            f"Policy mode {explanation.policy_mode.value} evaluated action {explanation.action.value} "
            f"with normalized risk score {explanation.risk_score}/100."
        )

    simulated_summary = SimulatedEvaluationSummary(
        risk_score=explanation.risk_score,
        risk_tier=explanation.risk_tier,
        decision_action=explanation.action,
        model_score=round(float(explanation.model_score), 6),
        base_value=round(float(explanation.base_value), 6) if explanation.base_value is not None else None,
        output_margin=round(float(explanation.output_margin), 6) if explanation.output_margin is not None else None,
        is_overridden=explanation.is_overridden,
        decision_reason=decision_reason,
        rule_matches=sim_rule_matches,
        reason_codes=sim_reason_codes,
        feature_attributions=sim_feature_attributions,
    )

    # 5. Compute Comparative Diff (if baseline exists)
    comparison_summary: Optional[SimulationComparisonSummary] = None
    if baseline_summary is not None:
        score_delta = simulated_summary.risk_score - baseline_summary.risk_score
        model_delta = round(simulated_summary.model_score - baseline_summary.model_score, 6)
        tier_changed = (simulated_summary.risk_tier != baseline_summary.risk_tier)
        action_changed = (simulated_summary.decision_action != baseline_summary.decision_action)

        # Feature diffs across all canonical 55 features
        sim_features_dict = validated_predict_req.model_dump()
        feature_diffs: List[FeatureDiffItem] = []
        modified_count = 0

        for f_name, sim_val in sim_features_dict.items():
            base_val = baseline_features.get(f_name)
            is_mod = (base_val is not None and sim_val != base_val) or (base_val is None)
            delta: Optional[float] = None
            if is_mod:
                modified_count += 1
                if isinstance(sim_val, (int, float)) and isinstance(base_val, (int, float)):
                    delta = round(float(sim_val) - float(base_val), 4)

            # Assign category
            cat = "General"
            if f_name.startswith("txn_count_") or f_name in ["time_since_prev_txn_seconds", "is_first_account_txn"]:
                cat = "Velocity & Frequency"
            elif f_name.startswith("amt_") and not f_name.startswith("amount_"):
                cat = "Spending & Monetary Volume"
            elif f_name.startswith("historical_") or f_name.startswith("amount_"):
                cat = "Spending Deviation & Z-Scores"
            elif f_name.startswith("account_") and not f_name.startswith("account_merchant_") and not f_name.startswith("account_category_"):
                cat = "Account History & Diversity"
            elif f_name.startswith("account_merchant_") or f_name.startswith("account_category_") or f_name.endswith("_txn_count_before"):
                cat = "Merchant & Category Interactions"
            elif "distance" in f_name or "speed" in f_name:
                cat = "Geographic & Impossible Travel"
            elif f_name.startswith("transaction_hour") or f_name.startswith("day_") or f_name.startswith("hour_") or f_name.startswith("is_"):
                cat = "Temporal & Periodicity"

            feature_diffs.append(
                FeatureDiffItem(
                    feature_name=f_name,
                    display_name=f_name.replace("_", " ").title(),
                    category=cat,
                    baseline_value=base_val,
                    simulated_value=sim_val,
                    is_modified=is_mod,
                    delta=delta,
                )
            )

        # Rule diffs (deduplicated by rule_id across catalog, baseline, and simulated matches)
        sim_rule_ids = {m.rule_id for m in explanation.rule_matches}
        sim_rule_map = {m.rule_id: m for m in explanation.rule_matches}
        standard_catalog = get_standard_rule_catalog()
        catalog_rule_map = {r.rule_id: r for r in standard_catalog}

        all_rule_ids = list(dict.fromkeys([r.rule_id for r in standard_catalog] + list(baseline_rule_ids) + list(sim_rule_ids)))
        rule_diffs: List[RuleDiffItem] = []

        for r_id in all_rule_ids:
            in_base = r_id in baseline_rule_ids
            in_sim = r_id in sim_rule_ids
            if in_sim and not in_base:
                diff_st = RuleDiffStatus.NEWLY_TRIGGERED
            elif in_base and not in_sim:
                diff_st = RuleDiffStatus.RESOLVED
            elif in_base and in_sim:
                diff_st = RuleDiffStatus.PERSISTENT
            else:
                diff_st = RuleDiffStatus.NEITHER

            if r_id in catalog_rule_map:
                r_meta = catalog_rule_map[r_id]
                desc = r_meta.description
                r_type = r_meta.rule_type.value if hasattr(r_meta.rule_type, "value") else str(r_meta.rule_type)
                priority = r_meta.priority
                outcome = r_meta.outcome.value if hasattr(r_meta.outcome, "value") else str(r_meta.outcome)
            elif r_id in sim_rule_map:
                s_meta = sim_rule_map[r_id]
                desc = s_meta.description
                r_type = s_meta.rule_type.value if hasattr(s_meta.rule_type, "value") else str(s_meta.rule_type)
                priority = s_meta.priority
                outcome = s_meta.outcome.value if hasattr(s_meta.outcome, "value") else str(s_meta.outcome)
            else:
                desc = f"Rule {r_id}"
                r_type = "CUSTOM"
                priority = 99
                outcome = "MONITOR"

            rule_diffs.append(
                RuleDiffItem(
                    rule_id=r_id,
                    description=desc,
                    rule_type=r_type,
                    priority=priority,
                    outcome=outcome,
                    baseline_triggered=in_base,
                    simulated_triggered=in_sim,
                    diff_status=diff_st,
                )
            )

        comparison_summary = SimulationComparisonSummary(
            risk_score_delta=score_delta,
            model_score_delta=model_delta,
            tier_changed=tier_changed,
            action_changed=action_changed,
            modified_features_count=modified_count,
            feature_diffs=feature_diffs,
            rule_diffs=rule_diffs,
        )

    return SimulationResponse(
        is_simulation=True,
        simulated_at=datetime.now(timezone.utc),
        evaluation_latency_ms=eval_latency_ms,
        baseline_transaction_id=request.baseline_transaction_id,
        baseline=baseline_summary,
        simulated=simulated_summary,
        comparison=comparison_summary,
    )



