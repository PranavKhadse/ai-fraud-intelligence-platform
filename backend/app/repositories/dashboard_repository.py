from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union
import logging
import uuid
from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import DecisionAction, RiskTier, RuleOutcome, RuleType
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.db.models.transaction import Transaction
from backend.app.repositories.exceptions import PersistenceError
from backend.app.schemas.dashboard import (
    AnalyticsDistributionsResponse,
    AnalyticsRulesResponse,
    AnalyticsTrendsResponse,
    AuditLogDetail,
    CategoryCount,
    DashboardOverviewResponse,
    DistributionBucket,
    EvaluationDetail,
    FeatureAttributionDetail,
    ReasonCodeDetail,
    RuleAnalyticsItem,
    RuleMatchDetail,
    RuleOutcomeBreakdown,
    TransactionDetailResponse,
    TransactionListItem,
    TransactionListResponse,
    TransactionMetadataDetail,
    TrendDataPoint,
    TrendInterval,
)

logger = logging.getLogger("fraud_api.dashboard_repository")


class DashboardRepository:
    """
    Asynchronous repository providing optimized SQL aggregation and filtering queries for dashboard views.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the DashboardRepository with an active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_overview_metrics(self) -> DashboardOverviewResponse:
        """
        Compute high-level operational KPIs using database-level SQL aggregations.

        Returns:
            DashboardOverviewResponse containing aggregated transaction volumes, rates,
            and mean scores. Safe zero-values are returned if the database is empty.

        Raises:
            PersistenceError: If an unexpected database aggregation query fails.
        """
        try:
            stmt = select(
                func.count(Transaction.id).label("total_transactions"),
                func.coalesce(func.sum(Transaction.amount), 0).label("total_amount"),
                func.coalesce(func.avg(RiskEvaluation.risk_score), 0.0).label("avg_risk_score"),
                func.coalesce(func.avg(RiskEvaluation.evaluation_latency_ms), 0.0).label("avg_latency_ms"),
                func.coalesce(
                    func.count().filter(RiskEvaluation.decision_action == DecisionAction.APPROVE), 0
                ).label("approval_count"),
                func.coalesce(
                    func.count().filter(RiskEvaluation.decision_action == DecisionAction.REVIEW), 0
                ).label("review_count"),
                func.coalesce(
                    func.count().filter(RiskEvaluation.decision_action == DecisionAction.BLOCK), 0
                ).label("block_count"),
            ).select_from(Transaction).join(
                RiskEvaluation, RiskEvaluation.transaction_id == Transaction.id, isouter=True
            )

            result = await self._session.execute(stmt)
            row = result.mappings().one_or_none()

            if not row or row["total_transactions"] == 0:
                return DashboardOverviewResponse.empty()

            total_tx = int(row["total_transactions"])
            total_amt = float(row["total_amount"])
            avg_score = float(row["avg_risk_score"])
            avg_lat = float(row["avg_latency_ms"])
            appr_cnt = int(row["approval_count"])
            rev_cnt = int(row["review_count"])
            blk_cnt = int(row["block_count"])

            appr_rate = round((appr_cnt / total_tx) * 100.0, 2) if total_tx > 0 else 0.0
            rev_rate = round((rev_cnt / total_tx) * 100.0, 2) if total_tx > 0 else 0.0
            blk_rate = round((blk_cnt / total_tx) * 100.0, 2) if total_tx > 0 else 0.0

            return DashboardOverviewResponse(
                total_transactions=total_tx,
                total_amount=round(total_amt, 2),
                approval_count=appr_cnt,
                approval_rate=appr_rate,
                review_count=rev_cnt,
                review_rate=rev_rate,
                block_count=blk_cnt,
                block_rate=blk_rate,
                average_risk_score=round(avg_score, 2),
                average_latency_ms=round(avg_lat, 2),
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute dashboard overview aggregation query: {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve dashboard overview metrics: {exc}"
            ) from exc

    async def get_transactions(
        self,
        limit: int = 20,
        offset: int = 0,
        decision_action: Optional[DecisionAction] = None,
        risk_tier: Optional[RiskTier] = None,
        search_term: Optional[str] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> TransactionListResponse:
        """
        Retrieve a paginated collection of compact transaction items with optional database-level filtering.

        Args:
            limit: Maximum items to return (1 <= limit <= 100).
            offset: Offset items to skip (offset >= 0).
            decision_action: Filter by decision action (APPROVE, REVIEW, BLOCK).
            risk_tier: Filter by risk tier (LOW, MEDIUM, HIGH, CRITICAL).
            search_term: Case-insensitive substring search matching external transaction ID,
                         account ID, merchant ID, or merchant category.
            min_score: Minimum calibrated risk score filter [0, 100].
            max_score: Maximum calibrated risk score filter [0, 100].
            start_date: Earliest transaction timestamp boundary.
            end_date: Latest transaction timestamp boundary.

        Returns:
            TransactionListResponse containing compact TransactionListItem entries, total count, limit, and offset.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        # Validate boundaries
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        try:
            # Build filters
            filters = []

            if decision_action is not None:
                filters.append(RiskEvaluation.decision_action == decision_action)

            if risk_tier is not None:
                filters.append(RiskEvaluation.risk_tier == risk_tier)

            if min_score is not None:
                filters.append(RiskEvaluation.risk_score >= min_score)

            if max_score is not None:
                filters.append(RiskEvaluation.risk_score <= max_score)

            if start_date is not None:
                filters.append(Transaction.transaction_timestamp >= start_date)

            if end_date is not None:
                filters.append(Transaction.transaction_timestamp <= end_date)

            if search_term and search_term.strip():
                clean_term = f"%{search_term.strip()}%"
                filters.append(
                    or_(
                        Transaction.external_transaction_id.ilike(clean_term),
                        Transaction.account_id.ilike(clean_term),
                        Transaction.merchant_id.ilike(clean_term),
                        Transaction.merchant_category.ilike(clean_term),
                    )
                )

            # Count total matching rows
            count_stmt = (
                select(func.count(Transaction.id))
                .select_from(Transaction)
                .join(RiskEvaluation, RiskEvaluation.transaction_id == Transaction.id, isouter=True)
            )
            if filters:
                count_stmt = count_stmt.where(and_(*filters))

            count_res = await self._session.execute(count_stmt)
            total_count = count_res.scalar() or 0

            if total_count == 0:
                return TransactionListResponse(
                    items=[],
                    total_count=0,
                    limit=safe_limit,
                    offset=safe_offset,
                )

            # Query paginated rows
            data_stmt = (
                select(
                    Transaction.id.label("id"),
                    Transaction.external_transaction_id.label("external_transaction_id"),
                    Transaction.transaction_timestamp.label("transaction_timestamp"),
                    Transaction.amount.label("amount"),
                    Transaction.currency.label("currency"),
                    Transaction.merchant_category.label("merchant_category"),
                    func.coalesce(RiskEvaluation.risk_score, 0).label("risk_score"),
                    func.coalesce(RiskEvaluation.risk_tier, RiskTier.LOW).label("risk_tier"),
                    func.coalesce(RiskEvaluation.decision_action, DecisionAction.APPROVE).label("decision_action"),
                    func.coalesce(RiskEvaluation.is_overridden, False).label("is_overridden"),
                    RiskEvaluation.evaluation_latency_ms.label("evaluation_latency_ms"),
                )
                .select_from(Transaction)
                .join(RiskEvaluation, RiskEvaluation.transaction_id == Transaction.id, isouter=True)
            )

            if filters:
                data_stmt = data_stmt.where(and_(*filters))

            # Default ordering: newest evaluated transactions first
            data_stmt = (
                data_stmt.order_by(Transaction.transaction_timestamp.desc(), Transaction.id.desc())
                .limit(safe_limit)
                .offset(safe_offset)
            )

            result = await self._session.execute(data_stmt)
            rows = result.mappings().all()

            items: List[TransactionListItem] = []
            for r in rows:
                items.append(
                    TransactionListItem(
                        id=str(r["id"]),
                        external_transaction_id=r["external_transaction_id"],
                        transaction_timestamp=r["transaction_timestamp"],
                        amount=float(r["amount"]),
                        currency=str(r["currency"]),
                        merchant_category=str(r["merchant_category"]),
                        risk_score=int(r["risk_score"]),
                        risk_tier=r["risk_tier"],
                        decision_action=r["decision_action"],
                        is_overridden=bool(r["is_overridden"]),
                        evaluation_latency_ms=float(r["evaluation_latency_ms"]) if r["evaluation_latency_ms"] is not None else None,
                    )
                )

            return TransactionListResponse(
                items=items,
                total_count=total_count,
                limit=safe_limit,
                offset=safe_offset,
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute dashboard transactions query: {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve dashboard transactions: {exc}"
            ) from exc

    async def get_transaction_detail(
        self,
        transaction_id: Union[uuid.UUID, str],
    ) -> Optional[TransactionDetailResponse]:
        """
        Retrieve full transaction investigation details including metadata, risk evaluation,
        55-feature snapshot, TreeSHAP attributions, reason codes, triggered rules, and audit trail.

        Args:
            transaction_id: Internal primary key UUID or external client transaction ID.

        Returns:
            TransactionDetailResponse if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            # 1. Parse identifier as UUID if possible
            parsed_uuid: Optional[uuid.UUID] = None
            if isinstance(transaction_id, uuid.UUID):
                parsed_uuid = transaction_id
            elif isinstance(transaction_id, str):
                try:
                    parsed_uuid = uuid.UUID(transaction_id.strip())
                except (ValueError, AttributeError):
                    parsed_uuid = None

            # 2. Build query with eager loading for evaluations and child collections
            stmt = (
                select(Transaction)
                .options(
                    selectinload(Transaction.evaluations).selectinload(RiskEvaluation.rule_matches),
                    selectinload(Transaction.evaluations).selectinload(RiskEvaluation.reason_codes),
                    selectinload(Transaction.evaluations).selectinload(RiskEvaluation.feature_attributions),
                )
            )

            if parsed_uuid is not None:
                stmt = stmt.where(Transaction.id == parsed_uuid)
            else:
                str_id = str(transaction_id).strip()
                stmt = stmt.where(Transaction.external_transaction_id == str_id)

            result = await self._session.execute(stmt)
            tx: Optional[Transaction] = result.scalar_one_or_none()

            # If not found by UUID, try external_transaction_id as fallback
            if tx is None and parsed_uuid is not None and isinstance(transaction_id, str):
                fallback_stmt = (
                    select(Transaction)
                    .where(Transaction.external_transaction_id == transaction_id.strip())
                    .options(
                        selectinload(Transaction.evaluations).selectinload(RiskEvaluation.rule_matches),
                        selectinload(Transaction.evaluations).selectinload(RiskEvaluation.reason_codes),
                        selectinload(Transaction.evaluations).selectinload(RiskEvaluation.feature_attributions),
                    )
                )
                fb_res = await self._session.execute(fallback_stmt)
                tx = fb_res.scalar_one_or_none()

            if tx is None:
                return None

            # 3. Map Transaction Metadata
            tx_metadata = TransactionMetadataDetail(
                id=str(tx.id),
                external_transaction_id=tx.external_transaction_id,
                account_id=tx.account_id,
                merchant_id=tx.merchant_id,
                merchant_category=tx.merchant_category,
                job_category=tx.job_category,
                amount=float(tx.amount),
                currency=tx.currency,
                cardholder_lat=float(tx.cardholder_lat),
                cardholder_long=float(tx.cardholder_long),
                merchant_lat=float(tx.merchant_lat),
                merchant_long=float(tx.merchant_long),
                city_pop=int(tx.city_pop),
                transaction_timestamp=tx.transaction_timestamp,
                created_at=tx.created_at,
            )

            # 4. Map 55-Feature Snapshot (Persisted point-in-time vector)
            features: dict = dict(tx.features_snapshot) if tx.features_snapshot else {}

            # 5. Map Latest Risk Evaluation (if exists)
            eval_record: Optional[RiskEvaluation] = None
            if tx.evaluations:
                # Sort newest evaluated_at first
                sorted_evals = sorted(
                    tx.evaluations,
                    key=lambda e: e.evaluated_at or e.created_at,
                    reverse=True,
                )
                eval_record = sorted_evals[0]

            eval_detail: Optional[EvaluationDetail] = None
            feature_attributions: List[FeatureAttributionDetail] = []
            reason_codes: List[ReasonCodeDetail] = []
            rule_matches: List[RuleMatchDetail] = []

            if eval_record is not None:
                eval_detail = EvaluationDetail(
                    id=str(eval_record.id),
                    model_version=eval_record.model_version,
                    policy_mode=eval_record.policy_mode,
                    model_score=float(eval_record.model_score),
                    risk_score=int(eval_record.risk_score),
                    risk_tier=eval_record.risk_tier,
                    decision_action=eval_record.decision_action,
                    baseline_action=eval_record.baseline_action,
                    is_overridden=bool(eval_record.is_overridden),
                    rule_action=eval_record.rule_action,
                    decision_reason=eval_record.decision_reason,
                    output_margin=float(eval_record.output_margin) if eval_record.output_margin is not None else None,
                    base_value=float(eval_record.base_value) if eval_record.base_value is not None else None,
                    evaluation_latency_ms=float(eval_record.evaluation_latency_ms) if eval_record.evaluation_latency_ms is not None else None,
                    correlation_id=eval_record.correlation_id,
                    evaluated_at=eval_record.evaluated_at,
                )

                # Deterministic sorting for Feature Attributions: rank ASC, abs(shap_value) DESC
                sorted_fa = sorted(
                    eval_record.feature_attributions,
                    key=lambda fa: (fa.rank, -abs(float(fa.shap_value))),
                )
                for fa in sorted_fa:
                    feature_attributions.append(
                        FeatureAttributionDetail(
                            id=str(fa.id),
                            feature_name=fa.feature_name,
                            display_name=fa.display_name,
                            raw_value=fa.raw_value,
                            shap_value=float(fa.shap_value),
                            direction=fa.direction,
                            relative_contribution_pct=float(fa.relative_contribution_pct),
                            rank=int(fa.rank),
                        )
                    )

                # Deterministic sorting for Reason Codes: rank ASC
                sorted_rc = sorted(
                    eval_record.reason_codes,
                    key=lambda rc: rc.rank,
                )
                for rc in sorted_rc:
                    reason_codes.append(
                        ReasonCodeDetail(
                            id=str(rc.id),
                            code=rc.code,
                            headline=rc.headline,
                            description=rc.description,
                            category=rc.category,
                            source=rc.source,
                            severity=rc.severity,
                            rank=int(rc.rank),
                        )
                    )

                # Deterministic sorting for Rule Matches: priority ASC
                sorted_rm = sorted(
                    eval_record.rule_matches,
                    key=lambda rm: rm.priority,
                )
                for rm in sorted_rm:
                    rule_matches.append(
                        RuleMatchDetail(
                            id=str(rm.id),
                            rule_id=rm.rule_id,
                            description=rm.description,
                            feature_name=rm.feature_name,
                            operator=rm.operator,
                            comparison_value=rm.comparison_value,
                            outcome=rm.outcome,
                            rule_type=rm.rule_type,
                            priority=int(rm.priority),
                        )
                    )

            # 6. Retrieve Associated Audit Logs (Deterministic order: event_timestamp DESC, id DESC)
            audit_conditions = [AuditLog.entity_id == tx.id]
            if eval_record is not None:
                audit_conditions.append(AuditLog.entity_id == eval_record.id)

            audit_stmt = (
                select(AuditLog)
                .where(or_(*audit_conditions))
                .order_by(AuditLog.event_timestamp.desc(), AuditLog.id.desc())
            )
            audit_res = await self._session.execute(audit_stmt)
            audit_records = audit_res.scalars().all()

            audit_trail: List[AuditLogDetail] = []
            for log in audit_records:
                actor_type_str = log.actor_type.value if hasattr(log.actor_type, "value") else str(log.actor_type)
                audit_trail.append(
                    AuditLogDetail(
                        id=str(log.id),
                        event_type=log.event_type,
                        action=log.action,
                        actor_type=actor_type_str,
                        actor_id=log.actor_id,
                        correlation_id=log.correlation_id,
                        client_ip=log.client_ip,
                        event_timestamp=log.event_timestamp,
                    )
                )

            return TransactionDetailResponse(
                transaction=tx_metadata,
                evaluation=eval_detail,
                features=features,
                feature_attributions=feature_attributions,
                reason_codes=reason_codes,
                rule_matches=rule_matches,
                audit_trail=audit_trail,
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute transaction detail query for '{transaction_id}': {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve transaction details for '{transaction_id}': {exc}"
            ) from exc

    async def get_analytics_distributions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        decision_action: Optional[DecisionAction] = None,
        risk_tier: Optional[RiskTier] = None,
    ) -> AnalyticsDistributionsResponse:
        """
        Aggregate risk score, model probability, risk tier, and decision action distributions via SQL.

        Args:
            start_date: Optional earliest evaluated timestamp boundary.
            end_date: Optional latest evaluated timestamp boundary.
            decision_action: Optional filter by final decision action.
            risk_tier: Optional filter by risk tier.

        Returns:
            AnalyticsDistributionsResponse containing 10-bucket histograms and categorical breakdowns.

        Raises:
            PersistenceError: If an unexpected database aggregation query fails.
        """
        try:
            filters = []
            if start_date is not None:
                filters.append(RiskEvaluation.evaluated_at >= start_date)
            if end_date is not None:
                filters.append(RiskEvaluation.evaluated_at <= end_date)
            if decision_action is not None:
                filters.append(RiskEvaluation.decision_action == decision_action)
            if risk_tier is not None:
                filters.append(RiskEvaluation.risk_tier == risk_tier)

            stmt = select(
                func.count(RiskEvaluation.id).label("total_evaluated"),
                # Risk score 10 buckets
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 0, RiskEvaluation.risk_score <= 9)), 0).label("rs_0_9"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 10, RiskEvaluation.risk_score <= 19)), 0).label("rs_10_19"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 20, RiskEvaluation.risk_score <= 29)), 0).label("rs_20_29"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 30, RiskEvaluation.risk_score <= 39)), 0).label("rs_30_39"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 40, RiskEvaluation.risk_score <= 49)), 0).label("rs_40_49"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 50, RiskEvaluation.risk_score <= 59)), 0).label("rs_50_59"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 60, RiskEvaluation.risk_score <= 69)), 0).label("rs_60_69"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 70, RiskEvaluation.risk_score <= 79)), 0).label("rs_70_79"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 80, RiskEvaluation.risk_score <= 89)), 0).label("rs_80_89"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.risk_score >= 90, RiskEvaluation.risk_score <= 100)), 0).label("rs_90_100"),
                # Model score 10 probability buckets
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.0, RiskEvaluation.model_score < 0.1)), 0).label("ms_0_1"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.1, RiskEvaluation.model_score < 0.2)), 0).label("ms_1_2"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.2, RiskEvaluation.model_score < 0.3)), 0).label("ms_2_3"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.3, RiskEvaluation.model_score < 0.4)), 0).label("ms_3_4"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.4, RiskEvaluation.model_score < 0.5)), 0).label("ms_4_5"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.5, RiskEvaluation.model_score < 0.6)), 0).label("ms_5_6"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.6, RiskEvaluation.model_score < 0.7)), 0).label("ms_6_7"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.7, RiskEvaluation.model_score < 0.8)), 0).label("ms_7_8"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.8, RiskEvaluation.model_score < 0.9)), 0).label("ms_8_9"),
                func.coalesce(func.count().filter(and_(RiskEvaluation.model_score >= 0.9, RiskEvaluation.model_score <= 1.0)), 0).label("ms_9_10"),
                # Risk tiers
                func.coalesce(func.count().filter(RiskEvaluation.risk_tier == RiskTier.LOW), 0).label("tier_low"),
                func.coalesce(func.count().filter(RiskEvaluation.risk_tier == RiskTier.MEDIUM), 0).label("tier_med"),
                func.coalesce(func.count().filter(RiskEvaluation.risk_tier == RiskTier.HIGH), 0).label("tier_high"),
                func.coalesce(func.count().filter(RiskEvaluation.risk_tier == RiskTier.CRITICAL), 0).label("tier_crit"),
                # Decisions
                func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.APPROVE), 0).label("act_appr"),
                func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.REVIEW), 0).label("act_rev"),
                func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.BLOCK), 0).label("act_blk"),
            ).select_from(RiskEvaluation)

            if filters:
                stmt = stmt.where(and_(*filters))

            result = await self._session.execute(stmt)
            row = result.mappings().one_or_none()

            total_eval = int(row["total_evaluated"]) if row else 0

            # 1. Map 10 risk score buckets
            rs_ranges = [
                ("0–9", 0.0, 9.0, "rs_0_9"),
                ("10–19", 10.0, 19.0, "rs_10_19"),
                ("20–29", 20.0, 29.0, "rs_20_29"),
                ("30–39", 30.0, 39.0, "rs_30_39"),
                ("40–49", 40.0, 49.0, "rs_40_49"),
                ("50–59", 50.0, 59.0, "rs_50_59"),
                ("60–69", 60.0, 69.0, "rs_60_69"),
                ("70–79", 70.0, 79.0, "rs_70_79"),
                ("80–89", 80.0, 89.0, "rs_80_89"),
                ("90–100", 90.0, 100.0, "rs_90_100"),
            ]
            rs_distribution: List[DistributionBucket] = []
            for label, low, high, key in rs_ranges:
                cnt = int(row[key]) if row else 0
                pct = round((cnt / total_eval) * 100.0, 2) if total_eval > 0 else 0.0
                rs_distribution.append(
                    DistributionBucket(
                        bucket_label=label,
                        lower_bound=low,
                        upper_bound=high,
                        count=cnt,
                        percentage=pct,
                    )
                )

            # 2. Map 10 model score probability buckets
            ms_ranges = [
                ("0.0–0.1", 0.0, 0.1, "ms_0_1"),
                ("0.1–0.2", 0.1, 0.2, "ms_1_2"),
                ("0.2–0.3", 0.2, 0.3, "ms_2_3"),
                ("0.3–0.4", 0.3, 0.4, "ms_3_4"),
                ("0.4–0.5", 0.4, 0.5, "ms_4_5"),
                ("0.5–0.6", 0.5, 0.6, "ms_5_6"),
                ("0.6–0.7", 0.6, 0.7, "ms_6_7"),
                ("0.7–0.8", 0.7, 0.8, "ms_7_8"),
                ("0.8–0.9", 0.8, 0.9, "ms_8_9"),
                ("0.9–1.0", 0.9, 1.0, "ms_9_10"),
            ]
            ms_distribution: List[DistributionBucket] = []
            for label, low, high, key in ms_ranges:
                cnt = int(row[key]) if row else 0
                pct = round((cnt / total_eval) * 100.0, 2) if total_eval > 0 else 0.0
                ms_distribution.append(
                    DistributionBucket(
                        bucket_label=label,
                        lower_bound=low,
                        upper_bound=high,
                        count=cnt,
                        percentage=pct,
                    )
                )

            # 3. Map Risk Tiers
            tier_keys = [
                (RiskTier.LOW.value, "tier_low"),
                (RiskTier.MEDIUM.value, "tier_med"),
                (RiskTier.HIGH.value, "tier_high"),
                (RiskTier.CRITICAL.value, "tier_crit"),
            ]
            tier_distribution: List[CategoryCount] = []
            for tier_name, key in tier_keys:
                cnt = int(row[key]) if row else 0
                pct = round((cnt / total_eval) * 100.0, 2) if total_eval > 0 else 0.0
                tier_distribution.append(CategoryCount(category=tier_name, count=cnt, percentage=pct))

            # 4. Map Decisions
            act_keys = [
                (DecisionAction.APPROVE.value, "act_appr"),
                (DecisionAction.REVIEW.value, "act_rev"),
                (DecisionAction.BLOCK.value, "act_blk"),
            ]
            decision_distribution: List[CategoryCount] = []
            for act_name, key in act_keys:
                cnt = int(row[key]) if row else 0
                pct = round((cnt / total_eval) * 100.0, 2) if total_eval > 0 else 0.0
                decision_distribution.append(CategoryCount(category=act_name, count=cnt, percentage=pct))

            return AnalyticsDistributionsResponse(
                total_evaluated=total_eval,
                risk_score_distribution=rs_distribution,
                model_score_distribution=ms_distribution,
                risk_tier_distribution=tier_distribution,
                decision_distribution=decision_distribution,
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute analytics distributions query: {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve analytics distributions: {exc}"
            ) from exc

    async def get_analytics_trends(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: Optional[TrendInterval] = None,
    ) -> AnalyticsTrendsResponse:
        """
        Aggregate time-series transaction and risk telemetry using date_trunc with contiguous zero-filling.

        Args:
            start_date: Earliest timestamp boundary (defaults to 7 days before end_date).
            end_date: Latest timestamp boundary (defaults to now UTC).
            interval: Optional 'hourly' or 'daily' interval (auto-selected if None).

        Returns:
            AnalyticsTrendsResponse containing chronologically ordered, contiguous TrendDataPoint entries.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            # Establish default time boundaries if not provided
            now = datetime.now(timezone.utc)
            effective_end = end_date if end_date is not None else now
            effective_start = start_date if start_date is not None else (effective_end - timedelta(days=7))

            # Auto-detect interval if not specified: <= 48h -> hourly, > 48h -> daily
            time_delta = effective_end - effective_start
            if interval is None:
                effective_interval = TrendInterval.HOURLY if time_delta.total_seconds() <= 48 * 3600 else TrendInterval.DAILY
            else:
                effective_interval = interval

            date_trunc_unit = "hour" if effective_interval == TrendInterval.HOURLY else "day"
            bucket_col = func.timezone(
                "UTC",
                func.date_trunc(date_trunc_unit, func.timezone("UTC", RiskEvaluation.evaluated_at)),
            ).label("bucket_time")

            stmt = (
                select(
                    bucket_col,
                    func.count(Transaction.id).label("total_count"),
                    func.coalesce(func.sum(Transaction.amount), 0).label("total_amount"),
                    func.coalesce(func.avg(RiskEvaluation.risk_score), 0.0).label("avg_risk_score"),
                    func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.APPROVE), 0).label("approval_count"),
                    func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.REVIEW), 0).label("review_count"),
                    func.coalesce(func.count().filter(RiskEvaluation.decision_action == DecisionAction.BLOCK), 0).label("block_count"),
                    func.coalesce(func.count().filter(RiskEvaluation.risk_tier.in_([RiskTier.HIGH, RiskTier.CRITICAL])), 0).label("high_critical_count"),
                )
                .select_from(RiskEvaluation)
                .join(Transaction, Transaction.id == RiskEvaluation.transaction_id)
                .where(
                    and_(
                        RiskEvaluation.evaluated_at >= effective_start,
                        RiskEvaluation.evaluated_at <= effective_end,
                    )
                )
                .group_by(bucket_col)
                .order_by(bucket_col.asc())
            )

            result = await self._session.execute(stmt)
            rows = result.mappings().all()

            # Index retrieved rows by datetime key (normalized to UTC)
            data_by_bucket: dict = {}
            for r in rows:
                b_time: datetime = r["bucket_time"]
                if b_time.tzinfo is None:
                    b_time = b_time.replace(tzinfo=timezone.utc)
                data_by_bucket[b_time] = r

            # Generate contiguous sequence of time buckets from start_date to end_date
            data_points: List[TrendDataPoint] = []
            if effective_interval == TrendInterval.HOURLY:
                # Truncate start to the hour
                cur_dt = effective_start.replace(minute=0, second=0, microsecond=0)
                step = timedelta(hours=1)
            else:
                # Truncate start to the day
                cur_dt = effective_start.replace(hour=0, minute=0, second=0, microsecond=0)
                step = timedelta(days=1)

            # Cap max generated points to prevent memory overflow on extreme spans
            max_points = 500
            points_generated = 0

            while cur_dt <= effective_end and points_generated < max_points:
                row_data = data_by_bucket.get(cur_dt)
                if row_data:
                    data_points.append(
                        TrendDataPoint(
                            timestamp=cur_dt,
                            total_count=int(row_data["total_count"]),
                            total_amount=round(float(row_data["total_amount"]), 2),
                            average_risk_score=round(float(row_data["avg_risk_score"]), 2),
                            approval_count=int(row_data["approval_count"]),
                            review_count=int(row_data["review_count"]),
                            block_count=int(row_data["block_count"]),
                            high_critical_count=int(row_data["high_critical_count"]),
                        )
                    )
                else:
                    # Contiguous zero-fill for periods with zero transaction volume
                    data_points.append(
                        TrendDataPoint(
                            timestamp=cur_dt,
                            total_count=0,
                            total_amount=0.0,
                            average_risk_score=0.0,
                            approval_count=0,
                            review_count=0,
                            block_count=0,
                            high_critical_count=0,
                        )
                    )
                cur_dt += step
                points_generated += 1

            return AnalyticsTrendsResponse(
                interval=effective_interval,
                start_date=effective_start,
                end_date=effective_end,
                data_points=data_points,
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute analytics trends query: {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve analytics trends: {exc}"
            ) from exc

    async def get_analytics_rules(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 20,
    ) -> AnalyticsRulesResponse:
        """
        Aggregate triggered business rules, distinct affected transactions, override impact, and outcome distributions.

        Args:
            start_date: Optional earliest evaluated timestamp boundary.
            end_date: Optional latest evaluated timestamp boundary.
            limit: Maximum number of ranked rule items to return (1 <= limit <= 100).

        Returns:
            AnalyticsRulesResponse with ranked rule items ordered by trigger_count DESC.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        safe_limit = max(1, min(limit, 100))

        try:
            # 1. Count total distinct risk evaluations processed in the period
            eval_filters = []
            if start_date is not None:
                eval_filters.append(RiskEvaluation.evaluated_at >= start_date)
            if end_date is not None:
                eval_filters.append(RiskEvaluation.evaluated_at <= end_date)

            total_evals_stmt = select(func.count(RiskEvaluation.id)).select_from(RiskEvaluation)
            if eval_filters:
                total_evals_stmt = total_evals_stmt.where(and_(*eval_filters))

            total_evals_res = await self._session.execute(total_evals_stmt)
            total_evaluations = total_evals_res.scalar() or 0

            # 2. Aggregate rule triggers, distinct affected evaluations, and override counts
            rule_filters = []
            if start_date is not None:
                rule_filters.append(RiskEvaluation.evaluated_at >= start_date)
            if end_date is not None:
                rule_filters.append(RiskEvaluation.evaluated_at <= end_date)

            stmt = (
                select(
                    EvaluationRuleMatch.rule_id,
                    EvaluationRuleMatch.description,
                    EvaluationRuleMatch.rule_type,
                    EvaluationRuleMatch.priority,
                    func.count(EvaluationRuleMatch.id).label("trigger_count"),
                    func.count(func.distinct(EvaluationRuleMatch.evaluation_id)).label("affected_transactions"),
                    func.count(
                        func.distinct(
                            case((RiskEvaluation.is_overridden == True, RiskEvaluation.id), else_=None)
                        )
                    ).label("override_count"),
                    func.coalesce(func.count().filter(EvaluationRuleMatch.outcome == RuleOutcome.BLOCK), 0).label("cnt_block"),
                    func.coalesce(func.count().filter(EvaluationRuleMatch.outcome == RuleOutcome.REVIEW), 0).label("cnt_review"),
                    func.coalesce(func.count().filter(EvaluationRuleMatch.outcome == RuleOutcome.MONITOR), 0).label("cnt_monitor"),
                )
                .select_from(EvaluationRuleMatch)
                .join(RiskEvaluation, RiskEvaluation.id == EvaluationRuleMatch.evaluation_id)
            )

            if rule_filters:
                stmt = stmt.where(and_(*rule_filters))

            stmt = (
                stmt.group_by(
                    EvaluationRuleMatch.rule_id,
                    EvaluationRuleMatch.description,
                    EvaluationRuleMatch.rule_type,
                    EvaluationRuleMatch.priority,
                )
                .order_by(
                    func.count(EvaluationRuleMatch.id).desc(),
                    EvaluationRuleMatch.priority.asc(),
                    EvaluationRuleMatch.rule_id.asc(),
                )
                .limit(safe_limit)
            )

            result = await self._session.execute(stmt)
            rows = result.mappings().all()

            rules: List[RuleAnalyticsItem] = []
            for r in rows:
                trig_cnt = int(r["trigger_count"])
                aff_tx = int(r["affected_transactions"])
                ovr_cnt = int(r["override_count"])
                trig_rate = round((aff_tx / total_evaluations) * 100.0, 2) if total_evaluations > 0 else 0.0

                cnt_blk = int(r["cnt_block"])
                cnt_rev = int(r["cnt_review"])
                cnt_mon = int(r["cnt_monitor"])

                outcomes: List[RuleOutcomeBreakdown] = []
                for out_type, out_cnt in [
                    (RuleOutcome.BLOCK, cnt_blk),
                    (RuleOutcome.REVIEW, cnt_rev),
                    (RuleOutcome.MONITOR, cnt_mon),
                ]:
                    if out_cnt > 0:
                        out_pct = round((out_cnt / trig_cnt) * 100.0, 2) if trig_cnt > 0 else 0.0
                        outcomes.append(
                            RuleOutcomeBreakdown(
                                outcome=out_type,
                                count=out_cnt,
                                percentage=out_pct,
                            )
                        )

                rules.append(
                    RuleAnalyticsItem(
                        rule_id=str(r["rule_id"]),
                        description=str(r["description"]),
                        rule_type=r["rule_type"],
                        priority=int(r["priority"]),
                        trigger_count=trig_cnt,
                        affected_transactions=aff_tx,
                        trigger_rate=trig_rate,
                        override_count=ovr_cnt,
                        outcomes=outcomes,
                    )
                )

            return AnalyticsRulesResponse(
                total_rules_active=len(rules),
                total_evaluations_analyzed=total_evaluations,
                rules=rules,
            )

        except SQLAlchemyError as exc:
            logger.error(f"Failed to execute analytics rules query: {exc}", exc_info=True)
            raise PersistenceError(
                f"Failed to retrieve analytics rules: {exc}"
            ) from exc

