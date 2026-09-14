"""
Reason Code Synthesis and Plain-English Explanation Generator.

Combines deterministic Phase 6 business rule triggers (source="RULE") with local
TreeSHAP feature attributions (source="MODEL") into prioritized, plain-English reason
codes for auditability, investigator triage, and case management.

Governance Principles:
- Rule-based reason codes originate strictly from deterministic RuleEngine evaluation.
- Model-based reason codes originate strictly from top local TreeSHAP positive attributions.
- Raw unencoded feature values are preserved and formatted with human-readable domain units.
- Categorical features are never formatted as ordinal integer codes.
"""

from typing import Dict, Any, List, Optional, Union, Mapping, Tuple, Sequence
import numpy as np

from ml.risk_engine.config import (
    RuleOutcome,
    DecisionAction,
)
from ml.risk_engine.rules import (
    RuleMatch,
)
from ml.explainability.schemas import (
    ReasonSource,
    ReasonSeverity,
    ReasonCodeDetail,
    FeatureAttribution,
    AttributionDirection,
)
from ml.explainability.config import (
    get_feature_metadata,
)


class ReasonCodeGenerator:
    """
    Synthesizes and formats standardized human-readable reason codes combining
    deterministic business rules and model feature attributions.
    """

    @staticmethod
    def _format_rule_reason(
        match: RuleMatch,
        rank: int,
        is_override: bool = False,
    ) -> ReasonCodeDetail:
        """
        Format a RuleMatch instance into a structured ReasonCodeDetail.

        Args:
            match: Triggered RuleMatch from Phase 6 RuleEngine.
            rank: Assigned presentation rank.
            is_override: Whether this rule overrode the baseline ML policy action.

        Returns:
            ReasonCodeDetail: Formatted rule reason code with source="RULE".
        """
        outcome = match.outcome
        feat_val = match.feature_value
        comp_val = match.comparison_value

        # Severity mapping based on rule outcome and override state
        if outcome == RuleOutcome.BLOCK:
            severity = ReasonSeverity.CRITICAL
        elif outcome == RuleOutcome.REVIEW:
            severity = ReasonSeverity.HIGH if is_override else ReasonSeverity.MEDIUM
        elif outcome == RuleOutcome.MONITOR:
            severity = ReasonSeverity.INFO
        else:
            severity = ReasonSeverity.INFO

        # Format human-readable description tailored to the specific rule ID
        rule_id = match.rule_id
        if rule_id == "RULE_VELOCITY_BURST_REVIEW":
            headline = "Rapid 1-Hour Velocity Burst (Policy Rule)"
            desc = (
                f"Triggered business rule: 1-hour transaction count ({feat_val:.0f}) exceeds "
                f"the review threshold of {comp_val:.0f} transactions."
            )
        elif rule_id == "RULE_AMT_ZSCORE_DEVIATION_REVIEW":
            headline = "Extreme Spending Deviation (Policy Rule)"
            desc = (
                f"Triggered business rule: Spending Z-score deviation ({feat_val:+.2f}σ) exceeds "
                f"the threshold of {comp_val:.2f}σ from cardholder baseline."
            )
        elif rule_id == "RULE_AMT_EXTREME_MONITOR":
            headline = "Large Transaction Amount Monitor"
            desc = (
                f"Triggered monitor rule: Transaction amount (${feat_val:,.2f}) exceeds "
                f"monitoring threshold of ${comp_val:,.2f}."
            )
        elif rule_id == "RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR":
            headline = "Audit Threshold Monitor"
            desc = (
                f"Triggered audit monitor: Transaction amount (${feat_val:,.2f}) exceeds "
                f"standard tracking threshold of ${comp_val:,.2f}."
            )
        elif rule_id == "RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR":
            headline = "Impossible Travel Speed Monitor"
            desc = (
                "Triggered monitor rule: Impossible travel speed flag detected between sequential locations."
            )
        elif rule_id == "RULE_GEO_EXTREME_DISTANCE_MONITOR":
            headline = "Upper-Tail Geographic Distance Monitor"
            desc = (
                f"Triggered monitor rule: Cardholder-to-merchant distance ({feat_val:,.1f} km) "
                f"exceeds upper-tail threshold of {comp_val:,.1f} km."
            )
        else:
            headline = f"Business Rule Triggered ({rule_id})"
            desc = match.description

        if is_override:
            desc += f" [Decision escalated to {outcome.value} by policy rule]."

        return ReasonCodeDetail(
            code=match.rule_id,
            headline=headline,
            description=desc,
            category=match.rule_type.value,
            source=ReasonSource.RULE,
            severity=severity,
            rank=rank,
        )

    @staticmethod
    def _format_model_reason(
        attribution: FeatureAttribution,
        rank: int,
    ) -> ReasonCodeDetail:
        """
        Format a model FeatureAttribution into a structured ReasonCodeDetail.

        Args:
            attribution: Positive (risk-increasing) FeatureAttribution instance.
            rank: Assigned presentation rank.

        Returns:
            ReasonCodeDetail: Formatted model reason code with source="MODEL".
        """
        meta = get_feature_metadata(attribution.feature_name)
        raw_val = attribution.raw_value

        # Determine severity based on relative contribution and factor rank
        if attribution.relative_contribution_pct >= 30.0 or attribution.rank == 1:
            severity = ReasonSeverity.HIGH
        elif attribution.relative_contribution_pct >= 15.0:
            severity = ReasonSeverity.MEDIUM
        else:
            severity = ReasonSeverity.LOW

        # Interpolate description template safely
        try:
            if meta["is_categorical"]:
                # Categorical features are strings (e.g. 'shopping_net', 'gas_transport')
                desc = meta["risk_template"].format(raw_value=str(raw_val))
            elif isinstance(raw_val, (int, float, np.number)) and not isinstance(raw_val, bool):
                desc = meta["risk_template"].format(raw_value=float(raw_val))
            elif isinstance(raw_val, bool):
                desc = meta["risk_template"].format(raw_value=raw_val)
            else:
                desc = meta["risk_template"].format(raw_value=str(raw_val))
        except Exception:
            desc = f"{meta['display_name']} ({raw_val}) increases statistical fraud risk (attribution: {attribution.shap_value:+.4f})."

        return ReasonCodeDetail(
            code=meta["reason_code"],
            headline=meta["risk_headline"],
            description=desc,
            category=meta["category"],
            source=ReasonSource.MODEL,
            severity=severity,
            rank=rank,
        )

    @classmethod
    def generate_reason_codes(
        cls,
        top_risk_factors: Sequence[FeatureAttribution],
        rule_matches: Sequence[RuleMatch] = (),
        is_overridden: bool = False,
        rule_action: Optional[RuleOutcome] = None,
        max_reasons: int = 5,
    ) -> Tuple[ReasonCodeDetail, ...]:
        """
        Synthesize combined reason codes prioritizing deterministic rules and top model drivers.

        Ordering Hierarchy:
        1. Overriding Business Rules (e.g. escalated to REVIEW).
        2. Non-overriding Actionable Rules (e.g. REVIEW or BLOCK matches on already REVIEW/BLOCK decisions).
        3. Top Model Risk Drivers (source="MODEL", sorted by attribution magnitude descending).
        4. Passive Monitoring Rules (source="RULE", outcome=MONITOR).

        Args:
            top_risk_factors: Sequence of top risk-increasing FeatureAttribution objects.
            rule_matches: Sequence of triggered RuleMatch objects from Phase 6 RuleEngine.
            is_overridden: Whether a rule overrode the baseline ML policy.
            rule_action: Rule outcome enacted if an override occurred.
            max_reasons: Maximum total reason codes to return.

        Returns:
            Tuple[ReasonCodeDetail, ...]: Ranked, structured reason codes.
        """
        reason_list: List[ReasonCodeDetail] = []
        current_rank = 1

        # Partition rule matches into actionable vs monitor
        overriding_rules: List[RuleMatch] = []
        actionable_rules: List[RuleMatch] = []
        monitor_rules: List[RuleMatch] = []

        for match in rule_matches:
            if is_overridden and match.outcome == rule_action:
                overriding_rules.append(match)
            elif match.outcome in (RuleOutcome.BLOCK, RuleOutcome.REVIEW):
                actionable_rules.append(match)
            elif match.outcome == RuleOutcome.MONITOR:
                monitor_rules.append(match)

        # 1. Overriding rules
        for match in overriding_rules:
            if len(reason_list) >= max_reasons:
                break
            reason_list.append(cls._format_rule_reason(match, rank=current_rank, is_override=True))
            current_rank += 1

        # 2. Actionable non-overriding rules
        for match in actionable_rules:
            if len(reason_list) >= max_reasons:
                break
            reason_list.append(cls._format_rule_reason(match, rank=current_rank, is_override=False))
            current_rank += 1

        # 3. Top Model Risk Drivers
        for factor in top_risk_factors:
            if len(reason_list) >= max_reasons:
                break
            reason_list.append(cls._format_model_reason(factor, rank=current_rank))
            current_rank += 1

        # 4. Passive Monitoring Rules (if space remains)
        for match in monitor_rules:
            if len(reason_list) >= max_reasons:
                break
            reason_list.append(cls._format_rule_reason(match, rank=current_rank, is_override=False))
            current_rank += 1

        return tuple(reason_list)
