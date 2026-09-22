"""
Ground-Truth Performance Monitoring Engine for Phase 13 ML & Model Monitoring.

Evaluates:
- Pure ML model performance at cost-optimal threshold tau* = 0.78 (and comparison tau = 0.50).
- Hybrid operational decision performance (BLOCK precision and intervention recall).
- Operational human review queue purity (CONFIRMED_FRAUD proportion among resolved REVIEW cases).
- FPR, PR-AUC, and ROC-AUC metrics.
- Mathematical degradation tracking against frozen v1.0.0 OOT benchmarks.
- 3-tier sample size semantics (<20 = INSUFFICIENT_DATA, 20-99 = LOW_SAMPLE, >=100 = NORMAL_CONFIDENCE).
- Strict relational join: Case -> evaluation_id -> RiskEvaluation -> transaction_id.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.schemas import (
    ConfusionMatrixData,
    MetricDegradationResult,
    OperationalDecisionMetrics,
    PerformanceBaselineProfile,
    PerformanceReport,
    ThresholdPerformanceMetrics,
)


class ModelPerformanceCalculator:
    """
    Evaluates pure model discrimination, operational hybrid decisions, and human review queue purity
    against delayed authoritative case dispositions.
    """

    def __init__(
        self,
        baseline_profile: Optional[PerformanceBaselineProfile] = None,
        config: Optional[MonitoringConfig] = None,
    ) -> None:
        self.config = config or default_monitoring_config
        if baseline_profile is not None:
            self.baseline_profile = baseline_profile
        else:
            self.baseline_profile = PerformanceBaselineProfile.load(self.config.performance_profile_path)

    def extract_ground_truth_labels(
        self, cases: Sequence[Any]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Extract ground truth records joining Case -> evaluation_id -> RiskEvaluation.

        Inclusion & Label Rules:
        - Only resolved cases with disposition IS NOT NULL are included.
        - CONFIRMED_FRAUD -> Positive Ground Truth (y = 1).
        - FALSE_POSITIVE, LEGITIMATE -> Negative Ground Truth (y = 0).
        - SUSPICIOUS_RESOLVED -> Excluded from binary classification (counted in throughput).
        - Unresolved cases (OPEN, IN_REVIEW, ESCALATED) where disposition is None -> Excluded.

        Returns:
            Tuple of (list_of_labeled_records, suspicious_resolved_count).
        """
        labeled_records: List[Dict[str, Any]] = []
        suspicious_resolved_count = 0

        for c in cases:
            # Extract disposition string
            disp = None
            if hasattr(c, "disposition"):
                raw_disp = getattr(c, "disposition")
                disp = getattr(raw_disp, "value", str(raw_disp)) if raw_disp is not None else None
            elif isinstance(c, dict):
                disp = c.get("disposition")

            if not disp or disp == "None":
                # Unresolved case
                continue

            disp_upper = str(disp).upper().strip()

            if disp_upper == "SUSPICIOUS_RESOLVED":
                suspicious_resolved_count += 1
                continue

            if disp_upper == "CONFIRMED_FRAUD":
                y_val = 1
            elif disp_upper in ("FALSE_POSITIVE", "LEGITIMATE"):
                y_val = 0
            else:
                # Unknown disposition
                continue

            # Extract linked RiskEvaluation metrics
            model_score = None
            decision_action = None
            evaluation_id = None
            transaction_id = None

            if hasattr(c, "evaluation") and getattr(c, "evaluation") is not None:
                eval_obj = getattr(c, "evaluation")
                model_score = float(getattr(eval_obj, "model_score", 0.0))
                raw_act = getattr(eval_obj, "decision_action", "APPROVE")
                decision_action = getattr(raw_act, "value", str(raw_act)).upper().strip()
                evaluation_id = str(getattr(eval_obj, "id", getattr(c, "evaluation_id", "")))
                transaction_id = str(getattr(eval_obj, "transaction_id", getattr(c, "transaction_id", "")))
            elif hasattr(c, "model_score"):
                model_score = float(getattr(c, "model_score", 0.0))
                raw_act = getattr(c, "decision_action", getattr(c, "action", "APPROVE"))
                decision_action = getattr(raw_act, "value", str(raw_act)).upper().strip()
                evaluation_id = str(getattr(c, "evaluation_id", ""))
                transaction_id = str(getattr(c, "transaction_id", ""))
            elif isinstance(c, dict):
                model_score = float(c.get("model_score", 0.0))
                raw_act = c.get("decision_action", c.get("action", "APPROVE"))
                decision_action = getattr(raw_act, "value", str(raw_act)).upper().strip()
                evaluation_id = str(c.get("evaluation_id", ""))
                transaction_id = str(c.get("transaction_id", ""))

            if model_score is not None:
                labeled_records.append({
                    "y_true": y_val,
                    "model_score": model_score,
                    "decision_action": decision_action or "APPROVE",
                    "evaluation_id": evaluation_id,
                    "transaction_id": transaction_id,
                    "disposition": disp_upper,
                })

        return labeled_records, suspicious_resolved_count

    def compute_threshold_metrics(
        self, y_true: np.ndarray, model_scores: np.ndarray, threshold: float
    ) -> ThresholdPerformanceMetrics:
        """
        Compute binary classification performance metrics at a specific operating threshold.
        """
        y_pred = (model_scores >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        total = int(len(y_true))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = float((tp + tn) / total) if total > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tpr = recall

        cm = ConfusionMatrixData(tp=tp, fp=fp, fn=fn, tn=tn, total=total)
        return ThresholdPerformanceMetrics(
            threshold=float(threshold),
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            fpr=fpr,
            tpr=tpr,
            confusion_matrix=cm,
        )

    def compute_operational_metrics(
        self, y_true: np.ndarray, actions: Sequence[str]
    ) -> OperationalDecisionMetrics:
        """
        Compute operational hybrid decision metrics and human review queue purity.
        """
        clean_actions = [str(a).upper().strip() for a in actions]

        total_blocks = int(sum(1 for a in clean_actions if a == "BLOCK"))
        fraud_in_block = int(sum(1 for y, a in zip(y_true, clean_actions) if a == "BLOCK" and y == 1))

        total_reviews = int(sum(1 for a in clean_actions if a == "REVIEW"))
        fraud_in_review = int(sum(1 for y, a in zip(y_true, clean_actions) if a == "REVIEW" and y == 1))

        total_frauds = int(np.sum(y_true == 1))
        fraud_in_intervention = fraud_in_block + fraud_in_review

        decision_precision_block = float(fraud_in_block / total_blocks) if total_blocks > 0 else 0.0
        decision_recall_intervention = float(fraud_in_intervention / total_frauds) if total_frauds > 0 else 0.0
        review_queue_purity = float(fraud_in_review / total_reviews) if total_reviews > 0 else 0.0

        return OperationalDecisionMetrics(
            decision_precision_block=decision_precision_block,
            decision_recall_intervention=decision_recall_intervention,
            review_queue_purity=review_queue_purity,
            total_reviews_count=total_reviews,
            fraud_in_review_count=fraud_in_review,
            total_blocks_count=total_blocks,
            fraud_in_block_count=fraud_in_block,
        )

    def evaluate_degradation(
        self,
        observed: float,
        baseline: float,
        metric_name: str,
        is_higher_better: bool = True,
        confidence: MetricConfidence = MetricConfidence.NORMAL_CONFIDENCE,
    ) -> MetricDegradationResult:
        """
        Calculate relative degradation and assign alert severity.
        """
        if confidence == MetricConfidence.INSUFFICIENT_DATA:
            return MetricDegradationResult(
                metric_name=metric_name,
                observed_value=float(observed),
                baseline_value=float(baseline),
                relative_delta=0.0,
                absolute_delta=float(abs(observed - baseline)),
                severity=DriftSeverity.INSUFFICIENT_DATA,
                confidence=confidence,
                alert_message=f"Insufficient sample size for {metric_name} degradation evaluation.",
            )

        if is_higher_better:
            rel_delta = float((baseline - observed) / max(baseline, 1e-4))
            is_critical = rel_delta > self.config.higher_is_better_rel_critical
            is_warning = rel_delta > self.config.higher_is_better_rel_warning
        else:
            # Lower is better (e.g. FPR)
            rel_delta = float((observed - baseline) / max(baseline, 1e-4))
            is_critical = (
                rel_delta > self.config.lower_is_better_rel_critical
                or observed > self.config.lower_is_better_abs_fpr_critical
            )
            is_warning = (
                rel_delta > self.config.lower_is_better_rel_warning
                or observed > self.config.lower_is_better_abs_fpr_warning
            )

        abs_delta = float(abs(observed - baseline))

        # Apply sample-size confidence guardrails:
        # LOW_SAMPLE (20-99) suppresses strong CRITICAL alerts down to WARNING/informational
        if confidence == MetricConfidence.LOW_SAMPLE:
            if is_critical or is_warning:
                severity = DriftSeverity.WARNING
                alert_msg = f"Low sample degradation warning on {metric_name}: observed={observed:.4f}, baseline={baseline:.4f} (relative delta={rel_delta:.2%})"
            else:
                severity = DriftSeverity.NORMAL
                alert_msg = None
        else:
            if is_critical:
                severity = DriftSeverity.CRITICAL
                alert_msg = f"Critical performance degradation on {metric_name}: observed={observed:.4f}, baseline={baseline:.4f} (relative delta={rel_delta:.2%})"
            elif is_warning:
                severity = DriftSeverity.WARNING
                alert_msg = f"Performance degradation warning on {metric_name}: observed={observed:.4f}, baseline={baseline:.4f} (relative delta={rel_delta:.2%})"
            else:
                severity = DriftSeverity.NORMAL
                alert_msg = None

        return MetricDegradationResult(
            metric_name=metric_name,
            observed_value=float(observed),
            baseline_value=float(baseline),
            relative_delta=rel_delta,
            absolute_delta=abs_delta,
            severity=severity,
            confidence=confidence,
            alert_message=alert_msg,
        )

    def compute_performance_from_labels(
        self,
        y_true: Sequence[int],
        model_scores: Sequence[float],
        actions: Sequence[str],
        total_raw_count: int,
        suspicious_count: int = 0,
        operating_threshold: Optional[float] = None,
        window_type: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> PerformanceReport:
        """
        Evaluate ground truth performance metrics, operational metrics, and degradation results.
        """
        n_labeled = len(y_true)
        threshold = operating_threshold or self.config.default_operating_threshold

        # 1. Determine Sample Size Confidence Grade
        if n_labeled < self.config.min_performance_samples_insufficient:
            confidence = MetricConfidence.INSUFFICIENT_DATA
        elif n_labeled < self.config.min_performance_samples_low_confidence:
            confidence = MetricConfidence.LOW_SAMPLE
        else:
            confidence = MetricConfidence.NORMAL_CONFIDENCE

        y_arr = np.asarray(y_true, dtype=int) if n_labeled > 0 else np.array([], dtype=int)
        scores_arr = np.asarray(model_scores, dtype=np.float64) if n_labeled > 0 else np.array([], dtype=np.float64)

        fraud_count = int(np.sum(y_arr == 1)) if n_labeled > 0 else 0
        legit_count = int(np.sum(y_arr == 0)) if n_labeled > 0 else 0

        # 2. Compute Threshold Metrics (Primary operating threshold 0.78, optional comparison 0.50)
        primary_metrics = self.compute_threshold_metrics(y_arr, scores_arr, threshold=threshold)
        comp_metrics = self.compute_threshold_metrics(y_arr, scores_arr, threshold=self.config.comparison_threshold)

        # 3. Compute Operational Decision & Review Queue Metrics
        ops_metrics = self.compute_operational_metrics(y_arr, actions)

        # 4. Compute PR-AUC & ROC-AUC if both classes present
        pr_auc_val: Optional[float] = None
        roc_auc_val: Optional[float] = None
        if n_labeled >= self.config.min_performance_samples_insufficient and fraud_count > 0 and legit_count > 0:
            try:
                roc_auc_val = float(roc_auc_score(y_arr, scores_arr))
                prec_c, rec_c, _ = precision_recall_curve(y_arr, scores_arr)
                pr_auc_val = float(auc(rec_c, prec_c))
            except Exception:
                pr_auc_val = None
                roc_auc_val = None

        # 5. Evaluate Degradations against baseline benchmarks
        degradation_results: Dict[str, MetricDegradationResult] = {}
        base_prim = self.baseline_profile.primary_operating_metrics
        base_ops = self.baseline_profile.operational_decision_metrics

        degradation_results["model_precision"] = self.evaluate_degradation(
            primary_metrics.precision, base_prim.precision, "model_precision", is_higher_better=True, confidence=confidence
        )
        degradation_results["model_recall"] = self.evaluate_degradation(
            primary_metrics.recall, base_prim.recall, "model_recall", is_higher_better=True, confidence=confidence
        )
        degradation_results["model_f1"] = self.evaluate_degradation(
            primary_metrics.f1, base_prim.f1, "model_f1", is_higher_better=True, confidence=confidence
        )
        degradation_results["model_fpr"] = self.evaluate_degradation(
            primary_metrics.fpr, base_prim.fpr, "model_fpr", is_higher_better=False, confidence=confidence
        )
        degradation_results["decision_precision_block"] = self.evaluate_degradation(
            ops_metrics.decision_precision_block, base_ops.decision_precision_block, "decision_precision_block", is_higher_better=True, confidence=confidence
        )
        degradation_results["decision_recall_intervention"] = self.evaluate_degradation(
            ops_metrics.decision_recall_intervention, base_ops.decision_recall_intervention, "decision_recall_intervention", is_higher_better=True, confidence=confidence
        )
        degradation_results["review_queue_purity"] = self.evaluate_degradation(
            ops_metrics.review_queue_purity, base_ops.review_queue_purity, "review_queue_purity", is_higher_better=True, confidence=confidence
        )

        if pr_auc_val is not None:
            degradation_results["pr_auc"] = self.evaluate_degradation(
                pr_auc_val, self.baseline_profile.pr_auc, "pr_auc", is_higher_better=True, confidence=confidence
            )
        if roc_auc_val is not None:
            degradation_results["roc_auc"] = self.evaluate_degradation(
                roc_auc_val, self.baseline_profile.roc_auc, "roc_auc", is_higher_better=True, confidence=confidence
            )

        active_alerts = [r for r in degradation_results.values() if r.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)]

        # 6. Overall Performance Status Precedence
        if confidence == MetricConfidence.INSUFFICIENT_DATA:
            overall_status = DriftSeverity.INSUFFICIENT_DATA
        elif any(r.severity == DriftSeverity.CRITICAL for r in degradation_results.values()):
            overall_status = DriftSeverity.CRITICAL
        elif any(r.severity == DriftSeverity.WARNING for r in degradation_results.values()):
            overall_status = DriftSeverity.WARNING
        else:
            overall_status = DriftSeverity.NORMAL

        created_at_iso = datetime.now(timezone.utc).isoformat()

        return PerformanceReport(
            model_version=self.baseline_profile.model_version,
            dataset_row_count=total_raw_count,
            labeled_sample_count=n_labeled,
            fraud_cases_count=fraud_count,
            legitimate_cases_count=legit_count,
            suspicious_resolved_count=suspicious_count,
            created_at=created_at_iso,
            overall_performance_status=overall_status,
            confidence=confidence,
            operating_threshold=float(threshold),
            primary_metrics=primary_metrics,
            comparison_metrics=comp_metrics,
            pr_auc=pr_auc_val,
            roc_auc=roc_auc_val,
            operational_metrics=ops_metrics,
            degradation_results=degradation_results,
            active_degradation_alerts=active_alerts,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
        )

    def compute_performance_report(
        self,
        cases: Sequence[Any],
        operating_threshold: Optional[float] = None,
        window_type: Optional[str] = None,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> PerformanceReport:
        """
        Evaluate performance metrics and degradations from a list of Case entities or dict records.
        """
        total_raw = len(cases)
        labeled_records, suspicious_count = self.extract_ground_truth_labels(cases)

        y_true = [r["y_true"] for r in labeled_records]
        model_scores = [r["model_score"] for r in labeled_records]
        actions = [r["decision_action"] for r in labeled_records]

        return self.compute_performance_from_labels(
            y_true=y_true,
            model_scores=model_scores,
            actions=actions,
            total_raw_count=total_raw,
            suspicious_count=suspicious_count,
            operating_threshold=operating_threshold,
            window_type=window_type,
            window_start=window_start,
            window_end=window_end,
        )
