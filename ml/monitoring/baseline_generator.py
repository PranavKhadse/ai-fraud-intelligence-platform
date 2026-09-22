"""
Deterministic Baseline Profile Generator for Phase 13 ML & Model Monitoring.

Generates:
1. baseline_feature_profile_v1.0.0.json (from train_features.parquet):
   - Precomputes 10 quantile bin edges (deciles) for PSI across all 1,296,675 training rows.
   - Stores deterministic reference samples (M = 5,000 points per numeric feature, seed 42) for 2-sample KS test.
   - Stores discrete category probabilities and known vocabularies for categoricals.
2. baseline_prediction_profile_v1.0.0.json (from val_features.parquet):
   - Executes frozen RiskEvaluator and standard 6-rule catalog strictly in-memory (zero persistence).
   - Precomputes reference 10-bin model score histograms, 10-bucket risk score distributions,
     tier proportions, and action proportions.
3. baseline_performance_profile_v1.0.0.json (from test_features.parquet):
   - Evaluates frozen RiskEvaluator at primary operating threshold tau* = 0.78 and comparison tau = 0.50.
   - Precomputes model classification benchmarks, FPR, hybrid decision precision/recall, and review queue purity.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ml.evaluation.metrics import compute_classification_metrics
from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    TARGET_COLUMN,
)
from ml.risk_engine.catalog import get_standard_rule_catalog
from ml.risk_engine.config import DecisionAction, RiskTier
from ml.risk_engine.evaluator import RiskEvaluator
from ml.risk_engine.rules import RuleEngine
from ml.monitoring.config import MonitoringConfig, default_monitoring_config
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    ConfusionMatrixData,
    FeatureBaselineProfile,
    NumericalFeatureProfile,
    OperationalDecisionMetrics,
    PerformanceBaselineProfile,
    PredictionBaselineProfile,
    RiskScoreBucket,
    ScoreHistogramBin,
    ThresholdPerformanceMetrics,
)

logger = logging.getLogger("ml.monitoring.baseline_generator")


def generate_feature_baseline_profile(
    config: Optional[MonitoringConfig] = None,
    train_df: Optional[pd.DataFrame] = None,
) -> FeatureBaselineProfile:
    """
    Generate deterministic baseline feature profile for all 55 predictive columns.

    Args:
        config: Monitoring configuration parameters (paths, seed, sample sizes).
        train_df: Optional pre-loaded training DataFrame. If None, loaded from config.train_dataset_path.

    Returns:
        FeatureBaselineProfile: Fully populated and validated profile object.
    """
    cfg = config or default_monitoring_config
    if train_df is None:
        logger.info("Loading training features from %s", cfg.train_dataset_path)
        train_df = pd.read_parquet(cfg.train_dataset_path)

    row_count = len(train_df)
    features_dict: Dict[str, Any] = {}

    rng = np.random.RandomState(cfg.random_seed)

    for col in PREDICTIVE_FEATURE_COLUMNS:
        assert col in train_df.columns, f"Required predictive column '{col}' missing from train dataset!"

        if col in CATEGORICAL_PREDICTORS:
            # Categorical feature profiling
            series = train_df[col].astype(str)
            val_counts = series.value_counts(normalize=True)
            vocab = sorted(series.unique().tolist())
            probabilities = {k: float(round(v, 6)) for k, v in val_counts.items()}
            missing_count = int(train_df[col].isna().sum())

            features_dict[col] = CategoricalFeatureProfile(
                feature_name=col,
                data_type="categorical",
                count=row_count,
                missing_count=missing_count,
                missing_rate=float(round(missing_count / row_count, 6)),
                vocabulary=vocab,
                probabilities=probabilities,
            )
        else:
            # Numerical feature profiling
            vals = train_df[col].to_numpy(dtype=np.float64)
            missing_count = int(np.isnan(vals).sum())
            non_null_vals = vals[~np.isnan(vals)]

            mean_val = float(np.mean(non_null_vals))
            std_val = float(np.std(non_null_vals))
            min_val = float(np.min(non_null_vals))
            max_val = float(np.max(non_null_vals))

            # Precompute 10 quantile bin edges (deciles: 0%, 10%, ..., 100%)
            quantile_levels = np.linspace(0.0, 1.0, cfg.num_deciles + 1)
            bin_edges = [float(round(q, 6)) for q in np.quantile(non_null_vals, quantile_levels)]

            # Extract deterministic stratified/uniform sample of size M = reference_sample_size
            sample_size = min(cfg.reference_sample_size, len(non_null_vals))
            sample_indices = rng.choice(len(non_null_vals), size=sample_size, replace=False)
            sample_indices.sort()
            ref_samples = [float(round(x, 6)) for x in non_null_vals[sample_indices]]

            features_dict[col] = NumericalFeatureProfile(
                feature_name=col,
                data_type="numerical",
                count=row_count,
                missing_count=missing_count,
                missing_rate=float(round(missing_count / row_count, 6)),
                mean=float(round(mean_val, 6)),
                std=float(round(std_val, 6)),
                min=float(round(min_val, 6)),
                max=float(round(max_val, 6)),
                bin_edges=bin_edges,
                ref_samples=ref_samples,
            )

    profile = FeatureBaselineProfile(
        model_version=cfg.model_version,
        dataset_name=str(cfg.train_dataset_path.name),
        dataset_row_count=row_count,
        created_at=datetime.now(timezone.utc).isoformat(),
        random_seed=cfg.random_seed,
        num_features=len(features_dict),
        features=features_dict,
    )

    profile.save(cfg.feature_profile_path)
    logger.info("Saved feature baseline profile to %s (SHA-256: %s)", cfg.feature_profile_path, profile.sha256_checksum)
    return profile


def generate_prediction_baseline_profile(
    config: Optional[MonitoringConfig] = None,
    val_df: Optional[pd.DataFrame] = None,
) -> PredictionBaselineProfile:
    """
    Generate deterministic baseline prediction profile using frozen RiskEvaluator in-memory.

    Args:
        config: Monitoring configuration parameters.
        val_df: Optional pre-loaded validation DataFrame. If None, loaded from config.val_dataset_path.

    Returns:
        PredictionBaselineProfile: Populated prediction baseline profile object.
    """
    cfg = config or default_monitoring_config
    if val_df is None:
        logger.info("Loading validation features from %s", cfg.val_dataset_path)
        val_df = pd.read_parquet(cfg.val_dataset_path)

    row_count = len(val_df)

    # Instantiate frozen RiskEvaluator with standard 6-rule catalog (read-only, in-memory)
    evaluator = RiskEvaluator(
        rule_engine=RuleEngine(get_standard_rule_catalog())
    )

    results = evaluator.evaluate_dataframe(val_df)
    assert len(results) == row_count, f"Evaluation count mismatch: {len(results)} vs {row_count}"

    model_scores = np.array([float(r.model_score) for r in results], dtype=np.float64)
    risk_scores = np.array([int(r.risk_score) for r in results], dtype=np.int64)
    actions = [r.action.value for r in results]
    tiers = [r.risk_tier.value for r in results]
    overridden_count = sum(1 for r in results if r.is_overridden)

    # 1. 10 Equal-Width Model Score Bins: [0.0, 0.1), ..., [0.9, 1.0]
    score_bins: List[ScoreHistogramBin] = []
    for b_idx in range(10):
        low = b_idx / 10.0
        high = (b_idx + 1) / 10.0
        if b_idx == 9:
            mask = (model_scores >= low) & (model_scores <= high)
        else:
            mask = (model_scores >= low) & (model_scores < high)
        b_count = int(np.sum(mask))
        score_bins.append(
            ScoreHistogramBin(
                bin_index=b_idx,
                lower_bound=round(low, 2),
                upper_bound=round(high, 2),
                count=b_count,
                proportion=float(round(b_count / row_count, 6)),
            )
        )

    # 2. 10 Risk Score Buckets: [0, 10), ..., [90, 100]
    risk_buckets: List[RiskScoreBucket] = []
    for b_idx in range(10):
        low = b_idx * 10
        high = (b_idx + 1) * 10
        label = f"{low}-{high - 1}" if b_idx < 9 else "90-100"
        if b_idx == 9:
            mask = (risk_scores >= low) & (risk_scores <= high)
        else:
            mask = (risk_scores >= low) & (risk_scores < high)
        b_count = int(np.sum(mask))
        risk_buckets.append(
            RiskScoreBucket(
                bucket_index=b_idx,
                label=label,
                lower_bound=low,
                upper_bound=high if b_idx == 9 else high - 1,
                count=b_count,
                proportion=float(round(b_count / row_count, 6)),
            )
        )

    # 3. Tier Proportions
    tier_counts = {t.value: 0 for t in RiskTier}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    tier_props = {k: float(round(v / row_count, 6)) for k, v in tier_counts.items()}

    # 4. Action Proportions
    action_counts = {a.value: 0 for a in DecisionAction}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1
    action_props = {k: float(round(v / row_count, 6)) for k, v in action_counts.items()}

    profile = PredictionBaselineProfile(
        model_version=cfg.model_version,
        dataset_name=str(cfg.val_dataset_path.name),
        dataset_row_count=row_count,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_score_bins=score_bins,
        risk_score_buckets=risk_buckets,
        tier_proportions=tier_props,
        action_proportions=action_props,
        is_overridden_rate=float(round(overridden_count / row_count, 6)),
        mean_model_score=float(round(float(np.mean(model_scores)), 6)),
        mean_risk_score=float(round(float(np.mean(risk_scores)), 6)),
    )

    profile.save(cfg.prediction_profile_path)
    logger.info("Saved prediction baseline profile to %s (SHA-256: %s)", cfg.prediction_profile_path, profile.sha256_checksum)
    return profile


def generate_performance_baseline_profile(
    config: Optional[MonitoringConfig] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> PerformanceBaselineProfile:
    """
    Generate authoritative baseline performance benchmarks using frozen RiskEvaluator on OOT test data.

    Args:
        config: Monitoring configuration parameters.
        test_df: Optional pre-loaded Out-of-Time test DataFrame with is_fraud column.

    Returns:
        PerformanceBaselineProfile: Populated performance baseline profile object.
    """
    cfg = config or default_monitoring_config
    if test_df is None:
        logger.info("Loading OOT test features from %s", cfg.test_dataset_path)
        test_df = pd.read_parquet(cfg.test_dataset_path)

    assert TARGET_COLUMN in test_df.columns, f"Target column '{TARGET_COLUMN}' required in test DataFrame!"
    y_true = test_df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    row_count = len(test_df)
    total_frauds = int(np.sum(y_true == 1))
    total_legit = int(np.sum(y_true == 0))

    # Evaluate in-memory using RiskEvaluator and standard 6-rule catalog
    evaluator = RiskEvaluator(
        rule_engine=RuleEngine(get_standard_rule_catalog())
    )

    results = evaluator.evaluate_dataframe(test_df)
    y_prob = np.array([float(r.model_score) for r in results], dtype=np.float64)
    actions = np.array([r.action.value for r in results])

    # 1. Primary Operating Metrics at cost-optimal threshold tau* = 0.78
    prim_raw = compute_classification_metrics(y_true, y_prob, threshold=cfg.default_operating_threshold)
    prim_cm = ConfusionMatrixData(**prim_raw["confusion_matrix"], total=row_count)
    prim_metrics = ThresholdPerformanceMetrics(
        threshold=cfg.default_operating_threshold,
        precision=prim_raw["precision"],
        recall=prim_raw["recall"],
        f1=prim_raw["f1"],
        accuracy=prim_raw["accuracy"],
        fpr=prim_raw["rates"]["fpr"],
        tpr=prim_raw["rates"]["tpr"],
        confusion_matrix=prim_cm,
    )

    # 2. Secondary Comparison Metrics at default threshold tau = 0.50
    comp_raw = compute_classification_metrics(y_true, y_prob, threshold=cfg.comparison_threshold)
    comp_cm = ConfusionMatrixData(**comp_raw["confusion_matrix"], total=row_count)
    comp_metrics = ThresholdPerformanceMetrics(
        threshold=cfg.comparison_threshold,
        precision=comp_raw["precision"],
        recall=comp_raw["recall"],
        f1=comp_raw["f1"],
        accuracy=comp_raw["accuracy"],
        fpr=comp_raw["rates"]["fpr"],
        tpr=comp_raw["rates"]["tpr"],
        confusion_matrix=comp_cm,
    )

    # 3. Operational Decision Metrics (Hybrid Engine)
    block_mask = (actions == DecisionAction.BLOCK.value)
    review_mask = (actions == DecisionAction.REVIEW.value)
    intervention_mask = (actions == DecisionAction.BLOCK.value) | (actions == DecisionAction.REVIEW.value)

    total_blocks = int(np.sum(block_mask))
    fraud_in_block = int(np.sum((y_true == 1) & block_mask))
    decision_prec_block = float(round(fraud_in_block / total_blocks, 5)) if total_blocks > 0 else 0.0

    total_reviews = int(np.sum(review_mask))
    fraud_in_review = int(np.sum((y_true == 1) & review_mask))
    review_queue_purity = float(round(fraud_in_review / total_reviews, 5)) if total_reviews > 0 else 0.0

    fraud_interventions = int(np.sum((y_true == 1) & intervention_mask))
    decision_recall_intervention = float(round(fraud_interventions / total_frauds, 5)) if total_frauds > 0 else 0.0

    ops_metrics = OperationalDecisionMetrics(
        decision_precision_block=decision_prec_block,
        decision_recall_intervention=decision_recall_intervention,
        review_queue_purity=review_queue_purity,
        total_reviews_count=total_reviews,
        fraud_in_review_count=fraud_in_review,
        total_blocks_count=total_blocks,
        fraud_in_block_count=fraud_in_block,
    )

    profile = PerformanceBaselineProfile(
        model_version=cfg.model_version,
        dataset_name=str(cfg.test_dataset_path.name),
        dataset_row_count=row_count,
        total_frauds=total_frauds,
        total_legitimate=total_legit,
        created_at=datetime.now(timezone.utc).isoformat(),
        default_operating_threshold=cfg.default_operating_threshold,
        pr_auc=prim_raw["pr_auc"],
        roc_auc=prim_raw["roc_auc"],
        primary_operating_metrics=prim_metrics,
        comparison_metrics=comp_metrics,
        operational_decision_metrics=ops_metrics,
    )

    profile.save(cfg.performance_profile_path)
    logger.info("Saved performance baseline profile to %s (SHA-256: %s)", cfg.performance_profile_path, profile.sha256_checksum)
    return profile


def generate_all_baseline_profiles(
    config: Optional[MonitoringConfig] = None,
) -> Dict[str, str]:
    """
    Generate and persist all 3 baseline monitoring profile artifacts.

    Returns:
        Dict[str, str]: Mapping of artifact name to computed SHA-256 hex digest.
    """
    cfg = config or default_monitoring_config
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    feat_profile = generate_feature_baseline_profile(cfg)
    pred_profile = generate_prediction_baseline_profile(cfg)
    perf_profile = generate_performance_baseline_profile(cfg)

    return {
        "baseline_feature_profile": feat_profile.sha256_checksum or "",
        "baseline_prediction_profile": pred_profile.sha256_checksum or "",
        "baseline_performance_profile": perf_profile.sha256_checksum or "",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Generating all Phase 13 baseline monitoring profiles...")
    results = generate_all_baseline_profiles()
    print("Generation complete! Generated artifact checksums:")
    for k, v in results.items():
        print(f"  {k}: {v}")
