"""
Reproducible Master Entry Point for Phase 2 Exploratory Data Analysis (EDA).

Usage:
    python -m ml.eda.run_eda
"""

import logging
from pathlib import Path

from ml.eda.load import load_processed_splits
from ml.eda.overview import compute_all_overviews
from ml.eda.class_imbalance import analyze_class_imbalance
from ml.eda.temporal import analyze_temporal_patterns
from ml.eda.amount import compute_amount_distribution
from ml.eda.merchant import analyze_merchants_and_categories
from ml.eda.geography import analyze_geography
from ml.eda.behavioral_signals import analyze_account_behavior
from ml.eda.correlations import analyze_correlations
from ml.eda.visualize import generate_all_figures
from ml.eda.report import build_eda_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RunEDA")


def main() -> None:
    logger.info("Starting Phase 2 Exploratory Data Analysis (EDA)...")

    # 1. Load Parquet Partitions
    logger.info("Loading processed Parquet partitions from data/processed/benchmark/...")
    train_df, val_df, test_df = load_processed_splits()
    logger.info(f"Loaded Train ({len(train_df):,} rows), Val ({len(val_df):,} rows), Test ({len(test_df):,} rows).")

    # 2. Dataset Overview
    logger.info("Computing partition overviews...")
    overviews = compute_all_overviews(train_df, val_df, test_df)

    # 3. Class Imbalance
    logger.info("Analyzing class imbalance...")
    imbalance = analyze_class_imbalance(train_df, val_df, test_df)

    # 4. Temporal Patterns
    logger.info("Analyzing temporal fraud patterns...")
    temporal = analyze_temporal_patterns(train_df)

    # 5. Amount Distributions
    logger.info("Analyzing monetary distributions...")
    amount = compute_amount_distribution(train_df)

    # 6. Merchant & Category Concentration
    logger.info("Analyzing merchant categories and concentration...")
    merchant = analyze_merchants_and_categories(train_df)

    # 7. Geographic Analysis
    logger.info("Analyzing geographic distance patterns...")
    geography = analyze_geography(train_df)

    # 8. Account Behavior
    logger.info("Analyzing account-level behavioral distributions...")
    account = analyze_account_behavior(train_df)

    # 9. Correlations & Associations
    logger.info("Computing correlation and association matrices...")
    correlations = analyze_correlations(train_df)

    # 10. Generate Visualizations (15 Matplotlib Figures)
    logger.info("Generating 15 publication-quality matplotlib charts into docs/eda/figures/...")
    fig_paths = generate_all_figures(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        overview_meta=overviews,
        temporal_meta=temporal,
        amount_meta=amount,
        merchant_meta=merchant,
        correlations_meta=correlations,
    )
    logger.info(f"Generated {len(fig_paths)} charts successfully.")

    # 11. Generate Markdown Report
    report_path = Path("docs/eda_report.md")
    logger.info(f"Compiling comprehensive report to {report_path}...")
    build_eda_report(
        overview=overviews,
        imbalance=imbalance,
        temporal=temporal,
        amount=amount,
        merchant=merchant,
        geography=geography,
        account=account,
        correlations=correlations,
        output_path=report_path,
    )
    logger.info(f"Report compiled successfully: {report_path}")
    logger.info("Phase 2 EDA Pipeline completed successfully.")


if __name__ == "__main__":
    main()
