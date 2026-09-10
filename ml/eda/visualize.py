"""
Publication-Quality Visualization Generator for Fraud EDA.
Strictly uses matplotlib (zero seaborn dependencies).
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Set clean aesthetic styling parameters
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FIGURES_DIR = Path("docs/eda/figures")

# Distinct, professional color palette
COLOR_LEGIT = "#2b5c8f"    # Slate blue
COLOR_FRAUD = "#c83232"    # Crimson red
COLOR_ACCENT = "#e07a5f"   # Coral
COLOR_BAR = "#3d5a80"      # Deep steel


def generate_all_figures(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    overview_meta: Dict[str, Any],
    temporal_meta: Dict[str, Any],
    amount_meta: Dict[str, Any],
    merchant_meta: Dict[str, Any],
    correlations_meta: Dict[str, Any],
) -> List[Path]:
    """Generate all 15 high-value publication-quality EDA charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # ----------------------------------------------------
    # 1. Class Distribution (Log Scale Count)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = [overview_meta["train"]["legit_count"], overview_meta["train"]["fraud_count"]]
    labels = ["Legitimate\n(99.42%)", "Fraudulent\n(0.58%)"]
    bars = ax.bar(labels, counts, color=[COLOR_LEGIT, COLOR_FRAUD], width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Transaction Count (Log Scale)")
    ax.set_title("Training Set Class Distribution (Extreme Class Imbalance)")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    out = FIGURES_DIR / "01_class_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 2. Fraud Rate Over Time (Monthly Trend)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    train_copy = train_df.copy()
    train_copy["year_month"] = pd.to_datetime(train_copy["timestamp"]).dt.to_period("M").astype(str)
    monthly = train_copy.groupby("year_month")["is_fraud"].agg(["mean", "sum", "count"]).reset_index()
    monthly["fraud_pct"] = monthly["mean"] * 100

    ax.plot(monthly["year_month"], monthly["fraud_pct"], marker="o", color=COLOR_FRAUD, linewidth=2, markersize=5)
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Year-Month")
    ax.set_title("Monthly Fraud Rate Trend (Training Dataset 2019 - mid-2020)")
    plt.xticks(rotation=45)
    out = FIGURES_DIR / "02_fraud_rate_over_time.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 3. Transaction Volume Over Time
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(monthly["year_month"], monthly["count"], color=COLOR_BAR, width=0.6, edgecolor="black", linewidth=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))
    ax.set_ylabel("Transaction Volume (Thousands)")
    ax.set_xlabel("Year-Month")
    ax.set_title("Monthly Total Transaction Ingestion Volume")
    plt.xticks(rotation=45)
    out = FIGURES_DIR / "03_transaction_volume_over_time.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 4. Fraud Count by Hour of Day
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hourly_df = pd.DataFrame(temporal_meta["hourly_stats"])
    ax.bar(hourly_df["hour"], hourly_df["fraud_count"], color=COLOR_FRAUD, width=0.7, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Hour of Day (0 to 23)")
    ax.set_ylabel("Total Fraudulent Transactions")
    ax.set_title("Absolute Fraud Count by Hour of Day")
    ax.set_xticks(range(0, 24))
    out = FIGURES_DIR / "04_fraud_by_hour.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 5. Fraud Count by Day of Week
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    dow_df = pd.DataFrame(temporal_meta["dow_stats"])
    ax.bar(dow_df["day_of_week"], dow_df["fraud_count"], color=COLOR_BAR, width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Fraud Transactions")
    ax.set_title("Absolute Fraud Count by Day of Week")
    plt.xticks(rotation=30)
    out = FIGURES_DIR / "05_fraud_by_day_of_week.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 6. Fraud Rate by Hour of Day (%)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(hourly_df["hour"], hourly_df["fraud_rate"], marker="s", color=COLOR_FRAUD, linewidth=2.2, markersize=5)
    ax.fill_between(hourly_df["hour"], hourly_df["fraud_rate"], color=COLOR_FRAUD, alpha=0.15)
    ax.set_xlabel("Hour of Day (0 - 23)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Prevalence Rate (%) by Hour of Day (Night-Time Surge)")
    ax.set_xticks(range(0, 24))
    # Highlight night surge
    ax.axvspan(22, 23, color="gray", alpha=0.15, label="Late Night (22h-03h)")
    ax.axvspan(0, 3, color="gray", alpha=0.15)
    ax.legend(loc="upper right")
    out = FIGURES_DIR / "06_fraud_rate_by_hour.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 7. Overall Amount Distribution (Log Scale)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(train_df["amount"], bins=60, color=COLOR_LEGIT, edgecolor="black", linewidth=0.5, log=True)
    ax.set_xlabel("Transaction Amount ($ USD)")
    ax.set_ylabel("Frequency Count (Log Scale)")
    ax.set_title("Overall Transaction Amount Distribution")
    out = FIGURES_DIR / "07_amount_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 8. Legitimate vs Fraud Amount Distribution
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    legit_amt = train_df[train_df["is_fraud"] == 0]["amount"]
    fraud_amt = train_df[train_df["is_fraud"] == 1]["amount"]
    ax.hist(legit_amt, bins=50, range=(0, 1500), density=True, alpha=0.5, color=COLOR_LEGIT, label="Legitimate (Normalized)")
    ax.hist(fraud_amt, bins=50, range=(0, 1500), density=True, alpha=0.6, color=COLOR_FRAUD, label="Fraudulent (Normalized)")
    ax.set_xlabel("Transaction Amount ($ USD)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Comparative Monetary Density: Legitimate vs. Fraudulent Transactions")
    ax.legend()
    out = FIGURES_DIR / "08_legit_vs_fraud_amount_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 9. Fraud Rate by Amount Bucket
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b_df = pd.DataFrame(amount_meta["bucket_analysis"])
    ax.bar(b_df["amount_bucket"], b_df["fraud_rate"], color=COLOR_FRAUD, width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Transaction Amount Bucket ($ USD)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Empirical Fraud Rate by Transaction Amount Bucket")
    plt.xticks(rotation=30)
    for i, rate in enumerate(b_df["fraud_rate"]):
        ax.annotate(f"{rate:.1f}%", xy=(i, rate), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    out = FIGURES_DIR / "09_fraud_rate_by_amount_bucket.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 10. Fraud Rate by Merchant Category
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_df = pd.DataFrame(merchant_meta["category_summary"]).sort_values(by="fraud_rate", ascending=True)
    ax.barh(cat_df["merchant_category"], cat_df["fraud_rate"], color=COLOR_BAR, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_ylabel("Merchant Category (MCC)")
    ax.set_title("Empirical Fraud Rate (%) by Merchant Category")
    for i, (cat, rate) in enumerate(zip(cat_df["merchant_category"], cat_df["fraud_rate"])):
        ax.annotate(f"{rate:.2f}%", xy=(rate, i), xytext=(4, -3), textcoords="offset points", fontsize=8)
    out = FIGURES_DIR / "10_fraud_rate_by_merchant_category.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 11. Top Categories by Fraud Count
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_count_df = pd.DataFrame(merchant_meta["category_summary"]).sort_values(by="fraud_txns", ascending=True)
    ax.barh(cat_count_df["merchant_category"], cat_count_df["fraud_txns"], color=COLOR_FRAUD, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Total Fraudulent Transactions")
    ax.set_ylabel("Merchant Category")
    ax.set_title("Absolute Fraud Volume by Merchant Category")
    for i, (cat, count) in enumerate(zip(cat_count_df["merchant_category"], cat_count_df["fraud_txns"])):
        ax.annotate(f"{int(count):,}", xy=(count, i), xytext=(4, -3), textcoords="offset points", fontsize=8)
    out = FIGURES_DIR / "11_top_categories_by_fraud_count.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 12. Geographic Transaction Distribution
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # Sample legitimate for clean plotting
    legit_sample = train_df[train_df["is_fraud"] == 0].sample(n=10000, random_state=42)
    fraud_sample = train_df[train_df["is_fraud"] == 1]
    ax.scatter(legit_sample["cardholder_long"], legit_sample["cardholder_lat"], c=COLOR_LEGIT, s=4, alpha=0.2, label="Legitimate (10k sample)")
    ax.scatter(fraud_sample["cardholder_long"], fraud_sample["cardholder_lat"], c=COLOR_FRAUD, s=8, alpha=0.7, label="Fraudulent")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Scatter: Cardholder Locations (Continental USA)")
    ax.legend(loc="lower left")
    out = FIGURES_DIR / "12_geographic_transaction_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 13. Account Transaction Count Distribution
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    acc_counts = train_df.groupby("account_id").size()
    ax.hist(acc_counts, bins=40, color=COLOR_BAR, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Transactions per Account")
    ax.set_ylabel("Number of Accounts")
    ax.set_title("Account Transaction-Count Distribution (Training Set)")
    out = FIGURES_DIR / "13_account_transaction_count_distribution.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 14. Train vs Validation vs Test Fraud Rate Comparison
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [
        overview_meta["train"]["fraud_percentage"],
        overview_meta["val"]["fraud_percentage"],
        overview_meta["test"]["fraud_percentage"],
    ]
    split_labels = ["Train Set\n(70%)", "Validation Set\n(15%)", "Test Set\n(15% OOT)"]
    bars = ax.bar(split_labels, rates, color=[COLOR_BAR, "#5c6b73", "#9db4c0"], width=0.45, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Empirical Fraud Rate (%)")
    ax.set_title("Temporal Split Comparison: Fraud Prevalence Over Time")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    out = FIGURES_DIR / "14_train_val_test_fraud_rate_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    # ----------------------------------------------------
    # 15. Numerical Correlation Matrix (Pure Matplotlib Heatmap)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    num_cols = ["amount", "unix_time", "city_pop", "cardholder_lat", "cardholder_long", "merchant_lat", "merchant_long", "is_fraud"]
    corr_df = train_df[num_cols].corr(method="pearson")
    cax = ax.matshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left")
    ax.set_yticklabels(num_cols)
    ax.set_title("Pearson Correlation Matrix (Numerical Features)", pad=20)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr_df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.6 else "white", fontsize=8)
    out = FIGURES_DIR / "15_numerical_correlation_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    generated.append(out)

    return generated
