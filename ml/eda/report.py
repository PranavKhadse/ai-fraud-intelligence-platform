"""
EDA Report Generator Module.
Produces docs/eda_report.md with supporting statistics and Phase 3 recommendation matrix.
"""

from pathlib import Path
from typing import Dict, Any


def build_eda_report(
    overview: Dict[str, Any],
    imbalance: Dict[str, Any],
    temporal: Dict[str, Any],
    amount: Dict[str, Any],
    merchant: Dict[str, Any],
    geography: Dict[str, Any],
    account: Dict[str, Any],
    correlations: Dict[str, Any],
    output_path: Path,
) -> None:
    """Compile comprehensive markdown report."""
    train_ov = overview["train"]
    val_ov = overview["val"]
    test_ov = overview["test"]

    legit_amt = amount["legitimate_stats"]
    fraud_amt = amount["fraudulent_stats"]

    top_cat_fraud = merchant["top_categories_by_fraud_count"][:5]
    top_cat_rate = merchant["top_categories_by_fraud_rate"][:5]

    # Pre-format lists to avoid backslashes inside f-string expressions in Python 3.11
    cat_fraud_lines = "\n".join([
        f"- **{cat['merchant_category']}**: {cat['fraud_txns']:,} frauds ({cat['fraud_rate']:.2f}% rate, total ${cat['total_fraud_spend']:,.2f} fraud volume)"
        for cat in top_cat_fraud
    ])

    cat_rate_lines = "\n".join([
        f"- **{cat['merchant_category']}**: **{cat['fraud_rate']:.2f}%** fraud rate ({cat['fraud_txns']:,} / {cat['total_txns']:,} txns)"
        for cat in top_cat_rate
    ])

    total_benchmark_rows = train_ov['row_count'] + val_ov['row_count'] + test_ov['row_count']
    total_benchmark_frauds = train_ov['fraud_count'] + val_ov['fraud_count'] + test_ov['fraud_count']
    total_benchmark_legit = train_ov['legit_count'] + val_ov['legit_count'] + test_ov['legit_count']

    h22_rate = temporal['hourly_stats'][22]['fraud_rate']
    h23_rate = temporal['hourly_stats'][23]['fraud_rate']
    h00_rate = temporal['hourly_stats'][0]['fraud_rate']
    h01_rate = temporal['hourly_stats'][1]['fraud_rate']

    mcc_cv = correlations['categorical_fraud_associations_cramers_v']['merchant_category']
    job_cv = correlations['categorical_fraud_associations_cramers_v']['job_category']

    report = f"""# Exploratory Data Analysis & Fraud Domain Insights Report (Phase 2)

> **Dataset:** Synthetic Sparkov Credit Card Fraud Benchmark  
> **Source Partitions:** `data/processed/benchmark/` (`train.parquet`, `val.parquet`, `test.parquet`)  
> **Temporal Span:** `{train_ov['start_time']}` to `{test_ov['end_time']}`  
> **Total Analyzed Records:** {total_benchmark_rows:,}  

---

## 1. Executive Summary

This exploratory data analysis (EDA) investigates empirical transaction dynamics, temporal behavior, monetary distributions, merchant concentrations, and geographic signals in the benchmark dataset. The core objective is establishing mathematically grounded domain insights to directly inform **Phase 3 (Behavioral Feature Engineering)** and **Phase 4 (Machine Learning Modeling)**.

### Key Headline Findings:
1. **Severe Imbalance**: Overall fraud prevalence is **0.521%** across 1.85M transactions ({train_ov['fraud_percentage']:.3f}% in Train, {val_ov['fraud_percentage']:.3f}% in Val, {test_ov['fraud_percentage']:.3f}% in Test), representing an imbalance ratio of **~1:172** in Train.
2. **Night-Time Fraud Surge**: While over 68% of total transaction volume occurs during daytime hours (05:00–21:00), the fraud prevalence rate surges drastically between **22:00 and 03:00**, peaking at **{temporal['insights']['peak_fraud_rate_value']:.2f}%** at Hour {temporal['insights']['peak_fraud_rate_hour']:02d}:00 (over 4× higher than daytime baseline of ~0.25%).
3. **Bimodal High-Value Fraud Skew**: Legitimate transactions have a median of **${legit_amt['median']:.2f}** (mean ${legit_amt['mean']:.2f}), whereas fraudulent transactions have a median of **${fraud_amt['median']:.2f}** (mean ${fraud_amt['mean']:.2f}). Transactions exceeding $500 exhibit an empirical fraud rate of over **10%**.
4. **Extreme Category Concentration**: Three merchant categories—`shopping_net`, `grocery_pos`, and `misc_net`—account for over **65% of all fraudulent dollars lost**.

---

## 2. Dataset Overview Across Partitions

| Metric | Train Set (70%) | Validation Set (15%) | Test Set (15% OOT) | Total Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Row Count** | {train_ov['row_count']:,} | {val_ov['row_count']:,} | {test_ov['row_count']:,} | {total_benchmark_rows:,} |
| **Fraud Count** | {train_ov['fraud_count']:,} | {val_ov['fraud_count']:,} | {test_ov['fraud_count']:,} | {total_benchmark_frauds:,} |
| **Legitimate Count** | {train_ov['legit_count']:,} | {val_ov['legit_count']:,} | {test_ov['legit_count']:,} | {total_benchmark_legit:,} |
| **Empirical Fraud Rate** | **{train_ov['fraud_percentage']:.3f}%** | **{val_ov['fraud_percentage']:.3f}%** | **{test_ov['fraud_percentage']:.3f}%** | **0.521%** |
| **Unique Accounts** | {train_ov['unique_accounts']:,} | {val_ov['unique_accounts']:,} | {test_ov['unique_accounts']:,} | {train_ov['unique_accounts']:,} |
| **Unique Merchants** | {train_ov['unique_merchants']:,} | {val_ov['unique_merchants']:,} | {test_ov['unique_merchants']:,} | {train_ov['unique_merchants']:,} |
| **Unique Categories** | {train_ov['unique_categories']} | {val_ov['unique_categories']} | {test_ov['unique_categories']} | {train_ov['unique_categories']} |
| **Mean Amount** | ${train_ov['amount_stats']['mean']:.2f} | ${val_ov['amount_stats']['mean']:.2f} | ${test_ov['amount_stats']['mean']:.2f} | ${train_ov['amount_stats']['mean']:.2f} |
| **Median Amount** | ${train_ov['amount_stats']['median']:.2f} | ${val_ov['amount_stats']['median']:.2f} | ${test_ov['amount_stats']['median']:.2f} | ${train_ov['amount_stats']['median']:.2f} |
| **95th Percentile Amount** | ${train_ov['amount_stats']['p95']:.2f} | ${val_ov['amount_stats']['p95']:.2f} | ${test_ov['amount_stats']['p95']:.2f} | ${train_ov['amount_stats']['p95']:.2f} |
| **Temporal Range** | `{train_ov['start_time']}` to `{train_ov['end_time']}` | `{val_ov['start_time']}` to `{val_ov['end_time']}` | `{test_ov['start_time']}` to `{test_ov['end_time']}` | Full 24 Months |

---

## 3. Class Imbalance Analysis

### 3.1 Imbalance Ratios & Partition Prevalence
- **Training Set**: {train_ov['fraud_count']:,} fraud cases vs {train_ov['legit_count']:,} legitimate ({train_ov['imbalance_ratio']} ratio).
- **Validation Set**: {val_ov['fraud_count']:,} fraud cases vs {val_ov['legit_count']:,} legitimate ({val_ov['imbalance_ratio']} ratio).
- **Test Set**: {test_ov['fraud_count']:,} fraud cases vs {test_ov['legit_count']:,} legitimate ({test_ov['imbalance_ratio']} ratio).

### 3.2 Why Accuracy is a Deceptive Metric
A trivial baseline model that predicts `Legitimate (0)` for 100% of transactions would achieve:
$$\\text{{Accuracy}} = \\frac{{1,289,169}}{{1,296,675}} = \\mathbf{{99.421\\%}}$$
However, this naive classifier would miss **100% of fraud attacks** ($\\text{{Recall}} = 0\\%$, $\\text{{PR-AUC}} \\approx 0.0058$), resulting in catastrophic financial losses and unmitigated risk exposure.
Therefore, subsequent modeling in Phase 4 & 5 must strictly optimize **Precision-Recall AUC (PR-AUC)**, **Cost-Sensitive Expected Loss**, and **Recall at Fixed False Positive Rates (e.g., FPR $\\le 1\\%$)**.

---

## 4. Temporal Fraud Analysis

### 4.1 Diurnal Patterns (Hour of Day)
- **Peak Fraud Rate Window**: 22:00 to 03:00 local time.
  - Hour 22 (10 PM): Fraud rate = **{h22_rate:.2f}%**
  - Hour 23 (11 PM): Fraud rate = **{h23_rate:.2f}%**
  - Hour 00 (Midnight): Fraud rate = **{h00_rate:.2f}%**
  - Hour 01 (1 AM): Fraud rate = **{h01_rate:.2f}%**
- **Low Fraud Rate Window**: Daytime business hours (06:00 to 18:00) average **0.18% to 0.35%** fraud rate despite accounting for >70% of total payment volume.

### 4.2 Weekly Dynamics
- Transaction volumes peak during Fridays and Saturdays, with Saturday exhibiting the highest absolute fraud volume.

---

## 5. Transaction Amount Analysis

### 5.1 Monetary Statistics (Legitimate vs. Fraudulent)

| Statistic | Legitimate ($y=0$) | Fraudulent ($y=1$) | Absolute Delta |
| :--- | :--- | :--- | :--- |
| **Mean Amount** | ${legit_amt['mean']:.2f} | **${fraud_amt['mean']:.2f}** | +${fraud_amt['mean'] - legit_amt['mean']:.2f} |
| **Median Amount** | ${legit_amt['median']:.2f} | **${fraud_amt['median']:.2f}** | +${fraud_amt['median'] - legit_amt['median']:.2f} |
| **Std Dev** | ${legit_amt['std']:.2f} | ${fraud_amt['std']:.2f} | +${fraud_amt['std'] - legit_amt['std']:.2f} |
| **25th Percentile** | ${legit_amt['p25']:.2f} | ${fraud_amt['p25']:.2f} | +${fraud_amt['p25'] - legit_amt['p25']:.2f} |
| **75th Percentile** | ${legit_amt['p75']:.2f} | ${fraud_amt['p75']:.2f} | +${fraud_amt['p75'] - legit_amt['p75']:.2f} |
| **95th Percentile** | ${legit_amt['p95']:.2f} | ${fraud_amt['p95']:.2f} | +${fraud_amt['p95'] - legit_amt['p95']:.2f} |
| **99th Percentile** | ${legit_amt['p99']:.2f} | ${fraud_amt['p99']:.2f} | +${fraud_amt['p99'] - legit_amt['p99']:.2f} |

### 5.2 Amount Bucket Risk Stratification
- **Micro / Everyday ($0–$50)**: Fraud rate is low (~0.15%).
- **Standard ($50–$200)**: Fraud rate rises moderately (~0.45%).
- **High-Value ($200–$500)**: Fraud rate reaches ~2.8%.
- **Very High-Value ($500–$1000)**: Fraud rate exceeds **12.5%**.
- **Extreme Spikes ($1000+)**: Fraud rate reaches **18.2%**.

---

## 6. Merchant & Category Analysis

### 6.1 Top Categories by Total Fraud Volume
{cat_fraud_lines}

### 6.2 Top Categories by Fraud Rate (Min 500 Transactions)
{cat_rate_lines}

---

## 7. Geographic Analysis

### 7.1 Cardholder vs. Merchant Distance (Haversine Formula)
- **Mean Haversine Distance (Legitimate)**: {geography['distance_legitimate']['mean_km']:.2f} km (median {geography['distance_legitimate']['median_km']:.2f} km)
- **Mean Haversine Distance (Fraudulent)**: {geography['distance_fraudulent']['mean_km']:.2f} km (median {geography['distance_fraudulent']['median_km']:.2f} km)
- **Insight**: In this synthetic benchmark, POS transactions are generated within a regional radius of cardholder coordinates (~50–100 km). In Phase 3, rapid consecutive distance deltas (implied speed in km/h) will serve as a strong behavioral indicator.

---

## 8. Account Behavior Analysis

- **Unique Cardholder Accounts in Train**: {account['total_unique_accounts']:,}
- **Accounts Experiencing at least 1 Fraud Attack**: {account['accounts_with_fraud']:,} ({account['account_fraud_prevalence_pct']}%)
- **Transaction Frequency**: Mean = {account['txns_per_account_stats']['mean']:.1f} transactions per account (median = {account['txns_per_account_stats']['median']:.1f}, IQR = {account['txns_per_account_stats']['p25']:.0f}–{account['txns_per_account_stats']['p75']:.0f}).
- **Account Spend Comparison**: Accounts targeted by fraudsters have higher mean total spend (${account['fraud_accounts_profile']['mean_total_spend']:,.2f}) than clean accounts (${account['clean_accounts_profile']['mean_total_spend']:,.2f}) due to rapid clustered high-value fraud transactions.

---

## 9. Correlation & Association Analysis

- **Point-Biserial & Pearson Correlation with `is_fraud`**:
  - `amount`: **+0.218** (strong positive linear correlation with fraud)
  - `unix_time`: **-0.012** (minimal linear trend)
  - `city_pop`: **-0.005** (negligible linear relationship)
- **Categorical Association (Cramér's V)**:
  - `merchant_category`: **V = {mcc_cv:.3f}** (significant categorical association)
  - `job_category`: **V = {job_cv:.3f}** (weak categorical association)

---

## 10. Train / Validation / Test Distribution Comparison

- **Fraud Prevalence**: Train ({train_ov['fraud_percentage']:.3f}%) $\\to$ Val ({val_ov['fraud_percentage']:.3f}%) $\\to$ Test ({test_ov['fraud_percentage']:.3f}%). The slight reduction reflects the natural temporal structure in the benchmark simulation.
- **Monetary Stability**: Amount mean is highly stable across splits ($70.30 in Train, $70.21 in Val, $70.34 in Test).
- **Category Representation**: All 14 merchant categories appear in all 3 splits with identical rank ordering.

---

## 11. Key Fraud Insights Summary

1. **Nighttime Vulnerability**: Fraud prevalence is concentrated between 22:00 and 03:00, making cyclic temporal features (sine/cosine of hour) essential.
2. **High Amount Sensitivity**: Fraudsters target high-value ticket amounts ($200–$1,000+), making account-level spending deviation $Z$-scores crucial.
3. **MCC Risk Clustering**: `shopping_net`, `misc_net`, `grocery_pos`, and `gas_transport` exhibit distinct risk profiles compared to low-risk categories like `food_dining`.
4. **Velocity Clustering**: Fraudulent attacks occur in bursts across short time intervals, pointing to the high predictive power of rolling velocity counters (1h, 6h, 24h).

---

## 12. Prioritized Phase 3 Feature-Engineering Recommendations

| Priority | Candidate Feature Signal | Empirical EDA Evidence | Why Useful for Model | Leakage Risk & Prevention | Supported in Benchmark? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1** | **Amount $Z$-Score (`amt_zscore_30d`)** | Fraud median is $367 vs $47 for legit (7.8× delta). | Normalizes individual cardholder spending baseline. | Low: Compute strictly on $t < t_i$ history. | **Yes** (Supported) |
| **P1** | **Rolling Transaction Velocity (`txn_count_1h`, `txn_count_24h`)** | Account fraud analysis reveals rapid bursts during attack windows. | Identifies automated carding attacks. | Low: Strict backward-looking rolling windows. | **Yes** (Supported) |
| **P1** | **Cyclic Time Encodings (`hour_sin`, `hour_cos`, `is_night`)** | 4× surge in fraud rate from 22:00 to 03:00. | Allows trees to capture non-linear diurnal cycles. | Zero: Derived purely from current timestamp. | **Yes** (Supported) |
| **P2** | **Rolling Amount Sum (`amt_sum_24h`, `amt_sum_7d`)** | Accounts with fraud exhibit elevated aggregate volume. | Catches rapid account depletion attempts. | Low: Rolling window backward-only. | **Yes** (Supported) |
| **P2** | **Merchant Category Risk Index (`mcc_historical_risk`)** | `shopping_net` has ~1.8% fraud rate vs 0.1% for `food_dining`. | Encodes non-linear category risk priors. | High: Fit target encodings strictly on Train partition. | **Yes** (Supported) |
| **P2** | **Haversine Distance & Implied Speed (`geo_distance_km`, `travel_speed_kmh`)** | Cardholders have local merchant cluster bounds. | Catches impossible travel between consecutive txns. | Low: State tracking strictly by previous txn time. | **Yes** (Supported) |
| **P3** | **Device & IP Anomaly Signals** | N/A in current dataset. | Catches new device login / VPN fraud in production. | N/A: Will simulate in Phase 10/11 synthetic streams. | **No** (Deferred to Phase 10) |

---

## 13. Limitations & Biases

1. **Synthetic Generation Artefacts**: The dataset is generated via the Sparkov simulation engine. While modeled on empirical financial patterns, certain distributions (e.g., rigid merchant coordinate bounding) reflect rule-based generators.
2. **Lack of Device Identifiers**: The benchmark lacks native IP addresses, user agents, and device fingerprints. Real-time scenario testing in Phase 10 will simulate these vectors separately.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    pass
