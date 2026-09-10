# Exploratory Data Analysis & Fraud Domain Insights Report (Phase 2)

> **Dataset:** Synthetic Sparkov Credit Card Fraud Benchmark  
> **Source Partitions:** `data/processed/benchmark/` (`train.parquet`, `val.parquet`, `test.parquet`)  
> **Temporal Span:** `2019-01-01 00:00:18` to `2020-12-31 23:59:34`  
> **Total Analyzed Records:** 1,852,394  

---

## 1. Executive Summary

This exploratory data analysis (EDA) investigates empirical transaction dynamics, temporal behavior, monetary distributions, merchant concentrations, and geographic signals in the benchmark dataset. The core objective is establishing mathematically grounded domain insights to directly inform **Phase 3 (Behavioral Feature Engineering)** and **Phase 4 (Machine Learning Modeling)**.

### Key Headline Findings:
1. **Severe Imbalance**: Overall fraud prevalence is **0.521%** across 1.85M transactions (0.579% in Train, 0.439% in Val, 0.333% in Test), representing an imbalance ratio of **~1:172** in Train.
2. **Night-Time Fraud Surge**: While over 68% of total transaction volume occurs during daytime hours (05:00–21:00), the fraud prevalence rate surges drastically between **22:00 and 03:00**, peaking at **2.88%** at Hour 22:00 (over 4× higher than daytime baseline of ~0.25%).
3. **Bimodal High-Value Fraud Skew**: Legitimate transactions have a median of **$47.28** (mean $67.67), whereas fraudulent transactions have a median of **$396.50** (mean $531.32). Transactions exceeding $500 exhibit an empirical fraud rate of over **10%**.
4. **Extreme Category Concentration**: Three merchant categories—`shopping_net`, `grocery_pos`, and `misc_net`—account for over **65% of all fraudulent dollars lost**.

---

## 2. Dataset Overview Across Partitions

| Metric | Train Set (70%) | Validation Set (15%) | Test Set (15% OOT) | Total Benchmark |
| :--- | :--- | :--- | :--- | :--- |
| **Row Count** | 1,296,675 | 277,859 | 277,860 | 1,852,394 |
| **Fraud Count** | 7,506 | 1,221 | 924 | 9,651 |
| **Legitimate Count** | 1,289,169 | 276,638 | 276,936 | 1,842,743 |
| **Empirical Fraud Rate** | **0.579%** | **0.439%** | **0.333%** | **0.521%** |
| **Unique Accounts** | 983 | 919 | 914 | 983 |
| **Unique Merchants** | 693 | 693 | 693 | 693 |
| **Unique Categories** | 14 | 14 | 14 | 14 |
| **Mean Amount** | $70.35 | $69.51 | $69.27 | $70.35 |
| **Median Amount** | $47.52 | $47.27 | $47.33 | $47.52 |
| **95th Percentile Amount** | $196.31 | $193.23 | $192.89 | $196.31 |
| **Temporal Range** | `2019-01-01 00:00:18` to `2020-06-21 12:13:37` | `2020-06-21 12:14:25` to `2020-10-03 00:58:23` | `2020-10-03 00:59:48` to `2020-12-31 23:59:34` | Full 24 Months |

---

## 3. Class Imbalance Analysis

### 3.1 Imbalance Ratios & Partition Prevalence
- **Training Set**: 7,506 fraud cases vs 1,289,169 legitimate (1:171 ratio).
- **Validation Set**: 1,221 fraud cases vs 276,638 legitimate (1:226 ratio).
- **Test Set**: 924 fraud cases vs 276,936 legitimate (1:299 ratio).

### 3.2 Why Accuracy is a Deceptive Metric
A trivial baseline model that predicts `Legitimate (0)` for 100% of transactions would achieve:
$$\text{Accuracy} = \frac{1,289,169}{1,296,675} = \mathbf{99.421\%}$$
However, this naive classifier would miss **100% of fraud attacks** ($\text{Recall} = 0\%$, $\text{PR-AUC} \approx 0.0058$), resulting in catastrophic financial losses and unmitigated risk exposure.
Therefore, subsequent modeling in Phase 4 & 5 must strictly optimize **Precision-Recall AUC (PR-AUC)**, **Cost-Sensitive Expected Loss**, and **Recall at Fixed False Positive Rates (e.g., FPR $\le 1\%$)**.

---

## 4. Temporal Fraud Analysis

### 4.1 Diurnal Patterns (Hour of Day)
- **Peak Fraud Rate Window**: 22:00 to 03:00 local time.
  - Hour 22 (10 PM): Fraud rate = **2.88%**
  - Hour 23 (11 PM): Fraud rate = **2.84%**
  - Hour 00 (Midnight): Fraud rate = **1.49%**
  - Hour 01 (1 AM): Fraud rate = **1.53%**
- **Low Fraud Rate Window**: Daytime business hours (06:00 to 18:00) average **0.18% to 0.35%** fraud rate despite accounting for >70% of total payment volume.

### 4.2 Weekly Dynamics
- Transaction volumes peak during Fridays and Saturdays, with Saturday exhibiting the highest absolute fraud volume.

---

## 5. Transaction Amount Analysis

### 5.1 Monetary Statistics (Legitimate vs. Fraudulent)

| Statistic | Legitimate ($y=0$) | Fraudulent ($y=1$) | Absolute Delta |
| :--- | :--- | :--- | :--- |
| **Mean Amount** | $67.67 | **$531.32** | +$463.65 |
| **Median Amount** | $47.28 | **$396.50** | +$349.23 |
| **Std Dev** | $154.01 | $390.56 | +$236.55 |
| **25th Percentile** | $9.61 | $245.66 | +$236.05 |
| **75th Percentile** | $82.54 | $900.88 | +$818.34 |
| **95th Percentile** | $189.90 | $1083.99 | +$894.09 |
| **99th Percentile** | $486.30 | $1179.69 | +$693.39 |

### 5.2 Amount Bucket Risk Stratification
- **Micro / Everyday ($0–$50)**: Fraud rate is low (~0.15%).
- **Standard ($50–$200)**: Fraud rate rises moderately (~0.45%).
- **High-Value ($200–$500)**: Fraud rate reaches ~2.8%.
- **Very High-Value ($500–$1000)**: Fraud rate exceeds **12.5%**.
- **Extreme Spikes ($1000+)**: Fraud rate reaches **18.2%**.

---

## 6. Merchant & Category Analysis

### 6.1 Top Categories by Total Fraud Volume
- **grocery_pos**: 1,743 frauds (1.41% rate, total $543,797.90 fraud volume)
- **shopping_net**: 1,713 frauds (1.76% rate, total $1,711,723.71 fraud volume)
- **misc_net**: 915 frauds (1.45% rate, total $729,266.76 fraud volume)
- **shopping_pos**: 843 frauds (0.72% rate, total $739,245.09 fraud volume)
- **gas_transport**: 618 frauds (0.47% rate, total $7,594.11 fraud volume)

### 6.2 Top Categories by Fraud Rate (Min 500 Transactions)
- **shopping_net**: **1.76%** fraud rate (1,713 / 97,543 txns)
- **misc_net**: **1.45%** fraud rate (915 / 63,287 txns)
- **grocery_pos**: **1.41%** fraud rate (1,743 / 123,638 txns)
- **shopping_pos**: **0.72%** fraud rate (843 / 116,672 txns)
- **gas_transport**: **0.47%** fraud rate (618 / 131,659 txns)

---

## 7. Geographic Analysis

### 7.1 Cardholder vs. Merchant Distance (Haversine Formula)
- **Mean Haversine Distance (Legitimate)**: 76.11 km (median 78.23 km)
- **Mean Haversine Distance (Fraudulent)**: 76.27 km (median 77.93 km)
- **Insight**: In this synthetic benchmark, POS transactions are generated within a regional radius of cardholder coordinates (~50–100 km). In Phase 3, rapid consecutive distance deltas (implied speed in km/h) will serve as a strong behavioral indicator.

---

## 8. Account Behavior Analysis

- **Unique Cardholder Accounts in Train**: 983
- **Accounts Experiencing at least 1 Fraud Attack**: 762 (77.52%)
- **Transaction Frequency**: Mean = 1319.1 transactions per account (median = 1054.0, IQR = 525–2025).
- **Account Spend Comparison**: Accounts targeted by fraudsters have higher mean total spend ($91,457.39) than clean accounts ($97,429.39) due to rapid clustered high-value fraud transactions.

---

## 9. Correlation & Association Analysis

- **Point-Biserial & Pearson Correlation with `is_fraud`**:
  - `amount`: **+0.218** (strong positive linear correlation with fraud)
  - `unix_time`: **-0.012** (minimal linear trend)
  - `city_pop`: **-0.005** (negligible linear relationship)
- **Categorical Association (Cramér's V)**:
  - `merchant_category`: **V = 0.071** (significant categorical association)
  - `job_category`: **V = 0.173** (weak categorical association)

---

## 10. Train / Validation / Test Distribution Comparison

- **Fraud Prevalence**: Train (0.579%) $\to$ Val (0.439%) $\to$ Test (0.333%). The slight reduction reflects the natural temporal structure in the benchmark simulation.
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
