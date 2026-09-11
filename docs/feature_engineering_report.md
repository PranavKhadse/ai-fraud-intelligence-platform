# Behavioral Feature Engineering Report (Phase 3)

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Dataset:** Simulated Sparkov Credit Card Fraud Benchmark (1,852,394 Total Transactions)  
> **Feature Catalog:** 47 Engineered Behavioral Features across 7 Groups ([docs/feature_catalog.md](file:///docs/feature_catalog.md))  
> **Output Datasets:** `data/processed/features/` (`train_features.parquet`, `val_features.parquet`, `test_features.parquet`)  

---

## 1. Executive Summary

Phase 3 successfully engineered **47 production-quality behavioral features** across **7 feature groups**, enriching the 15 canonical transaction columns to establish an enterprise-grade, **62-column feature store**.

### Key Milestones Achieved:
1. **Mathematical Point-in-Time Correctness**: Strictly enforced $\text{timestamp}_j < \text{timestamp}_i$ using vectorized `np.searchsorted` interval indexing.
2. **Identical-Timestamp Isolation**: Transactions sharing identical timestamps ($t_j == t_i$) are guaranteed to never see each other as historical data.
3. **Cross-Partition Historical Continuity**: Validation transactions leverage Train + prior Validation history; Test transactions leverage Train + Validation + prior Test history. Future rows and labels are never accessible.
4. **Zero Target Leakage**: Feature engineering operates strictly on non-target transaction events. The `is_fraud` column is never accessed or referenced during feature generation.
5. **High Computational Performance**: Vectorized execution completed feature extraction on **1.85M transactions in 4056.27 seconds** with a peak memory allocation of **2821.0 MB**.

---

## 2. Partition & Performance Metrics

| Partition | Input Rows | Output Rows | Total Columns | Processing Runtime | Parquet File Size |
|---|---|---|---|---|---|
| **Train Features** | 1,296,675 | 1,296,675 | 62 | 1138.33s | 203.06 MB |
| **Validation Features** | 277,859 | 277,859 | 62 | 1348.91s | 49.20 MB |
| **Test Features (OOT)** | 277,860 | 277,860 | 62 | 1569.02s | 49.45 MB |
| **Total Benchmark** | **1,852,394** | **1,852,394** | **62 Columns** | **4056.27s** | **301.71 MB** |

---

## 3. Point-in-Time Leakage Audit Results

An independent, programmatic ground-truth leakage audit was executed across **1,000 randomly sampled transactions**:

- **Audited Sample Size:** 1,000 transactions
- **Exact Ground-Truth Matches:** 1,000 / 1,000
- **Pass Rate:** **100.00% (100% PASS)**
- **Lookahead Violations Detected:** **0**
- **Equal-Timestamp Leakages Detected:** **0**
- **Features Independently Audited:**
  `account_txn_count_before`, `account_total_spend_before`, `txn_count_1h`, `txn_count_24h`, `amt_sum_24h`, `account_unique_merchant_count_before`, `account_unique_category_count_before`.

---

## 4. Schema & Finiteness Validation

All 3 feature partitions passed comprehensive data integrity checks:
- **Canonical Columns Preserved:** 15 / 15 (100%)
- **Engineered Columns Added:** 47 / 47 (100%)
- **Total Columns:** 62
- **Null Values (NaN):** 0 across all partitions
- **Infinite Values (Inf):** 0 across all partitions
- **Speed Flag Rule Check (`is_impossible_travel_speed == (speed > 800)`):** 100% Consistent
- **Cold-Start Indicator Check (`is_first_account_txn == (count == 0)`):** 100% Consistent

---

## 5. Visual Behavioral Diagnostic Distributions

### 5.1 Amount Deviation Z-Score
The lifetime expanding $Z$-score strongly separates legitimate transactions (centered near 0) from fraudulent transactions (exhibiting heavy positive tail $\ge 3\sigma$).

![Amount Z-Score Distribution](file:///docs/features/figures/amount_zscore_distribution.png)

### 5.2 24-Hour Transaction Velocity Burst
Fraudulent attacks exhibit noticeable clustering in rolling transaction frequency, capturing high-velocity carding behavior.

![24-Hour Velocity Distribution](file:///docs/features/figures/velocity_24h_distribution.png)

### 5.3 Diurnal Fraud Prevalence
The engineered `is_night` feature aligns with the empirical 22:00–04:00 surge discovered in Phase 2 EDA.

![Diurnal Fraud Rate](file:///docs/features/figures/diurnal_fraud_rate.png)

---

## 6. Cold-Start Policy Verification

| Domain | Default Value | Measured Verification in Output |
|---|---|---|
| First Transaction Indicator | `is_first_account_txn = 1` | Verified on all 983 accounts' initial transactions |
| Time Since Previous Txn | `0.0s` | Verified for first transactions |
| Rolling Counts & Sums | `0` / `0.0` | Verified on initial windows |
| Spending Deviation Z-Score | `0.0` | Verified for $N < 2$ history |
| Spending Ratio to Historical Mean | `1.0` | Verified neutral ratio for $N=0$ |
| Distance & Speed from Prev Merchant | `0.0` / `0.0 km/h` | Verified for initial transactions |

---

## 7. Next Steps: Phase 4 Machine Learning Readiness

With the completion of Phase 3, the feature-engineered dataset is fully prepared for **Phase 4 (Baseline & Advanced ML Models)**:
- **Baseline Models**: Logistic Regression, Random Forest.
- **Production Classifiers**: LightGBM and XGBoost gradient-boosted decision trees.
- **Optimization Focus**: Precision-Recall AUC (PR-AUC) and Cost-Weighted Loss under the ~1:172 class imbalance regime.
