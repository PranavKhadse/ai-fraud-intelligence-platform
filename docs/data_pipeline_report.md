# Data Pipeline & Ingestion Report (Phase 1)

> **Dataset:** Synthetic Sparkov Credit Card Fraud Benchmark  
> **Source:** Brandon Harris / `kartik2112/fraud-detection` (Hugging Face Datasets)  
> **Schema Standard:** Canonical 15-Field Transaction Schema ([PROJECT_SPEC.md](file:///PROJECT_SPEC.md))  

---

## 1. Executive Ingestion Summary

- **Total Ingested Records:** 1,852,394
- **Cleaned & Deduplicated Records:** 1,852,394
- **Total Fraud Instances:** 9,651 (0.521%)
- **Dataset Time Span:** `2019-01-01 00:00:18` to `2020-12-31 23:59:34`
- **Temporal Ordering:** Verified non-decreasing `unix_time` with deterministic secondary sort on `transaction_id`.

---

## 2. Time-Aware Out-of-Time (OOT) Split Statistics

| Partition | Row Count | % Total | Fraud Cases | Fraud % | Temporal Start | Temporal End |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 1,296,675 | 70.0% | 7,506 | 0.579% | `2019-01-01 00:00:18` | `2020-06-21 12:13:37` |
| **Validation Set** | 277,859 | 15.0% | 1,221 | 0.439% | `2020-06-21 12:14:25` | `2020-10-03 00:58:23` |
| **Test Set (OOT)** | 277,860 | 15.0% | 924 | 0.333% | `2020-10-03 00:59:48` | `2020-12-31 23:59:34` |
| **Total** | **1,852,394** | **100.0%** | **9,651** | **0.521%** | `2019-01-01 00:00:18` | `2020-12-31 23:59:34` |

---

## 3. Data Leakage & Temporal Disjointness Audit

- **Train vs. Validation Disjointness:** `PASS (Train max <= Val min)`
- **Validation vs. Test Disjointness:** `PASS (Val max <= Test min)`
- **Data Leakage Risk:** **Zero Lookahead Leakage**. Random K-Fold partitioning is strictly avoided; all future evaluations will be conducted out-of-time.

---

## 4. Data Quality & Cleaning Decisions

- **Null Value Counts:** {"transaction_id": 0, "account_id": 0, "timestamp": 0, "unix_time": 0, "amount": 0, "currency": 0, "merchant_id": 0, "merchant_category": 0, "cardholder_lat": 0, "cardholder_long": 0, "merchant_lat": 0, "merchant_long": 0, "city_pop": 0, "job_category": 0, "is_fraud": 0}
- **Duplicate Transaction IDs Found:** 0
- **Negative Monetary Amounts:** 0
- **Invalid Class Labels:** 0
- **Invalid Coordinates Out of Bounds:** Lat: 0, Long: 0
- **Cleaning Actions Applied:**
  - Clean baseline: No invalid rows or coordinate anomalies detected.

---

## 5. Output Storage Details

Processed columnar files saved in Apache Parquet format:
- `data/processed/benchmark/train.parquet`
- `data/processed/benchmark/val.parquet`
- `data/processed/benchmark/test.parquet`

*All raw and processed data artifacts are excluded from Git tracking via `.gitignore`.*
