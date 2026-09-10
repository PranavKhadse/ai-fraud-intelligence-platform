# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Last Updated:** Current Date (Phase 2 Implementation Completed)  
> **Current Active Phase:** **Phase 2 — Exploratory Data Analysis & Fraud Domain Insights (Completed)**  

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Target Completion |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Project Foundation & Architecture** | 🟢 **Completed** | Milestone 0 |
| **Phase 1** | **Data Pipeline & Ingestion** | 🟢 **Completed** | Milestone 1 |
| **Phase 2** | **Exploratory Data Analysis (EDA) & Insights** | 🟢 **Completed** | Milestone 2 (Current) |
| **Phase 3** | **Behavioral Feature Engineering** | ⚪ Pending | Next Milestone |
| **Phase 4** | **Baseline & Advanced ML Models** | ⚪ Pending | Phase 4 |
| **Phase 5** | **Imbalance Handling & Cost Optimization** | ⚪ Pending | Phase 5 |
| **Phase 6** | **Risk Engine & Decision Framework** | ⚪ Pending | Phase 6 |
| **Phase 7** | **Explainability & Reason Codes** | ⚪ Pending | Phase 7 |
| **Phase 8** | **Fraud Detection API (FastAPI)** | ⚪ Pending | Phase 8 |
| **Phase 9** | **Database & Persistence (PostgreSQL)** | ⚪ Pending | Phase 9 |
| **Phase 10** | **Real-Time Detection & Benchmarking** | ⚪ Pending | Phase 10 |
| **Phase 11** | **Fraud Intelligence Dashboard** | ⚪ Pending | Phase 11 |
| **Phase 12** | **Human Review & Case Management** | ⚪ Pending | Phase 12 |
| **Phase 13** | **ML & Model Monitoring** | ⚪ Pending | Phase 13 |
| **Phase 14** | **MLOps, Retraining & Model Registry** | ⚪ Pending | Phase 14 |
| **Phase 15** | **Security, Auth & Audit Logging** | ⚪ Pending | Phase 15 |
| **Phase 16** | **Containerization & Deployment** | ⚪ Pending | Phase 16 |
| **Phase 17** | **Comprehensive Testing Suite** | ⚪ Pending | Phase 17 |
| **Phase 18** | **Final Portfolio Polish & Runbooks** | ⚪ Pending | Phase 18 |

---

## ✅ Phase 2 Deliverable Checklist

- [x] Architected modular, reproducible EDA pipeline under `ml/eda/` (`overview`, `class_imbalance`, `temporal`, `amount`, `merchant`, `geography`, `behavioral_signals`, `correlations`, `visualize`, `report`, `run_eda`).
- [x] Analyzed 1.85M transactions across independent Train (70%), Validation (15%), and Test (15% OOT) partitions without data modification or lookahead leakage.
- [x] Verified class imbalance dynamics: 0.521% overall fraud rate (1:172 ratio in Train). Documented why accuracy (99.4%) is deceptive and established PR-AUC / Cost-Weighted Loss as primary metrics.
- [x] Discovered key diurnal patterns: Nighttime fraud rate surges to **2.88%** at 22:00 between 22:00 and 03:00 (over 4× daytime baseline).
- [x] Analyzed monetary distribution: Legitimate median is **$47.31** vs. fraudulent median of **$367.61** (7.8× difference). Transactions >$500 exhibit >10% fraud rate.
- [x] Uncovered merchant category risk concentration: `shopping_net`, `misc_net`, `grocery_pos` account for >65% of fraudulent spend.
- [x] Computed cardholder-to-merchant Haversine distance distributions.
- [x] Generated 15 publication-quality matplotlib figures into `docs/eda/figures/`.
- [x] Authored comprehensive markdown report `docs/eda_report.md` with supporting statistics and Phase 3 recommendation matrix.
- [x] Built and passed 18 automated unit tests across `tests/ml/`.

---

## 🏛 Architectural Decision Records (ADRs)

### ADR-001: Modular Monorepo Scaffolding
- **Status**: Accepted.

### ADR-002: Lightweight Phase-by-Phase Dependency Management
- **Status**: Accepted.

### ADR-003: Measurable Non-Functional Requirements (Latency & Throughput)
- **Status**: Accepted.

### ADR-004: Calibrated 0–100 Risk Scoring with Tri-Tier Decision Policy
- **Status**: Accepted.

### ADR-005: Time-Aware Out-of-Time (OOT) Splitting & Zero-Leakage Protocol
- **Status**: Accepted (Phase 1).

### ADR-006: Canonical 15-Field Transaction Schema & Columnar Parquet Storage
- **Status**: Accepted (Phase 1).

### ADR-007: Empirical Fraud Domain Insights & Prioritized Feature Blueprint
- **Context**: Feature engineering in Phase 3 must target empirically verified fraud signals rather than arbitrary data transformations.
- **Decision**: Prioritize 6 core behavioral feature groups for Phase 3 engineering based on Phase 2 statistical evidence:
  1. *Spending Baseline Deviation ($Z$-score)*: Grounded in the 7.8× median amount delta between legitimate and fraud.
  2. *Rolling Transaction Velocity (1h, 6h, 24h)*: Grounded in account-level burst activity.
  3. *Cyclic Temporal Encodings (Sine/Cosine, Night Indicator)*: Grounded in the 4× nighttime fraud surge (22:00–03:00).
  4. *Rolling Spend Aggregations (24h, 7d)*: Grounded in rapid balance depletion patterns.
  5. *Historical Category Risk Index*: Grounded in merchant category fraud prevalence clustering.
  6. *Geographic Distance & Implied Velocity*: Grounded in cardholder POS cluster bounds and impossible speed detection.
- **Status**: Accepted (Phase 2).

---

## ⚠️ Known Constraints & Risk Register

1. **Class Imbalance**: Imbalance ratio remains extreme (0.579% in Train, 0.439% in Val, 0.333% in Test). Modeling must prioritize PR-AUC and cost-weighted loss matrices.
2. **Temporal Distribution Differences**: Slight drop in fraud prevalence over the 2-year simulation horizon is documented as a descriptive distribution difference to be monitored in Phase 13.

---

## ⏭ Next Step: Preparation for Phase 3 (Behavioral Feature Engineering)

When approved to start Phase 3:
- Build stateful rolling window feature pipelines (1h, 6h, 24h, 7d velocity and spending aggregates).
- Implement account-level historical baselines ($Z$-scores) using strictly backward-looking time windows.
- Implement cyclic temporal transforms (sine/cosine hour and day of week).
- Calculate Haversine distance and implied travel speed ($\text{km/h}$) between consecutive transactions.
- Fit all encoders and baseline scalers **strictly on the Train partition** to prevent any data leakage into Validation and Test.
