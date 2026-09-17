# Phase 11 — Fraud Intelligence Dashboard & What-If Simulator Report

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Phase:** 11 (Fraud Intelligence Dashboard)  
> **Status:** 🟢 **Completed & Fully Verified**  
> **Increments:** 11.1, 11.2, 11.3, 11.4, 11.5  
> **Zero-Write Guarantee:** Verified (No DB writes, no audit logs, no persistence calls during simulation)  

---

## 1. Executive Summary

Phase 11 delivers a production-grade, responsive **Fraud Intelligence & Risk Analysis Dashboard** built on modern **React 18 + TypeScript + Vite**, powered by an asynchronous **FastAPI backend (PostgreSQL + SQLAlchemy 2.0)**. 

The dashboard enables fraud analysts, risk officers, and data scientists to:
1. **Monitor Live Risk Stream**: View near-real-time streaming transaction feeds with sub-second polling, tier filters, time windows, and instant KPI metrics.
2. **Investigate Transactions Deeply**: Slide-over drawer inspection of transaction metadata, 55-feature point-in-time snapshots, persisted TreeSHAP attributions, rule matches, plain-English reason codes, and complete tamper-evident audit logs.
3. **Analyze Trends & Explainability**: 10-bucket risk score distributions, hourly/daily transaction volumes, top-triggered business rules, and high-resolution TreeSHAP margin waterfall charts.
4. **Simulate What-If Counterfactuals (Increment 11.5)**: Interactive sandbox to modify any of the 55 canonical features across 7 categories, evaluate in-memory ML + TreeSHAP models against production rules, and view itemized rule diffs and SHAP shifts with a **strict zero-write guarantee**.

---

## 2. Architecture & Component Breakdown

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   REACT 18 + TYPESCRIPT SPA                                     │
 │  ┌─────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  ┌──────────────────────┐  │
 │  │  Overview Feed  │  │ Transaction Drawer  │  │  Trend Analytics  │  │   What-If Simulator  │  │
 │  │  (Live Polling) │  │  (Deep Inspection)  │  │  (Charts & SHAP)  │  │  (Counterfactuals)   │  │
 │  └────────┬────────┘  └──────────┬──────────┘  └─────────┬─────────┘  └──────────┬───────────┘  │
 └───────────┼──────────────────────┼───────────────────────┼───────────────────────┼──────────────┘
             │                      │                       │                       │
             ▼                      ▼                       ▼                       ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                      FASTAPI REST API LAYER                                     │
 │   GET /dashboard/overview    GET /dashboard/tx/:id     GET /dashboard/analytics     POST /simulate│
 └───────────┬──────────────────────┬───────────────────────┬───────────────────────┬──────────────┘
             │                      │                       │                       │
             │                      │                       │        ┌──────────────┴──────────────┐
             ▼                      ▼                       ▼        ▼                             │
 ┌─────────────────────────────────────────────────────────────┐ ┌───────────────────────────────┐ │
 │            PostgreSQL Database (Read-Only Path)             │ │ In-Memory ML Scoring Engine   │ │
 │  - transactions            - evaluation_reason_codes        │ │ - RiskEvaluator (Champion XGB)│ │
 │  - risk_evaluations        - evaluation_feature_attributions│ │ - RuleEngine (6-Rule Catalog) │ │
 │  - evaluation_rule_matches - audit_logs                     │ │ - Native C++ TreeSHAP Engine  │ │
 └─────────────────────────────────────────────────────────────┘ └───────────────────────────────┘ │
                                                                 │      ZERO-WRITE GUARANTEE     │ │
                                                                 │ (No INSERT / UPDATE / DELETE) │ │
                                                                 └───────────────────────────────┘ │
```

---

## 3. Increment Deliverables Summary

### Increment 11.1: Dashboard Foundation & UI Scaffolding
- Modular React 18 + TypeScript + Vite project structure under `frontend/`.
- Modern dark-mode design system with CSS custom properties (`#0B0F19` canvas, `#111827` cards, cyan/emerald/amber/rose risk badges).
- High-level overview endpoint `GET /api/v1/dashboard/overview` providing 24h summary metrics (total count, approval rate, block rate, review rate, average risk score, fraud loss avoided).

### Increment 11.2: Near-Real-Time Risk Intelligence & Live Feed
- Filterable transaction feed with auto-polling toggle (5s / 10s / 30s / off) with near-real-time badge indicator.
- Search by transaction ID, customer ID, or merchant category with risk tier filtering (`ALL`, `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- Responsive pagination and dynamic sorting.

### Increment 11.3: Transaction Investigation & Deep Inspection
- Slide-over `TransactionDrawer` displaying:
  - Header with risk score meter, status badges, and rule override indicators.
  - Formatted currency, location travel speeds, timestamp formatting.
  - Canonical 55-feature point-in-time snapshot categorized into 7 domain groups.
  - Persisted TreeSHAP feature attributions with direction and magnitude bars.
  - Triggered business rules with severity and priority.
  - Complete chronological audit log trail with actor metadata.

### Increment 11.4: Explainability Visualizer & Trend Analytics
- Dedicated `AnalyticsView` with:
  - 10-bucket risk score distribution histogram ($0\text{--}9, 10\text{--}19, \dots, 90\text{--}100$) correctly handling boundary $100$.
  - Hourly & daily volume trend charts with approval/review/block composition.
  - Top-triggered business rules distribution.
  - Interactive TreeSHAP `ShapWaterfall` chart faithfully visualizing base log-odds margin ($\phi_0 \approx 0.2437$), individual feature attributions, residual delta, and final model score.

### Increment 11.5: What-If Transaction Simulator & Zero-Write Hardening
- Canonical 55-feature counterfactual editor grouped across 7 categories:
  1. Transaction Core (`amount`, `transaction_hour`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_weekend`, `is_night`)
  2. Spatiotemporal & Velocity (`hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `cardholder_lat`, `cardholder_long`, `merchant_lat`, `merchant_long`, `city_pop`)
  3. Short-Term Velocity (`txn_count_1h`, `txn_count_6h`, `txn_count_24h`, `txn_count_7d`, `txn_count_30d`, `time_since_prev_txn_seconds`, `is_first_account_txn`)
  4. Spending Velocity (`amt_sum_1h`, `amt_sum_24h`, `amt_sum_7d`, `amt_sum_30d`, `amt_mean_24h`, `amt_mean_7d`, `amt_max_24h`, `amt_median_30d`)
  5. Behavioral Deviations (`historical_amount_mean`, `historical_amount_std`, `historical_amount_median`, `amount_zscore`, `amount_ratio_to_historical_mean`)
  6. Merchant & Category Histories (`account_txn_count_before`, `account_total_spend_before`, `account_avg_amount_before`, `account_max_amount_before`, `account_unique_merchant_count_before`, `account_unique_category_count_before`, `account_merchant_txn_count_before`, `account_category_txn_count_before`, `account_merchant_spend_before`, `account_category_spend_before`, `merchant_txn_count_before`, `category_txn_count_before`, `merchant_category`, `job_category`)
  7. Geospatial Travel Speed (`cardholder_merchant_distance_km`, `distance_from_prev_merchant_km`, `implied_travel_speed_kmh`, `is_impossible_travel_speed`)
- **Interactive Sandbox Features**:
  - Live modified-field indicators, category modified count badges, and one-click category resets.
  - 4 Demo Preset Scenarios (Baseline Low-Risk, High Velocity Surge, Impossible Travel Spike, Excessive Amount Spike).
  - Seamless "Simulate in What-If" transition from `TransactionDrawer` pre-populating baseline transaction snapshot.
  - Side-by-side comparison summary card displaying $\Delta\text{Risk Score}$, $\Delta\text{Model Score}$, tier changes, and action changes.
  - Deduplicated Rule Impact Analysis classifying rules into `NEWLY_TRIGGERED`, `RESOLVED`, `PERSISTENT`, or `NEITHER`.
  - Itemized feature delta breakdown table.
  - Live `ShapWaterfall` render of simulated TreeSHAP log-odds contributions.
  - Prominent "SIMULATION SCENARIO — READ-ONLY / NOT PERSISTED" watermark banner.

---

## 4. Zero-Write Guarantee & State Governance

The What-If Simulator enforces a strict zero-mutation policy:
- **No Database Persistence**: `POST /api/v1/dashboard/simulate` executes entirely in-memory using the singleton `RiskService` and `RiskEvaluator`.
- **No Side Effects**: Never calls `FraudPersistenceService`, never issues `INSERT`, `UPDATE`, or `DELETE`, and creates no audit logs.
- **Read-Only Baseline Lookup**: Baseline references (by UUID or external transaction ID) use read-only queries with stored risk evaluations; persisted data is never modified or overwritten.
- **Verification**: Verified via dedicated integration tests asserting identical table row counts and field-level entity invariance across all database tables before and after simulation runs.

---

## 5. Automated Verification & Test Results

- **Backend Pytest Suite**: 100% pass rate across unit, integration, and ML suites.
- **Frontend TypeScript/Vite**: Zero type errors (`tsc -b`) and successful production bundle build (`npm run build`).
- **Phase 10 Integrity**: All benchmarking suites and latency SLAs remain intact.
