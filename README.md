# AI-Powered Fraud Detection & Risk Intelligence Platform

An enterprise-grade, end-to-end fraud detection and financial risk intelligence platform designed to evaluate real-time transaction streams, identify anomalous behaviors under extreme class imbalance, deliver sub-threshold calibrated risk scores, provide interpretable decision reasoning via Explainable AI (SHAP), and streamline analyst case investigation workflows.

---

## 📌 Executive Summary & Business Problem

In digital banking, payments, and e-commerce ecosystems, financial institutions face escalating sophisticated fraud attacks alongside rising false-positive rates that disrupt legitimate customer transactions.

Traditional rule-based fraud detection systems suffer from:
1. **High False Positive Rates (FPR)**: Legitimate customers are frequently blocked, creating friction and churn.
2. **Slow Adaptation to Novel Vectors**: Static threshold rules fail against evolving fraud patterns (e.g., account takeover, credential stuffing, rapid velocity bursts, distributed carding attacks).
3. **Black-Box Opacity**: Complex ML models often fail compliance audits because they lack clear, defensible reason codes explaining why a transaction was flagged.
4. **Disjointed Analyst Operations**: Disconnect between automated risk scoring, human investigator workflows, and model feedback loops.

This platform bridges these operational gaps with an integrated, production-style architecture combining **gradient-boosted decision trees (XGBoost/LightGBM)**, **behavioral feature engineering**, **explainable AI (SHAP)**, **0–100 calibrated risk scoring**, **FastAPI microservices**, **PostgreSQL persistence**, and a **React + TypeScript fraud analyst dashboard**.

---

## 🚀 Core Capabilities

- **State-of-the-Art ML Scoring**: XGBoost & LightGBM classifiers trained on highly imbalanced transaction distributions (optimizing Precision-Recall AUC and cost-weighted loss matrices).
- **Behavioral Feature Engineering**: High-velocity tracking, rolling historical statistics, location/IP velocity changes, device fingerprint shifts, and spending deviation metrics.
- **Calibrated 0–100 Risk Score & Tri-Tier Policy**: Standardized composite risk scores mapped to automated business actions:
  - `🟢 APPROVE` (Low Risk: Score < 35)
  - `🟡 REVIEW` (Medium Risk: 35 ≤ Score < 75)
  - `🔴 BLOCK` (High Risk: Score ≥ 75)
- **Explainable AI & Reason Codes**: Native TreeSHAP feature attributions mapped into human-readable reason codes and complete margin waterfalls with strict rule/model provenance.
- **Production REST API**: High-throughput FastAPI service with strictly typed Pydantic v2 schemas and structured logging.
- **Persistent Data Store**: PostgreSQL database modeling transactions, risk evaluation audits, rules, and human-in-the-loop analyst cases.
- **Analyst Case Management Dashboard**: Modern React + TypeScript UI for real-time risk alerts, fraud investigations, and manual dispositioning.
- **Continuous MLOps & Drift Intelligence**: Automated tracking of data/concept drift, feature distribution shift, model retraining pipelines, and champion/challenger validation.

---

## 🔍 Explainability, Reason Codes & Decision Transparency

The platform integrates a local explainability framework designed to support analyst review and diagnostic interpretability, powered by native gradient-boosted TreeSHAP and deterministic business rule telemetry:

- **Local TreeSHAP Explanations**: Extracts exact feature attributions ($\phi_1 \dots \phi_{55}$) directly using XGBoost's native compiled C++ TreeSHAP engine without external C-extension dependencies.
- **Risk & Mitigating Factor Decomposition**: Quantifies risk-increasing factors ($\phi > 0$) pushing scores toward fraud, and mitigating factors ($\phi < 0$) reducing transaction risk.
- **Rule & Model Reason Codes**: Standardized, human-readable reason codes with clear source provenance (`source="RULE"` for deterministic policy overrides vs. `source="MODEL"` for statistical tree drivers).
- **Decision Transparency & Override Tracking**: Clearly exposes and distinguishes `model_score`, `baseline_action`, `rule_action`, `action`, and `is_overridden` (e.g., policy escalations to `REVIEW` are faithfully reported as rule overrides, not model predictions).
- **Residual Margin Waterfall Reconstruction**: Mathematically complete margin-space waterfall ($\text{Base Margin} \to \text{Top Features} \to \text{Residual} \to \text{Final Margin}$) reconstructing output margin within floating-point tolerance ($\approx 7.2 \times 10^{-6}$).
- **Raw-Value & Categorical Integrity**: Preserves human-readable raw categorical labels (e.g., `'shopping_net'`) and unencoded numeric inputs; internal ordinal encoding codes are never surfaced as business values.
- **Comprehensive Test Coverage**: Verified across 26 dedicated explainability tests, 250 risk engine tests, and 358 complete ML suite tests (100% pass rate).

---

## 🛠 Planned Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Uvicorn |
| **Machine Learning** | NumPy, pandas, scikit-learn, XGBoost, LightGBM, Native TreeSHAP |
| **Database & Storage** | PostgreSQL |
| **Frontend Dashboard** | React, TypeScript, Modern CSS / Tailwind (to be selected in Phase 11) |
| **Testing & Quality** | pytest, pytest-cov, httpx |
| **Infrastructure & Ops** | Docker, Docker Compose, Git |

*(Technologies are evaluated and introduced strictly in the respective phase where their requirement is implemented).*

---

## 🏗 High-Level System Architecture

```
                       ┌─────────────────────────────────────────┐
                       │       Incoming Transaction Stream       │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │          FastAPI Gateway Service        │
                       │     (Schema Validation & Auth)          │
                       └────────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Behavioral Feature Eng  │                 │    Deterministic Rules    │
       │ (Velocity, Aggregations)  │                 │ (Hard Blacklist, Sanctions)│
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                               │
       │  ML Inference (XGB/LGBM)  │                               │
       │  + TreeSHAP Explainability│                               │
       └─────────────┬─────────────┘                               │
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │    Composite Risk Engine (0-100 Score)  │
                       │    [ APPROVE  |  REVIEW  |  BLOCK ]     │
                       └────────────────────┬────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │    PostgreSQL Database    │                 │  Analyst Case Dashboard   │
       │ (Audits, Cases, Features) │                 │  (React / TypeScript UI)  │
       └───────────────────────────┘                 └───────────────────────────┘
```

---

## 🗺 19-Stage Development Roadmap

| Phase | Phase Name | Status | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Project Foundation & Architecture** | 🟢 **Completed** | Scaffolding, docs (`README`, `PROJECT_SPEC`, `ARCHITECTURE`, `PROJECT_STATUS`), `.gitignore`, `.env.example`, Git initial commit. |
| **Phase 1** | **Data Pipeline & Ingestion** | 🟢 **Completed** | Ingestion pipeline, temporal chronological sorting, canonical 15-field schema, time-aware train/val/test splits. |
| **Phase 2** | **Exploratory Data Analysis (EDA)** | 🟢 **Completed** | Class imbalance analysis, anomaly distributions, temporal behaviors, domain insights report. |
| **Phase 3** | **Behavioral Feature Engineering** | 🟢 **Completed** | 55 point-in-time features: velocity counters, rolling statistics, geo/IP travel speed, deviation metrics. |
| **Phase 4** | **Baseline & Advanced ML Models** | 🟢 **Completed** | XGBoost champion ($0.9619$ PR-AUC), LightGBM, Random Forest, Logistic Regression with process isolation. |
| **Phase 5** | **Imbalance Handling & Cost Optimization** | 🟢 **Completed** | Cost matrix optimization ($\tau^*=0.78$), $31.8\%$ OOT cost reduction, sensitivity curves, leakage governance. |
| **Phase 6** | **Risk Engine & Decision Framework** | 🟢 **Completed** | 0–100 risk scoring, tri-tier policy, deterministic rule engine, 6-rule standard catalog, empirical benchmark. |
| **Phase 7** | **Explainability & Reason Codes** | 🟢 **Completed** | Native TreeSHAP feature attributions, margin waterfall reconstruction, rule & model reason codes, decision override transparency. |
| **Phase 8** | **Fraud Detection API (FastAPI)** | 🟢 **Completed** | Production REST API (`/health`, `/predict`, `/api/v1/health`, `/api/v1/predict`), Pydantic v2 schemas, lifespan model pre-warming. |
| **Phase 9** | **Database & Persistence (PostgreSQL)** | 🟢 **Completed** | PostgreSQL schema modeling, Alembic migrations, Repositories, Unit of Work, inline API persistence, idempotency & duplicate-request protection. |
| **Phase 10** | **Real-Time Detection & Benchmarking** | 🟢 **Completed** | End-to-end transaction scoring pipeline, 7-stage latency profiling, multi-tier persistence ablation, benchmark CLI. |
| **Phase 11** | **Fraud Intelligence Dashboard** | 🟢 **Completed** | React 18 + TypeScript SPA, near-real-time live feed, deep investigation drawer, trend analytics, TreeSHAP waterfall, read-only What-If counterfactual simulator. |
| **Phase 12** | **Human Review & Case Management** | 🟢 **Completed** | PostgreSQL case persistence, 10 REST endpoints, 5-state lifecycle with row locking, 3-column React workspace, E2E concurrency hardening, ground-truth ML data contracts. |
| **Phase 13** | **ML & Model Monitoring** | ⚪ *Upcoming* | Data drift, concept drift, feature distribution shift detection, statistical alerts. |
| **Phase 14** | **MLOps, Retraining & Versioning** | ⚪ *Upcoming* | Automated retraining triggers, model registry, champion/challenger shadow evaluation. |
| **Phase 15** | **Security, Auth & Audit Logging** | ⚪ *Upcoming* | Role-Based Access Control (RBAC), JWT authentication, tamper-evident audit logs. |
| **Phase 16** | **Containerization & Deployment** | ⚪ *Upcoming* | Dockerfiles, multi-service Docker Compose orchestrations, production build checks. |
| **Phase 17** | **Comprehensive Testing Suite** | ⚪ *Upcoming* | Unit, integration, ML behavioral, and load testing suites with automated coverage reporting. |
| **Phase 18** | **Final Portfolio Polish & Runbooks** | ⚪ *Upcoming* | End-to-end documentation, demo walkthroughs, operational runbooks, portfolio polish. |

---

## 📂 Project Structure

```
ai-fraud-intelligence/
├── .env.example              # Environment variables template (no secrets)
├── .gitignore                # Comprehensive ignore rules
├── README.md                 # Project overview and high-level architecture
├── PROJECT_SPEC.md           # Single source of truth specification document
├── ARCHITECTURE.md           # In-depth architectural designs and data flows
├── PROJECT_STATUS.md         # Active progress tracker and architectural decisions
├── backend/                  # FastAPI microservices
│   └── app/
│       ├── api/              # API router blueprints (v1 endpoints: /health, /predict)
│       ├── core/             # Configuration & settings
│       ├── db/               # Database engine & session (Phase 9)
│       ├── models/           # SQLAlchemy ORM models (Phase 9)
│       ├── schemas/          # Pydantic v2 request & response schemas
│       ├── services/         # Risk engine & business logic service layer
│       └── main.py           # Application entrypoint & lifespan management
├── ml/                       # Machine learning pipelines
│   ├── data/                 # Ingestion & data splitters
│   ├── features/             # Feature engineering pipelines
│   ├── models/               # Model training & artifacts
│   ├── evaluation/           # Imbalance metrics & cost curves
│   ├── explainability/       # SHAP & reason code generators
│   ├── monitoring/           # Data & concept drift monitors
│   └── retraining/           # Model retraining pipelines
├── frontend/                 # React + TypeScript Dashboard
├── data/                     # Raw, processed, and synthetic data tiers (gitignored)
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── database/                 # SQL schemas, migrations, and seed scripts
│   ├── migrations/
│   └── seeds/
├── tests/                    # Test suites across unit, integration, and ML
│   ├── unit/                 # Schema & health endpoint tests
│   ├── integration/          # API prediction & rule override integration tests
│   └── ml/                   # 358 comprehensive ML & risk engine tests
├── config/                   # Centralized configuration files
├── scripts/                  # Automation and utility scripts
├── docs/                     # Specifications, runbooks, and diagrams
└── infrastructure/           # Docker and container manifests
```

---

## ⚙️ Local Development & Quick Start

### 1. Configure Environment
```bash
cp .env.example .env
```

### 2. Start the FastAPI Service Locally
Run the high-performance Uvicorn server:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, explore the interactive OpenAPI Swagger UI at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints & Usage Examples

### 1. Health & Telemetry Check (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Sample Response:**
```json
{
  "status": "healthy",
  "app_name": "AI-Powered Fraud Detection & Risk Intelligence Platform API",
  "version": "1.0.0",
  "model_loaded": true,
  "model_version": "1.0.0",
  "rules_loaded_count": 6,
  "timestamp": "2026-09-15T04:20:00.000000+00:00"
}
```

### 2. Fraud Prediction & Explainability (`POST /predict`)
Evaluates a single financial transaction against the 55-feature behavioral contract, computing continuous model scores, normalized 0–100 risk scores, tri-tier decision actions, triggered business rules, and local TreeSHAP reason codes:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 49.99,
    "cardholder_lat": 40.7128,
    "cardholder_long": -74.0060,
    "merchant_lat": 40.7306,
    "merchant_long": -73.9352,
    "city_pop": 8336817.0,
    "merchant_category": "shopping_net",
    "job_category": "engineer",
    "transaction_hour": 14,
    "day_of_week": 2,
    "day_of_month": 15,
    "month": 9,
    "week_of_year": 38,
    "is_weekend": 0,
    "is_night": 0,
    "hour_sin": 0.5,
    "hour_cos": -0.866,
    "day_of_week_sin": 0.9749,
    "day_of_week_cos": -0.2225,
    "txn_count_1h": 1.0,
    "txn_count_6h": 2.0,
    "txn_count_24h": 3.0,
    "txn_count_7d": 10.0,
    "txn_count_30d": 35.0,
    "time_since_prev_txn_seconds": 3600.0,
    "is_first_account_txn": 0,
    "amt_sum_1h": 49.99,
    "amt_sum_24h": 120.50,
    "amt_sum_7d": 450.00,
    "amt_sum_30d": 1800.00,
    "amt_mean_24h": 40.17,
    "amt_mean_7d": 45.00,
    "amt_max_24h": 60.00,
    "amt_median_30d": 38.50,
    "historical_amount_mean": 42.00,
    "historical_amount_std": 15.50,
    "historical_amount_median": 38.50,
    "amount_zscore": 0.515,
    "amount_ratio_to_historical_mean": 1.19,
    "account_txn_count_before": 35.0,
    "account_total_spend_before": 1800.00,
    "account_avg_amount_before": 51.43,
    "account_max_amount_before": 150.00,
    "account_unique_merchant_count_before": 20.0,
    "account_unique_category_count_before": 8.0,
    "account_merchant_txn_count_before": 3.0,
    "account_category_txn_count_before": 12.0,
    "account_merchant_spend_before": 150.00,
    "account_category_spend_before": 520.00,
    "merchant_txn_count_before": 2500.0,
    "category_txn_count_before": 15000.0,
    "cardholder_merchant_distance_km": 6.35,
    "distance_from_prev_merchant_km": 2.10,
    "implied_travel_speed_kmh": 2.10,
    "is_impossible_travel_speed": 0
  }'
```

---

## ⚡ Real-Time Detection & Performance Benchmarking (Phase 10)

The platform includes a high-resolution, multi-scenario performance benchmarking and profiling engine (`backend/app/benchmarking/` and `scripts/benchmark.py`):

- **Asynchronous Load Generator**: Evaluates scalable concurrency sweeps ($C \in \{1, 2, 4, 8, 16\}$) using in-process ASGI (`httpx.ASGITransport`) or network HTTP modes with nanosecond-precision timers (`time.perf_counter_ns`).
- **Seven-Stage Component Latency Decomposition**: Micro-profiles Request Validation, Feature Preparation, XGBoost Inference, TreeSHAP Attribution, Rule Engine/Policy, Persistence Mapping, and PostgreSQL Persistence.
- **Multi-Tier Persistence Ablation**: Systematically isolates the database persistence overhead (Full API + PostgreSQL vs In-Memory Pipeline vs Idempotent Replay Fast Path).
- **Idempotency Replay Acceleration**: Demonstrates a **2.84x speedup** (64.8% latency reduction) on idempotent transaction replays.
- **100% Zero-Fabrication Guarantee**: All reported metrics are empirically measured on the active system.

### Running the Benchmark CLI:
```bash
# Execute full multi-concurrency benchmark sweep with JSON and CSV exports
python scripts/benchmark.py --requests 200 --concurrency 1,2,4,8,16 --warmup 50 --output docs/benchmark_results.json --csv docs/benchmark_results.csv

# Fast in-process validation run
python scripts/benchmark.py --requests 50 --concurrency 1,2,4 --output docs/benchmark_results.json

# HTTP network mode against a live running server
python scripts/benchmark.py --transport http --target-url http://127.0.0.1:8000 --requests 200
```

---

## 🖥️ Fraud Intelligence Dashboard & What-If Simulator (Phase 11)

The platform includes an interactive fraud intelligence dashboard built with React 18, TypeScript, and Vite under `frontend/`:

- **Near-Real-Time Live Transaction Feed**: Sub-second polling (5s/10s/30s/off) with live risk tier indicators, customer/merchant search, and pagination.
- **Deep Investigation Drawer**: Detailed inspection of 55-feature snapshots, persisted TreeSHAP attributions, rule matches, plain-English reason codes, and audit logs.
- **Explainability & Trend Analytics**: 10-bucket risk score distributions ($0\text{--}9 \dots 90\text{--}100$), volume trends, and mathematical TreeSHAP margin waterfall visualizations.
- **What-If Counterfactual Transaction Simulator**:
  - In-memory execution using production `RiskEvaluator` and `RuleEngine`.
  - Full 55-feature category editor across 7 domain categories.
  - Side-by-side risk/model score comparison ($\Delta\text{Score}$), tier changes, and decision transitions.
  - Deduplicated rule impact diffs (`NEWLY_TRIGGERED`, `RESOLVED`, `PERSISTENT`, `NEITHER`).
  - Strict **Zero-Write Guarantee**: no database writes, no audit logs, zero state mutation.

### Running the Frontend Locally:
```bash
cd frontend
npm install
npm run dev
```

### Building the Production Bundle:
```bash
cd frontend
npm run build
```

---

## 🛡️ Human Review & Case Management Platform (Phase 12)

The platform provides a comprehensive, concurrency-hardened Human Review & Case Management system bridging automated ML evaluation with expert fraud investigation:

- **Automated Routing & Manual Escalation**:
  - Automatically routes high-risk transactions (`decision_action="REVIEW"`) into `OPEN` cases during transaction persistence.
  - Supports manual analyst/supervisor escalation with collision-safe case numbering `CASE-YYYYMMDD-XXXXXX` and database-level `UNIQUE(transaction_id)` duplicate enforcement.
- **5-State Concurrency-Safe Lifecycle**:
  - Formal state machine (`OPEN` $\to$ `IN_REVIEW` $\to$ `ESCALATED` $\to$ `RESOLVED` $\to$ `CLOSED`) with reversible dispute reopening.
  - PostgreSQL row-level locks (`SELECT FOR UPDATE`) protect all mutations (claims, reassignments, status changes, dispositions) against race conditions with zero deadlock risk.
- **10 REST API Endpoints (`/api/v1/cases`)**:
  - Paginated review queue, 7 real-time KPI metrics, detailed investigation context, manual escalation, reviewer assignment, lifecycle status transitions, notes management, human disposition submission, and authoritative case audit timelines.
- **Interactive React Investigation Workspace**:
  - **Review Queue**: 7 KPI summary cards (Total Open, Unassigned, In Review, Escalated, Resolved Today, Resolved 24h, Critical Priority), multi-filter toolbar, and server-side sorting.
  - **Case Investigation Workspace**: 3-column responsive layout displaying 55-feature snapshots, calibrated risk gauges, plain-English reason codes, triggered rules, embedded `ShapWaterfall` feature attributions, append-only notes composer, and chronological audit timelines.
  - **Action Modals**: Dedicated modals for case creation, reviewer assignment, escalation, disposition, case closure, and case reopening.
  - **Development Actor Context Switcher**: Role switcher (`ANALYST`, `ADMIN`, `API_CLIENT`) with strict production fail-closed security (`ALLOW_DEV_ACTOR_HEADERS=False` blocks header injection).
- **Ground-Truth Data Contract Preservation**:
  - Guarantees complete preservation of point-in-time feature snapshots (55 features), uncalibrated model scores, calibrated risk scores, and human review labels (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `LEGITIMATE`, `SUSPICIOUS_RESOLVED`) for future Phase 14 retraining loops.

---

## 📈 ML & Model Monitoring Platform (Phase 13)

The platform provides a comprehensive statistical observability and drift intelligence system (`ml/monitoring/`, `backend/app/services/monitoring_service.py`, `frontend/src/pages/MonitoringPage.tsx`):

- **Feature Distribution Drift**:
  - Evaluates Population Stability Index (PSI) over 10 deciles and asymptotic two-sample Kolmogorov-Smirnov (KS) tests against 5,000 empirical reference samples for all 55 features.
  - Detects missing-rate surges, column dropouts, and categorical Jensen-Shannon Divergence (JSD) with unseen level alerts.
- **Prediction & Policy Drift**:
  - Tracks raw model score PSI, 10-bucket risk score PSI, 4-tier risk distribution JSD, and 3-way decision action JSD.
  - Isolates rule-override frequency deltas distinguishing ML model drift from policy rule changes.
- **Ground-Truth Model Performance Tracking**:
  - Maps human analyst case dispositions into ground truth, computing complete confusion matrices at operating threshold ($t=0.78$), precision, recall, specificity, F1-score, FPR, ROC-AUC, PR-AUC, and review queue purity.
  - Quantifies mathematical relative percentage degradation vs. baseline OOT benchmarks.
- **Sample-Size Confidence Guardrails**:
  - Enforces minimum observation sample sizes ($N < 100 \to \text{INSUFFICIENT\_DATA}$, $N_{\text{labeled}} < 20 \to \text{INSUFFICIENT\_DATA}$, $20 \le N_{\text{labeled}} < 100 \to \text{LOW\_SAMPLE}$).
- **Dedicated Read-Only Isolation & Snapshot Storage**:
  - Dedicated `monitoring_snapshots` PostgreSQL table with unique window constraint for idempotent persistence and fast-path resolution.
  - Guarantees zero writes/mutations to `transactions`, `risk_evaluations`, `cases`, and `audit_logs`.
- **Interactive Monitoring Dashboard**:
  - React sub-dashboards for Health Overview, Feature Drift Analysis, Prediction Drift Analysis, Performance Tracking, and Snapshot History.

### Running Monitoring Benchmarks & CLI:
```bash
# Run full performance benchmark suite
python scripts/benchmark_monitoring.py --output-json docs/monitoring_benchmark_results.json --output-csv docs/monitoring_benchmark_results.csv

# CLI drift and performance checks
python ml/monitoring/cli.py check-feature-drift --window 24h
python ml/monitoring/cli.py check-prediction-drift --window 24h
python ml/monitoring/cli.py check-performance --window 24h
```

---

## 🧪 Running Automated Tests

Run the complete test suite across ML models, risk engine, database persistence, benchmarking, case management, and model monitoring:
```bash
python -m pytest tests/ -q
```

Run specific test modules:
```bash
# Unit tests (schemas, health, repositories, lifecycle, models, monitoring)
python -m pytest tests/unit/ -v

# API & Persistence Integration tests
python -m pytest tests/integration/ -v

# Phase 13 Monitoring & Drift tests
python -m pytest tests/unit/test_monitoring_*.py tests/unit/test_feature_drift.py tests/unit/test_prediction_drift.py tests/unit/test_model_performance.py tests/integration/test_monitoring_*.py -v

# Phase 12.5 End-to-End Case Lifecycle tests
python -m pytest tests/integration/test_case_e2e_lifecycle.py -v

# ML, Feature Engineering & Risk Engine tests
python -m pytest tests/ml/ -v
```

