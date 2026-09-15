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
| **Phase 9** | **Database & Persistence (PostgreSQL)** | ⚪ *Upcoming* | PostgreSQL schema modeling, migrations, transaction & audit logging, case records. |
| **Phase 10** | **Real-Time Detection & Benchmarking** | ⚪ *Upcoming* | End-to-end transaction scoring pipeline, latency profiling & throughput benchmarking. |
| **Phase 11** | **Fraud Intelligence Dashboard** | ⚪ *Upcoming* | React + TypeScript web app, real-time alerts feed, risk distribution charts, audit views. |
| **Phase 12** | **Human Review & Case Management** | ⚪ *Upcoming* | Analyst review queue, manual confirmation/dismissal workflow, feedback capture. |
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

## 🧪 Running Automated Tests

Run the complete test suite across ML models, risk engine, and FastAPI REST endpoints:
```bash
python -m pytest tests/
```

Run specific test modules:
```bash
# Unit tests (schemas & health endpoint)
python -m pytest tests/unit/

# API Integration tests (/predict, rule overrides, error handling)
python -m pytest tests/integration/

# ML & Risk Engine test suite
python -m pytest tests/ml/
```
