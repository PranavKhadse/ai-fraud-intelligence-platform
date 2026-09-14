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
| **Phase 8** | **Fraud Detection API (FastAPI)** | ⚪ *Upcoming* | Production REST API endpoints (`/evaluate`, `/score`, `/health`), Pydantic validation schemas. |
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
│       ├── api/              # API router blueprints
│       ├── core/             # Configuration & security
│       ├── db/               # Database engine & session
│       ├── models/           # SQLAlchemy ORM models
│       ├── schemas/          # Pydantic schemas
│       └── services/         # Risk engine & business logic
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
│   ├── unit/
│   ├── integration/
│   └── ml/
├── config/                   # Centralized configuration files
├── scripts/                  # Automation and utility scripts
├── docs/                     # Specifications, runbooks, and diagrams
└── infrastructure/           # Docker and container manifests
```

---

## ⚙️ Future Setup Instructions

> [!NOTE]
> Phase 0 is foundational. Heavy dependencies and application runtime code will be installed incrementally in subsequent phases.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "AI-Powered Fraud Detection & Risk Intelligence Platform"
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   ```

3. **Follow the Phased Implementation**:
   Refer to [PROJECT_STATUS.md](PROJECT_STATUS.md) for the active milestone and [PROJECT_SPEC.md](PROJECT_SPEC.md) for technical requirements.
