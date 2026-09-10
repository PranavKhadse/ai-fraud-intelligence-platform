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
- **Explainable AI & Reason Codes**: TreeSHAP feature attributions mapped into human-readable reason codes (e.g., `"HIGH_VELOCITY_1H"`, `"UNUSUAL_GEO_DISTANCE"`, `"TRANSACTION_AMOUNT_SPIKE"`).
- **Production REST API**: High-throughput FastAPI service with strictly typed Pydantic v2 schemas and structured logging.
- **Persistent Data Store**: PostgreSQL database modeling transactions, risk evaluation audits, rules, and human-in-the-loop analyst cases.
- **Analyst Case Management Dashboard**: Modern React + TypeScript UI for real-time risk alerts, fraud investigations, and manual dispositioning.
- **Continuous MLOps & Drift Intelligence**: Automated tracking of data/concept drift, feature distribution shift, model retraining pipelines, and champion/challenger validation.

---

## 🛠 Planned Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Uvicorn |
| **Machine Learning** | NumPy, pandas, scikit-learn, XGBoost, LightGBM, SHAP |
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
| **Phase 0** | **Project Foundation & Architecture** | 🟡 **Active** | Scaffolding, docs (`README`, `PROJECT_SPEC`, `ARCHITECTURE`, `PROJECT_STATUS`), `.gitignore`, `.env.example`, Git initial commit. |
| **Phase 1** | **Data Pipeline & Ingestion** | ⚪ *Upcoming* | Fraud transaction ingestion, synthetic generation, schema validation, data partitioning. |
| **Phase 2** | **Exploratory Data Analysis (EDA)** | ⚪ *Upcoming* | Class imbalance analysis, anomaly distributions, temporal behaviors, domain insight report. |
| **Phase 3** | **Behavioral Feature Engineering** | ⚪ *Upcoming* | Velocity features, rolling statistics, geo/IP distance delta, device consistency metrics. |
| **Phase 4** | **Baseline & Advanced ML Models** | ⚪ *Upcoming* | Logistic Regression, Random Forest, LightGBM, and XGBoost training & cross-validation. |
| **Phase 5** | **Imbalance Handling & Cost Optimization** | ⚪ *Upcoming* | PR-AUC optimization, focal/weighted loss, cost-sensitive threshold matrix tuning. |
| **Phase 6** | **Risk Engine & Decision Framework** | ⚪ *Upcoming* | 0–100 risk score calibration, rule engine overrides, Approve / Review / Block policy engine. |
| **Phase 7** | **Explainability & Reason Codes** | ⚪ *Upcoming* | TreeSHAP feature attributions, local explanation extraction, human-readable reason codes. |
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
   Refer to [PROJECT_STATUS.md](file:///d:/Users/Pranav%20Khadse/Downloads/VIT/Coding/AI-Powered%20Fraud%20Detection%20%26%20Risk%20Intelligence%20Platform/PROJECT_STATUS.md) for the active milestone and [PROJECT_SPEC.md](file:///d:/Users/Pranav%20Khadse/Downloads/VIT/Coding/AI-Powered%20Fraud%20Detection%20%26%20Risk%20Intelligence%20Platform/PROJECT_SPEC.md) for technical requirements.
