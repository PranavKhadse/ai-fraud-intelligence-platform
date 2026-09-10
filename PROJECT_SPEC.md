# PROJECT_SPEC.md — AI-Powered Fraud Detection & Risk Intelligence Platform

> **Status:** Authoritative Single Source of Truth  
> **Current Revision:** Phase 0 (Foundation & Architecture)  
> **Author:** Lead Software Architect & Senior ML Engineer  

---

## 1. Project Objective & Vision

The objective of this project is to architect, build, and deploy an enterprise-grade, end-to-end **AI-Powered Fraud Detection and Risk Intelligence Platform**. The platform processes financial transactions, applies advanced behavioral feature extraction, performs gradient-boosted machine learning inference under severe class imbalance, calculates a calibrated 0–100 composite risk score, produces explainable decision reasoning via SHAP feature attributions, and provides a modern analyst case management interface backed by a robust REST API and relational database.

---

## 2. Business Problem & Target Personas

### 2.1 The Business Problem
Financial fraud imposes multi-billion-dollar losses across global payment networks. However, naive fraud defenses often result in high False Positive Rates (FPR), where legitimate cardholders are mistakenly declined, leading to cart abandonment and customer churn. 

A high-performing risk platform must balance two opposing costs:
1. **Cost of False Negatives (FN)**: Uncaught fraud leading to direct financial chargebacks and reputational damage.
2. **Cost of False Positives (FP)**: Legitimate user friction, customer support load, and lost transaction revenue.

### 2.2 Target Personas
- **Fraud Operations Analysts**: Review flagged suspicious transactions, inspect human-readable reason codes and SHAP waterwheel charts, and execute disposition decisions (`Confirm Fraud` vs. `Dismiss / Whitelist`).
- **Risk Policy Managers**: Define business threshold boundaries (e.g., Approve vs. Review vs. Block), establish hard compliance rules (sanctions, blacklists), and monitor aggregate fraud rates.
- **Machine Learning & MLOps Engineers**: Monitor feature distribution shifts, track model performance over time, detect data/concept drift, and manage automated candidate model retraining.
- **Upstream Transaction Systems / Merchants**: Consume sub-threshold decision APIs to authorize, hold, or reject payments.

---

## 3. Functional Requirements

### 3.1 Transaction Ingestion & Validation
- Accept structured transaction payloads (User ID, Card/Account Token, Amount, Currency, Timestamp, Merchant Category Code [MCC], Geo Coordinates / Country, IP Address, Device Fingerprint).
- Strict schema validation using Pydantic models with descriptive error responses for malformed payloads.

### 3.2 Behavioral Feature Engineering
- **Velocity Tracking**: Number and dollar volume of transactions by user/card in rolling windows ($1\text{h}$, $6\text{h}$, $24\text{h}$, $7\text{d}$).
- **Historical Spending Baselines**: Deviation of current transaction amount from user’s 30-day mean and standard deviation ($Z\text{-score}$).
- **Geo & IP Velocity / Anomaly**: Distance delta and implied travel speed ($\text{km/h}$) between consecutive transactions from the same account; detection of foreign/high-risk jurisdictions.
- **Device & Channel Consistency**: Identification of new device fingerprints, user-agent changes, or unexpected payment channels.

### 3.3 Hybrid Risk Scoring Engine
- **Composite 0–100 Risk Score**: Transform calibrated model fraud probability into an intuitive 0–100 scale.
- **Rule Engine Overrides**: Deterministic rules (e.g., hard blacklist, sanctioned entity, single transaction > hard threshold) that can override or escalate ML predictions.
- **Tri-Tier Decision Policies**:
  - `APPROVE` (Score $< 35$): Transaction clears automatically.
  - `REVIEW` (Score $35 \le \text{Score} < 75$): Transaction flagged and routed to the Human Investigation Case Queue.
  - `BLOCK` (Score $\ge 75$): Transaction automatically declined with immediate risk logging.

### 3.4 Explainable AI & Reason Codes
- Compute local feature attributions using TreeSHAP for each scored transaction.
- Map top positive SHAP contributions into human-understandable reason codes (e.g., `REASON_HIGH_VELOCITY_1H`, `REASON_UNUSUAL_GEO_DISTANCE`, `REASON_AMOUNT_OUTLIER`).
- Expose feature importance weights in API responses and investigator dashboard views.

### 3.5 Human Review & Case Management
- Auto-generate investigation cases for transactions resulting in `REVIEW` decisions or rule-triggered escalations.
- Support analyst workflows: view transaction details, historical account activity, SHAP feature waterfall, assign case, add investigator notes, and record final disposition.
- Store analyst dispositions as ground-truth feedback labels for future model retraining cycles.

---

## 4. Machine Learning Requirements

### 4.1 Algorithms & Modeling
- Baseline models: Logistic Regression, Random Forest.
- Production models: **XGBoost** and **LightGBM** Gradient Boosted Decision Trees.
- Calibration: Isotonic Regression or Platt Scaling to ensure predicted probabilities match true empirical event frequencies.

### 4.2 Handling Severe Class Imbalance
- Fraud datasets typically exhibit extreme imbalance (often $< 1\%$ positive fraud rate).
- Implement cost-sensitive learning techniques:
  - Class weighting / scale-pos-weight parameter tuning.
  - Evaluation primarily driven by **PR-AUC (Precision-Recall AUC)**, Cost-Sensitive Utility Matrices, and False Positive Rate at high True Positive Rate (e.g., $\text{TPR} \ge 90\%$), rather than misleading accuracy or raw ROC-AUC.

### 4.3 Data Leakage Prevention
- Strict time-series / temporal train-validation-test splitting to prevent lookahead bias.
- Stateful feature engineering calculated only using historical information strictly preceding the transaction timestamp.

---

## 5. Backend & Database Requirements

### 5.1 FastAPI Backend
- RESTful JSON APIs structured across modular routers (`/api/v1/transactions`, `/api/v1/cases`, `/api/v1/analytics`, `/api/v1/health`).
- Dependency injection for database sessions, authentication context, and ML model inference engines.
- Comprehensive request validation, structured JSON logging, and standardized HTTP status codes.

### 5.2 PostgreSQL Database
- Relational schema enforcing referential integrity:
  - `users` / `accounts`: Account metadata and risk tiers.
  - `transactions`: Core transaction details, amounts, timestamps, metadata.
  - `risk_evaluations`: Calculated risk score, decision action, model version, SHAP reason codes, raw probabilities.
  - `cases`: Case workflow tracking, assigned analyst, review status (`PENDING`, `IN_REVIEW`, `RESOLVED_FRAUD`, `RESOLVED_LEGITIMATE`), resolution notes.
  - `audit_logs`: Immutable audit trails recording system evaluations and analyst actions.

---

## 6. Frontend Requirements (Analyst Dashboard)

- **Technology**: React + TypeScript.
- **Real-Time Transaction Stream**: Visual feed of incoming transactions with color-coded risk badges.
- **Case Management Console**: Filterable queue of pending review cases with sorting by risk score, timestamp, and amount.
- **Deep-Dive Investigation View**: Transaction breakdown, historical account transaction graph, device/location map visual, and SHAP explainability waterfall charts.
- **Analytics & Metrics Overview**: Visualizing key risk KPIs: total processed volume, fraud detection rate, approval rate, false positive estimates, and score distribution histograms.

---

## 7. Security, Governance & Audit Requirements

- **Authentication & Authorization**: Role-Based Access Control (RBAC) separating `Admin`, `Analyst`, and `API_Consumer` roles using JWT tokens.
- **Zero Hardcoded Secrets**: All configuration, DB credentials, and keys strictly loaded via environment variables (`.env`).
- **Immutable Audit Logging**: Every automated decision and manual analyst disposition logged with timestamp, user ID, and action payload.
- **Data Protection**: PII masking for account numbers, card numbers, and sensitive identifiers.

---

## 8. Non-Functional Requirements (NFRs)

- **Measurable Latency**: Transaction scoring pipeline designed with profiling hooks to benchmark latency and throughput during Phase 10.
- **Modularity & Decoupling**: Clear architectural boundaries allowing ML pipelines, backend APIs, and database components to evolve independently.
- **Reproducibility**: Deterministic data preprocessing, random seed locking, and versioned configuration artifacts.
- **High Availability & Fault Tolerance**: Graceful fallback strategies (e.g., heuristic rule fallback if ML model service is temporarily unreachable).

---

## 9. Comprehensive 19-Stage Project Roadmap

```
Phase 0  ──► Project Foundation & Architecture (Current)
Phase 1  ──► Data Pipeline & Ingestion
Phase 2  ──► Exploratory Data Analysis (EDA) & Domain Insights
Phase 3  ──► Behavioral Feature Engineering
Phase 4  ──► Baseline & Advanced ML Models (XGBoost / LightGBM)
Phase 5  ──► Imbalance Handling & Cost-Sensitive Optimization
Phase 6  ──► Risk Engine & Decision Framework (0-100 Scoring)
Phase 7  ──► Explainability & Reason Codes (TreeSHAP)
Phase 8  ──► Fraud Detection REST API (FastAPI)
Phase 9  ──► Database & Persistence (PostgreSQL)
Phase 10 ──► Real-Time Detection & Benchmarking
Phase 11 ──► Fraud Intelligence Dashboard (React + TypeScript)
Phase 12 ──► Human Review & Case Management System
Phase 13 ──► ML & Model Monitoring (Data/Concept Drift)
Phase 14 ──► MLOps, Automated Retraining & Model Registry
Phase 15 ──► Security, Authentication, Authorization & Audit Logging
Phase 16 ──► Containerization & Deployment (Docker)
Phase 17 ──► Comprehensive Testing Suite (Unit, Integration, ML)
Phase 18 ──► Final Portfolio Polish, Runbooks & Documentation
```

### Roadmap Details:
1. **Phase 0: Project Foundation & Architecture** *(Active)*: Scaffolding, repository setup, Git configuration, architectural specs, environment templates.
2. **Phase 1: Data Pipeline & Ingestion**: Synthetic/real transaction ingestion engine, schema enforcement, data partitioning.
3. **Phase 2: Exploratory Data Analysis (EDA)**: Class distribution, feature correlations, temporal patterns, fraud topology analysis report.
4. **Phase 3: Behavioral Feature Engineering**: Time-window velocity features, spending deviation $Z$-scores, geo/device anomaly indicators.
5. **Phase 4: Baseline & Advanced ML Models**: Model training (Logistic Regression, Random Forest, XGBoost, LightGBM) with cross-validation.
6. **Phase 5: Imbalance Handling & Cost Optimization**: PR-AUC optimization, cost-weighted loss matrices, decision threshold tuning.
7. **Phase 6: Risk Engine & Decision Framework**: 0–100 calibrated risk scoring, heuristic overrides, Approve/Review/Block policy engine.
8. **Phase 7: Explainability & Reason Codes**: TreeSHAP feature attributions, local explanation extraction, human-readable reason codes.
9. **Phase 8: Fraud Detection API (FastAPI)**: REST endpoints (`/evaluate`, `/score`, `/health`), Pydantic validation, error handling.
10. **Phase 9: Database & Persistence (PostgreSQL)**: Relational schema, SQLAlchemy models, migration setup, audit trails.
11. **Phase 10: Real-Time Detection & Benchmarking**: End-to-end transaction processing pipeline, latency & throughput benchmarking.
12. **Phase 11: Fraud Intelligence Dashboard**: React + TypeScript web interface, live transaction feed, risk analytics visualizations.
13. **Phase 12: Human Review & Case Management**: Analyst workflow, case queue, manual dispositioning (confirm/dismiss), feedback loop.
14. **Phase 13: ML & Model Monitoring**: Data drift, concept drift, feature distribution shifts, prediction drift tracking.
15. **Phase 14: MLOps, Retraining & Versioning**: Automated trigger pipeline, candidate vs. champion evaluation, model registry.
16. **Phase 15: Security, Auth & Audit Logging**: RBAC, JWT authentication, immutable audit logging for fraud investigations.
17. **Phase 16: Containerization & Deployment**: Dockerfiles, multi-service Docker Compose, production readiness checks.
18. **Phase 17: Comprehensive Testing Suite**: Unit tests, integration tests, ML behavioral tests, load & stress tests.
19. **Phase 18: Final Portfolio Polish & Runbooks**: Architecture walkthrough, live demo guide, portfolio documentation, final cleanups.
