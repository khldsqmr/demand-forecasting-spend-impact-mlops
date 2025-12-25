# Demand Forecasting & Marketing Spend Impact (MLOps)

## Overview
This repository demonstrates an **end-to-end demand forecasting and spend impact system** designed using modern MLOps principles.

The goal is not only to forecast demand, but to **translate forecast accuracy into business and financial impact**, enabling better marketing budget allocation, operational planning, and ROI optimization.

This project emphasizes:
- Realistic data behavior
- Business-aligned modeling assumptions
- Production-ready system design
- Clear separation between data, modeling, and orchestration

---

## Business Problem

Marketing and operations teams rely on demand forecasts to:
- Allocate marketing budgets across channels
- Plan inventory, staffing, and fulfillment
- Optimize spend efficiency and ROI

Forecast errors are **asymmetric**:
- **Over-forecasting** → wasted spend, excess capacity
- **Under-forecasting** → lost revenue, missed demand

This system explicitly models:
- Baseline (organic) demand
- Incremental demand driven by marketing spend
- Lagged and diminishing returns
- Macro-economic effects on demand

---

## Data Architecture

### Upstream Data (Production Concept)
In a real production environment, data would be sourced from:
- Marketing platforms (search, social, display, CRM)
- Analytics pipelines
- Public macro-economic APIs
- Centralized data warehouse (e.g. BigQuery)

### Demonstration Setup (This Repository)
To keep the project:
- Reproducible
- Reviewable
- Lightweight

This repository uses **representative CSV snapshots** that simulate BigQuery exports.

> The focus is on **forecasting logic, feature engineering, and business impact** — not raw ingestion mechanics.

---

## Data Domains

The system operates across four core data domains:

| Domain | Description |
|------|------------|
| Baseline Demand | Organic demand influenced by macro factors |
| Marketing Spend | Channel-level spend and response |
| Product Mix | Allocation of demand across products |
| Macroeconomics | External demand drivers |

---

## Dataset Breakdown

### 1️⃣ Country-Level Demand & Macro Data
**File:** `data/raw/demand_spend_country_daily.csv`

**Key fields:**
- `DATE`
- `COUNTRY`
- `ECONOMIC_INDEX`
- `INFLATION_RATE`
- `UNEMPLOYMENT_RATE`
- `BASELINE_DEMAND`

---

### 2️⃣ Channel-Level Spend Response
**File:** `data/processed/demand_spend_by_channel_daily.csv`

**Key fields:**
- `DATE`
- `COUNTRY`
- `CHANNEL`
- `SPEND`
- `CHANNEL_RESPONSE`

Models diminishing returns and lag effects.

---

### 3️⃣ Product-Level Demand Allocation
**File:** `data/processed/demand_by_product_daily.csv`

**Key fields:**
- `DATE`
- `COUNTRY`
- `PRODUCT`
- `PRODUCT_DEMAND`

---

### 4️⃣ Model Training Dataset
**File:** `data/processed/model_training_dataset.csv`

Feature-engineered dataset containing:
- Aggregated spend & response
- Lagged features
- Rolling demand windows

Used for forecasting and impact modeling.

---

## Data Validation & Analysis

**Notebook:** `analysis/data_validation.ipynb`

This notebook validates:
- Data completeness
- Statistical realism
- Spend-response correlation
- Diminishing returns behavior
- Product mix consistency
- Feature distributions
- Lag and rolling feature integrity

The notebook serves as a **data contract and sanity check** before modeling.

---

## MLOps & System Design

### Containerization (`docker/`)
- Dockerfile templates included
- Demonstrates environment reproducibility
- Not built or pushed in this demo

---

### Orchestration (`kubeflow/`)
- Conceptual Kubeflow pipeline definitions
- Shows how training and forecasting would be orchestrated
- Included for architectural clarity, not execution

---

### Infrastructure (`terraform/`)
- GCP-oriented infrastructure layout
- Demonstrates production readiness
- Not applied or deployed

---

## Execution Model (Production Intent)

In a production environment, the system would:
1. Ingest new data into BigQuery
2. Trigger feature pipelines
3. Retrain models on schedule
4. Generate demand forecasts
5. Quantify financial impact of forecast errors
6. Feed results into planning & budgeting workflows

---

## Notes on Data
All data in this repository is **synthetic but statistically realistic**, created solely for demonstration purposes.

No proprietary or confidential data is used.

---

## Future Extensions
- Model benchmarking (ARIMA, Prophet, ML, DL)
- Budget optimization simulations
- Automated retraining pipelines
- Forecast monitoring & drift detection
- Full Kubeflow pipeline execution

---

## Key Takeaway
This project demonstrates **how a production-grade forecasting system should be designed**, not just how to train a model.

It bridges:
- Data engineering
- Forecasting
- Business impact
- MLOps architecture
