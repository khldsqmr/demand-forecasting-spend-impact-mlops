# Demand Forecasting & Marketing Spend Impact (MLOps)

## Executive Summary

This repository demonstrates a **production-grade demand forecasting system** designed using modern **MLOps and decision-science principles**.

The system goes beyond traditional forecast accuracy metrics and **quantifies forecast error directly in financial terms**, enabling better decisions around marketing spend, inventory planning, and operational risk.

### Key Outcomes

| Metric | Result |
|------|--------|
| Average WAPE (TimeSeries CV) | **0.46%** |
| Best Fold WAPE | 0.09% |
| Worst Fold WAPE | 0.80% |
| Total Forecast Cost | **$264,345** |
| Under-Forecast Cost Share | **~74%** |

**Key Insight:**  
Under-forecasting is significantly more expensive than over-forecasting, highlighting the need for **cost-aware modeling**, not just accuracy optimization.

---

## Business Problem

Marketing and operations teams rely on demand forecasts to:

- Allocate marketing budgets across channels
- Plan inventory, staffing, and fulfillment
- Optimize spend efficiency and ROI

However, **forecast errors are asymmetric**:

| Error Type | Business Impact |
|---------|----------------|
| Over-forecasting | Wasted marketing spend, excess capacity |
| Under-forecasting | Lost revenue, missed demand |

Most forecasting systems stop at MAE / RMSE.  
This system explicitly models **error asymmetry** and converts forecast performance into **dollars at risk**.

---

## Results & Financial Impact

### Forecast Accuracy (Leakage-Safe Evaluation)

Forecasts were evaluated using **TimeSeriesSplit cross-validation**, ensuring:

- No future leakage
- Realistic production-like evaluation
- Stable performance across time

#### Cross-Validation Summary

| Metric | Value |
|------|------|
| Average WAPE | **0.46%** |
| Std Dev (WAPE) | 0.33% |
| Best Fold | 0.09% |
| Worst Fold | 0.80% |

---

### Financial Impact of Forecast Errors

Using explicit business cost assumptions:

| Metric | Value |
|------|------|
| Total Actual Demand | 5,445,072 units |
| Total Predicted Demand | 5,444,953 units |
| Under-Forecast Units | 2,435.69 |
| Over-Forecast Units | 2,316.33 |
| Under-Forecast Cost | **$194,855** |
| Over-Forecast Cost | $69,490 |
| **Total Forecast Cost** | **$264,345** |

---

## Data Architecture

### Production Concept (Upstream)

In a real deployment, data would be ingested from:

- Marketing platforms (Search, Social, Display, CRM)
- Digital analytics pipelines
- Public macroeconomic APIs
- Centralized data warehouse (e.g. BigQuery)

---

### Demonstration Setup (This Repository)

To keep the project:

- Reproducible
- Lightweight
- Review-friendly

The system uses **representative CSV snapshots** that simulate BigQuery exports.

> Focus is on **forecasting logic, feature engineering, and business impact** — not ingestion mechanics.

---

## Data Domains

| Domain | Description |
|------|------------|
| Baseline Demand | Organic demand driven by macro factors |
| Marketing Spend | Channel-level spend and response |
| Product Mix | Allocation of demand across products |
| Macroeconomics | External demand drivers |

---

## Dataset Breakdown

### 1️⃣ Country-Level Demand & Macro Data
**Path:** `data/raw/demand_spend_country_daily.csv`

| Column | Description |
|------|------------|
| DATE | Observation date |
| COUNTRY | Market |
| ECONOMIC_INDEX | Composite economic signal |
| INFLATION_RATE | Inflation |
| UNEMPLOYMENT_RATE | Labor market signal |
| BASELINE_DEMAND | Organic demand |

---

### 2️⃣ Channel-Level Spend Response
**Path:** `data/processed/demand_spend_by_channel_daily.csv`

| Column | Description |
|------|------------|
| DATE | Date |
| COUNTRY | Market |
| CHANNEL | Marketing channel |
| SPEND | Daily spend |
| CHANNEL_RESPONSE | Modeled response |

Includes **lag effects** and **diminishing returns**.

---

### 3️⃣ Product-Level Demand Allocation
**Path:** `data/processed/demand_by_product_daily.csv`

| Column | Description |
|------|------------|
| DATE | Date |
| COUNTRY | Market |
| PRODUCT | Product |
| PRODUCT_DEMAND | Allocated demand |

---

### 4️⃣ Model Training Dataset
**Path:** `data/processed/model_training_dataset.csv`

Aggregated, model-ready dataset including:

- Spend aggregation
- Lagged features
- Rolling demand windows

---

## Data Validation & Quality Checks

**Notebook:** `analysis/data_validation.ipynb`

Validates:

- Data completeness
- Statistical realism
- Spend-response correlation
- Diminishing returns behavior
- Product mix consistency
- Lag & rolling feature integrity

Acts as a **data contract** before modeling.

---

## Feature Engineering

**Module:** `src/features/feature_engineering.py`

Features include:

- Calendar seasonality (cyclical encoding)
- Spend efficiency metrics
- Macro × demand interactions
- Trend & momentum signals

Design principles:

- Deterministic
- No data leakage
- Explainable
- Production-safe

---

## Modeling Approach

### Baseline Model
- Algorithm: **RandomForestRegressor**
- Evaluation: **TimeSeriesSplit**
- Training:
  - Cross-validated baseline
  - Final model trained on full history

### Artifacts
- Trained model
- Encoder
- Feature metadata

---

## Financial Impact Analysis

**Module:** `src/evaluation/spend_impact.py`

Computes:

- Forecast error by direction
- Cost of under-forecasting
- Cost of over-forecasting
- Aggregate forecast cost

This reframes forecasting as a **risk management problem**, not just prediction.

---

## MLOps & System Design

### Containerization (`docker/`)
- Dockerfile templates included
- Demonstrates environment reproducibility

### Orchestration (`kubeflow/`)
- Conceptual Kubeflow pipeline
- Included for architectural clarity

### Infrastructure (`terraform/`)
- GCP-oriented IaC layout
- Demonstrates production readiness

---

## Execution Flow

```text
Raw Data Generation
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
TimeSeries Cross-Validation
        ↓
Final Model Training
        ↓
Baseline Predictions
        ↓
Financial Impact Analysis
