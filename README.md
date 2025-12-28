# Demand Forecasting & Marketing Spend Impact (MLOps)

## Executive Summary
This repository demonstrates a **production-grade demand forecasting system** designed using modern **MLOps principles**, with a strong emphasis on **measurable business and financial impact**.

Rather than stopping at forecast accuracy, this project **quantifies the real dollar cost of forecast errors**, enabling data-driven marketing budget allocation, operational planning, and ROI optimization.

---

## Business Problem

Marketing and operations teams rely on demand forecasts to:
- Allocate marketing budgets across channels
- Plan inventory, staffing, and fulfillment
- Optimize spend efficiency and ROI

Forecast errors are **asymmetric**:

| Error Type | Business Impact |
|----------|----------------|
| Over-forecasting | Wasted marketing spend, excess capacity |
| Under-forecasting | Lost revenue, unmet demand |

This system explicitly models:
- Baseline (organic) demand
- Incremental demand driven by marketing spend
- Lagged and diminishing returns
- Macroeconomic effects on demand
- **Financial cost of forecast errors**

---

## Data Architecture

### Demonstration Setup
This repository uses **synthetic but statistically realistic CSV snapshots** that simulate BigQuery exports.

> Focus: forecasting logic, financial impact, and MLOps design — not raw ingestion mechanics.

---

## Data Domains

| Domain | Description |
|------|------------|
| Baseline Demand | Organic demand influenced by macro factors |
| Marketing Spend | Channel-level spend and response |
| Product Mix | Allocation of demand across products |
| Macroeconomics | External demand drivers |

---

## Data Validation

**Notebook:** `analysis/data_validation.ipynb`

Validates:
- Data completeness
- Spend–response correlation
- Diminishing returns behavior
- Product mix consistency
- Lag & rolling feature integrity

Acts as a **data contract** before modeling.

---

## Modeling Approach

### Baseline Model
- RandomForest Regressor
- TimeSeriesSplit cross-validation (5 folds)
- No future data leakage
- Stable, interpretable baseline

### Cross-Validation Results

| Metric | Value |
|------|------|
| Average WAPE | **0.46%** |
| Best Fold WAPE | **0.09%** |
| Worst Fold WAPE | **0.80%** |

✔ Performance is stable across time  
✔ Features and assumptions validated

---

## Financial Impact Analysis (KEY OUTPUT)

Forecast errors are converted into **explicit financial costs**.

### Aggregate Results (2021–2024)

| Metric | Value |
|------|------|
| Total Actual Demand | **5,445,072 units** |
| Total Predicted Demand | **5,444,953 units** |
| Total Under-Forecast Units | **2,436 units** |
| Total Over-Forecast Units | **2,316 units** |
| Under-Forecast Cost | **$194,855** |
| Over-Forecast Cost | **$69,490** |
| **Total Forecast Cost** | **$264,345** |

---

## Executive Interpretation (So What?)

- Even with **<0.5% WAPE**, forecast errors resulted in:
  - **$195K in lost revenue** (under-forecasting)
  - **$69K in wasted spend / excess capacity**
- **Total measurable financial impact:** **$264K**
- This proves:
  - Small forecast improvements can unlock **six-figure savings**
  - Accuracy metrics alone are insufficient — **dollars matter**

This framework enables:
- Budget reallocation simulations
- Risk-aware forecasting
- ROI-driven decision making

---

## MLOps & Engineering

### Modular Codebase
- Deterministic pipelines
- Clear separation of concerns
- Production-safe feature handling

### Containerization (`docker/`)
- Dockerfile included
- Demonstrates environment reproducibility
- Not built or pushed in this demo

### Orchestration (`kubeflow/`)
- Kubeflow pipeline defined and compiled
- Demonstrates production orchestration readiness

### Infrastructure (`terraform/`)
- GCP-oriented infrastructure layout
- Included for architectural completeness

---

## Production Execution Model

In production, the system would:
1. Ingest new data into BigQuery
2. Trigger feature pipelines
3. Retrain models on schedule
4. Generate demand forecasts
5. Quantify financial impact
6. Feed results into planning & budgeting workflows

---

## Key Takeaway

This project demonstrates **how forecasting systems should be built in production**:

✔ Time-safe modeling  
✔ Financially interpretable outputs  
✔ MLOps-ready architecture  
✔ Executive-aligned metrics  

**Forecasting is not about accuracy alone — it’s about money.**

---

## Extensions
- Advanced models (XGBoost, Prophet, DL)
- Budget optimization simulations
- Forecast drift monitoring


## System Architecture

```text
Raw Data (CSV / BigQuery exports)
↓
Data Validation & Sanity Checks
↓
Feature Engineering
↓
Baseline Model Training (TimeSeries CV)
↓
Final Model Training
↓
Demand Predictions
↓
Financial Impact Analysis
↓
Kubeflow Pipeline (Orchestration)

