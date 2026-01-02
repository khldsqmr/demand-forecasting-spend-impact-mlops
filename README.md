# Demand Forecasting & Spend Impact — Production MLOps System

## Executive Summary

This repository demonstrates a **production-grade forecasting and spend-impact system**
designed using modern **MLOps principles**, with a strong emphasis on **measurable business
and financial impact**.

Rather than stopping at forecast accuracy, this project **quantifies the real dollar cost
of forecast errors**, enabling data-driven marketing budget allocation, operational
planning, and ROI optimization.

---

## Business Problem

Marketing and operations teams rely on demand forecasts to:

- Allocate marketing budgets across channels
- Plan inventory, staffing, and fulfillment
- Optimize spend efficiency and return on investment (ROI)

Forecast errors are **asymmetric** in nature:

| Error Type | Business Impact |
|-----------|-----------------|
| Over-forecasting | Wasted marketing spend, excess capacity |
| Under-forecasting | Lost revenue, unmet demand |

This system explicitly models:

- Baseline (organic) demand
- Incremental demand driven by marketing spend
- Lagged and diminishing returns
- Macroeconomic effects on demand
- **Explicit financial cost of forecast errors**

---

## Data Architecture

### Demonstration Setup

This repository uses **synthetic but statistically realistic CSV snapshots** that simulate
BigQuery exports.

> Focus: forecasting logic, financial impact modeling, and MLOps system design — not raw
> ingestion mechanics.

---

## Data Domains

| Domain | Description |
|------|------------|
| Baseline Demand | Organic demand influenced by macroeconomic factors |
| Marketing Spend | Channel-level spend and response behavior |
| Product Mix | Allocation of demand across product categories |
| Macroeconomics | External demand drivers |

---

## Data Validation

**Notebook:** `analysis/data_validation.ipynb`

The validation layer acts as a **data contract** prior to modeling and verifies:

- Data completeness and continuity
- Spend–response correlation integrity
- Diminishing returns behavior
- Product mix consistency
- Lagged and rolling feature correctness

No model training proceeds unless these checks pass.

---

## Modeling Approach

### Baseline Model

- RandomForest Regressor
- TimeSeriesSplit cross-validation (5 folds)
- No future data leakage
- Stable, interpretable baseline model

### Cross-Validation Results

| Metric | Value |
|------|------|
| Average WAPE | **0.46%** |
| Best Fold WAPE | **0.09%** |
| Worst Fold WAPE | **0.80%** |

✔ Performance is stable across time  
✔ Features and assumptions are validated  
✔ No fold exhibits structural instability  

---

## Financial Impact Analysis (Key Output)

Forecast errors are converted into **explicit financial costs** rather than abstract
accuracy metrics.

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

Despite achieving **<0.5% WAPE**, forecast errors still resulted in:

- **$195K in lost revenue** due to under-forecasting
- **$69K in wasted spend or excess capacity** due to over-forecasting
- **Total measurable financial impact:** **$264K**

This demonstrates that:

- Accuracy metrics alone are insufficient
- Even small forecast errors can create **six-figure business impact**
- Forecasting systems must be evaluated in **financial terms**

This framework enables:

- Risk-aware forecasting
- Budget reallocation simulations
- ROI-driven decision-making

---

## MLOps & Engineering

### Modular Codebase

- Deterministic, repeatable pipelines
- Clear separation of concerns
- Production-safe feature handling
- Time-series–aware training and evaluation

### Containerization (`docker/`)

- Dockerfile included
- Demonstrates environment reproducibility
- Image not built or pushed in this demo

### Orchestration (`kubeflow/`)

- Kubeflow pipeline defined and compiled
- Demonstrates production orchestration readiness
- Supports retraining and evaluation workflows

### Infrastructure (`terraform/`)

- GCP-oriented infrastructure layout
- Included for architectural completeness

---

## Production Execution Model

In a real production environment, this system would:

1. Ingest new data into BigQuery
2. Trigger deterministic feature pipelines
3. Retrain models on a controlled schedule
4. Generate demand forecasts
5. Quantify forecast-related financial impact
6. Feed outputs into planning and budgeting workflows

---

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
