# Demand Forecasting & Spend Impact (MLOps)

## Overview
This project demonstrates an end-to-end MLOps system designed to forecast weekly demand by marketing channel and quantify the downstream financial impact on marketing spend efficiency and revenue outcomes.

The primary objective is not just to predict demand, but to enable better budgeting and planning decisions by translating forecast accuracy into monetary impact.

---

## Business Problem
Marketing and operations teams rely on demand forecasts to:
- Allocate marketing budgets across channels
- Plan inventory and staffing
- Optimize spend efficiency and ROI

Forecast errors are asymmetric:
- **Over-forecasting** leads to wasted spend and operational inefficiencies
- **Under-forecasting** results in lost revenue and missed opportunities

This system explicitly models that asymmetry and converts forecast performance into business-relevant financial metrics.

---

## Data Architecture

### Upstream Data Sources (Conceptual)
In a production environment, this system would ingest historical data from upstream data platforms such as **BigQuery**, populated by:
- Digital analytics pipelines
- Marketing spend platforms
- Public macroeconomic datasets

### Demonstration Setup
For clarity and reproducibility, this repository uses **representative CSV snapshots** to simulate upstream data inputs.

The focus of this project is on:
- Forecasting logic
- Spend impact analysis
- MLOps design and orchestration

Not on raw data ingestion mechanics.

---

## Data Domains
The system operates on four primary data domains:

1. **Demand Activations**
   - Weekly demand by marketing channel

2. **Marketing Spend**
   - Weekly spend allocation by channel

3. **Unit Economics**
   - Cost and revenue per activation

4. **Macroeconomic Indicators**
   - External demand drivers (e.g. unemployment, inflation, interest rates)

---

## System Components

### Core Forecasting Logic (`src/`)
- Feature preparation
- Model training
- Forecast generation
- Financial impact analysis

### Orchestration (`kubeflow/`)
- End-to-end pipeline definition
- Reproducible execution of training and forecasting steps

### Containerization (`docker/`)
- Environment consistency across local and production runs

### Infrastructure Design (`terraform/`)
- GCP-ready infrastructure definitions
- Included for architectural completeness (not deployed)

---

## Execution Model
The system is designed to:
1. Ingest historical demand, spend, and macroeconomic data
2. Train forecasting models at a weekly cadence
3. Generate demand forecasts by channel
4. Quantify financial impact of forecast errors
5. Enable repeatable, automated execution via pipelines

---

## Notes on Data
The data used in this repository is representative and anonymized, intended solely to demonstrate system design, forecasting methodology, and financial impact analysis.

---

## Future Extensions
- Model comparison and performance benchmarking
- Scenario simulations for budget optimization
- Deployment to managed cloud services
