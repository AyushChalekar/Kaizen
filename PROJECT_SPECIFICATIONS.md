<!-- docs/PROJECT_SPECIFICATION.md -->

# Predictive Resource Optimization System for Healthcare (PROSH)
## System Requirements, Technical Specifications, and Team Meeting Reference

---

## 1. Document Control and Revision History

This document serves as the single source of truth for system architecture, technical requirements, data contracts, and development progress. All project changes during implementation must be recorded in this section.

### 1.1 Change Management Protocol
1. Any architectural modification, schema change, or optimization parameter update must be submitted via a Pull Request modifying this file.
2. Update the version number following Semantic Versioning (`MAJOR.MINOR.PATCH`):
   - `MAJOR`: Fundamental architectural changes (e.g., switching from MILP to DRL).
   - `MINOR`: New module additions, schema extensions, or new UI views.
   - `PATCH`: Parameter adjustments, bug fixes, or minor documentation updates.

### 1.2 Revision History

| Version | Date (YYYY-MM-DD) | Author | Scope of Change | Status |
| :--- | :--- | :--- | :--- | :--- |
| v0.1.0 | 2026-09-18 | Team | Initial draft and consolidation from working notes | Approved |
| v0.2.0 | Pending | Shridhar | Finalize MILP vs. DRL selection and solver engine | Proposed |
| v0.3.0 | Pending | Suraj | Export synthetic stays and census datasets to Parquet | Proposed |
| v0.4.0 | Pending | Ayush | Wireframe review for Digital Twin and XAI views | Proposed |

---

## 2. Project Overview and System Intent

### 2.1 Executive Summary
Standard acute care hospital scheduling operates reactively or deterministically, resulting in emergency department (ED) boarding bottlenecks, nurse burnout, and unmanaged surge periods. 

The **Predictive Resource Optimization System for Healthcare (PROSH)** is an uncertainty-aware, closed-loop decision engine designed for co-optimizing clinical throughput and operational capacity. PROSH connects patient arrival and length-of-stay (LOS) forecasting with mathematical optimization and explainable AI (XAI) to automate staff rostering and bed allocation dynamically.

### 2.2 Core Technical Pipeline
1. **Data Layer and Digital Twin**: Simulates realistic patient flow through discrete-event modeling (SimPy) calibrated against real-world clinical benchmarks (MIMIC-IV, eICU).
2. **Predictive Engine**: Leverages gradient boosting (LightGBM/XGBoost) and Conformal Prediction (MAPIE) to output point predictions alongside mathematically guaranteed interval bounds `[y_lower, y_upper]`.
3. **Optimization Engine**: Ingests prediction intervals and facility constraints into a Robust Mixed-Integer Linear Program (MILP) or Deep Reinforcement Learning (DRL) agent to balance beds and staffing.
4. **Explainable AI (XAI) Layer**: Converts mathematical allocation vectors and active constraints into natural language clinical rationales for hospital administrators.

### 2.3 Research Differentiators

| Capability | Standard Literature / Existing Systems | PROSH Framework |
| :--- | :--- | :--- |
| **Prediction + Optimization** | Handled sequentially or in isolation | Co-optimized end-to-end in a closed loop |
| **Uncertainty Bounds** | Single point estimates only | Rigorous prediction intervals `[y_min, y_max]` |
| **Resource Scope** | Single resource (e.g., bed allocation OR nursing roster) | Multi-resource (Doctor, Nurse, General/ICU beds) |
| **Decision Output** | Black-box recommendations or static schedules | Human-interpretable, explainable clinical rationales |

### 2.4 Target Optimization KPIs
- **Patient Queue Time**: Mean and 95th percentile wait time in ED triage and bed queues.
- **Staff Overtime and Utilization**: Overtime cost reduction and nurse workload variance across wards.
- **Bed Occupancy Rates (BOR)**: Balanced utilization across General, HDU, and ICU units without bottleneck violations.
- **Surge Resilience**: Throughput degradation index under simulated multi-casualty or epidemic spikes.

---

## 3. Technology Stack and Repository Structure

### 3.1 Technology Matrix

| Layer | Technology | Primary Libraries / Frameworks | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite | `pnpm`, Lucide React, Recharts, TailwindCSS | Operational dashboards and simulation UI |
| **Simulation** | Python 3.11+ | SimPy, NumPy, SciPy | Discrete Event Simulation of patient flow |
| **Data Engine** | Python 3.11+ | Pandas, Polars, PyArrow | Data synthesis, ETL, and validation |
| **Predictive ML** | Python 3.11+ | LightGBM, XGBoost, MAPIE, Prophet | Volume and LOS forecasting with intervals |
| **Optimization** | Python 3.11+ | Pyomo, PuLP, OR-Tools, Stable-Baselines3 | Robust MILP or DRL decisioning |
| **XAI / Rationale** | Python 3.11+ | Instructor, OpenAI / Anthropic / Local LLM APIs | Allocation translation into clinical text |
| **Package Tooling** | `uv` (Python), `pnpm` (Node) | Lockfile-enforced reproducibility | Cross-platform environment synchronization |

kaizen/
|-- PROJECT_SPECIFICATIONS.md
|-- README.md
|-- clineprompt.txt
|-- backend/
|   |-- Dockerfile
|   |-- README.md
|   |-- pyproject.toml
|   |-- requirements.txt
|   |-- uv.lock
|   |-- data/
|   |   |-- synthetic_hourly_census.csv
|   |   `-- synthetic_patient_stays_100k_fixed.csv
|   |-- ml/
|   |   |-- clean.py
|   |   |-- eval.py
|   |   `-- train_los_classifier.py
|   |-- models/                          # Serialized model weights & pipeline artifacts
|   |-- outputs/                         # Prediction CSVs and evaluation benchmarks
|   |   |-- los_final_test_predictions.csv
|   |   |-- los_predictions_model7_test.csv
|   |   |-- los_predictions_model9_test.csv
|   |   |-- los_predictions_test.csv
|   |   |-- los_predictions_test_log.csv
|   |   |-- los_predictions_test_quantile_specific.csv
|   |   |-- los_predictions_test_tail_weighted.csv
|   |   |-- los_predictions_test_tuned.csv
|   |   |-- model_benchmark_comparison.csv
|   |   `-- model_benchmark_comparison.json
|   |-- scripts/
|   |   |-- models/                      # Training & diagnostic iteration scripts
|   |   |   |-- diagnose_los_model6.py
|   |   |   |-- evaluate_los_model.py
|   |   |   |-- train_los_final.py
|   |   |   |-- train_los_model.py
|   |   |   |-- train_los_model9.py
|   |   |   |-- train_los_model_feature_engineered.py
|   |   |   |-- train_los_model_log.py
|   |   |   |-- train_los_model_quantile_weighted.py
|   |   |   |-- train_los_model_tail_weighted.py
|   |   |   `-- train_los_model_tuned.py
|   |   `-- output/
|   `-- src/                             # Core SimPy and solver modules (to be populated)
`-- frontend/
    |-- README.md
    |-- eslint.config.js
    |-- index.html
    |-- package.json
    |-- pnpm-lock.yaml
    |-- tsconfig.app.json
    |-- tsconfig.json
    |-- tsconfig.node.json
    |-- vite.config.ts
    |-- public/
    |   |-- favicon.svg
    |   `-- icons.svg
    `-- src/
        |-- App.css
        |-- App.tsx
        |-- index.css
        |-- main.tsx
        `-- assets/
            |-- hero.png
            |-- react.svg
            `-- vite.svg