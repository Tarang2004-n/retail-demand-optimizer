# Retail Demand Forecasting & Inventory Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data science portfolio project demonstrating **scientifically rigorous retail demand forecasting** and **simulated inventory replenishment optimization** using the Walmart M5 competition dataset.

---

## 📌 Business Question & Executive Problem

*How much operational and financial value does machine-learning-driven demand forecasting deliver compared to standard heuristic replenishment baselines under realistic supply chain lead times?*

In retail grocery and consumer goods, stockouts directly cause lost revenue and brand defection, while excessive holding creates working capital drag and spoilage risk. This project quantifies whether replacing standard weekly seasonal naive heuristics with recursive gradient-boosted trees (LightGBM) translates into measurable inventory cost savings and improved customer fill rates.

---

## 📊 Dataset: Walmart M5 Forecasting

- **Source**: [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- **Scope**: Store `CA_1` (California), top 100 products by volume, across 1,941 consecutive days (January 29, 2011 to May 22, 2016).
- **Files**:
  - `sales_train_evaluation.csv`: Unit sales per product/day across 1,941 days.
  - `calendar.csv`: Dates, Walmart retail week numbers (`wm_yr_wk`), event flags, and SNAP supplement purchase eligibility.
  - `sell_prices.csv`: Weekly product prices per store.

---

## 🔬 Scientific Methodology & Guardrails

To prevent the subtle methodological flaws that frequently plague retail time-series projects, this system adheres to strict scientific requirements:

1. **Strict Temporal Integrity (No Future Leakage)**:
   - **Never use random train/test splits**.
   - Product selection deterministically ranks items using **only** the initial training window (`d_1` to `d_1829`).
   - All lag features are shifted prior to feature computation (`sales_lag_1` at day $t$ references sales at $t-1$).
   - Future prices are clamped to the last observed price at the forecast origin.
2. **Expanding Walk-Forward Validation**:
   - Three consecutive, non-overlapping 28-day validation windows:
     - `val1`: `d_1830` – `d_1857` (Origin `d_1829`)
     - `val2`: `d_1858` – `d_1885` (Origin `d_1857`)
     - `val3`: `d_1886` – `d_1913` (Origin `d_1885`)
3. **Sealed Final Test Set**:
   - The final 28 days (`d_1914` to `d_1941`) were strictly sealed and held out until Stage 4 final evaluation.
   - All model hyperparameters and safety stock parameters were frozen before evaluating on the test set.
4. **Recursive Multi-Step Forecasting**:
   - For steps $h \in [2, 28]$, lag and rolling statistics are updated recursively with **model predictions**, never actual future sales.
5. **Replenishment Simulation With In-Transit Order Tracking**:
   - Daily review Order-Up-To $(R, S)$ policy with lead time $L = 7$ days.
   - In-transit orders are explicitly tracked in the inventory position ($IP = \text{on\_hand} + \sum \text{outstanding\_orders}$) to eliminate artificial double-ordering.
   - Safety stock $\sigma$ is computed from validation window error variances, never test set errors.

---

## 📈 Measured Results (Sealed Test Set: `d_1914` to `d_1941`)

### 1. Forecast Accuracy (100 Products, 28 Days, 2,800 Predictions)

| Model | MAE | RMSE | WAPE (%) | Signed Bias | Accuracy vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive Baseline** | 5.6371 | 8.1608 | 50.13% | -0.7743 | Baseline |
| **LightGBM (Recursive)** | **4.9356** | **6.7144** | **43.89%** | +0.9277 | **+12.44% MAE / +17.72% RMSE** |

### 2. Simulated Inventory Performance ($L=7$ Days, Holding: $0.05/day, Penalty: $2.00)

| Metric | Seasonal Naive Baseline | LightGBM Recursive | Impact / Difference |
| :--- | :---: | :---: | :---: |
| **Customer Fill Rate (%)** | 96.63% | **97.86%** | **+1.23 percentage points** |
| **Total Demand Units** | 31,488.0 | 31,488.0 | Exact demand stream |
| **Units Served** | 30,426.0 | **30,813.6** | +387.6 units served |
| **Lost Sales Units** | 1,062.0 | **674.4** | **-36.5% reduction in stockouts** |
| **Stockout Item-Days** | 140 days (5.00%) | **100 days (3.57%)** | **40 fewer stockout events** |
| **Holding Cost ($)** | **$6,939.68** | $7,469.62 | +$529.94 (safety stock buffer) |
| **Lost Sales Penalty ($)** | $2,124.01 | **$1,348.85** | **-$775.16 in penalty savings** |
| **Total Operational Cost ($)** | $9,063.69 | **$8,818.47** | **-$245.22 net savings (-2.7%)** |

---

## ⚠️ Concrete Limitations & Explicit Assumptions

1. **Demand Proxy Limitation**: Historical sales serve as a proxy for customer demand. During out-of-stock periods, true unconstrained demand is truncated.
2. **Illustrative Cost Parameters**: Holding ($0.05/unit/day) and lost-sale penalty ($2.00/unit) costs are illustrative scenario parameters, not actual retail contracts.
3. **Single Store & Scope**: Evaluated on 100 high-velocity products in store `CA_1`. Cross-regional transferability requires testing on other stores.
4. **Deterministic Lead Time**: Lead time is assumed constant at 7 days; supplier disruptions and variability would necessitate higher variance buffers.
5. **No Supply Capacity Constraints**: Assumes infinite supplier fulfillment capacity and no minimum order quantities (MOQs).
6. **28-Day Horizon & Horizon Effects**: Orders placed during the final 7 days do not arrive before simulation end and remain in the transit pipeline.

---

## 🚀 Setup & Execution

### Installation

```bash
# Clone repository
git clone https://github.com/example/retail-demand-optimizer.git
cd retail-demand-optimizer

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce All Pipeline Stages

```bash
# Stage 1: Data preparation, validation splits, and DuckDB exploratory analysis
python src/run_stage1.py

# Stage 2: Recursive feature engineering, LightGBM tuning, and 3-window validation
python src/run_stage2.py

# Stage 3: Inventory simulation and sensitivity analysis on validation window 3
python src/run_stage3.py

# Stage 4: Sealed test set evaluation (executed once on d_1914 to d_1941)
python src/run_final_evaluation.py

# Launch interactive Streamlit dashboard
streamlit run app.py

# Execute full pytest suite (21 unit and integration tests)
pytest tests/ -v
```

---

## 📁 Project Structure

```
retail-demand-optimizer/
├── app.py                             # Interactive 4-tab Streamlit dashboard
├── config.yaml                        # Explicit configuration file
├── requirements.txt                   # Production dependencies
├── pytest.ini                         # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── config.py                      # Typed configuration classes & loaders
│   ├── data_preparation.py           # Pipeline: validation, selection, join, parquet
│   ├── exploratory_analysis.py        # DuckDB SQL queries & Plotly EDA charts
│   ├── features.py                    # Shifted lag, rolling, calendar, price features
│   ├── models.py                      # SeasonalNaiveModel & LightGBMForecaster
│   ├── evaluation.py                  # MAE, RMSE, WAPE, bias, permutation importance
│   ├── inventory_simulator.py         # Order-Up-To replenishment engine
│   ├── run_stage1.py                  # CLI Stage 1 runner
│   ├── run_stage2.py                  # CLI Stage 2 runner
│   ├── run_stage3.py                  # CLI Stage 3 runner
│   └── run_final_evaluation.py        # CLI Stage 4 sealed test runner
├── tests/
│   ├── test_data_pipeline.py          # Data preparation & temporal split tests
│   ├── test_forecasting.py            # Leakage, prediction count, recursive tests
│   ├── test_inventory.py              # Conservation, timing, pipeline order tests
│   └── test_final.py                  # Sealed test verification & synthetic E2E
├── artifacts/                         # Generated model pickles, parquet, and JSON
├── data/
│   ├── raw/                           # Raw M5 CSV files
│   └── processed/                     # sales_processed.parquet
└── reports/
    ├── business_memo.md               # 1-page executive memo to VP Supply Chain
    ├── resume_bullets.md              # 3 verified resume bullets with exact metrics
    ├── interview_notes.md             # In-depth technical interview talking points
    └── figures/                       # Plotly HTML and PNG charts
```
