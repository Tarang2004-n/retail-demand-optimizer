# Technical Interview Talking Points & Deep-Dive Notes

This document provides structured responses for technical and system design interview discussions based on the Retail Demand Forecasting and Inventory Optimization system.

---

### 1. How Data Leakage Was Prevented

- **Strict Temporal Boundary Enforcement**: Standard random train-test splitting is fatal for time-series evaluation because future observations bleed into past training sets. All splits are strictly chronological.
- **Unbiased Product Selection**: Products were selected based *exclusively* on total sales during the initial training window (`d_1` to `d_1829`). Sales from validation (`d_1830`–`d_1913`) and test (`d_1914`–`d_1941`) periods were zeroed out during product filtering.
- **Shifted Feature Transformations**:
  - For any day $t$, `sales_lag_1` is explicitly defined as sales at $t-1$.
  - Rolling mean, std, min, and max over windows of 7, 14, and 28 days are computed strictly on shifted series (excluding day $t$).
- **Recursive Multi-Step Feature Generation**: When forecasting a 28-day horizon, day $t+2$ cannot see actual sales from $t+1$. The simulator maintains an internal history buffer where model predictions $\hat{y}$ from earlier steps are recursively fed into the lag and rolling calculators for subsequent steps.
- **Pricing Leakage Prevention**: Future prices are clamped to the last known price observed at the forecast origin.
- **Safety Stock Leakage Prevention**: Safety stock standard deviations for test simulation were estimated strictly from forecast errors on earlier validation windows (`val1` + `val2` + `val3`), never from test set errors.

---

### 2. Validation Strategy Rationale: Expanding Walk-Forward Windows

- **Why Not Single Train/Validation Split?** Retail demand exhibits non-stationary seasonality (payday cycles, SNAP issuance, holidays). A single 28-day validation window can suffer from idiosyncratic shocks (e.g. Easter date shifts).
- **Three Consecutive Non-Overlapping Windows**:
  - `val1`: `d_1830` to `d_1857` (February 2016)
  - `val2`: `d_1858` to `d_1885` (March 2016)
  - `val3`: `d_1886` to `d_1913` (April 2016)
- **Expanding Window Training**: For each window, the model is trained on all historical observations up to that window's origin. This mirrors actual production operations where models are periodically retrained with expanding history.
- **Sealed Test Set**: The final 28 days (`d_1914` to `d_1941`) were sealed until the very final step, ensuring hyperparameter tuning on validation windows did not overfit the test set.

---

### 3. Model Architecture Rationale: Why LightGBM Over Deep Learning?

- **Tabular Efficiency and Inductive Biases**: Retail scanner data is tabular, featuring strong categorical interactions (`item_id`, `dept_id`, `cat_id`) and calendar regimes (`wday`, `snap_CA`, `month`). Gradient-boosted decision trees (GBDTs) consistently match or exceed neural network accuracy on tabular data without extensive feature normalization.
- **Categorical Handling**: LightGBM natively partitions categorical splits using Fisher's exact test / optimal subset algorithms, bypassing high-cardinality one-hot explosion.
- **Inference Speed**: Recursive 28-day multi-step inference requires 28 sequential predictions per item. LightGBM generates 2,800 predictions across 100 products in **0.13 seconds** on CPU, enabling instant scenario simulation.
- **Why Not Deep Learning (e.g. Temporal Fusion Transformer or DeepAR)?** Deep learning time-series models require orders of magnitude more training time, GPU dependencies, and extensive hyperparameter tuning. For a 100-product single-store deployment, LightGBM provided optimal accuracy-to-compute efficiency.

---

### 4. Inventory Replenishment Mechanics: Order-Up-To $(R, S)$ & In-Transit Orders

- **Daily Review Cycle ($R=1$)**: Every evening, inventory positions are evaluated and replenishment orders are placed.
- **Lead Time ($L=7$) & Protection Period ($L + R = 8$)**: An order placed today arrives in 7 days; the next order arrives in 8 days. Therefore, inventory must buffer demand across the full 8-day protection window.
- **Why Tracking In-Transit Orders Is Crucial**:
  - Without subtracting outstanding orders, the order formula would calculate: $Q = S - \text{on\_hand}$.
  - During lead time days 1 through 6, no orders have arrived yet. If outstanding orders are ignored, the system will place a fresh full order *every single day*, massively inflating inventory by $7 \times$ demand.
  - Correct formula: $Q = \max(0, S - \text{on\_hand} - \sum \text{outstanding\_orders})$.
- **Units Conservation**: Enforced through rigorous unit tests:
  $$\text{Ending Stock} = \text{Starting Stock} + \text{Arrivals} - \text{Served Demand}$$
  $$\text{Demand} = \text{Served Demand} + \text{Lost Sales}$$

---

### 5. Failure Modes, Edge Cases & Real-World Caveats

- **Censored Demand (Stockout Masking)**: POS scanner data records sales, not true customer intent. When on-hand stock hits 0, observed sales drop to 0, biasing future models to predict lower demand during stockouts.
- **Cannibalization and Halo Effects**: Promotions on one SKU frequently cannibalize adjacent SKUs within the same department. Point models treating items independently miss intra-basket substitution.
- **Bullwhip Amplification**: Upstream suppliers experience volatile ordering swings when store-level algorithms over-react to transitory demand spikes.
- **Perishability & Shelf Life Constraints**: Fast-moving grocery products (e.g. dairy, produce in `FOODS_3`) have hard expiration dates that would convert aging on-hand stock into physical spoilage write-offs.

---

### 6. Production Extensions & Next Steps

1. **Censored Demand Estimation**: Implement a Tobit regression or Expectation-Maximization (EM) hurdle model to impute unconstrained demand during stockout periods.
2. **Probabilistic Forecasting**: Transition from point predictions to quantile regression ($\tau \in \{0.50, 0.75, 0.90, 0.95\}$) using LightGBM pinball loss. This directly calculates safety stock without assuming Gaussian error distributions.
3. **Multi-Echelon Optimization**: Model the flow from regional Distribution Centers (DCs) to retail stores, jointly optimizing replenishment quantities and cross-dock transportation schedules.
4. **Constrained Optimization**: Add Mixed-Integer Linear Programming (MILP) layers to accommodate pallet rounding, minimum order quantities (MOQs), truckload capacities, and warehouse shelf limits.
