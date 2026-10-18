# Executive Memorandum

**TO:** Vice President of Supply Chain  
**FROM:** Retail Data Science & Operations Research Team  
**DATE:** October 18, 2026  
**SUBJECT:** Operational Evaluation of Machine Learning Demand Forecasting for Inventory Replenishment  

---

### Executive Context
Accurate forward-looking demand forecasting is critical for optimizing store-level working capital and mitigating lost revenue from retail stockouts. This evaluation measures the financial and service-level impact of upgrading store replenishment from a standard 7-day seasonal naive baseline to a recursive gradient-boosted tree model (LightGBM) across 100 fast-moving products in store `CA_1` over a 28-day sealed holdout test window (`d_1914` to `d_1941`).

---

### Key Measured Findings

1. **Substantial Forecasting Accuracy Gains**:
   - The recursive LightGBM model reduced Mean Absolute Error (MAE) from **5.64 to 4.94 units/day**, representing a **12.4% error reduction** over the seasonal baseline.
   - Root Mean Squared Error (RMSE) dropped from **8.16 to 6.71** (a **17.7% reduction**), indicating significant reduction in extreme forecast variance.
   - Weighted Absolute Percentage Error (WAPE) improved by **6.2 percentage points** (43.9% vs. 50.1%).

2. **Quantifiable Inventory Service & Cost Improvements**:
   Under a simulated daily review Order-Up-To replenishment policy with a 7-day supplier lead time (holding cost: $0.05/unit/day; lost sale penalty: $2.00/unit):
   - **Customer Fill Rate**: Rose from **96.63% to 97.86%** (+1.23 percentage points).
   - **Stockout Reduction**: Unmet customer demand dropped from **1,062 to 674 units**, achieving a **36.5% reduction in lost sales**.
   - **Stockout Days**: Decreased from **140 to 100 item-days** (stockout rate fell from 5.00% to 3.57%).
   - **Net Cost Savings**: Despite a modest holding cost increase (+$529.94) to maintain safety stock, penalty savings of -$775.16 produced a net operational savings of **$245.22 (-2.7%)** over the 28-day period.

---

### Strategic Recommendations

1. **Deploy LightGBM-Driven Replenishment for High-Volume Grocery**:
   Implement recursive LightGBM replenishment across high-velocity and perishable categories (such as `FOODS_3`), where day-of-week seasonality and promotion sensitivity drive large stockout penalties.
2. **Maintain Heuristic Baselines for Low-Velocity Long-Tail SKUs**:
   For slow-moving items with intermittent demand, the performance spread between LightGBM and Seasonal Naive is modest. Retaining the heuristic baseline for low-velocity tiers avoids unnecessary model maintenance and training overhead.
3. **Shorten Supplier Lead Times Where Feasible**:
   Sensitivity analysis reveals that reducing lead time from 7 days to 3 days decreases total operational costs by **31.7%** (from $8,818 to $6,021) and increases fill rate to **99.18%**, providing a stronger operational ROI than algorithmic tuning alone.

---

### Critical Operational Caveats & Next Steps

- **Demand Truncation**: Historical sales reflect fulfilled demand rather than unconstrained customer appetite; true lost sales during stockouts remain partially unobserved.
- **Contractual Cost Alignment**: Current cost metrics rely on illustrative accounting assumptions ($0.05 holding, $2.00 penalty). Procurement contracts and item-specific gross margins should replace illustrative figures prior to capital allocation.
- **Immediate Next Step**: Conduct a 60-day A/B field pilot across 5 retail stores, measuring actual on-shelf availability and labor replenishment costs.
