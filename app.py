"""Interactive Streamlit Portfolio Dashboard.

Retail Demand Forecasting & Inventory Optimization (M5 Dataset).
Presents data insights, recursive forecasting evaluations, inventory simulations, and business recommendations.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="Retail Demand & Inventory Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@st.cache_data
def load_all_data():
    """Load precomputed parquet datasets and JSON artifacts."""
    # Parquet files
    sales_df = pd.read_parquet(PROCESSED_DIR / "sales_processed.parquet")
    val_preds_df = pd.read_parquet(ARTIFACTS_DIR / "validation_predictions.parquet")
    test_preds_df = pd.read_parquet(ARTIFACTS_DIR / "test_predictions.parquet")
    val_inv_ledger = pd.read_parquet(ARTIFACTS_DIR / "inventory_simulation_ledger.parquet")
    test_inv_ledger = pd.read_parquet(ARTIFACTS_DIR / "test_inventory_ledger.parquet")

    # JSON artifacts
    with open(ARTIFACTS_DIR / "temporal_splits.json") as f:
        splits = json.load(f)
    with open(ARTIFACTS_DIR / "data_quality_report.json") as f:
        data_quality = json.load(f)
    with open(ARTIFACTS_DIR / "validation_metrics.json") as f:
        val_metrics = json.load(f)
    with open(ARTIFACTS_DIR / "test_metrics.json") as f:
        test_metrics = json.load(f)
    with open(ARTIFACTS_DIR / "inventory_simulation_results.json") as f:
        val_inv_results = json.load(f)
    with open(ARTIFACTS_DIR / "test_inventory_results.json") as f:
        test_inv_results = json.load(f)
    with open(ARTIFACTS_DIR / "inventory_sensitivity_analysis.json") as f:
        sensitivity = json.load(f)
    with open(ARTIFACTS_DIR / "final_report.json") as f:
        final_report = json.load(f)

    return {
        "sales": sales_df,
        "val_preds": val_preds_df,
        "test_preds": test_preds_df,
        "val_inv_ledger": val_inv_ledger,
        "test_inv_ledger": test_inv_ledger,
        "splits": splits,
        "data_quality": data_quality,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_inv_results": val_inv_results,
        "test_inv_results": test_inv_results,
        "sensitivity": sensitivity,
        "final_report": final_report,
    }


try:
    data = load_all_data()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please ensure Stages 1-4 have been executed.")
    st.stop()

# Header
st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.caption(
    "M5 Dataset (Store CA_1, Top 100 Products) • Strict Temporal Walk-Forward Validation • "
    "Order-Up-To (R, S) Replenishment Simulation"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales Overview & Data Quality",
    "📈 Forecast Performance",
    "🏭 Inventory Policy Simulation",
    "💼 Business Recommendations",
])


# ==============================================================================
# TAB 1: SALES OVERVIEW & DATA QUALITY
# ==============================================================================
with tab1:
    st.subheader("Historical Demand Trajectory & Data Quality Audit")

    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.markdown("#### Filters & Settings")
        selected_cat = st.selectbox(
            "Category Filter",
            options=["All Categories"] + sorted(list(data["sales"]["cat_id"].unique())),
        )
        if selected_cat != "All Categories":
            cat_items = sorted(data["sales"][data["sales"]["cat_id"] == selected_cat]["item_id"].unique())
        else:
            cat_items = sorted(data["sales"]["item_id"].unique())

        selected_item = st.selectbox(
            "Product Selector (Optional Detail)",
            options=["Aggregate Across Items"] + cat_items,
        )

        st.info(
            f"**Store Scope**: CA_1 (California)\n\n"
            f"**Products**: Top 100 by volume\n\n"
            f"**Timeline**: 1,941 Days ({data['data_quality']['dataset_scope']['min_date']} to {data['data_quality']['dataset_scope']['max_date']})"
        )

        st.markdown("---")
        st.markdown("#### Temporal Split Boundaries")
        splits_df = pd.DataFrame([
            {"Period": "Initial Train (Selection)", "Days": "d_1 - d_1829", "Length": "1,829 days", "Dates": "2011-01-29 to 2016-01-31"},
            {"Period": "Validation Window 1 (val1)", "Days": "d_1830 - d_1857", "Length": "28 days", "Dates": "2016-02-01 to 2016-02-28"},
            {"Period": "Validation Window 2 (val2)", "Days": "d_1858 - d_1885", "Length": "28 days", "Dates": "2016-02-29 to 2016-03-27"},
            {"Period": "Validation Window 3 (val3)", "Days": "d_1886 - d_1913", "Length": "28 days", "Dates": "2016-03-28 to 2016-04-24"},
            {"Period": "Sealed Holdout Test Set", "Days": "d_1914 - d_1941", "Length": "28 days", "Dates": "2016-04-25 to 2016-05-22"},
        ])
        st.dataframe(splits_df, hide_index=True, use_container_width=True)

    with col_main:
        # Time series chart
        filtered_sales = data["sales"].copy()
        if selected_cat != "All Categories":
            filtered_sales = filtered_sales[filtered_sales["cat_id"] == selected_cat]

        if selected_item != "Aggregate Across Items":
            ts_data = (
                filtered_sales[filtered_sales["item_id"] == selected_item]
                .groupby("date")["sales"]
                .sum()
                .reset_index()
            )
            chart_title = f"Daily Sales Trajectory: {selected_item}"
        else:
            ts_data = filtered_sales.groupby("date")["sales"].sum().reset_index()
            chart_title = f"Total Daily Sales: {selected_cat}"

        ts_data["date"] = pd.to_datetime(ts_data["date"])
        ts_data = ts_data.sort_values("date")

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=ts_data["date"],
                y=ts_data["sales"],
                mode="lines",
                name="Sales",
                line=dict(color="#1f77b4", width=1.5),
            )
        )

        # Mark test boundary
        test_start_date = "2016-04-25"
        fig_ts.add_shape(
            type="line",
            x0=test_start_date,
            x1=test_start_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
        fig_ts.add_annotation(
            x=test_start_date,
            y=1.02,
            yref="paper",
            text="Test Set Origin (2016-04-25 / d_1914)",
            showarrow=False,
            font=dict(color="#d62728", size=11),
        )
        fig_ts.add_shape(
            type="rect",
            x0=test_start_date,
            x1=str(ts_data["date"].max().date()),
            y0=0,
            y1=1,
            yref="paper",
            fillcolor="rgba(214, 39, 40, 0.08)",
            line_width=0,
        )

        fig_ts.update_layout(
            title=chart_title,
            xaxis=dict(title="Timeline", showgrid=True),
            yaxis=dict(title="Daily Units Sold", showgrid=True),
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=30, t=50, b=40),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Data Quality & Zero-Sales Heatmap
        st.markdown("#### Data Quality Findings & Scientific Notes")
        q_cols = st.columns(4)
        qm = data["data_quality"]["quality_metrics"]
        q_cols[0].metric("Missing Prices", f"{qm['missing_prices_percentage']}%", f"{qm['missing_prices_count']:,} rows")
        q_cols[1].metric("Zero-Sales Days", f"{qm['zero_sales_percentage']}%", f"{qm['zero_sales_count']:,} days")
        q_cols[2].metric("Missing Dates", f"{qm['missing_dates_count']}", "100% complete")
        q_cols[3].metric("Negative Sales", f"{qm['negative_sales_count']}", "Zero errors")

        st.warning(
            "**Scientific Distinction Notice**: \n"
            "- **Missing Sell Prices**: Reflected periods before product launch or delisting. Must *not* be interpreted as stockouts.\n"
            "- **Zero Sales Days**: In POS retail scanner data, zero sales represents unobserved demand. Without telemetry on shelf availability, "
            "zero sales cannot be distinguished from out-of-stock events without assumptions."
        )


# ==============================================================================
# TAB 2: FORECAST PERFORMANCE
# ==============================================================================
with tab2:
    st.subheader("Forecast Model Evaluation: Seasonal Naive vs. Recursive LightGBM")

    # KPI summary cards: Validation vs Test
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    val_naive_mae = data["val_metrics"]["models"]["seasonal_naive"]["overall"]["mae"]
    val_lgb_mae = data["val_metrics"]["models"]["lightgbm"]["overall"]["mae"]
    test_naive_mae = data["test_metrics"]["test_metrics"]["seasonal_naive"]["overall"]["mae"]
    test_lgb_mae = data["test_metrics"]["test_metrics"]["lightgbm"]["overall"]["mae"]

    val_lgb_wape = data["val_metrics"]["models"]["lightgbm"]["overall"]["wape"]
    test_lgb_wape = data["test_metrics"]["test_metrics"]["lightgbm"]["overall"]["wape"]

    kpi1.metric("Validation MAE (LightGBM)", f"{val_lgb_mae:.3f}", f"Baseline: {val_naive_mae:.3f} (-9.2%)")
    kpi2.metric("Test MAE (LightGBM)", f"{test_lgb_mae:.3f}", f"Baseline: {test_naive_mae:.3f} (-12.4%)")
    kpi3.metric("Validation WAPE", f"{val_lgb_wape*100:.1f}%", "-5.2 pp vs Naive")
    kpi4.metric("Test WAPE", f"{test_lgb_wape*100:.1f}%", "-6.2 pp vs Naive")

    st.markdown("---")

    fc_col1, fc_col2 = st.columns([1, 1])

    with fc_col1:
        st.markdown("#### Error Metrics Breakdown Across Windows")
        window_rows = []
        for w in ["val1", "val2", "val3"]:
            nw = data["val_metrics"]["models"]["seasonal_naive"]["by_window"][w]
            lw = data["val_metrics"]["models"]["lightgbm"]["by_window"][w]
            window_rows.append({
                "Window": w.upper(),
                "Seasonal Naive MAE": nw["mae"],
                "LightGBM MAE": lw["mae"],
                "MAE Δ (%)": round(((lw["mae"] - nw["mae"]) / nw["mae"]) * 100, 1),
                "Naive WAPE (%)": round(nw["wape"] * 100, 1),
                "LightGBM WAPE (%)": round(lw["wape"] * 100, 1),
            })
        # Add test window
        tn = data["test_metrics"]["test_metrics"]["seasonal_naive"]["overall"]
        tl = data["test_metrics"]["test_metrics"]["lightgbm"]["overall"]
        window_rows.append({
            "Window": "TEST (Sealed)",
            "Seasonal Naive MAE": tn["mae"],
            "LightGBM MAE": tl["mae"],
            "MAE Δ (%)": round(((tl["mae"] - tn["mae"]) / tn["mae"]) * 100, 1),
            "Naive WAPE (%)": round(tn["wape"] * 100, 1),
            "LightGBM WAPE (%)": round(tl["wape"] * 100, 1),
        })
        st.dataframe(pd.DataFrame(window_rows), hide_index=True, use_container_width=True)

        st.markdown("#### Performance by Volume Segment & Category (Validation)")
        seg_rows = []
        for seg in ["high", "medium", "low"]:
            ns = data["val_metrics"]["models"]["seasonal_naive"]["by_volume_segment"][seg]
            ls = data["val_metrics"]["models"]["lightgbm"]["by_volume_segment"][seg]
            seg_rows.append({
                "Segment": f"Volume: {seg.capitalize()}",
                "Naive MAE": ns["mae"],
                "LightGBM MAE": ls["mae"],
                "Naive WAPE": f"{ns['wape']*100:.1f}%",
                "LightGBM WAPE": f"{ls['wape']*100:.1f}%",
            })
        for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
            nc = data["val_metrics"]["models"]["seasonal_naive"]["by_category"][cat]
            lc = data["val_metrics"]["models"]["lightgbm"]["by_category"][cat]
            seg_rows.append({
                "Segment": f"Cat: {cat}",
                "Naive MAE": nc["mae"],
                "LightGBM MAE": lc["mae"],
                "Naive WAPE": f"{nc['wape']*100:.1f}%",
                "LightGBM WAPE": f"{lc['wape']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(seg_rows), hide_index=True, use_container_width=True)

    with fc_col2:
        st.markdown("#### Permutation Feature Importance (Validation Set ΔMAE)")
        st.caption("Notice: Permutation importance reflects model predictive reliance, NOT real-world causality.")
        top_feats = pd.DataFrame(data["val_metrics"]["top_features_by_importance"])
        if not top_feats.empty:
            top_feats = top_feats.sort_values("importance_mean", ascending=True).tail(10)
            fig_imp = go.Figure(
                go.Bar(
                    x=top_feats["importance_mean"],
                    y=top_feats["feature"],
                    orientation="h",
                    marker_color="#1f77b4",
                    error_x=dict(type="data", array=top_feats["importance_std"], visible=True),
                )
            )
            fig_imp.update_layout(
                xaxis=dict(title="Increase in Validation MAE When Permuted"),
                yaxis=dict(title=""),
                template="plotly_white",
                height=320,
                margin=dict(l=120, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # Actual vs Predicted Plot for Selected Item
    st.markdown("#### Actual vs. Forecasted Daily Sales (Sealed Test Window: d_1914 to d_1941)")
    test_products = sorted(data["test_preds"]["item_id"].unique())
    inspect_item = st.selectbox("Select Product to Inspect Forecasts", test_products, index=0)

    act_slice = data["sales"][data["sales"]["day_num"].between(1914, 1941)]
    act_slice_item = act_slice[act_slice["item_id"] == inspect_item].sort_values("day_num")

    test_preds_item = data["test_preds"][data["test_preds"]["item_id"] == inspect_item].sort_values("day_num")

    fig_cmp = go.Figure()
    fig_cmp.add_trace(
        go.Scatter(
            x=act_slice_item["date"],
            y=act_slice_item["sales"],
            mode="lines+markers",
            name="Actual Demand (Observed)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=5),
        )
    )
    fig_cmp.add_trace(
        go.Scatter(
            x=act_slice_item["date"],
            y=test_preds_item["pred_seasonal_naive"],
            mode="lines",
            name="Seasonal Naive",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig_cmp.add_trace(
        go.Scatter(
            x=act_slice_item["date"],
            y=test_preds_item["pred_lightgbm"],
            mode="lines",
            name="LightGBM Recursive",
            line=dict(color="#2ca02c", width=2.5),
        )
    )
    fig_cmp.update_layout(
        title=f"Test Set Demand Trajectory: {inspect_item} (2016-04-25 to 2016-05-22)",
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Daily Units Sold", showgrid=True),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ==============================================================================
# TAB 3: INVENTORY SIMULATION
# ==============================================================================
with tab3:
    st.subheader("Inventory Policy Simulation: Order-Up-To (R, S)")

    inv_tab_choice = st.radio(
        "Evaluation View",
        ["Sealed Test Set Simulation (Final Evaluated)", "Validation Window 3 Simulation (Tuning Benchmark)"],
        horizontal=True,
    )

    if inv_tab_choice.startswith("Sealed"):
        inv_res = data["test_inv_results"]["policies"]
        ledger = data["test_inv_ledger"]
        sim_window_label = "Sealed Test Set (d_1914 - d_1941)"
    else:
        inv_res = data["val_inv_results"]["policies"]
        ledger = data["val_inv_ledger"]
        sim_window_label = "Validation Window 3 (d_1886 - d_1913)"

    s_naive = inv_res["seasonal_naive"]
    s_lgb = inv_res["lightgbm"]

    # Main comparison metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customer Fill Rate", f"{s_lgb['fill_rate_percentage']:.2f}%", f"Naive: {s_naive['fill_rate_percentage']:.2f}% (+{s_lgb['fill_rate_percentage']-s_naive['fill_rate_percentage']:.2f}%)")
    m2.metric("Total Lost Sales", f"{s_lgb['total_lost_units']:,.0f} units", f"{s_lgb['total_lost_units']-s_naive['total_lost_units']:,.0f} units (Fewer Stockouts)")
    m3.metric("Holding Cost", f"${s_lgb['total_holding_cost']:,.2f}", f"+${s_lgb['total_holding_cost']-s_naive['total_holding_cost']:,.2f} (Buffer)")
    m4.metric("Total Operational Cost", f"${s_lgb['total_cost']:,.2f}", f"-${s_naive['total_cost']-s_lgb['total_cost']:,.2f} (Savings)", delta_color="inverse")

    st.markdown("---")

    inv_col1, inv_col2 = st.columns([1, 1])

    with inv_col1:
        st.markdown(f"#### Policy Cost & Service Comparison ({sim_window_label})")
        comp_df = pd.DataFrame([
            {"Policy": "Seasonal Naive", "Fill Rate (%)": s_naive["fill_rate_percentage"], "Lost Units": s_naive["total_lost_units"], "Holding Cost ($)": s_naive["total_holding_cost"], "Penalty Cost ($)": s_naive["total_lost_sales_penalty"], "Total Cost ($)": s_naive["total_cost"]},
            {"Policy": "LightGBM", "Fill Rate (%)": s_lgb["fill_rate_percentage"], "Lost Units": s_lgb["total_lost_units"], "Holding Cost ($)": s_lgb["total_holding_cost"], "Penalty Cost ($)": s_lgb["total_lost_sales_penalty"], "Total Cost ($)": s_lgb["total_cost"]},
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        # Bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Holding Cost", x=["Seasonal Naive", "LightGBM"], y=[s_naive["total_holding_cost"], s_lgb["total_holding_cost"]], marker_color="#1f77b4"))
        fig_bar.add_trace(go.Bar(name="Lost Sales Penalty", x=["Seasonal Naive", "LightGBM"], y=[s_naive["total_lost_sales_penalty"], s_lgb["total_lost_sales_penalty"]], marker_color="#d62728"))
        fig_bar.update_layout(barmode="stack", title="Operational Cost Breakdown ($)", template="plotly_white", height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Horizon effects display
        st.info(
            f"**Horizon Effects & Outstanding Orders**:\n\n"
            f"- **Ending Inventory**: Naive = {s_naive['ending_inventory']:,.1f} units | LightGBM = {s_lgb['ending_inventory']:,.1f} units\n"
            f"- **Orders In-Transit at Simulation End**: Naive = {s_naive['ending_outstanding_orders']:,.1f} units | LightGBM = {s_lgb['ending_outstanding_orders']:,.1f} units\n\n"
            f"*Orders placed in the final 7 days do not arrive before simulation end, accounting for capital tied up in the transit pipeline.*"
        )

    with inv_col2:
        st.markdown("#### Interactive Trajectory for Product")
        traj_items = sorted(ledger["item_id"].unique())
        selected_traj_item = st.selectbox("Product Trajectory", traj_items, index=0)

        item_sub = ledger[ledger["item_id"] == selected_traj_item]
        naive_sub = item_sub[item_sub["model_name"] == "Seasonal_Naive"].sort_values("day_num")
        lgb_sub = item_sub[item_sub["model_name"] == "LightGBM"].sort_values("day_num")

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(x=naive_sub["day_num"], y=naive_sub["ending_stock"], mode="lines", name="Seasonal Naive Stock", line=dict(color="#ff7f0e", width=2)))
        fig_inv.add_trace(go.Scatter(x=lgb_sub["day_num"], y=lgb_sub["ending_stock"], mode="lines", name="LightGBM Stock", line=dict(color="#2ca02c", width=2)))
        fig_inv.add_trace(go.Scatter(x=naive_sub["day_num"], y=naive_sub["demand"], mode="lines+markers", name="Observed Demand", line=dict(color="#1f77b4", width=1.5, dash="dot"), marker=dict(size=4)))
        fig_inv.update_layout(
            title=f"On-Hand Inventory Trajectory: {selected_traj_item}",
            xaxis=dict(title="Day Number", showgrid=True),
            yaxis=dict(title="Units On Hand", showgrid=True),
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    # Sensitivity analysis section
    st.markdown("---")
    st.markdown("#### Operational Sensitivity Analysis (Validation Benchmark)")
    st.caption("⚠️ **Post-Evaluation Scenario Exploration** — Explores operational resilience under varying supplier lead times and safety stock levels. This is NOT the frozen test result.")

    lead_sens = pd.DataFrame(data["sensitivity"]["lead_time_sensitivity"])
    ss_sens = pd.DataFrame(data["sensitivity"]["safety_stock_multiplier_sensitivity"])

    sens_col1, sens_col2 = st.columns(2)

    with sens_col1:
        fig_sens1 = go.Figure()
        for m, c in [("Seasonal_Naive", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
            sub = lead_sens[lead_sens["model_name"] == m]
            fig_sens1.add_trace(go.Scatter(x=sub["lead_time_days"], y=sub["total_cost"], mode="lines+markers", name=m, line=dict(color=c, width=2.5)))
        fig_sens1.update_layout(title="Total Cost ($) vs Supplier Lead Time (Days)", xaxis=dict(title="Lead Time (Days)", dtick=1), yaxis=dict(title="Cost ($)"), template="plotly_white", height=320)
        st.plotly_chart(fig_sens1, use_container_width=True)

    with sens_col2:
        fig_sens2 = go.Figure()
        for m, c in [("Seasonal_Naive", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
            sub = ss_sens[ss_sens["model_name"] == m]
            fig_sens2.add_trace(go.Scatter(x=sub["safety_stock_multiplier"], y=sub["fill_rate_percentage"], mode="lines+markers", name=m, line=dict(color=c, width=2.5, dash="dash")))
        fig_sens2.update_layout(title="Fill Rate (%) vs Safety Stock Multiplier (k)", xaxis=dict(title="Multiplier (k)"), yaxis=dict(title="Fill Rate (%)"), template="plotly_white", height=320)
        st.plotly_chart(fig_sens2, use_container_width=True)


# ==============================================================================
# TAB 4: BUSINESS RECOMMENDATIONS & LIMITATIONS
# ==============================================================================
with tab4:
    st.subheader("Executive Business Recommendations & Scientific Disclosures")

    rec_col1, rec_col2 = st.columns([3, 2])

    with rec_col1:
        st.markdown("""
        ### Executive Summary
        
        1. **Did ML Forecasting Improve Accuracy?**
           - **Yes.** On the sealed holdout test set (`d_1914` to `d_1941`), the recursive LightGBM model achieved an **MAE of 4.936** (vs. **5.637** for the Seasonal Naive baseline), delivering a **12.4% reduction in forecasting error** and an **17.7% reduction in RMSE** (6.71 vs. 8.16).
           - LightGBM consistently captured weekday/weekend demand swings and item velocities without overfitting.

        2. **Did Better Forecasts Translate to Better Inventory Decisions?**
           - **Yes.** Under an Order-Up-To replenishment policy with a 7-day supplier lead time:
             - **Customer Fill Rate**: Increased from **96.63% to 97.86%** (+1.23 percentage points).
             - **Lost Sales Units**: Reduced from **1,062 to 674 units** (a **36.5% reduction** in unmet demand).
             - **Stockout Incidents**: Decreased from **140 to 100 item-days** (-28.6%).
             - **Net Operational Cost**: Total cost decreased from **$9,063.69 to $8,818.47** (saving **$245.22**, a 2.7% net reduction over 28 days).

        3. **Key Trade-Offs**:
           - Higher service levels naturally require carrying slightly more safety stock (average on-hand of 53.4 vs. 49.6 units per product, increasing holding cost by **+$529.94**).
           - However, the **-$775.16 reduction in lost-sale penalties** heavily offset the holding increase.

        4. **Policy Recommendation**:
           - **Deploy LightGBM-Driven Replenishment** for Fast-Moving and High-Volume retail categories.
           - For slow-moving or highly intermittent items with low error penalties, maintaining the Seasonal Naive baseline is operationally viable with minimal complexity.
        """)

    with rec_col2:
        st.markdown("""
        ### Project Limitations & Next Steps
        
        #### Stated Scientific Limitations
        1. **Demand Proxy**: Historical sales serve as a proxy for customer demand. True unconstrained demand during out-of-stock periods is unobserved.
        2. **Illustrative Cost Parameters**: Holding ($0.05/day) and lost-sale ($2.00) costs are illustrative assumptions, not actual retailer contracts.
        3. **Store & Product Scope**: Evaluated on 100 products from a single store (`CA_1`). Findings should be validated across other geographic locations.
        4. **Forecast Horizon**: The evaluation spans a 28-day window. Long-term seasonal regime shifts (e.g. holidays, pandemics) require rolling retraining.
        5. **No Supply Constraints**: Assumes infinite supplier capacity and zero minimum order quantities (MOQs).
        6. **Lead Time Determinism**: Supplier lead time is assumed fixed at 7 days; real-world stochastic lead times would require additional variance buffers.

        #### Recommended Next Steps
        - **Censored Demand Imputation**: Implement Tobit regression or EM algorithms to infer unconstrained demand during zero-sales days.
        - **Probabilistic Forecasting**: Replace point forecasts with quantile predictions (e.g. LightGBM pinball loss) for data-driven safety stocks.
        - **Multi-Echelon Simulation**: Extend inventory logic from store level up to regional distribution centers.
        """)

    st.markdown("---")
    st.caption("Portfolio Project: Retail Demand Forecasting & Inventory Optimization • Built with Python, DuckDB, LightGBM, and Streamlit.")
