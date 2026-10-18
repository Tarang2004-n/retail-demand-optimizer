"""CLI Entry Point for Stage 3: Inventory Policy Simulation & Sensitivity Analysis.

Usage:
    python src/run_stage3.py
    python src/run_stage3.py --config config.yaml

SCIENTIFIC INTEGRITY & CONSTRAINTS:
- Simulation results are scenario-based under stated illustrative cost parameters.
- Cost figures ($0.05 holding, $2.00 penalty) are illustrative assumptions.
- Historical sales are used as a proxy for unconstrained customer demand.
- The holdout test set (d_1914 to d_1941) remains strictly untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config
from src.inventory_simulator import (
    InventorySimulator,
    compute_initial_inventory,
    compute_safety_stock_stds,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 3: Inventory Policy Simulation & Sensitivity Analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML (default: config.yaml).",
    )
    return parser.parse_args()


def save_figure(fig: go.Figure, base_name: str, figures_dir: Path) -> Tuple[Path, Path]:
    """Save a Plotly figure as both HTML and PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{base_name}.html"
    png_path = figures_dir / f"{base_name}.png"

    fig.write_html(str(html_path), include_plotlyjs=True)
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        asyncio.run(kaleido.write_fig(fig, str(png_path)))

    return html_path, png_path


def plot_inventory_trajectories(
    results_naive: pd.DataFrame,
    results_lgb: pd.DataFrame,
    product_ids: List[str],
    figures_dir: Path,
) -> go.Figure:
    """Plot daily on-hand inventory and replenishment orders for representative products."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Product: {pid}" for pid in product_ids],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, pid in enumerate(product_ids[:4]):
        row_idx, col_idx = positions[idx]
        naive_item = results_naive[results_naive["item_id"] == pid].sort_values("day_num")
        lgb_item = results_lgb[results_lgb["item_id"] == pid].sort_values("day_num")

        show_legend = (idx == 0)

        # Naive on-hand stock
        fig.add_trace(
            go.Scatter(
                x=naive_item["day_num"],
                y=naive_item["ending_stock"],
                mode="lines",
                name="Seasonal Naive Stock",
                line=dict(color="#ff7f0e", width=2),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

        # LightGBM on-hand stock
        fig.add_trace(
            go.Scatter(
                x=lgb_item["day_num"],
                y=lgb_item["ending_stock"],
                mode="lines",
                name="LightGBM Stock",
                line=dict(color="#2ca02c", width=2),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

        # Demand trace
        fig.add_trace(
            go.Scatter(
                x=naive_item["day_num"],
                y=naive_item["demand"],
                mode="lines+markers",
                name="Actual Daily Demand",
                line=dict(color="#1f77b4", width=1.5, dash="dot"),
                marker=dict(size=4),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text="Simulated Daily On-Hand Inventory Trajectory (Validation Window 3: d_1886-d_1913)<br>"
                 "<sup>Order-Up-To Policy (Lead Time: 7 Days, Daily Review). Demand proxy: observed sales.</sup>",
            font=dict(size=16, family="Arial"),
        ),
        xaxis=dict(title="Simulation Day", showgrid=True),
        yaxis=dict(title="Units On Hand", showgrid=True),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=50, r=40, t=80, b=70),
        width=1050,
        height=650,
    )

    save_figure(fig, "inventory_trajectories", figures_dir)
    return fig


def plot_policy_comparison(
    summary_naive: Dict[str, Any],
    summary_lgb: Dict[str, Any],
    figures_dir: Path,
) -> go.Figure:
    """Create comparison bar charts for cost components and fill rate."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Operational Cost Breakdown ($)", "Customer Demand Fill Rate (%)"],
        horizontal_spacing=0.15,
    )

    models = ["Seasonal Naive", "LightGBM"]
    holding_costs = [summary_naive["total_holding_cost"], summary_lgb["total_holding_cost"]]
    penalty_costs = [summary_naive["total_lost_sales_penalty"], summary_lgb["total_lost_sales_penalty"]]
    fill_rates = [summary_naive["fill_rate_percentage"], summary_lgb["fill_rate_percentage"]]

    # Stacked cost bars
    fig.add_trace(
        go.Bar(
            name="Holding Cost",
            x=models,
            y=holding_costs,
            marker_color="#1f77b4",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Lost Sales Penalty",
            x=models,
            y=penalty_costs,
            marker_color="#d62728",
        ),
        row=1,
        col=1,
    )

    # Fill rate bar
    fig.add_trace(
        go.Bar(
            name="Fill Rate (%)",
            x=models,
            y=fill_rates,
            marker_color="#2ca02c",
            text=[f"{fr:.2f}%" for fr in fill_rates],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Inventory Policy Performance Comparison (28-Day Simulation on Val3)<br>"
                 "<sup>Note: Costs are illustrative model assumptions ($0.05 holding, $2.00 lost-sale penalty).</sup>",
            font=dict(size=16, family="Arial"),
        ),
        template="plotly_white",
        yaxis=dict(title="Total Cost ($)", showgrid=True),
        yaxis2=dict(title="Fill Rate (%)", range=[90, 100], showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.3),
        margin=dict(l=60, r=40, t=80, b=80),
        width=1000,
        height=500,
    )

    save_figure(fig, "inventory_policy_comparison", figures_dir)
    return fig


def plot_sensitivity_analysis(
    sens_lead_time_df: pd.DataFrame,
    sens_safety_stock_df: pd.DataFrame,
    figures_dir: Path,
) -> go.Figure:
    """Plot sensitivity of total cost and fill rate across lead times and safety stock multipliers."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Total Cost vs. Lead Time (Days)",
            "Fill Rate vs. Safety Stock Multiplier",
        ],
        horizontal_spacing=0.12,
    )

    # Panel 1: Lead Time vs Cost
    for model_name, color in [("Seasonal_Naive", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
        sub = sens_lead_time_df[sens_lead_time_df["model_name"] == model_name]
        fig.add_trace(
            go.Scatter(
                x=sub["lead_time_days"],
                y=sub["total_cost"],
                mode="lines+markers",
                name=f"{model_name} Total Cost",
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Safety Stock Multiplier vs Fill Rate
    for model_name, color in [("Seasonal_Naive", "#ff7f0e"), ("LightGBM", "#2ca02c")]:
        sub = sens_safety_stock_df[sens_safety_stock_df["model_name"] == model_name]
        fig.add_trace(
            go.Scatter(
                x=sub["safety_stock_multiplier"],
                y=sub["fill_rate_percentage"],
                mode="lines+markers",
                name=f"{model_name} Fill Rate",
                line=dict(color=color, width=2.5, dash="dash"),
                marker=dict(size=7),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="Inventory Policy Sensitivity Analysis (Validation Window 3)<br>"
                 "<sup>Evaluating operational resilience under varying supplier lead times and safety stock levels.</sup>",
            font=dict(size=16, family="Arial"),
        ),
        xaxis=dict(title="Lead Time (Days)", dtick=1, showgrid=True),
        yaxis=dict(title="Total Operational Cost ($)", showgrid=True),
        xaxis2=dict(title="Safety Stock Multiplier (k)", showgrid=True),
        yaxis2=dict(title="Fill Rate (%)", showgrid=True),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=80, b=80),
        width=1050,
        height=520,
    )

    save_figure(fig, "inventory_sensitivity", figures_dir)
    return fig


def run_stage3_pipeline(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Execute complete Stage 3 inventory simulation and sensitivity analysis.

    Args:
        config: Typed ProjectConfig instance.

    Returns:
        Dictionary of simulation summaries, sensitivity results, and saved artifact paths.
    """
    if config is None:
        config = load_config()

    start_time = time.time()
    print("=" * 75)
    print("STAGE 3: INVENTORY POLICY SIMULATION & SENSITIVITY ANALYSIS")
    print("=" * 75)

    # 1. Load Data and Prior Stage Artifacts
    pred_path = config.artifacts_path / "validation_predictions.parquet"
    act_path = config.processed_data_path / "sales_processed.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing validation predictions at {pred_path}. Run Stage 2 first.")
    if not act_path.exists():
        raise FileNotFoundError(f"Missing processed data at {act_path}. Run Stage 1 first.")

    print(f"Loading validation predictions from {pred_path}...")
    preds_df = pd.read_parquet(pred_path)
    val3_preds = preds_df[preds_df["window"] == "val3"].copy()

    print(f"Loading processed actual sales from {act_path}...")
    actuals_df = pd.read_parquet(act_path)

    # 2. Compute Safety Stock Standard Deviations from Earlier Windows ONLY (val1, val2)
    print("\nEstimating safety stock error variance from earlier windows (val1 & val2)...")
    naive_stds, lgb_stds = compute_safety_stock_stds(
        validation_predictions_path=pred_path,
        actual_sales_path=act_path,
        earlier_windows=["val1", "val2"],
    )
    print("  ✓ Forecast error variance calculated with zero leakage from val3 or test set.")

    # 3. Compute Deterministic Initial Inventory (Identical for Both Policies)
    val3_days = list(range(1886, 1914))  # 28 days
    start_sim_day = val3_days[0]
    lead_time = config.inventory.lead_time_days

    print(f"\nComputing identical initial stock from pre-simulation history (days {start_sim_day-28} to {start_sim_day-1})...")
    initial_inventory = compute_initial_inventory(
        actual_sales_path=act_path,
        start_sim_day=start_sim_day,
        lead_time=lead_time,
        review_period=config.inventory.review_period_days,
        lookback_days=28,
        safety_stock_buffer=2.0,
    )
    total_initial_stock = sum(initial_inventory.values())
    print(f"  ✓ Initial inventory initialized: {total_initial_stock:,.1f} total units across 100 products.")

    # 4. Run Baseline Simulation on val3 for Both Policies
    print("\nRunning daily replenishment simulation on Validation Window 3 (d_1886 to d_1913)...")
    # Seasonal Naive
    sim_naive = InventorySimulator(
        config=config,
        forecast_df=val3_preds.rename(columns={"pred_seasonal_naive": "predicted_sales"}),
        actual_sales_df=actuals_df,
        model_name="Seasonal_Naive",
        safety_stock_stds=naive_stds,
        initial_inventory=initial_inventory,
    )
    res_naive = sim_naive.run_simulation(val3_days)
    summary_naive = sim_naive.get_summary()

    # LightGBM
    sim_lgb = InventorySimulator(
        config=config,
        forecast_df=val3_preds.rename(columns={"pred_lightgbm": "predicted_sales"}),
        actual_sales_df=actuals_df,
        model_name="LightGBM",
        safety_stock_stds=lgb_stds,
        initial_inventory=initial_inventory,
    )
    res_lgb = sim_lgb.run_simulation(val3_days)
    summary_lgb = sim_lgb.get_summary()
    print("  ✓ Simulations completed successfully for both policies.")

    # 5. Sensitivity Analysis
    print("\nRunning sensitivity analysis across lead times, safety stocks, and cost ratios...")
    lead_times = [3, 5, 7, 10]
    multipliers = [1.0, 1.5, 2.0]
    cost_ratios = [
        {"holding": 0.02, "penalty": 2.00, "label": "Low Holding ($0.02)"},
        {"holding": 0.05, "penalty": 2.00, "label": "Baseline ($0.05)"},
        {"holding": 0.10, "penalty": 2.00, "label": "High Holding ($0.10)"},
        {"holding": 0.05, "penalty": 4.00, "label": "High Penalty ($4.00)"},
    ]

    # Vary lead time
    sens_lead_records = []
    for lt in lead_times:
        init_inv_lt = compute_initial_inventory(
            act_path, start_sim_day=start_sim_day, lead_time=lt
        )
        for model_name, f_col, stds in [
            ("Seasonal_Naive", "pred_seasonal_naive", naive_stds),
            ("LightGBM", "pred_lightgbm", lgb_stds),
        ]:
            sim = InventorySimulator(
                config=config,
                forecast_df=val3_preds.rename(columns={f_col: "predicted_sales"}),
                actual_sales_df=actuals_df,
                model_name=model_name,
                safety_stock_stds=stds,
                initial_inventory=init_inv_lt,
                lead_time_days=lt,
            )
            sim.run_simulation(val3_days)
            summ = sim.get_summary()
            sens_lead_records.append(summ)

    sens_lead_df = pd.DataFrame(sens_lead_records)

    # Vary safety stock multiplier
    sens_ss_records = []
    for mult in multipliers:
        for model_name, f_col, stds in [
            ("Seasonal_Naive", "pred_seasonal_naive", naive_stds),
            ("LightGBM", "pred_lightgbm", lgb_stds),
        ]:
            sim = InventorySimulator(
                config=config,
                forecast_df=val3_preds.rename(columns={f_col: "predicted_sales"}),
                actual_sales_df=actuals_df,
                model_name=model_name,
                safety_stock_stds=stds,
                initial_inventory=initial_inventory,
                safety_stock_multiplier=mult,
            )
            sim.run_simulation(val3_days)
            summ = sim.get_summary()
            sens_ss_records.append(summ)

    sens_ss_df = pd.DataFrame(sens_ss_records)

    # Vary cost ratios
    sens_cost_records = []
    for c_spec in cost_ratios:
        for model_name, f_col, stds in [
            ("Seasonal_Naive", "pred_seasonal_naive", naive_stds),
            ("LightGBM", "pred_lightgbm", lgb_stds),
        ]:
            sim = InventorySimulator(
                config=config,
                forecast_df=val3_preds.rename(columns={f_col: "predicted_sales"}),
                actual_sales_df=actuals_df,
                model_name=model_name,
                safety_stock_stds=stds,
                initial_inventory=initial_inventory,
                holding_cost_per_unit_per_day=c_spec["holding"],
                lost_sale_penalty_per_unit=c_spec["penalty"],
            )
            sim.run_simulation(val3_days)
            summ = sim.get_summary()
            summ["cost_scenario"] = c_spec["label"]
            sens_cost_records.append(summ)

    sens_cost_df = pd.DataFrame(sens_cost_records)
    print("  ✓ Sensitivity analysis complete.")

    # 6. Generate Figures
    figures_dir = config.figures_path
    figures_dir.mkdir(parents=True, exist_ok=True)

    representative_items = ["FOODS_3_090", "FOODS_1_018", "FOODS_3_313", "FOODS_3_329"]
    plot_inventory_trajectories(res_naive, res_lgb, representative_items, figures_dir)
    print(f"  ✓ Saved inventory trajectories figure: {figures_dir / 'inventory_trajectories.png'}")

    plot_policy_comparison(summary_naive, summary_lgb, figures_dir)
    print(f"  ✓ Saved policy comparison figure: {figures_dir / 'inventory_policy_comparison.png'}")

    plot_sensitivity_analysis(sens_lead_df, sens_ss_df, figures_dir)
    print(f"  ✓ Saved sensitivity analysis figure: {figures_dir / 'inventory_sensitivity.png'}")

    # 7. Save Artifacts
    artifacts_dir = config.artifacts_path
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save full daily simulation ledger
    combined_ledger = pd.concat([res_naive, res_lgb], ignore_index=True)
    ledger_path = artifacts_dir / "inventory_simulation_ledger.parquet"
    combined_ledger.to_parquet(ledger_path, index=False)
    print(f"  ✓ Saved daily simulation ledger to {ledger_path}")

    # Save baseline summary results
    results_payload = {
        "simulation_parameters": {
            "validation_window": "val3",
            "start_day": start_sim_day,
            "end_day": val3_days[-1],
            "n_days": len(val3_days),
            "n_products": len(initial_inventory),
            "lead_time_days": lead_time,
            "review_period_days": config.inventory.review_period_days,
            "holding_cost_per_unit_per_day": config.inventory.holding_cost_per_unit_per_day,
            "lost_sale_penalty_per_unit": config.inventory.lost_sale_penalty_per_unit,
            "safety_stock_multiplier": config.inventory.safety_stock_multiplier,
            "initial_inventory_total_units": total_initial_stock,
        },
        "scientific_integrity_notices": {
            "demand_proxy_notice": (
                "Historical sales are used as a proxy for customer demand. True unconstrained demand "
                "is unobserved in retail POS data when inventory reaches zero."
            ),
            "illustrative_costs_notice": (
                "Holding costs ($0.05/unit/day) and lost sale penalties ($2.00/unit) are illustrative "
                "scenario parameters. Results must not be interpreted as proven real-world dollar savings."
            ),
            "horizon_effect_notice": (
                "Orders placed within lead_time days of the simulation horizon end do not arrive before "
                "simulation termination. Ending inventory and outstanding orders are explicitly reported."
            ),
        },
        "policies": {
            "seasonal_naive": summary_naive,
            "lightgbm": summary_lgb,
        },
    }
    summary_path = artifacts_dir / "inventory_simulation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"  ✓ Saved simulation summary to {summary_path}")

    # Save sensitivity analysis report
    sens_payload = {
        "lead_time_sensitivity": sens_lead_records,
        "safety_stock_multiplier_sensitivity": sens_ss_records,
        "cost_ratio_sensitivity": sens_cost_records,
    }
    sens_path = artifacts_dir / "inventory_sensitivity_analysis.json"
    with open(sens_path, "w", encoding="utf-8") as f:
        json.dump(sens_payload, f, indent=2)
    print(f"  ✓ Saved sensitivity analysis to {sens_path}")

    # 8. Print Results Tables
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("INVENTORY SIMULATION RESULTS (28-DAY SIMULATION ON VAL3)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Seasonal Naive':>16} | {'LightGBM':>16}")
    print("-" * 70)
    print(f"{'Fill Rate (%)':<32} | {summary_naive['fill_rate_percentage']:>15.2f}% | {summary_lgb['fill_rate_percentage']:>15.2f}%")
    print(f"{'Total Demand (Units)':<32} | {summary_naive['total_demand']:>16,.1f} | {summary_lgb['total_demand']:>16,.1f}")
    print(f"{'Units Served':<32} | {summary_naive['total_units_served']:>16,.1f} | {summary_lgb['total_units_served']:>16,.1f}")
    print(f"{'Lost Sales (Units)':<32} | {summary_naive['total_lost_units']:>16,.1f} | {summary_lgb['total_lost_units']:>16,.1f}")
    print(f"{'Stockout Days (Instances)':<32} | {summary_naive['stockout_days_count']:>16} | {summary_lgb['stockout_days_count']:>16}")
    print(f"{'Stockout Rate (%)':<32} | {summary_naive['stockout_rate']*100:>15.2f}% | {summary_lgb['stockout_rate']*100:>15.2f}%")
    print(f"{'Average On-Hand / Product':<32} | {summary_naive['avg_on_hand_per_product']:>16.2f} | {summary_lgb['avg_on_hand_per_product']:>16.2f}")
    print(f"{'Total Holding Cost ($)':<32} | ${summary_naive['total_holding_cost']:>15,.2f} | ${summary_lgb['total_holding_cost']:>15,.2f}")
    print(f"{'Total Lost Sales Penalty ($)':<32} | ${summary_naive['total_lost_sales_penalty']:>15,.2f} | ${summary_lgb['total_lost_sales_penalty']:>15,.2f}")
    print(f"{'Total Operational Cost ($)':<32} | ${summary_naive['total_cost']:>15,.2f} | ${summary_lgb['total_cost']:>15,.2f}")
    print(f"{'Ending On-Hand Inventory':<32} | {summary_naive['ending_inventory']:>16,.1f} | {summary_lgb['ending_inventory']:>16,.1f}")
    print(f"{'Ending Outstanding Orders':<32} | {summary_naive['ending_outstanding_orders']:>16,.1f} | {summary_lgb['ending_outstanding_orders']:>16,.1f}")
    print("=" * 75)
    print(f"Stage 3 completed in {elapsed:.2f} seconds.")
    print("=" * 75)

    return {
        "summary_naive": summary_naive,
        "summary_lgb": summary_lgb,
        "sens_lead_time": sens_lead_df,
        "sens_safety_stock": sens_ss_df,
    }


if __name__ == "__main__":
    sys.exit(0 if run_stage3_pipeline() else 1)
