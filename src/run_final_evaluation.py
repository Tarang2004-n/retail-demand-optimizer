"""Final Evaluation CLI on the Sealed Holdout Test Set (d_1914 to d_1941).

Freezes all hyperparameters, product selections, and inventory parameters.
Executes the final test-set forecast evaluation and inventory simulation ONCE.

SCIENTIFIC INTEGRITY:
- Models are trained strictly on data up to d_1913 (train + val1 + val2 + val3).
- Test set (d_1914 to d_1941) was never touched during EDA, model development, or tuning.
- Safety stock error standard deviations are computed from validation windows (val1+val2+val3), NOT test errors.
- No tuning is performed on test results. Results are recorded honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config
from src.evaluation import (
    assign_volume_segments,
    evaluate_forecasts,
)
from src.features import (
    build_feature_matrix,
    get_feature_columns,
)
from src.inventory_simulator import (
    InventorySimulator,
    compute_initial_inventory,
)
from src.models import LightGBMForecaster, SeasonalNaiveModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Final Evaluation on the Sealed Holdout Test Set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML (default: config.yaml).",
    )
    return parser.parse_args()


def compute_validation_error_stds(
    val_preds_path: Path | str,
    actual_sales_path: Path | str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-item forecast error std across all 3 validation windows (val1+val2+val3).

    Zero test set data is used.

    Args:
        val_preds_path: Path to validation_predictions.parquet.
        actual_sales_path: Path to sales_processed.parquet.

    Returns:
        Tuple of (naive_stds_dict, lgb_stds_dict).
    """
    preds_df = pd.read_parquet(val_preds_path)
    actuals_df = pd.read_parquet(actual_sales_path)

    val_days = preds_df["day_num"].unique()
    val_actuals = actuals_df[actuals_df["day_num"].isin(val_days)][
        ["item_id", "day_num", "sales"]
    ].copy()

    merged = pd.merge(preds_df, val_actuals, on=["item_id", "day_num"], how="inner")
    merged["naive_err"] = merged["sales"] - merged["pred_seasonal_naive"]
    merged["lgb_err"] = merged["sales"] - merged["pred_lightgbm"]

    naive_stds = (
        merged.groupby("item_id", observed=False)["naive_err"]
        .std()
        .fillna(1.0)
        .to_dict()
    )
    lgb_stds = (
        merged.groupby("item_id", observed=False)["lgb_err"]
        .std()
        .fillna(1.0)
        .to_dict()
    )
    return naive_stds, lgb_stds


def run_final_evaluation(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Execute the sealed test set evaluation end-to-end.

    Args:
        config: Typed ProjectConfig instance.

    Returns:
        Dictionary containing final evaluation report.
    """
    if config is None:
        config = load_config()

    start_time = time.time()
    print("=" * 78)
    print("STAGE 4: FINAL EVALUATION ON SEALED TEST SET (d_1914 to d_1941)")
    print("=" * 78)

    # 1. Load Data and Check Frozen Validation Artifacts
    parquet_path = config.processed_data_path / "sales_processed.parquet"
    val_metrics_path = config.artifacts_path / "validation_metrics.json"
    val_preds_path = config.artifacts_path / "validation_predictions.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing processed data at {parquet_path}. Run Stage 1 first.")
    if not val_metrics_path.exists():
        raise FileNotFoundError(f"Missing validation metrics at {val_metrics_path}. Run Stage 2 first.")
    if not val_preds_path.exists():
        raise FileNotFoundError(f"Missing validation predictions at {val_preds_path}. Run Stage 2 first.")

    with open(val_metrics_path, "r", encoding="utf-8") as f:
        val_metrics_json = json.load(f)

    best_lgb_params = val_metrics_json["best_lightgbm_parameters"]
    print(f"Loaded frozen LightGBM hyperparameters from validation: {best_lgb_params}")

    df_raw = pd.read_parquet(parquet_path)
    test_start_d = config.temporal.test_start_d  # 1914
    test_end_d = config.temporal.test_end_d      # 1941
    origin_d = test_start_d - 1                  # 1913
    horizon = config.temporal.forecast_horizon   # 28

    print(f"Test Set Window: d_{test_start_d} to d_{test_end_d} ({horizon} days)")
    print(f"Forecast Origin: d_{origin_d}")

    # 2. Build Features up to Test End (Price frozen at origin_d to prevent future price leakage)
    print("\nBuilding feature matrix with prices clamped to origin (d_1913)...")
    df_feat = build_feature_matrix(df_raw, config=config, origin_day=origin_d)
    feature_cols = get_feature_columns(df_feat)
    volume_segments = assign_volume_segments(df_raw, train_end_d=config.temporal.train_end_d)

    # 3. Train Models on ALL Historical Data (d_35 to d_1913: train + 3 validation windows)
    train_mask = (df_feat["day_num"] >= 35) & (df_feat["day_num"] <= origin_d)
    X_train = df_feat.loc[train_mask, feature_cols]
    y_train = df_feat.loc[train_mask, "sales"]

    print(f"Training models on full historical data ({len(X_train):,} rows across 100 products)...")

    # Train Seasonal Naive
    seasonal_naive = SeasonalNaiveModel()
    seasonal_naive.fit(df_feat[df_feat["day_num"] <= origin_d])

    # Train LightGBM
    forecaster = LightGBMForecaster(params=best_lgb_params, random_seed=config.data.random_seed)
    t_fit_start = time.time()
    forecaster.fit(X_train, y_train, feature_cols=feature_cols)
    print(f"  ✓ LightGBM fitted in {time.time() - t_fit_start:.2f}s.")

    # 4. Generate Recursive 28-day Forecasts on Test Set (d_1914 to d_1941)
    print(f"\nGenerating recursive 28-day forecasts for test window (d_{test_start_d} to d_{test_end_d})...")
    # Seasonal Naive
    test_naive_preds = seasonal_naive.predict(
        origin_day=origin_d,
        horizon=horizon,
        full_df=df_feat,
    )
    test_naive_preds["window"] = "test"

    # LightGBM Recursive
    t_pred_start = time.time()
    test_lgb_preds = forecaster.predict_recursive(
        origin_day=origin_d,
        full_df=df_feat,
        horizon=horizon,
        lags=config.forecasting.lightgbm.lag_features,
        windows=config.forecasting.lightgbm.rolling_windows,
    )
    test_lgb_preds["window"] = "test"
    print(f"  ✓ LightGBM recursive test forecasts generated in {time.time() - t_pred_start:.2f}s.")

    # 5. Evaluate Forecast Performance on Test Set
    print("\nEvaluating test set forecast accuracy against actual demand...")
    eval_test_naive = evaluate_forecasts(
        actual_df=df_feat,
        pred_df=test_naive_preds,
        model_name="Seasonal_Naive",
        volume_segments=volume_segments,
    )
    eval_test_lgb = evaluate_forecasts(
        actual_df=df_feat,
        pred_df=test_lgb_preds,
        model_name="LightGBM_Recursive",
        volume_segments=volume_segments,
    )

    # Save test predictions
    test_preds_combined = pd.merge(
        test_naive_preds.rename(columns={"predicted_sales": "pred_seasonal_naive"}),
        test_lgb_preds.rename(columns={"predicted_sales": "pred_lightgbm"}),
        on=["item_id", "day_num", "d", "window"],
    )
    test_preds_path = config.artifacts_path / "test_predictions.parquet"
    test_preds_combined.to_parquet(test_preds_path, index=False)
    print(f"  ✓ Saved test predictions to {test_preds_path}")

    # Save test metrics JSON
    test_metrics_payload = {
        "dataset_scope": {
            "store_id": config.data.store_id,
            "n_products": config.data.n_products,
            "origin_d": origin_d,
            "test_start_d": test_start_d,
            "test_end_d": test_end_d,
            "n_test_days": horizon,
            "frozen_hyperparameters": best_lgb_params,
        },
        "test_metrics": {
            "seasonal_naive": eval_test_naive,
            "lightgbm": eval_test_lgb,
        },
    }
    test_metrics_path = config.artifacts_path / "test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, indent=2)
    print(f"  ✓ Saved test metrics to {test_metrics_path}")

    # 6. Inventory Simulation on the Test Set
    print("\nRunning inventory replenishment simulation on the Test Set...")
    # Calculate safety stock stds strictly from validation windows
    naive_stds, lgb_stds = compute_validation_error_stds(
        val_preds_path=val_preds_path,
        actual_sales_path=parquet_path,
    )

    # Compute identical initial inventory from pre-simulation history (val3: days 1886-1913)
    test_days = list(range(test_start_d, test_end_d + 1))
    initial_stock_test = compute_initial_inventory(
        actual_sales_path=parquet_path,
        start_sim_day=test_start_d,
        lead_time=config.inventory.lead_time_days,
        review_period=config.inventory.review_period_days,
        lookback_days=28,
        safety_stock_buffer=2.0,
    )

    # Simulate Seasonal Naive policy
    sim_naive = InventorySimulator(
        config=config,
        forecast_df=test_naive_preds.rename(columns={"predicted_sales": "predicted_sales"}),
        actual_sales_df=df_raw,
        model_name="Seasonal_Naive",
        safety_stock_stds=naive_stds,
        initial_inventory=initial_stock_test,
    )
    ledger_naive = sim_naive.run_simulation(test_days)
    inv_summary_naive = sim_naive.get_summary()

    # Simulate LightGBM policy
    sim_lgb = InventorySimulator(
        config=config,
        forecast_df=test_lgb_preds.rename(columns={"predicted_sales": "predicted_sales"}),
        actual_sales_df=df_raw,
        model_name="LightGBM",
        safety_stock_stds=lgb_stds,
        initial_inventory=initial_stock_test,
    )
    ledger_lgb = sim_lgb.run_simulation(test_days)
    inv_summary_lgb = sim_lgb.get_summary()

    # Save test inventory results
    test_inv_payload = {
        "simulation_parameters": {
            "period": "test_set (d_1914 to d_1941)",
            "lead_time_days": config.inventory.lead_time_days,
            "review_period_days": config.inventory.review_period_days,
            "holding_cost_per_unit_per_day": config.inventory.holding_cost_per_unit_per_day,
            "lost_sale_penalty_per_unit": config.inventory.lost_sale_penalty_per_unit,
            "safety_stock_multiplier": config.inventory.safety_stock_multiplier,
            "initial_inventory_units": sum(initial_stock_test.values()),
        },
        "policies": {
            "seasonal_naive": inv_summary_naive,
            "lightgbm": inv_summary_lgb,
        },
    }
    test_inv_path = config.artifacts_path / "test_inventory_results.json"
    with open(test_inv_path, "w", encoding="utf-8") as f:
        json.dump(test_inv_payload, f, indent=2)
    print(f"  ✓ Saved test inventory results to {test_inv_path}")

    # Save test ledger
    test_ledger_combined = pd.concat([ledger_naive, ledger_lgb], ignore_index=True)
    test_ledger_path = config.artifacts_path / "test_inventory_ledger.parquet"
    test_ledger_combined.to_parquet(test_ledger_path, index=False)
    print(f"  ✓ Saved test inventory ledger to {test_ledger_path}")

    # 7. Build Combined Final Report (Validation + Test Side-by-Side)
    final_report = {
        "project_title": "Retail Demand Forecasting and Inventory Optimization",
        "dataset_scope": {
            "store_id": config.data.store_id,
            "n_products": config.data.n_products,
            "validation_windows": [w.name for w in config.temporal.validation_windows],
            "test_window": f"d_{test_start_d} to d_{test_end_d}",
            "frozen_model_parameters": best_lgb_params,
        },
        "forecast_comparison": {
            "validation_overall": {
                "seasonal_naive": val_metrics_json["models"]["seasonal_naive"]["overall"],
                "lightgbm": val_metrics_json["models"]["lightgbm"]["overall"],
            },
            "test_set": {
                "seasonal_naive": eval_test_naive["overall"],
                "lightgbm": eval_test_lgb["overall"],
            },
        },
        "inventory_comparison": {
            "test_set": {
                "seasonal_naive": inv_summary_naive,
                "lightgbm": inv_summary_lgb,
            },
        },
        "scientific_integrity_notices": {
            "no_leakage": "The test set was evaluated once with all parameters strictly frozen.",
            "demand_proxy": "Historical sales are used as a proxy for demand. Stockout periods truncate observed sales.",
            "illustrative_costs": "Holding cost ($0.05/day) and penalty ($2.00) are illustrative scenario assumptions.",
        },
    }

    final_report_path = config.artifacts_path / "final_report.json"
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"  ✓ Saved combined final report to {final_report_path}")

    # 8. Print Executive Comparison Tables
    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print("FINAL TEST SET FORECAST PERFORMANCE (d_1914 to d_1941)")
    print("=" * 78)
    print(f"{'Model':<24} | {'MAE':>8} | {'RMSE':>8} | {'WAPE':>8} | {'Bias':>9}")
    print("-" * 65)
    for res in [eval_test_naive, eval_test_lgb]:
        ov = res["overall"]
        print(
            f"{res['model_name']:<24} | {ov['mae']:>8.4f} | {ov['rmse']:>8.4f} | "
            f"{ov['wape']:>8.4f} | {ov['bias']:>+9.4f}"
        )

    print("\n" + "=" * 78)
    print("FINAL TEST SET INVENTORY PERFORMANCE (d_1914 to d_1941)")
    print("=" * 78)
    print(f"{'Metric':<34} | {'Seasonal Naive':>16} | {'LightGBM':>16}")
    print("-" * 72)
    print(f"{'Fill Rate (%)':<34} | {inv_summary_naive['fill_rate_percentage']:>15.2f}% | {inv_summary_lgb['fill_rate_percentage']:>15.2f}%")
    print(f"{'Total Demand (Units)':<34} | {inv_summary_naive['total_demand']:>16,.1f} | {inv_summary_lgb['total_demand']:>16,.1f}")
    print(f"{'Units Served':<34} | {inv_summary_naive['total_units_served']:>16,.1f} | {inv_summary_lgb['total_units_served']:>16,.1f}")
    print(f"{'Lost Sales (Units)':<34} | {inv_summary_naive['total_lost_units']:>16,.1f} | {inv_summary_lgb['total_lost_units']:>16,.1f}")
    print(f"{'Stockout Days (Instances)':<34} | {inv_summary_naive['stockout_days_count']:>16} | {inv_summary_lgb['stockout_days_count']:>16}")
    print(f"{'Stockout Rate (%)':<34} | {inv_summary_naive['stockout_rate']*100:>15.2f}% | {inv_summary_lgb['stockout_rate']*100:>15.2f}%")
    print(f"{'Average On-Hand / Product':<34} | {inv_summary_naive['avg_on_hand_per_product']:>16.2f} | {inv_summary_lgb['avg_on_hand_per_product']:>16.2f}")
    print(f"{'Total Holding Cost ($)':<34} | ${inv_summary_naive['total_holding_cost']:>15,.2f} | ${inv_summary_lgb['total_holding_cost']:>15,.2f}")
    print(f"{'Total Lost Sales Penalty ($)':<34} | ${inv_summary_naive['total_lost_sales_penalty']:>15,.2f} | ${inv_summary_lgb['total_lost_sales_penalty']:>15,.2f}")
    print(f"{'Total Operational Cost ($)':<34} | ${inv_summary_naive['total_cost']:>15,.2f} | ${inv_summary_lgb['total_cost']:>15,.2f}")
    print(f"{'Ending On-Hand Inventory':<34} | {inv_summary_naive['ending_inventory']:>16,.1f} | {inv_summary_lgb['ending_inventory']:>16,.1f}")
    print(f"{'Ending Outstanding Orders':<34} | {inv_summary_naive['ending_outstanding_orders']:>16,.1f} | {inv_summary_lgb['ending_outstanding_orders']:>16,.1f}")
    print("=" * 78)
    print(f"Final evaluation completed in {elapsed:.2f} seconds.")
    print("=" * 78)

    return final_report


if __name__ == "__main__":
    sys.exit(0 if run_final_evaluation() else 1)
