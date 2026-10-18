"""CLI Entry Point for Stage 2: Forecasting and Validation.

Usage:
    python src/run_stage2.py
    python src/run_stage2.py --config config.yaml

SCIENTIFIC INTEGRITY:
- Uses ONLY training and validation windows (d_1 to d_1913).
- Strictly HOLDS OUT test set (d_1914 to d_1941) — untouched.
- Recursive 28-day forecasts update lag/rolling features with predictions, never future sales.
- Evaluates Seasonal Naive baseline vs. LightGBM without bias.
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
    plot_actual_vs_predicted,
    plot_permutation_importance,
)
from src.features import (
    build_feature_matrix,
    get_feature_columns,
)
from src.models import LightGBMForecaster, SeasonalNaiveModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 2: Retail Demand Forecasting & Multi-Window Validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML (default: config.yaml).",
    )
    return parser.parse_args()


def tune_lightgbm_grid(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    config: ProjectConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate a small, laptop-friendly candidate grid on validation window 1.

    Args:
        df_feat: Full engineered feature DataFrame.
        feature_cols: List of feature column names.
        config: Typed ProjectConfig instance.

    Returns:
        Tuple of (best_params_dict, tuning_results_df).
    """
    train_end = config.temporal.train_end_d  # 1829
    val1_start = 1830
    val1_end = 1857

    # Training slice (excluding initial incomplete lookback days)
    train_mask = (df_feat["day_num"] >= 35) & (df_feat["day_num"] <= train_end)
    X_train = df_feat.loc[train_mask, feature_cols]
    y_train = df_feat.loc[train_mask, "sales"]

    # Validation 1 actuals
    val1_actuals = df_feat[df_feat["day_num"].between(val1_start, val1_end)][
        ["item_id", "day_num", "sales"]
    ].copy()

    # Small, laptop-friendly candidate grid
    candidate_configs = [
        {
            "name": "Config_1_Fast",
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        {
            "name": "Config_2_Balanced",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        {
            "name": "Config_3_Capacity",
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
    ]

    print("\n--- Hyperparameter Grid Search (Evaluated on Val1: d_1830-d_1857) ---")
    results = []

    best_mae = float("inf")
    best_config = candidate_configs[0]["params"]

    for cand in candidate_configs:
        t0 = time.time()
        forecaster = LightGBMForecaster(params=cand["params"], random_seed=config.data.random_seed)
        forecaster.fit(X_train, y_train, feature_cols=feature_cols)

        # Generate recursive forecast for Val1
        preds = forecaster.predict_recursive(
            origin_day=train_end,
            full_df=df_feat,
            horizon=config.temporal.forecast_horizon,
            lags=config.forecasting.lightgbm.lag_features,
            windows=config.forecasting.lightgbm.rolling_windows,
        )

        merged = pd.merge(preds, val1_actuals, on=["item_id", "day_num"])
        mae = float(np.mean(np.abs(merged["sales"] - merged["predicted_sales"])))
        rmse = float(np.sqrt(np.mean((merged["sales"] - merged["predicted_sales"]) ** 2)))
        wape = float(np.sum(np.abs(merged["sales"] - merged["predicted_sales"])) / np.sum(merged["sales"]))
        bias = float(np.mean(merged["predicted_sales"] - merged["sales"]))

        elapsed = time.time() - t0
        print(
            f"  • {cand['name']:<18} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
            f"WAPE: {wape:.4f} | Bias: {bias:+.4f} | Time: {elapsed:.2f}s"
        )

        results.append({
            "name": cand["name"],
            "params": cand["params"],
            "val1_mae": mae,
            "val1_rmse": rmse,
            "val1_wape": wape,
            "val1_bias": bias,
            "fit_time_sec": elapsed,
        })

        if mae < best_mae:
            best_mae = mae
            best_config = cand["params"]

    results_df = pd.DataFrame(results)
    return best_config, results_df


def run_stage2_pipeline(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Execute Stage 2: Feature Engineering, Model Training, Validation, and Evaluation.

    Args:
        config: Typed ProjectConfig instance.

    Returns:
        Dictionary of validation results, model paths, and performance summaries.
    """
    if config is None:
        config = load_config()

    start_time = time.time()
    print("=" * 75)
    print("STAGE 2: FORECASTING & MULTI-WINDOW VALIDATION")
    print("=" * 75)

    # 1. Load Processed Parquet
    parquet_path = config.processed_data_path / "sales_processed.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {parquet_path}. Run Stage 1 first."
        )
    print(f"Loading processed dataset from {parquet_path}...")
    df_raw = pd.read_parquet(parquet_path)
    print(f"  ✓ Loaded {len(df_raw):,} records.")

    # 2. Build Feature Matrix (Excluding Test Set)
    print("\nEngineering features (lags, rolling stats, calendar, price)...")
    # Clamp future prices at origin of test set (d_1913) so validation features are pure
    df_feat = build_feature_matrix(df_raw, config=config, origin_day=1913)
    feature_cols = get_feature_columns(df_feat)
    print(f"  ✓ Engineered {len(feature_cols)} predictor features.")

    # Assign volume segments (high, medium, low) based strictly on training sales
    volume_segments = assign_volume_segments(df_raw, train_end_d=config.temporal.train_end_d)

    # 3. Hyperparameter Tuning
    best_params, tuning_df = tune_lightgbm_grid(df_feat, feature_cols, config)
    print(f"\nSelected Best LightGBM Parameters: {best_params}")

    # 4. Multi-Window Expanding Validation
    # Windows:
    # val1: origin 1829, forecast 1830-1857
    # val2: origin 1857, forecast 1858-1885
    # val3: origin 1885, forecast 1886-1913
    val_windows = config.temporal.validation_windows
    horizon = config.temporal.forecast_horizon

    seasonal_naive = SeasonalNaiveModel()
    seasonal_naive.fit(df_feat[df_feat["day_num"] <= config.temporal.train_end_d])

    naive_predictions = []
    lgb_predictions = []

    print("\nGenerating recursive 28-day forecasts across 3 validation windows...")
    last_trained_forecaster: Optional[LightGBMForecaster] = None

    for w in val_windows:
        origin = w.start_d - 1
        t0 = time.time()
        print(f"  Window {w.name} (Origin d_{origin} -> Forecast d_{w.start_d} to d_{w.end_d}):")

        # Seasonal Naive predictions
        naive_w_preds = seasonal_naive.predict(
            origin_day=origin,
            horizon=horizon,
            full_df=df_feat,
        )
        naive_w_preds["window"] = w.name
        naive_predictions.append(naive_w_preds)

        # LightGBM Expanding Window Training
        # Train on all available days up to origin (starting at day 35 for complete lookbacks)
        train_mask = (df_feat["day_num"] >= 35) & (df_feat["day_num"] <= origin)
        X_train_w = df_feat.loc[train_mask, feature_cols]
        y_train_w = df_feat.loc[train_mask, "sales"]

        forecaster = LightGBMForecaster(params=best_params, random_seed=config.data.random_seed)
        forecaster.fit(X_train_w, y_train_w, feature_cols=feature_cols)
        last_trained_forecaster = forecaster

        # Recursive prediction using ONLY predictions for h > 1
        lgb_w_preds = forecaster.predict_recursive(
            origin_day=origin,
            full_df=df_feat,
            horizon=horizon,
            lags=config.forecasting.lightgbm.lag_features,
            windows=config.forecasting.lightgbm.rolling_windows,
        )
        lgb_w_preds["window"] = w.name
        lgb_predictions.append(lgb_w_preds)

        print(f"    ✓ Completed 28-day forecasts in {time.time() - t0:.2f}s.")

    naive_all_preds = pd.concat(naive_predictions, ignore_index=True)
    lgb_all_preds = pd.concat(lgb_predictions, ignore_index=True)

    # 5. Evaluate Forecasts
    print("\nEvaluating forecast performance across models and segments...")
    eval_naive = evaluate_forecasts(
        actual_df=df_feat,
        pred_df=naive_all_preds,
        model_name="Seasonal_Naive",
        volume_segments=volume_segments,
    )
    eval_lgb = evaluate_forecasts(
        actual_df=df_feat,
        pred_df=lgb_all_preds,
        model_name="LightGBM_Recursive",
        volume_segments=volume_segments,
    )

    # 6. Generate Diagnostic Plots
    figures_dir = config.figures_path
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pick 4 representative products: high volume, medium volume, low volume, intermittent
    # Determine item with highest zero-sales rate
    pre_test_data = df_feat[df_feat["day_num"] <= 1913]
    zero_rates = (
        pre_test_data.groupby("item_id", observed=False)["sales"]
        .apply(lambda s: (s == 0).mean())
        .sort_values(ascending=False)
    )
    intermittent_item = zero_rates.index[0]  # item with highest zero %

    # Find highest volume item
    train_totals = (
        df_feat[df_feat["day_num"] <= config.temporal.train_end_d]
        .groupby("item_id", observed=False)["sales"]
        .sum()
        .sort_values(ascending=False)
    )
    high_item = train_totals.index[0]
    medium_item = train_totals.index[len(train_totals) // 2]
    low_item = train_totals.index[-5]

    representative_products = [high_item, medium_item, low_item, intermittent_item]
    print(f"Representative products selected for visual diagnostics:")
    print(f"  • High volume:    {high_item}")
    print(f"  • Medium volume:  {medium_item}")
    print(f"  • Low volume:     {low_item}")
    print(f"  • Intermittent:   {intermittent_item}")

    plot_actual_vs_predicted(
        actual_df=df_feat,
        naive_pred_df=naive_all_preds,
        lgb_pred_df=lgb_all_preds,
        product_ids=representative_products,
        figures_dir=figures_dir,
        window_name="val3",
    )
    print("  ✓ Saved actual vs. predicted diagnostic plot (val3).")

    # Permutation Importance on val3
    val3_mask = df_feat["day_num"].between(1886, 1913)
    X_val3 = df_feat.loc[val3_mask, feature_cols].copy()
    y_val3 = df_feat.loc[val3_mask, "sales"]

    if last_trained_forecaster is not None and last_trained_forecaster.model is not None:
        print("Computing permutation feature importance on validation window 3...")
        imp_df, _ = plot_permutation_importance(
            model=last_trained_forecaster.model,
            X_val=X_val3,
            y_val=y_val3,
            figures_dir=figures_dir,
            n_repeats=3,
            top_n=12,
            seed=config.data.random_seed,
        )
        print("  ✓ Saved permutation feature importance plot.")
    else:
        imp_df = pd.DataFrame()

    # 7. Save Models, Predictions, and Evaluation Metrics
    artifacts_dir = config.artifacts_path
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save best LightGBM model
    model_save_path = artifacts_dir / "best_lightgbm_model.pkl"
    if last_trained_forecaster is not None:
        last_trained_forecaster.save(model_save_path)
        print(f"  ✓ Saved trained LightGBM model to {model_save_path}")

    # Save validation predictions
    val_preds_combined = pd.merge(
        naive_all_preds.rename(columns={"predicted_sales": "pred_seasonal_naive"}),
        lgb_all_preds.rename(columns={"predicted_sales": "pred_lightgbm"}),
        on=["item_id", "day_num", "d", "window"],
    )
    preds_save_path = artifacts_dir / "validation_predictions.parquet"
    val_preds_combined.to_parquet(preds_save_path, index=False)
    print(f"  ✓ Saved validation predictions to {preds_save_path}")

    # Save comprehensive validation metrics JSON
    metrics_payload = {
        "dataset_scope": {
            "store_id": config.data.store_id,
            "n_products": config.data.n_products,
            "train_end_d": config.temporal.train_end_d,
            "validation_windows": [w.name for w in val_windows],
            "test_set_touched": False,
        },
        "best_lightgbm_parameters": best_params,
        "models": {
            "seasonal_naive": eval_naive,
            "lightgbm": eval_lgb,
        },
        "top_features_by_importance": imp_df.head(10).to_dict(orient="records"),
    }
    metrics_save_path = artifacts_dir / "validation_metrics.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  ✓ Saved validation metrics report to {metrics_save_path}")

    # 8. Print Results Tables
    print("\n" + "=" * 75)
    print("STAGE 2 VALIDATION RESULTS COMPARISON")
    print("=" * 75)

    # Overall Table
    print(f"\n{'Model':<22} | {'MAE':>8} | {'RMSE':>8} | {'WAPE':>8} | {'Bias':>9}")
    print("-" * 65)
    for res in [eval_naive, eval_lgb]:
        ov = res["overall"]
        print(
            f"{res['model_name']:<22} | {ov['mae']:>8.4f} | {ov['rmse']:>8.4f} | "
            f"{ov['wape']:>8.4f} | {ov['bias']:>+9.4f}"
        )

    # By Window Table
    print("\n--- Performance by Validation Window (MAE / WAPE) ---")
    print(f"{'Window':<10} | {'Seasonal Naive MAE':>18} | {'LightGBM MAE':>14} | {'Naive WAPE':>11} | {'LGBM WAPE':>10}")
    print("-" * 72)
    for w in [win.name for win in val_windows]:
        n_w = eval_naive["by_window"][w]
        l_w = eval_lgb["by_window"][w]
        print(
            f"{w:<10} | {n_w['mae']:>18.4f} | {l_w['mae']:>14.4f} | "
            f"{n_w['wape']:>11.4f} | {l_w['wape']:>10.4f}"
        )

    # By Sales-Volume Segment Table
    print("\n--- Performance by Sales Volume Segment (MAE / WAPE) ---")
    print(f"{'Segment':<10} | {'Seasonal Naive MAE':>18} | {'LightGBM MAE':>14} | {'Naive WAPE':>11} | {'LGBM WAPE':>10}")
    print("-" * 72)
    for seg in ["high", "medium", "low"]:
        n_s = eval_naive["by_volume_segment"][seg]
        l_s = eval_lgb["by_volume_segment"][seg]
        print(
            f"{seg:<10} | {n_s['mae']:>18.4f} | {l_s['mae']:>14.4f} | "
            f"{n_s['wape']:>11.4f} | {l_s['wape']:>10.4f}"
        )

    # By Category Table
    print("\n--- Performance by Category (MAE / WAPE) ---")
    print(f"{'Category':<10} | {'Seasonal Naive MAE':>18} | {'LightGBM MAE':>14} | {'Naive WAPE':>11} | {'LGBM WAPE':>10}")
    print("-" * 72)
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        n_c = eval_naive["by_category"][cat]
        l_c = eval_lgb["by_category"][cat]
        print(
            f"{cat:<10} | {n_c['mae']:>18.4f} | {l_c['mae']:>14.4f} | "
            f"{n_c['wape']:>11.4f} | {l_c['wape']:>10.4f}"
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"Stage 2 pipeline completed in {elapsed:.2f} seconds.")
    print("=" * 75)

    return {
        "best_params": best_params,
        "eval_naive": eval_naive,
        "eval_lgb": eval_lgb,
        "tuning_results": tuning_df,
    }


if __name__ == "__main__":
    sys.exit(0 if run_stage2_pipeline() else 1)
