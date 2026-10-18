"""Unit and End-to-End Integration Tests for Stage 4 Final Evaluation.

Tests:
1. test_test_set_sealed_until_final: Verify test predictions cover exactly d_1914 to d_1941.
2. test_frozen_config: Verify test evaluation used the exact frozen hyperparameters from validation.
3. test_synthetic_fixture: End-to-end execution of features, modeling, recursive forecasting,
   and inventory simulation on a minimal synthetic dataset (5 products, 50 days).
   DISCLOSURE: Synthetic test fixture only; never presented as real M5 competition results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.config import ProjectConfig, load_config
from src.features import build_feature_matrix, get_feature_columns
from src.inventory_simulator import InventorySimulator
from src.models import LightGBMForecaster, SeasonalNaiveModel


def test_test_set_sealed_until_final():
    """Verify test predictions artifact exists and covers exactly d_1914 to d_1941."""
    config = load_config()
    test_preds_path = config.artifacts_path / "test_predictions.parquet"
    test_metrics_path = config.artifacts_path / "test_metrics.json"

    assert test_preds_path.exists(), f"Missing {test_preds_path}"
    assert test_metrics_path.exists(), f"Missing {test_metrics_path}"

    df_test = pd.read_parquet(test_preds_path)
    assert len(df_test) == 100 * 28, f"Expected 2,800 test predictions, got {len(df_test)}"
    assert df_test["day_num"].min() == 1914
    assert df_test["day_num"].max() == 1941
    assert set(df_test["d"].unique()) == {f"d_{i}" for i in range(1914, 1942)}

    # Verify predictions are non-negative
    assert (df_test["pred_seasonal_naive"] >= 0).all()
    assert (df_test["pred_lightgbm"] >= 0).all()


def test_frozen_config():
    """Verify the final evaluation strictly used the same frozen hyperparameters as validation."""
    config = load_config()
    val_path = config.artifacts_path / "validation_metrics.json"
    test_path = config.artifacts_path / "test_metrics.json"

    with open(val_path, "r", encoding="utf-8") as f:
        val_json = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    frozen_val_params = val_json["best_lightgbm_parameters"]
    frozen_test_params = test_json["dataset_scope"]["frozen_hyperparameters"]

    assert frozen_val_params == frozen_test_params, (
        f"Hyperparameters mutated between validation and test! {frozen_val_params} vs {frozen_test_params}"
    )


def test_synthetic_fixture():
    """End-to-end integration test on synthetic data (5 products, 50 days).

    Explicit Disclosure: This test uses mock synthetic time series to verify code paths.
    It does NOT represent actual M5 results.
    """
    n_products = 5
    n_days = 50
    items = [f"SYNTH_PROD_{i}" for i in range(n_products)]
    rows = []

    for item in items:
        for d in range(1, n_days + 1):
            wday = (d % 7) + 1
            rows.append({
                "id": f"{item}_STORE_1",
                "item_id": item,
                "dept_id": "SYNTH_DEPT",
                "cat_id": "SYNTH_CAT",
                "store_id": "STORE_1",
                "state_id": "ST",
                "d": f"d_{d}",
                "day_num": d,
                "date": str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=d - 1)),
                "wm_yr_wk": 12001 + (d // 7),
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wday - 1],
                "wday": wday,
                "month": 1,
                "year": 2020,
                "event_name_1": None,
                "event_type_1": None,
                "snap_CA": 0,
                "sell_price": 3.50,
                "sales": max(0, 10 + (wday * 2) + np.random.randint(-2, 3)),
            })

    synth_df = pd.DataFrame(rows)

    # 1. Feature Engineering
    config = load_config()
    synth_feat = build_feature_matrix(synth_df, config=config, origin_day=40)
    feat_cols = get_feature_columns(synth_feat)
    assert len(feat_cols) > 10

    # 2. Train LightGBM on days 15 to 40
    train_mask = (synth_feat["day_num"] >= 15) & (synth_feat["day_num"] <= 40)
    forecaster = LightGBMForecaster(params={"n_estimators": 10, "max_depth": 3, "verbose": -1})
    forecaster.fit(synth_feat.loc[train_mask, feat_cols], synth_feat.loc[train_mask, "sales"])

    # 3. Recursive forecast for days 41 to 50 (10 days)
    preds = forecaster.predict_recursive(
        origin_day=40,
        full_df=synth_feat,
        horizon=10,
        lags=[1, 7],
        windows=[7],
    )
    assert len(preds) == n_products * 10
    assert (preds["predicted_sales"] >= 0.0).all()

    # 4. Inventory Simulation
    initial_inv = {item: 20.0 for item in items}
    mock_stds = {item: 2.5 for item in items}
    sim = InventorySimulator(
        config=config,
        forecast_df=preds,
        actual_sales_df=synth_df,
        model_name="SynthPolicy",
        safety_stock_stds=mock_stds,
        initial_inventory=initial_inv,
        lead_time_days=3,
    )
    sim.run_simulation(list(range(41, 51)))
    summary = sim.get_summary()

    assert 0.0 <= summary["fill_rate"] <= 1.0
    assert summary["total_cost"] >= 0.0
    assert summary["total_units_served"] > 0
