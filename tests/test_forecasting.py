"""Unit and Integration Tests for Stage 2 Forecasting & Validation.

Tests:
1. No future leakage: Mutating future actual sales doesn't change forecasts at earlier origins.
2. Prediction count: Exactly 28 non-negative predictions per product per origin.
3. No train-eval overlap: Training and evaluation day ranges never overlap.
4. Lag features shifted: lag_1 at day t equals sales at day t-1.
5. Recursive uses predictions: Step h>1 features strictly use predictions from h-1, not actuals.
"""

from __future__ import annotations

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
from src.features import (
    build_feature_matrix,
    compute_step_lag_rolling_features,
    create_calendar_features,
    create_category_features,
    create_lag_features,
    create_price_features,
    create_rolling_features,
    get_feature_columns,
)
from src.models import LightGBMForecaster, SeasonalNaiveModel


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Create a minimal synthetic retail time series for deterministic testing."""
    n_days = 80
    items = ["ITEM_A", "ITEM_B"]
    rows = []

    for item in items:
        base_level = 10 if item == "ITEM_A" else 25
        for d_idx in range(1, n_days + 1):
            # Day of week cycle: 0..6
            weekday_val = (d_idx - 1) % 7
            sales_val = base_level + weekday_val * 2
            rows.append({
                "id": f"{item}_CA_1",
                "item_id": item,
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "d": f"d_{d_idx}",
                "day_num": d_idx,
                "date": str(pd.Timestamp("2016-01-01") + pd.Timedelta(days=d_idx - 1)),
                "wm_yr_wk": 11601 + (d_idx // 7),
                "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday_val],
                "wday": weekday_val + 1,
                "month": 1 if d_idx <= 31 else 2,
                "year": 2016,
                "event_name_1": None,
                "event_type_1": None,
                "snap_CA": 1 if d_idx % 5 == 0 else 0,
                "sell_price": 2.50 if item == "ITEM_A" else 4.00,
                "sales": sales_val,
            })

    df = pd.DataFrame(rows)
    return df


def test_lag_features_shifted(synthetic_data: pd.DataFrame):
    """Verify lag_1 at day t strictly equals sales at day t-1, never day t."""
    df = create_lag_features(synthetic_data, lags=[1, 7], target_col="sales")

    item_a = df[df["item_id"] == "ITEM_A"].sort_values("day_num").reset_index(drop=True)

    # Day 1: lag 1 should be NaN (no prior day)
    assert pd.isna(item_a.loc[0, "sales_lag_1"])

    # Day 2: lag 1 should equal Day 1 sales
    assert item_a.loc[1, "sales_lag_1"] == item_a.loc[0, "sales"]

    # Day 10: lag 1 should equal Day 9 sales, lag 7 should equal Day 3 sales
    assert item_a.loc[9, "sales_lag_1"] == item_a.loc[8, "sales"]
    assert item_a.loc[9, "sales_lag_7"] == item_a.loc[2, "sales"]


def test_rolling_features_shifted(synthetic_data: pd.DataFrame):
    """Verify rolling statistics exclude day t sales and use only shifted history."""
    df = create_rolling_features(synthetic_data, windows=[3], target_col="sales")
    item_a = df[df["item_id"] == "ITEM_A"].sort_values("day_num").reset_index(drop=True)

    # Day 1: rolling mean of shifted sales should be NaN
    assert pd.isna(item_a.loc[0, "rolling_mean_3"])

    # Day 4: rolling mean over window 3 should be mean of days 1, 2, 3
    expected_mean_day4 = (
        item_a.loc[0, "sales"] + item_a.loc[1, "sales"] + item_a.loc[2, "sales"]
    ) / 3.0
    assert np.isclose(item_a.loc[3, "rolling_mean_3"], expected_mean_day4)


def test_no_future_leakage(synthetic_data: pd.DataFrame):
    """Verify forecasts from origin day T do not change when future actual sales are mutated."""
    origin_day = 50
    horizon = 14

    # 1. Build features and train on original synthetic data
    config = load_config()
    df_feat = build_feature_matrix(synthetic_data, config=config, origin_day=origin_day)
    feature_cols = get_feature_columns(df_feat)

    train_mask = (df_feat["day_num"] >= 15) & (df_feat["day_num"] <= origin_day)
    X_train = df_feat.loc[train_mask, feature_cols]
    y_train = df_feat.loc[train_mask, "sales"]

    forecaster = LightGBMForecaster(params={"n_estimators": 20, "max_depth": 4, "verbose": -1})
    forecaster.fit(X_train, y_train, feature_cols=feature_cols)

    preds_original = forecaster.predict_recursive(
        origin_day=origin_day,
        full_df=df_feat,
        horizon=horizon,
        lags=[1, 7],
        windows=[7],
    )

    # 2. Mutate future sales (day > origin_day) with massive spikes
    mutated_data = synthetic_data.copy()
    future_mask = mutated_data["day_num"] > origin_day
    mutated_data.loc[future_mask, "sales"] = 999999

    # Rebuild feature matrix with mutated future data
    df_feat_mutated = build_feature_matrix(mutated_data, config=config, origin_day=origin_day)

    # Future prices and future actuals must not influence predictions from origin
    preds_mutated = forecaster.predict_recursive(
        origin_day=origin_day,
        full_df=df_feat_mutated,
        horizon=horizon,
        lags=[1, 7],
        windows=[7],
    )

    # Compare predictions
    np.testing.assert_allclose(
        preds_original["predicted_sales"].values,
        preds_mutated["predicted_sales"].values,
        rtol=1e-5,
        err_msg="Predictions changed when future sales were mutated! Data leakage detected.",
    )


def test_prediction_count(synthetic_data: pd.DataFrame):
    """Verify exactly 28 non-negative predictions per product per origin."""
    origin_day = 50
    horizon = 28
    config = load_config()
    df_feat = build_feature_matrix(synthetic_data, config=config, origin_day=origin_day)
    feature_cols = get_feature_columns(df_feat)

    train_mask = (df_feat["day_num"] >= 15) & (df_feat["day_num"] <= origin_day)
    forecaster = LightGBMForecaster(params={"n_estimators": 10, "max_depth": 3, "verbose": -1})
    forecaster.fit(df_feat.loc[train_mask, feature_cols], df_feat.loc[train_mask, "sales"])

    preds = forecaster.predict_recursive(
        origin_day=origin_day,
        full_df=df_feat,
        horizon=horizon,
        lags=[1, 7],
        windows=[7],
    )

    n_items = synthetic_data["item_id"].nunique()
    expected_total_preds = n_items * horizon

    assert len(preds) == expected_total_preds
    for item in synthetic_data["item_id"].unique():
        item_preds = preds[preds["item_id"] == item]
        assert len(item_preds) == horizon
        # Verify non-negativity
        assert (item_preds["predicted_sales"] >= 0).all()
        # Verify no NaN or infinite values
        assert not item_preds["predicted_sales"].isna().any()
        assert not np.isinf(item_preds["predicted_sales"]).any()


def test_no_train_eval_overlap():
    """Verify training and evaluation intervals never overlap in validation windows or test set."""
    config = load_config()
    val_windows = config.temporal.validation_windows
    train_end = config.temporal.train_end_d
    test_start = config.temporal.test_start_d
    test_end = config.temporal.test_end_d

    # Selection training ends before first validation window
    assert train_end < val_windows[0].start_d

    for i in range(len(val_windows)):
        w = val_windows[i]
        # Window start must be <= window end
        assert w.start_d <= w.end_d
        assert (w.end_d - w.start_d + 1) == config.temporal.forecast_horizon

        # Consecutive validation windows must be strictly sequential
        if i > 0:
            assert w.start_d == val_windows[i - 1].end_d + 1

    # Test set strictly follows last validation window
    last_val_end = val_windows[-1].end_d
    assert test_start == last_val_end + 1
    assert test_end == config.temporal.total_days


def test_recursive_uses_predictions():
    """Verify that in recursive forecasting, step h>1 features strictly use predictions, not actuals."""
    lags = [1]
    windows = [3]

    # History contains known sales for days 1, 2, 3: [10.0, 10.0, 10.0]
    history = [10.0, 10.0, 10.0]

    # At step 1: features are computed from history
    feats_step1 = compute_step_lag_rolling_features(np.array(history), lags, windows)
    assert feats_step1["sales_lag_1"] == 10.0

    # Step 1 predicts 50.0 (even if actual was 0.0)
    pred_step1 = 50.0
    history.append(pred_step1)

    # At step 2: lag_1 must equal the prediction 50.0!
    feats_step2 = compute_step_lag_rolling_features(np.array(history), lags, windows)
    assert feats_step2["sales_lag_1"] == 50.0

    # Rolling mean at step 2 must include the predicted value 50.0
    # Window of 3 includes: [10.0, 10.0, 50.0] -> mean = 70 / 3
    expected_rolling_mean = (10.0 + 10.0 + 50.0) / 3.0
    assert np.isclose(feats_step2["rolling_mean_3"], expected_rolling_mean)
