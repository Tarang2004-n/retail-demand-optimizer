"""Model implementations for Retail Demand Forecasting.

Includes:
1. SeasonalNaiveModel: Baseline repeating last 7 observed days across the 28-day horizon.
2. LightGBMForecaster: LightGBM regressor with recursive 28-day forecasting and native categorical support.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config
from src.features import compute_step_lag_rolling_features, get_feature_columns


class SeasonalNaiveModel:
    """Seasonal Naive baseline model for weekly seasonal retail demand.

    Repeats the most recent 7 observed days across the entire 28-day horizon.
    For forecast origin T:
    - Step 1 (T+1): Sales at T-6 (same weekday 7 days prior)
    - Step 2 (T+2): Sales at T-5
    ...
    - Step 7 (T+7): Sales at T
    - Step 8 (T+8): Sales at T-6 again (modulo 7 cycle)
    ...
    - Step 28 (T+28): Sales at T
    """

    def __init__(self) -> None:
        self.history_df: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> SeasonalNaiveModel:
        """Store historical sales for lookups.

        Args:
            train_df: DataFrame containing at least ['item_id', 'day_num', 'sales'].
        """
        self.history_df = train_df[["item_id", "day_num", "sales"]].copy()
        return self

    def predict(
        self,
        origin_day: int,
        horizon: int = 28,
        full_df: Optional[pd.DataFrame] = None,
        item_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate seasonal naive forecasts for the specified horizon from origin_day.

        Args:
            origin_day: Day number of the forecast origin (e.g., 1829).
            horizon: Number of days to forecast (default: 28).
            full_df: Optional DataFrame with recent sales if history not fitted.
            item_ids: Optional list of item IDs to forecast for.

        Returns:
            DataFrame with columns ['item_id', 'day_num', 'd', 'predicted_sales'].
        """
        df_source = full_df if full_df is not None else self.history_df
        if df_source is None:
            raise ValueError("No historical sales data available. Call fit() first or provide full_df.")

        # Slice 7 observed days strictly preceding and including origin: [origin - 6, origin]
        start_day = origin_day - 6
        recent_sales = df_source[
            (df_source["day_num"] >= start_day) & (df_source["day_num"] <= origin_day)
        ].copy()

        if item_ids is None:
            item_ids = sorted(recent_sales["item_id"].unique().tolist())

        preds_list = []
        # Group by item and sort by day_num to ensure Monday..Sunday alignment
        recent_sales = recent_sales.sort_values(by=["item_id", "day_num"])

        for item_id, group in recent_sales.groupby("item_id", observed=False):
            if item_id not in item_ids:
                continue
            item_7_days = group["sales"].values
            if len(item_7_days) < 7:
                # Fallback if fewer than 7 days observed
                item_7_days = np.pad(
                    item_7_days,
                    (7 - len(item_7_days), 0),
                    mode="edge" if len(item_7_days) > 0 else "constant",
                )

            # Cycle through the 7 days for the full horizon
            for h in range(1, horizon + 1):
                target_day = origin_day + h
                # (h - 1) % 7 maps step 1 -> index 0 (origin-6), step 7 -> index 6 (origin)
                pred_val = float(item_7_days[(h - 1) % 7])
                preds_list.append({
                    "item_id": item_id,
                    "day_num": target_day,
                    "d": f"d_{target_day}",
                    "predicted_sales": max(0.0, pred_val),
                })

        pred_df = pd.DataFrame(preds_list)
        return pred_df


class LightGBMForecaster:
    """LightGBM regressor with recursive 28-day forecasting and native categorical support."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ) -> None:
        self.random_seed = random_seed
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params is not None:
            default_params.update(params)
        self.params = default_params
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> LightGBMForecaster:
        """Fit LightGBM model on training data.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
            feature_cols: List of feature names to use.
            categorical_cols: List of categorical feature names.

        Returns:
            self.
        """
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = get_feature_columns(X_train)

        if categorical_cols is not None:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = [
                c for c in ["item_id", "dept_id", "cat_id", "event_type_1"]
                if c in self.feature_cols
            ]

        # Ensure categoricals are category dtype
        X = X_train[self.feature_cols].copy()
        for col in self.categorical_cols:
            if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X,
            y_train,
            categorical_feature=self.categorical_cols,
        )
        return self

    def predict_recursive(
        self,
        origin_day: int,
        full_df: pd.DataFrame,
        horizon: int = 28,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate recursive 28-day forecasts starting from origin_day.

        CRITICAL SCIENTIFIC REQUIREMENT:
        At each step h > 1, lag and rolling features are updated with model PREDICTIONS,
        NEVER actual future sales.
        Calendar features are known in advance.
        Prices are frozen at the last known price at origin_day.

        Args:
            origin_day: Day number of forecast origin (e.g., 1829).
            full_df: DataFrame with all items and days up to origin_day + horizon.
            horizon: Number of forecast days (default: 28).
            lags: Lag list (default: [1, 7, 14, 21, 28]).
            windows: Rolling window list (default: [7, 14, 28]).

        Returns:
            DataFrame with columns ['item_id', 'day_num', 'd', 'predicted_sales'].
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() before predicting.")

        if lags is None:
            lags = [1, 7, 14, 21, 28]
        if windows is None:
            windows = [7, 14, 28]

        max_lookback = max(max(lags), max(windows)) + 14  # lookback buffer

        # Build initial history buffer strictly up to origin_day for each product
        history_mask = (full_df["day_num"] <= origin_day) & (
            full_df["day_num"] > origin_day - max_lookback
        )
        hist_df = full_df[history_mask].sort_values(by=["item_id", "day_num"])

        item_history: Dict[str, List[float]] = {}
        for item_id, group in hist_df.groupby("item_id", observed=False):
            item_history[item_id] = list(group["sales"].astype(float).values)

        # Slice future metadata template (calendar, price frozen at origin, categoricals)
        future_mask = full_df["day_num"].between(origin_day + 1, origin_day + horizon)
        future_meta = full_df[future_mask].copy()

        all_predictions: List[Dict[str, Any]] = []

        # Recursive loop for step h = 1 .. horizon
        for h in range(1, horizon + 1):
            curr_day = origin_day + h
            day_slice = future_meta[future_meta["day_num"] == curr_day].copy()

            # Compute recursive lag & rolling statistics from item_history
            for lag in lags:
                day_slice[f"sales_lag_{lag}"] = [
                    float(item_history[item][-lag])
                    if len(item_history[item]) >= lag
                    else float(item_history[item][0])
                    for item in day_slice["item_id"]
                ]

            for w in windows:
                for stat in ["mean", "std", "min", "max"]:
                    vals = []
                    for item in day_slice["item_id"]:
                        w_vals = (
                            item_history[item][-w:]
                            if len(item_history[item]) >= w
                            else item_history[item]
                        )
                        if stat == "mean":
                            vals.append(float(np.mean(w_vals)))
                        elif stat == "std":
                            vals.append(
                                float(np.std(w_vals, ddof=1)) if len(w_vals) > 1 else 0.0
                            )
                        elif stat == "min":
                            vals.append(float(np.min(w_vals)))
                        elif stat == "max":
                            vals.append(float(np.max(w_vals)))
                    day_slice[f"rolling_{stat}_{w}"] = vals

            # Prepare feature matrix for step h
            X_step = day_slice[self.feature_cols].copy()
            for col in self.categorical_cols:
                if col in X_step.columns and not isinstance(X_step[col].dtype, pd.CategoricalDtype):
                    X_step[col] = X_step[col].astype("category")

            # Predict step h for all products
            preds = self.model.predict(X_step)
            # Clip at 0 (sales cannot be negative)
            preds = np.clip(preds, 0.0, None)

            # Update history buffer with predictions (NEVER actual sales)
            for i, item in enumerate(day_slice["item_id"]):
                pred_val = float(preds[i])
                item_history[item].append(pred_val)
                all_predictions.append({
                    "item_id": item,
                    "day_num": curr_day,
                    "d": f"d_{curr_day}",
                    "predicted_sales": pred_val,
                })

        pred_df = pd.DataFrame(all_predictions)
        return pred_df

    def save(self, filepath: Path | str) -> None:
        """Save model object to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "params": self.params,
                "model": self.model,
                "feature_cols": self.feature_cols,
                "categorical_cols": self.categorical_cols,
                "random_seed": self.random_seed,
            }, f)

    @classmethod
    def load(cls, filepath: Path | str) -> LightGBMForecaster:
        """Load model object from disk."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        forecaster = cls(params=data["params"], random_seed=data.get("random_seed", 42))
        forecaster.model = data["model"]
        forecaster.feature_cols = data["feature_cols"]
        forecaster.categorical_cols = data["categorical_cols"]
        return forecaster
