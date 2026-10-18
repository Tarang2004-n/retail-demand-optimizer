"""Feature engineering for Retail Demand Forecasting.

Creates lag, rolling, calendar, price, and category features.
CRITICAL:
- Lag features are shifted before computation (lag_1 at day t is sales at day t-1).
- Rolling statistics are computed strictly on shifted sales.
- Calendar and event features are known in advance.
- Price features use only the last known price at origin to prevent future leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config


def create_lag_features(
    df: pd.DataFrame,
    lags: List[int],
    target_col: str = "sales",
    group_col: str = "item_id",
) -> pd.DataFrame:
    """Create lag features for sales.

    CRITICAL SCIENTIFIC REQUIREMENT:
    Sales are shifted BEFORE calculating. Lag 1 at day t uses sales at day t-1,
    never sales at day t.

    Args:
        df: Input DataFrame containing target and group columns, sorted by (group_col, day_num).
        lags: List of integer lag offsets (e.g., [1, 7, 14, 21, 28]).
        target_col: Column to lag (default: 'sales').
        group_col: Product group identifier (default: 'item_id').

    Returns:
        DataFrame with added lag feature columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_col, observed=False)[target_col].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int],
    target_col: str = "sales",
    group_col: str = "item_id",
) -> pd.DataFrame:
    """Create rolling statistics (mean, std, min, max) over specified windows.

    CRITICAL SCIENTIFIC REQUIREMENT:
    Rolling features are computed strictly on SHIFTED sales (t-1 and older).
    Day t sales is excluded from day t rolling statistics.

    Args:
        df: Input DataFrame sorted by (group_col, day_num).
        windows: List of window sizes (e.g., [7, 14, 28]).
        target_col: Column to compute rolling statistics on.
        group_col: Product group identifier.

    Returns:
        DataFrame with added rolling feature columns.
    """
    df = df.copy()
    grouped = df.groupby(group_col, observed=False)[target_col]

    for w in windows:
        # Use transform with lambda to preserve exact DataFrame index alignment
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std().fillna(0.0)
        )
        df[f"rolling_min_{w}"] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).min()
        )
        df[f"rolling_max_{w}"] = grouped.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).max()
        )
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, day-of-week, event, and SNAP indicator features.

    Calendar features are known in advance and valid across forecast horizons.

    Args:
        df: Input DataFrame containing date, weekday, events, snap columns.

    Returns:
        DataFrame with enriched calendar features.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    df["day_of_week"] = dates.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["year"] = dates.dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Event indicators
    if "event_name_1" in df.columns:
        df["has_event"] = df["event_name_1"].notna().astype(int)
    else:
        df["has_event"] = 0

    if "event_type_1" in df.columns:
        df["event_type_1"] = df["event_type_1"].fillna("None").astype("category")

    # SNAP indicators
    if "snap_CA" in df.columns:
        df["snap_CA"] = df["snap_CA"].fillna(0).astype(int)

    return df


def create_price_features(
    df: pd.DataFrame,
    max_known_day: Optional[int] = None,
    group_col: str = "item_id",
) -> pd.DataFrame:
    """Create price features ensuring zero future price leakage.

    If max_known_day is set (e.g. at forecast origin), future prices (day_num > max_known_day)
    are strictly clamped to the last observed price at or before max_known_day.

    Args:
        df: Input DataFrame containing sell_price, day_num, item_id.
        max_known_day: Optional forecast origin day cutoff.
        group_col: Product group identifier.

    Returns:
        DataFrame with cleaned price and price momentum features.
    """
    df = df.copy()

    # Forward fill then backward fill missing prices per item
    df["sell_price"] = df.groupby(group_col, observed=False)["sell_price"].ffill().bfill()

    if max_known_day is not None and "day_num" in df.columns:
        # Determine last known price per item at or before max_known_day
        known_mask = df["day_num"] <= max_known_day
        last_prices = (
            df[known_mask]
            .groupby(group_col, observed=False)["sell_price"]
            .last()
            .to_dict()
        )
        # Apply last known price to any future days
        future_mask = df["day_num"] > max_known_day
        for item, price in last_prices.items():
            item_future = future_mask & (df[group_col] == item)
            df.loc[item_future, "sell_price"] = price

    # Price momentum: current price divided by 7-day rolling mean price
    grouped_price = df.groupby(group_col, observed=False)["sell_price"]
    price_rolling_7 = grouped_price.transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["price_change_ratio"] = (df["sell_price"] / price_rolling_7).fillna(1.0)

    return df


def create_category_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Encode categorical features as pandas 'category' dtype for native LightGBM handling.

    Args:
        df: Input DataFrame.
        categorical_cols: List of column names to cast to category.

    Returns:
        DataFrame with categorical dtypes.
    """
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = ["cat_id", "dept_id", "item_id"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
    origin_day: Optional[int] = None,
) -> pd.DataFrame:
    """End-to-end feature pipeline creating all features on the dataset.

    Args:
        df: Cleaned sales DataFrame.
        config: Typed ProjectConfig instance.
        origin_day: Optional cutoff day to freeze prices and prevent future leaks.

    Returns:
        DataFrame containing all engineered features.
    """
    if config is None:
        config = load_config()

    lags = config.forecasting.lightgbm.lag_features
    windows = config.forecasting.lightgbm.rolling_windows

    # Sort deterministically
    df = df.sort_values(by=["item_id", "day_num"]).reset_index(drop=True)

    # 1. Price features (with optional cutoff)
    df = create_price_features(df, max_known_day=origin_day)

    # 2. Calendar features
    df = create_calendar_features(df)

    # 3. Category features
    df = create_category_features(df)

    # 4. Lag features
    df = create_lag_features(df, lags=lags)

    # 5. Rolling features
    df = create_rolling_features(df, windows=windows)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of feature column names used for model training/inference."""
    excluded = {
        "id", "store_id", "state_id", "d", "date", "sales", "day_num",
        "wm_yr_wk", "weekday", "event_name_1", "event_name_2", "event_type_2",
        "snap_TX", "snap_WI",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    return feature_cols


def compute_step_lag_rolling_features(
    sales_history: np.ndarray,
    lags: List[int],
    windows: List[int],
) -> dict[str, float]:
    """Compute lag and rolling features for a single product at step h using sales history.

    Used inside the recursive forecasting loop.
    sales_history is a 1D array of sales up to step h-1.
    The most recent value in sales_history is at index -1 (corresponds to day t-1).

    Args:
        sales_history: Array of historical (or recursively predicted) sales.
        lags: List of lag steps.
        windows: List of window lengths.

    Returns:
        Dictionary of feature names and scalar values.
    """
    feats = {}
    history_len = len(sales_history)

    # Lags: lag 1 is sales_history[-1], lag 7 is sales_history[-7], etc.
    for lag in lags:
        if history_len >= lag:
            feats[f"sales_lag_{lag}"] = float(sales_history[-lag])
        else:
            feats[f"sales_lag_{lag}"] = float(sales_history[0])

    # Rolling: computed on shifted history (which is exactly sales_history!)
    for w in windows:
        window_vals = sales_history[-w:] if history_len >= w else sales_history
        feats[f"rolling_mean_{w}"] = float(np.mean(window_vals))
        feats[f"rolling_std_{w}"] = float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
        feats[f"rolling_min_{w}"] = float(np.min(window_vals))
        feats[f"rolling_max_{w}"] = float(np.max(window_vals))

    return feats
