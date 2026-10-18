"""Evaluation metrics, segmented performance reporting, and diagnostics.

Implements MAE, RMSE, WAPE (with zero-denominator guard), signed bias,
segmented analysis (by window, category, and sales volume), and Plotly visualizations.

SCIENTIFIC INTEGRITY:
- Permutation importance measures predictive reliance, NOT real-world causal impact.
- WAPE includes a zero-denominator guard (returns NaN if total actual demand is 0).
- Distinguishes observed sales from unobserved demand.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def calculate_mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error (MAE)."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_wape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error (WAPE).

    Includes zero-denominator guard: returns NaN if sum of actuals is zero.
    WAPE = sum(|actual - predicted|) / sum(actual)
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    sum_actual = np.sum(y_true)
    if sum_actual <= 0.0 or np.isnan(sum_actual):
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / sum_actual)


def calculate_bias(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Signed mean forecast bias (predicted - actual).

    Positive bias indicates systematic over-forecasting.
    Negative bias indicates systematic under-forecasting.
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return float(np.mean(y_pred - y_true))


def compute_metrics_dict(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Calculate all standard metrics and return as dictionary."""
    return {
        "mae": round(calculate_mae(actual, predicted), 4),
        "rmse": round(calculate_rmse(actual, predicted), 4),
        "wape": round(calculate_wape(actual, predicted), 4),
        "bias": round(calculate_bias(actual, predicted), 4),
    }


def assign_volume_segments(
    df: pd.DataFrame,
    train_end_d: int = 1829,
) -> Dict[str, str]:
    """Segment products into 'high', 'medium', and 'low' volume based on training period average.

    Args:
        df: Processed sales DataFrame.
        train_end_d: Last day of product selection/training period.

    Returns:
        Dictionary mapping item_id to segment name.
    """
    train_mask = df["day_num"] <= train_end_d
    avg_sales = (
        df[train_mask]
        .groupby("item_id", observed=False)["sales"]
        .mean()
        .sort_values(ascending=False)
    )

    q67 = avg_sales.quantile(0.67)
    q33 = avg_sales.quantile(0.33)

    segments = {}
    for item_id, mean_val in avg_sales.items():
        if mean_val >= q67:
            segments[item_id] = "high"
        elif mean_val >= q33:
            segments[item_id] = "medium"
        else:
            segments[item_id] = "low"
    return segments


def evaluate_forecasts(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_name: str = "model",
    volume_segments: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate forecasts across validation windows, product categories, and sales volume tiers.

    Args:
        actual_df: DataFrame with ['item_id', 'day_num', 'sales', 'cat_id', 'dept_id', 'date'].
        pred_df: DataFrame with ['item_id', 'day_num', 'predicted_sales', 'window'].
        model_name: Name of model being evaluated.
        volume_segments: Optional dict mapping item_id to 'high'/'medium'/'low'.

    Returns:
        Structured evaluation results dictionary.
    """
    merged = pd.merge(
        pred_df,
        actual_df[["item_id", "day_num", "sales", "cat_id", "dept_id", "date"]],
        on=["item_id", "day_num"],
        how="inner",
    )

    if volume_segments is not None:
        merged["volume_segment"] = merged["item_id"].map(volume_segments).fillna("medium")
    else:
        merged["volume_segment"] = "unknown"

    # Overall metrics
    overall = compute_metrics_dict(merged["sales"], merged["predicted_sales"])
    overall["n_predictions"] = int(len(merged))

    # Metrics by Validation Window
    by_window: Dict[str, Any] = {}
    for window, group in merged.groupby("window", observed=False):
        metrics = compute_metrics_dict(group["sales"], group["predicted_sales"])
        metrics["n_predictions"] = int(len(group))
        by_window[str(window)] = metrics

    # Metrics by Category
    by_category: Dict[str, Any] = {}
    for cat, group in merged.groupby("cat_id", observed=False):
        metrics = compute_metrics_dict(group["sales"], group["predicted_sales"])
        metrics["n_predictions"] = int(len(group))
        by_category[str(cat)] = metrics

    # Metrics by Sales-Volume Segment
    by_segment: Dict[str, Any] = {}
    for seg, group in merged.groupby("volume_segment", observed=False):
        metrics = compute_metrics_dict(group["sales"], group["predicted_sales"])
        metrics["n_predictions"] = int(len(group))
        by_segment[str(seg)] = metrics

    return {
        "model_name": model_name,
        "overall": overall,
        "by_window": by_window,
        "by_category": by_category,
        "by_volume_segment": by_segment,
    }


def save_figure(fig: go.Figure, base_name: str, figures_dir: Path) -> Tuple[Path, Path]:
    """Save a Plotly figure as HTML and PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{base_name}.html"
    png_path = figures_dir / f"{base_name}.png"

    fig.write_html(str(html_path), include_plotlyjs=True)
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        asyncio.run(kaleido.write_fig(fig, str(png_path)))

    return html_path, png_path


def plot_actual_vs_predicted(
    actual_df: pd.DataFrame,
    naive_pred_df: pd.DataFrame,
    lgb_pred_df: pd.DataFrame,
    product_ids: List[str],
    figures_dir: Path,
    window_name: str = "val3",
) -> go.Figure:
    """Create a multi-panel Plotly chart comparing actual sales vs forecasts for representative products.

    Args:
        actual_df: DataFrame of actual sales.
        naive_pred_df: Seasonal naive predictions.
        lgb_pred_df: LightGBM predictions.
        product_ids: List of item IDs (e.g. [high, medium, low, intermittent]).
        figures_dir: Output figures directory.
        window_name: Validation window to visualize.

    Returns:
        Plotly Figure.
    """
    n_products = len(product_ids)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Product: {pid}" for pid in product_ids],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Filter to specific window
    naive_w = naive_pred_df[naive_pred_df["window"] == window_name]
    lgb_w = lgb_pred_df[lgb_pred_df["window"] == window_name]
    days_in_window = sorted(lgb_w["day_num"].unique())
    act_w = actual_df[actual_df["day_num"].isin(days_in_window)]

    colors = {
        "actual": "#1f77b4",
        "naive": "#ff7f0e",
        "lightgbm": "#2ca02c",
    }

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, pid in enumerate(product_ids[:4]):
        row_idx, col_idx = positions[idx]
        act_item = act_w[act_w["item_id"] == pid].sort_values("day_num")
        naive_item = naive_w[naive_w["item_id"] == pid].sort_values("day_num")
        lgb_item = lgb_w[lgb_w["item_id"] == pid].sort_values("day_num")

        show_legend = (idx == 0)

        # Actual trace
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(act_item["date"]),
                y=act_item["sales"],
                mode="lines+markers",
                name="Actual Demand (Observed)",
                line=dict(color=colors["actual"], width=2),
                marker=dict(size=4),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

        # Seasonal Naive trace
        naive_dates = act_item.set_index("day_num")["date"].to_dict()
        naive_x = [naive_dates.get(d) for d in naive_item["day_num"]]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(naive_x),
                y=naive_item["predicted_sales"],
                mode="lines",
                name="Seasonal Naive Baseline",
                line=dict(color=colors["naive"], width=1.5, dash="dash"),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

        # LightGBM trace
        lgb_dates = act_item.set_index("day_num")["date"].to_dict()
        lgb_x = [lgb_dates.get(d) for d in lgb_item["day_num"]]
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(lgb_x),
                y=lgb_item["predicted_sales"],
                mode="lines",
                name="LightGBM Recursive",
                line=dict(color=colors["lightgbm"], width=2),
                showlegend=show_legend,
            ),
            row=row_idx,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text=f"Validation Window ({window_name}) Actual vs. Forecasted Daily Sales",
            font=dict(size=17, family="Arial"),
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=50, r=40, t=70, b=70),
        width=1050,
        height=650,
    )

    save_figure(fig, f"actual_vs_predicted_{window_name}", figures_dir)
    return fig


def plot_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    figures_dir: Path,
    n_repeats: int = 5,
    top_n: int = 15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, go.Figure]:
    """Compute and plot permutation feature importance on validation data.

    SCIENTIFIC INTEGRITY NOTICE:
    Explicitly states in chart title and annotations that feature importance reflects
    predictive association in the model, NOT real-world causal mechanisms.

    Args:
        model: Fitted forecaster or booster.
        X_val: Validation feature matrix.
        y_val: Validation target series.
        figures_dir: Target output directory.
        n_repeats: Number of permutations per feature.
        top_n: Number of top features to display.
        seed: Random seed.

    Returns:
        Tuple of (importance_df, Plotly Figure).
    """
    rng = np.random.RandomState(seed)
    preds_base = np.clip(model.predict(X_val), 0, None)
    base_mae = float(np.mean(np.abs(y_val - preds_base)))

    results = []
    # Evaluate permutation on subset of columns to be laptop-friendly
    cols = list(X_val.columns)
    for col in cols:
        col_deltas = []
        orig_vals = X_val[col].values
        is_cat = isinstance(X_val[col].dtype, pd.CategoricalDtype)
        cat_categories = X_val[col].cat.categories if is_cat else None

        X_perm = X_val.copy()
        for _ in range(n_repeats):
            permuted = rng.permutation(orig_vals)
            if is_cat:
                X_perm[col] = pd.Categorical(permuted, categories=cat_categories)
            else:
                X_perm[col] = permuted
            p_perm = np.clip(model.predict(X_perm), 0, None)
            delta = float(np.mean(np.abs(y_val - p_perm)) - base_mae)
            col_deltas.append(delta)
        results.append({
            "feature": col,
            "importance_mean": float(np.mean(col_deltas)),
            "importance_std": float(np.std(col_deltas)),
        })

    imp_df = pd.DataFrame(results).sort_values(by="importance_mean", ascending=True)
    top_df = imp_df.tail(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_df["importance_mean"],
            y=top_df["feature"],
            orientation="h",
            error_x=dict(type="data", array=top_df["importance_std"], visible=True),
            marker=dict(color="#1f77b4"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Permutation Feature Importance (Validation Set ΔMAE)<br>"
                 "<sup>Notice: Predictive reliance reflects correlational patterns, NOT real-world causal impact.</sup>",
            font=dict(size=16, family="Arial"),
        ),
        xaxis=dict(
            title="Increase in Validation MAE When Feature is Permuted",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(title="Engineered Feature", showgrid=False),
        template="plotly_white",
        margin=dict(l=150, r=40, t=90, b=60),
        width=950,
        height=550,
    )

    save_figure(fig, "permutation_importance", figures_dir)
    return imp_df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True), fig
