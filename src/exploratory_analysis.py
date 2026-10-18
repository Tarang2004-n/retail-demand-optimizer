"""Exploratory Data Analysis using DuckDB SQL and Plotly.

Provides fast, in-process analytical querying via DuckDB on processed parquet data,
and creates publication-ready interactive and static visualizations.

SCIENTIFIC INTEGRITY:
- Excludes test period data (d_1914 to d_1941) from all exploratory decisions.
- Clearly marks the test set boundary on any time-series visualization.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional, Tuple
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kaleido

from src.config import ProjectConfig, load_config


def save_figure(
    fig: go.Figure,
    base_name: str,
    figures_dir: Path,
) -> Tuple[Path, Path]:
    """Save a Plotly figure as both interactive HTML and high-resolution static PNG.

    Args:
        fig: Plotly Figure instance.
        base_name: Filename prefix (without extension).
        figures_dir: Target directory path.

    Returns:
        Tuple of (html_path, png_path).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{base_name}.html"
    png_path = figures_dir / f"{base_name}.png"

    # Save HTML
    fig.write_html(str(html_path), include_plotlyjs=True)

    # Save PNG (using kaleido with fallback to async API)
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        asyncio.run(kaleido.write_fig(fig, str(png_path)))

    return html_path, png_path


def get_parquet_path(config: ProjectConfig) -> Path:
    """Return path to processed sales parquet file."""
    parquet_path = config.processed_data_path / "sales_processed.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed parquet not found at {parquet_path}. Run data_preparation.py first."
        )
    return parquet_path


def weekly_sales_summary(
    parquet_path: str | Path,
    by_category: bool = False,
) -> pd.DataFrame:
    """Query weekly aggregated sales using DuckDB SQL.

    Args:
        parquet_path: Path to processed parquet file.
        by_category: If True, aggregate by week and category.

    Returns:
        DataFrame with weekly sales aggregations.
    """
    con = duckdb.connect()
    parquet_str = str(parquet_path)

    if by_category:
        query = f"""
            SELECT 
                wm_yr_wk,
                cat_id,
                MIN(date) AS week_start_date,
                MAX(date) AS week_end_date,
                MIN(day_num) AS min_day_num,
                MAX(day_num) AS max_day_num,
                SUM(sales) AS total_weekly_sales,
                ROUND(AVG(sales), 3) AS avg_item_daily_sales,
                ROUND(SUM(sales * sell_price), 2) AS estimated_revenue
            FROM '{parquet_str}'
            GROUP BY wm_yr_wk, cat_id
            ORDER BY wm_yr_wk, cat_id
        """
    else:
        query = f"""
            SELECT 
                wm_yr_wk,
                MIN(date) AS week_start_date,
                MAX(date) AS week_end_date,
                MIN(day_num) AS min_day_num,
                MAX(day_num) AS max_day_num,
                SUM(sales) AS total_weekly_sales,
                ROUND(AVG(sales), 3) AS avg_item_daily_sales,
                ROUND(SUM(sales * sell_price), 2) AS estimated_revenue,
                COUNT(DISTINCT item_id) AS active_items
            FROM '{parquet_str}'
            GROUP BY wm_yr_wk
            ORDER BY wm_yr_wk
        """
    return con.execute(query).df()


def category_summary(
    parquet_path: str | Path,
    test_start_d: int = 1914,
) -> pd.DataFrame:
    """Query category-level summary statistics using DuckDB SQL.

    STRICT REQUIREMENT: Excludes the test period (day_num >= test_start_d)
    to prevent data leakage in exploratory decisions.

    Args:
        parquet_path: Path to processed parquet file.
        test_start_d: Day number where test period begins (d_1914).

    Returns:
        DataFrame with category metrics.
    """
    con = duckdb.connect()
    parquet_str = str(parquet_path)

    query = f"""
        SELECT 
            cat_id,
            COUNT(DISTINCT item_id) AS n_items,
            SUM(sales) AS total_units_sold,
            ROUND(AVG(sales), 2) AS avg_daily_sales,
            ROUND(AVG(sell_price), 2) AS avg_sell_price,
            ROUND(SUM(sales * sell_price), 2) AS estimated_revenue,
            ROUND(COUNT(CASE WHEN sales = 0 THEN 1 END) * 100.0 / COUNT(*), 2) AS zero_sales_pct
        FROM '{parquet_str}'
        WHERE day_num < {test_start_d}
        GROUP BY cat_id
        ORDER BY total_units_sold DESC
    """
    return con.execute(query).df()


def zero_sales_rates(
    parquet_path: str | Path,
    test_start_d: int = 1914,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Query product-level zero-sales rates using DuckDB SQL.

    STRICT REQUIREMENT: Excludes the test period (day_num >= test_start_d)
    to prevent data leakage in exploratory decisions.

    Args:
        parquet_path: Path to processed parquet file.
        test_start_d: Day number where test period begins (d_1914).
        limit: Optional integer limit for returned records.

    Returns:
        DataFrame with item zero sales rates.
    """
    con = duckdb.connect()
    parquet_str = str(parquet_path)

    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    query = f"""
        SELECT 
            item_id,
            cat_id,
            dept_id,
            COUNT(*) AS total_days,
            COUNT(CASE WHEN sales = 0 THEN 1 END) AS zero_sales_days,
            ROUND(COUNT(CASE WHEN sales = 0 THEN 1 END) * 100.0 / COUNT(*), 2) AS zero_sales_pct,
            ROUND(AVG(sales), 2) AS avg_sales,
            MAX(sales) AS max_sales
        FROM '{parquet_str}'
        WHERE day_num < {test_start_d}
        GROUP BY item_id, cat_id, dept_id
        ORDER BY zero_sales_pct DESC
        {limit_clause}
    """
    return con.execute(query).df()


def plot_sales_trends(
    weekly_cat_df: pd.DataFrame,
    weekly_total_df: pd.DataFrame,
    config: ProjectConfig,
) -> go.Figure:
    """Create a Plotly line chart of weekly sales trends with test boundary marked.

    Args:
        weekly_cat_df: Weekly sales by category.
        weekly_total_df: Weekly sales total.
        config: Typed ProjectConfig instance.

    Returns:
        Plotly Figure.
    """
    test_start_d = config.temporal.test_start_d
    test_end_d = config.temporal.test_end_d

    # Find date boundary corresponding to test_start_d
    # Find row where min_day_num <= test_start_d <= max_day_num
    test_rows = weekly_total_df[
        (weekly_total_df["max_day_num"] >= test_start_d)
    ]
    test_start_date = test_rows["week_start_date"].iloc[0] if len(test_rows) > 0 else "2016-04-25"
    last_date = weekly_total_df["week_end_date"].max()

    fig = go.Figure()

    # Total sales line
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(weekly_total_df["week_start_date"]),
            y=weekly_total_df["total_weekly_sales"],
            mode="lines",
            name="Total (All 100 Products)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    # Category lines
    category_colors = {
        "FOODS": "#2ca02c",
        "HOBBIES": "#ff7f0e",
        "HOUSEHOLD": "#d62728",
    }
    for cat_id, group in weekly_cat_df.groupby("cat_id"):
        color = category_colors.get(cat_id, None)
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(group["week_start_date"]),
                y=group["total_weekly_sales"],
                mode="lines",
                name=f"Category: {cat_id}",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.8,
            )
        )

    # Add vertical line for test set start
    fig.add_shape(
        type="line",
        x0=str(test_start_date),
        x1=str(test_start_date),
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#d62728", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=str(test_start_date),
        y=1.03,
        yref="paper",
        text="Test Set Origin (d_1914 / 2016-04-25)",
        showarrow=False,
        xanchor="right",
        font=dict(color="#d62728", size=11, family="Arial"),
    )

    # Highlight the test period with a subtle shaded rectangle
    fig.add_shape(
        type="rect",
        x0=str(test_start_date),
        x1=str(last_date),
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="rgba(214, 39, 40, 0.08)",
        layer="below",
        line_width=0,
    )
    fig.add_annotation(
        x=str(last_date),
        y=1.03,
        yref="paper",
        text="Untouched Test Window",
        showarrow=False,
        xanchor="right",
        font=dict(color="#d62728", size=10, family="Arial"),
    )

    fig.update_layout(
        title=dict(
            text="Weekly Sales Trends Across Top 100 Products (Store CA_1)",
            font=dict(size=18, family="Arial"),
        ),
        xaxis=dict(
            title="Timeline (Week Start Date)",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            title="Weekly Total Units Sold",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=40, t=80, b=80),
        width=1050,
        height=550,
    )

    return fig


def plot_weekday_patterns(
    parquet_path: str | Path,
    config: ProjectConfig,
) -> go.Figure:
    """Create a Plotly bar chart of average daily sales by weekday (pre-test data only).

    Args:
        parquet_path: Path to processed parquet file.
        config: Typed ProjectConfig instance.

    Returns:
        Plotly Figure.
    """
    con = duckdb.connect()
    parquet_str = str(parquet_path)
    test_start_d = config.temporal.test_start_d

    # Query average sales by weekday and category strictly on pre-test data
    query = f"""
        SELECT 
            weekday,
            wday,
            cat_id,
            ROUND(AVG(sales), 2) AS avg_daily_sales,
            ROUND(STDDEV(sales), 2) AS std_sales
        FROM '{parquet_str}'
        WHERE day_num < {test_start_d}
        GROUP BY weekday, wday, cat_id
        ORDER BY wday, cat_id
    """
    df = con.execute(query).df()

    # Map weekday order (Monday to Sunday standard display)
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    category_colors = {
        "FOODS": "#2ca02c",
        "HOBBIES": "#ff7f0e",
        "HOUSEHOLD": "#1f77b4",
    }

    fig = px.bar(
        df,
        x="weekday",
        y="avg_daily_sales",
        color="cat_id",
        barmode="group",
        category_orders={"weekday": weekday_order},
        color_discrete_map=category_colors,
        labels={
            "weekday": "Day of Week",
            "avg_daily_sales": "Average Daily Sales Per Product",
            "cat_id": "Category",
        },
        title="Day-of-Week Sales Patterns by Category (Pre-Test Period: d_1 to d_1913)",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Average Daily Units Sold / Product", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        width=950,
        height=500,
    )

    return fig


def plot_product_variation(
    parquet_path: str | Path,
    config: ProjectConfig,
) -> go.Figure:
    """Create a Plotly box plot showing daily sales variation across product categories.

    STRICT REQUIREMENT: Uses only pre-test data (day_num < test_start_d).

    Args:
        parquet_path: Path to processed parquet file.
        config: Typed ProjectConfig instance.

    Returns:
        Plotly Figure.
    """
    con = duckdb.connect()
    parquet_str = str(parquet_path)
    test_start_d = config.temporal.test_start_d

    # Calculate mean daily sales and zero-sales percentage per individual item
    query = f"""
        SELECT 
            item_id,
            cat_id,
            dept_id,
            ROUND(AVG(sales), 2) AS mean_daily_sales,
            ROUND(STDDEV(sales), 2) AS std_daily_sales,
            ROUND(COUNT(CASE WHEN sales = 0 THEN 1 END) * 100.0 / COUNT(*), 1) AS zero_sales_pct,
            MAX(sales) AS max_daily_sales
        FROM '{parquet_str}'
        WHERE day_num < {test_start_d}
        GROUP BY item_id, cat_id, dept_id
        ORDER BY cat_id, mean_daily_sales DESC
    """
    product_stats = con.execute(query).df()

    category_colors = {
        "FOODS": "#2ca02c",
        "HOBBIES": "#ff7f0e",
        "HOUSEHOLD": "#1f77b4",
    }

    fig = px.box(
        product_stats,
        x="cat_id",
        y="mean_daily_sales",
        color="cat_id",
        points="all",
        hover_data=["item_id", "dept_id", "zero_sales_pct", "max_daily_sales"],
        color_discrete_map=category_colors,
        labels={
            "cat_id": "Product Category",
            "mean_daily_sales": "Average Daily Sales (Units / Day)",
        },
        title="Product-Level Demand Variation Across Categories (Pre-Test Period: d_1 to d_1913)",
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis=dict(title="Category", showgrid=False),
        yaxis=dict(
            title="Product Average Daily Units Sold",
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        width=900,
        height=500,
    )

    return fig


def run_exploratory_analysis(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Execute complete exploratory data analysis workflow and generate all charts.

    Args:
        config: Optional ProjectConfig. If None, loaded from default config.yaml.

    Returns:
        Dictionary with analytical tables and saved figure file paths.
    """
    if config is None:
        config = load_config()

    parquet_path = get_parquet_path(config)
    figures_dir = config.figures_path
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Running DuckDB analytical queries...")
    # 1. Weekly sales summary
    weekly_total_df = weekly_sales_summary(parquet_path, by_category=False)
    weekly_cat_df = weekly_sales_summary(parquet_path, by_category=True)

    # 2. Category summary (pre-test only)
    cat_summary_df = category_summary(parquet_path, test_start_d=config.temporal.test_start_d)

    # 3. Zero sales rates (pre-test only)
    zero_rates_df = zero_sales_rates(parquet_path, test_start_d=config.temporal.test_start_d)

    print("Generating Plotly charts...")
    # 4. Sales trends plot
    trends_fig = plot_sales_trends(weekly_cat_df, weekly_total_df, config)
    trends_html, trends_png = save_figure(trends_fig, "sales_trends", figures_dir)
    print(f"  ✓ Saved sales trends chart: {trends_png.name} & {trends_html.name}")

    # 5. Weekday patterns plot
    weekday_fig = plot_weekday_patterns(parquet_path, config)
    weekday_html, weekday_png = save_figure(weekday_fig, "weekday_patterns", figures_dir)
    print(f"  ✓ Saved weekday patterns chart: {weekday_png.name} & {weekday_html.name}")

    # 6. Product variation plot
    variation_fig = plot_product_variation(parquet_path, config)
    variation_html, variation_png = save_figure(variation_fig, "product_variation", figures_dir)
    print(f"  ✓ Saved product variation chart: {variation_png.name} & {variation_html.name}")

    results = {
        "weekly_total_summary": weekly_total_df,
        "weekly_cat_summary": weekly_cat_df,
        "category_summary": cat_summary_df,
        "zero_sales_rates": zero_rates_df,
        "figures": {
            "sales_trends": {"html": str(trends_html), "png": str(trends_png)},
            "weekday_patterns": {"html": str(weekday_html), "png": str(weekday_png)},
            "product_variation": {"html": str(variation_html), "png": str(variation_png)},
        },
    }

    return results


if __name__ == "__main__":
    cfg = load_config()
    run_exploratory_analysis(cfg)
