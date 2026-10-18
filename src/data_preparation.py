"""Data Preparation Pipeline for Retail Demand Forecasting & Inventory Optimization.

Handles validation, deterministic product selection, wide-to-long transformation,
joining with calendar and pricing data, data quality auditing, and parquet export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.config import ProjectConfig, load_config


def validate_input_files(config: ProjectConfig) -> Dict[str, Any]:
    """Validate that all required raw data files exist and contain expected columns.

    Args:
        config: Typed ProjectConfig instance.

    Returns:
        Dictionary containing validation status and file metadata.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If required columns are missing from any file.
    """
    raw_dir = config.raw_data_path

    required_files = {
        "calendar": raw_dir / "calendar.csv",
        "sales": raw_dir / "sales_train_evaluation.csv",
        "prices": raw_dir / "sell_prices.csv",
    }

    validation_report: Dict[str, Any] = {
        "files_checked": {},
        "all_valid": True,
    }

    # Verify file existence
    for file_key, file_path in required_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required raw data file: {file_path}")
        validation_report["files_checked"][file_key] = {
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
        }

    # Verify Calendar columns
    cal_cols = pd.read_csv(required_files["calendar"], nrows=1).columns.tolist()
    expected_cal_cols = {"date", "wm_yr_wk", "weekday", "wday", "month", "year", "d"}
    missing_cal = expected_cal_cols - set(cal_cols)
    if missing_cal:
        raise ValueError(f"calendar.csv missing required columns: {missing_cal}")
    validation_report["files_checked"]["calendar"]["columns_found"] = len(cal_cols)

    # Verify Sales columns (first 6 metadata cols + d_1 through d_{total_days})
    sales_cols = pd.read_csv(required_files["sales"], nrows=1).columns.tolist()
    expected_sales_meta = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing_sales_meta = expected_sales_meta - set(sales_cols)
    if missing_sales_meta:
        raise ValueError(f"sales_train_evaluation.csv missing required metadata columns: {missing_sales_meta}")

    expected_day_cols = {f"d_{i}" for i in range(1, config.temporal.total_days + 1)}
    missing_days = expected_day_cols - set(sales_cols)
    if missing_days:
        raise ValueError(f"sales_train_evaluation.csv missing {len(missing_days)} day columns")
    validation_report["files_checked"]["sales"]["columns_found"] = len(sales_cols)

    # Verify Sell Prices columns
    price_cols = pd.read_csv(required_files["prices"], nrows=1).columns.tolist()
    expected_price_cols = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    missing_prices = expected_price_cols - set(price_cols)
    if missing_prices:
        raise ValueError(f"sell_prices.csv missing required columns: {missing_prices}")
    validation_report["files_checked"]["prices"]["columns_found"] = len(price_cols)

    return validation_report


def define_temporal_splits(
    config: ProjectConfig,
    calendar_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Define temporal split boundaries and save them to artifacts/temporal_splits.json.

    Strictly enforces temporal ordering:
    - Never uses random splits.
    - Reserves the final 28 days (d_1914 to d_1941) as the untouched test set.
    - Sets up 3 preceding non-overlapping 28-day validation windows.
    - Training period for product selection is strictly d_1 to d_1829.

    Args:
        config: Typed ProjectConfig instance.
        calendar_df: Optional calendar DataFrame to map 'd' values to dates.

    Returns:
        Dictionary detailing the split boundaries.
    """
    artifacts_dir = config.artifacts_path
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load calendar if not provided to enrich splits with human-readable dates
    if calendar_df is None:
        cal_path = config.raw_data_path / "calendar.csv"
        calendar_df = pd.read_csv(cal_path, usecols=["d", "date"])

    d_to_date = dict(zip(calendar_df["d"], calendar_df["date"]))

    train_end_d = config.temporal.train_end_d
    test_start_d = config.temporal.test_start_d
    test_end_d = config.temporal.test_end_d

    splits_data: Dict[str, Any] = {
        "dataset": "M5 Forecasting",
        "total_days": config.temporal.total_days,
        "forecast_horizon_days": config.temporal.forecast_horizon,
        "scientific_rules": [
            "Strict temporal ordering: no random train/test splits.",
            "Final 28 days (d_1914-d_1941) held out as untouched test set.",
            "Three preceding non-overlapping 28-day validation windows.",
            "Product selection uses ONLY initial training window (d_1-d_1829).",
        ],
        "train_selection_period": {
            "start_d": "d_1",
            "end_d": f"d_{train_end_d}",
            "start_day_num": 1,
            "end_day_num": train_end_d,
            "start_date": d_to_date.get("d_1"),
            "end_date": d_to_date.get(f"d_{train_end_d}"),
            "num_days": train_end_d,
        },
        "validation_windows": [],
        "test_period": {
            "name": "test",
            "start_d": f"d_{test_start_d}",
            "end_d": f"d_{test_end_d}",
            "start_day_num": test_start_d,
            "end_day_num": test_end_d,
            "start_date": d_to_date.get(f"d_{test_start_d}"),
            "end_date": d_to_date.get(f"d_{test_end_d}"),
            "num_days": test_end_d - test_start_d + 1,
        },
    }

    # Validation windows
    for w in config.temporal.validation_windows:
        splits_data["validation_windows"].append({
            "name": w.name,
            "start_d": f"d_{w.start_d}",
            "end_d": f"d_{w.end_d}",
            "start_day_num": w.start_d,
            "end_day_num": w.end_d,
            "start_date": d_to_date.get(f"d_{w.start_d}"),
            "end_date": d_to_date.get(f"d_{w.end_d}"),
            "num_days": w.end_d - w.start_d + 1,
        })

    # Save to artifacts
    out_file = artifacts_dir / "temporal_splits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    return splits_data


def select_products(
    config: ProjectConfig,
    sales_df: Optional[pd.DataFrame] = None,
    save_artifact: bool = True,
) -> List[str]:
    """Deterministically select top N products from specified store using ONLY training sales.

    NO future leakage: Sales from validation and test periods (d_1830..d_1941) are strictly
    excluded from the calculation.

    Args:
        config: Typed ProjectConfig instance.
        sales_df: Optional DataFrame with sales. If None, loaded from raw sales file.
        save_artifact: If True, saves artifact to artifacts/selected_products.json.

    Returns:
        List of selected item_id strings.
    """
    artifacts_dir = config.artifacts_path
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_end_d = config.temporal.train_end_d
    store_id = config.data.store_id
    n_products = config.data.n_products

    # Columns representing strictly the training period
    train_day_cols = [f"d_{i}" for i in range(1, train_end_d + 1)]
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    if sales_df is None:
        sales_path = config.raw_data_path / "sales_train_evaluation.csv"
        # Optimize memory by reading only required columns for product selection
        sales_df = pd.read_csv(sales_path, usecols=meta_cols + train_day_cols)

    # Filter to target store
    store_mask = sales_df["store_id"] == store_id
    df_store = sales_df[store_mask].copy()

    # Sum only training days sales
    df_store["total_train_sales"] = df_store[train_day_cols].sum(axis=1)

    # Deterministic sorting: highest sales first, tie-breaking by item_id ascending
    df_store = df_store.sort_values(
        by=["total_train_sales", "item_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    top_df = df_store.head(n_products).copy()
    top_df["rank"] = range(1, len(top_df) + 1)

    selected_items: List[str] = top_df["item_id"].tolist()

    # Create detailed artifact if requested
    if save_artifact:
        artifact_payload = {
            "store_id": store_id,
            "n_products": len(selected_items),
            "selection_period": f"d_1 to d_{train_end_d}",
            "category_counts": top_df["cat_id"].value_counts().to_dict(),
            "department_counts": top_df["dept_id"].value_counts().to_dict(),
            "selected_products": top_df[
                ["rank", "item_id", "cat_id", "dept_id", "total_train_sales"]
            ].to_dict(orient="records"),
        }

        out_file = artifacts_dir / "selected_products.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(artifact_payload, f, indent=2)

    return selected_items


def wide_to_long(
    sales_df: pd.DataFrame,
    selected_items: List[str],
    store_id: str,
    total_days: int,
) -> pd.DataFrame:
    """Transform wide sales format to long format for selected products and store.

    Args:
        sales_df: Full sales DataFrame containing d_1 through d_N columns.
        selected_items: List of selected item_id strings.
        store_id: Store identifier (e.g., 'CA_1').
        total_days: Total number of days (e.g., 1941).

    Returns:
        Melted DataFrame with columns: [id, item_id, dept_id, cat_id, store_id, state_id, d, sales].
    """
    # Filter to selected store and items
    mask = (sales_df["store_id"] == store_id) & (sales_df["item_id"].isin(selected_items))
    filtered_df = sales_df[mask].copy()

    day_cols = [f"d_{i}" for i in range(1, total_days + 1)]
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    melted = pd.melt(
        filtered_df,
        id_vars=id_vars,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    expected_rows = len(selected_items) * total_days
    if len(melted) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows after melting, but got {len(melted)}"
        )

    return melted


def join_data(
    melted_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join melted sales data with calendar and sell prices.

    Asserts no row duplication at each step.

    Args:
        melted_df: Melted sales DataFrame.
        calendar_df: Calendar DataFrame.
        prices_df: Sell prices DataFrame.

    Returns:
        Enriched DataFrame with calendar and price attributes.
    """
    initial_row_count = len(melted_df)

    # 1. Join with calendar on 'd'
    # Calendar has columns: date, wm_yr_wk, weekday, wday, month, year, d, event_name_1...
    with_cal = pd.merge(melted_df, calendar_df, on="d", how="left")
    assert len(with_cal) == initial_row_count, (
        f"Row count changed during calendar join: {len(with_cal)} vs {initial_row_count}"
    )

    # 2. Filter prices to relevant store and items to optimize memory
    store_id = melted_df["store_id"].iloc[0]
    items_set = set(melted_df["item_id"].unique())
    prices_filtered = prices_df[
        (prices_df["store_id"] == store_id) & (prices_df["item_id"].isin(items_set))
    ].drop_duplicates(subset=["store_id", "item_id", "wm_yr_wk"])

    # Join with prices on (store_id, item_id, wm_yr_wk)
    with_prices = pd.merge(
        with_cal,
        prices_filtered,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    assert len(with_prices) == initial_row_count, (
        f"Row count changed during prices join: {len(with_prices)} vs {initial_row_count}"
    )

    # Add numeric day integer column for easy temporal filtering and indexing
    with_prices["day_num"] = with_prices["d"].str[2:].astype(int)

    # Ensure date column is formatted as standard ISO string and datetime
    with_prices["date"] = pd.to_datetime(with_prices["date"])

    # Sort deterministically
    with_prices = with_prices.sort_values(by=["item_id", "day_num"]).reset_index(drop=True)

    return with_prices


def audit_data_quality(df: pd.DataFrame, config: ProjectConfig) -> Dict[str, Any]:
    """Audit data quality and save findings to artifacts/data_quality_report.json.

    Audits:
    - Missing prices (and why they occur)
    - Missing dates
    - Zero sales days
    - Negative values
    - Row duplication

    IMPORTANT: Neither missing prices nor zero sales days can be interpreted as confirmed
    stockouts without on-shelf availability data.

    Args:
        df: Enriched sales DataFrame.
        config: Typed ProjectConfig instance.

    Returns:
        Dictionary containing quality metrics.
    """
    artifacts_dir = config.artifacts_path
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    n_products = df["item_id"].nunique()
    total_days = df["day_num"].nunique()

    # Missing values
    missing_prices_count = int(df["sell_price"].isna().sum())
    missing_prices_pct = round((missing_prices_count / total_rows) * 100, 2)

    missing_dates_count = int(df["date"].isna().sum())
    zero_sales_count = int((df["sales"] == 0).sum())
    zero_sales_pct = round((zero_sales_count / total_rows) * 100, 2)

    negative_sales_count = int((df["sales"] < 0).sum())
    negative_prices_count = int((df["sell_price"] < 0).sum())

    # Row uniqueness verification
    dup_rows = int(df.duplicated(subset=["item_id", "d"]).sum())

    # Date continuity check
    expected_days = config.temporal.total_days
    has_all_days = (total_days == expected_days)

    report: Dict[str, Any] = {
        "dataset_scope": {
            "store_id": config.data.store_id,
            "n_products": n_products,
            "total_days": total_days,
            "expected_days": expected_days,
            "total_rows": total_rows,
            "min_date": str(df["date"].min().date()),
            "max_date": str(df["date"].max().date()),
        },
        "quality_metrics": {
            "missing_prices_count": missing_prices_count,
            "missing_prices_percentage": missing_prices_pct,
            "missing_dates_count": missing_dates_count,
            "zero_sales_count": zero_sales_count,
            "zero_sales_percentage": zero_sales_pct,
            "negative_sales_count": negative_sales_count,
            "negative_prices_count": negative_prices_count,
            "duplicate_product_day_rows": dup_rows,
            "date_continuity_verified": has_all_days,
        },
        "category_distribution": df.groupby("cat_id")["item_id"].nunique().to_dict(),
        "scientific_integrity_notes": {
            "missing_prices_interpretation": (
                "Missing sell prices reflect periods before product launch or after delisting "
                "in this store. They MUST NOT be interpreted as confirmed stockouts."
            ),
            "zero_sales_interpretation": (
                "Zero sales days represent unobserved demand or true zero customer demand. "
                "Because on-shelf availability and inventory levels are not recorded in M5, "
                "zero sales CANNOT be distinguished from out-of-stock events without assumptions."
            ),
            "no_leakage_guarantee": (
                "Product selection and validation splits strictly respect temporal boundaries. "
                "No information from validation or test periods was used to filter or curate items."
            ),
        },
    }

    out_file = artifacts_dir / "data_quality_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def save_processed(df: pd.DataFrame, config: ProjectConfig) -> Path:
    """Save processed DataFrame as parquet with pyarrow.

    Args:
        df: Enriched sales DataFrame.
        config: Typed ProjectConfig instance.

    Returns:
        Path to the saved parquet file.
    """
    processed_dir = config.processed_data_path
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "sales_processed.parquet"

    # Convert date back to string or timestamp for standard parquet compatibility
    save_df = df.copy()
    save_df["date"] = save_df["date"].astype(str)

    save_df.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def run_data_preparation(config: Optional[ProjectConfig] = None) -> pd.DataFrame:
    """Execute the full data preparation pipeline end-to-end.

    Args:
        config: Optional ProjectConfig. If None, loaded from default config.yaml.

    Returns:
        Final processed DataFrame.
    """
    if config is None:
        config = load_config()

    print("Step 1: Validating raw input files...")
    validation = validate_input_files(config)
    print(f"  ✓ Files validated: {list(validation['files_checked'].keys())}")

    print("Step 2: Defining temporal splits...")
    cal_path = config.raw_data_path / "calendar.csv"
    calendar_df = pd.read_csv(cal_path)
    splits = define_temporal_splits(config, calendar_df=calendar_df)
    print(f"  ✓ Saved temporal splits to {config.artifacts_path / 'temporal_splits.json'}")

    print("Step 3: Selecting products deterministically (training sales only)...")
    sales_path = config.raw_data_path / "sales_train_evaluation.csv"
    # Read full sales to keep for melting
    sales_full_df = pd.read_csv(sales_path)
    selected_items = select_products(config, sales_df=sales_full_df)
    print(f"  ✓ Selected {len(selected_items)} products for store {config.data.store_id}")

    print("Step 4: Transforming wide format to long format...")
    melted_df = wide_to_long(
        sales_df=sales_full_df,
        selected_items=selected_items,
        store_id=config.data.store_id,
        total_days=config.temporal.total_days,
    )
    print(f"  ✓ Melted shape: {melted_df.shape} ({len(melted_df):,} rows)")

    print("Step 5: Joining calendar and pricing data...")
    prices_path = config.raw_data_path / "sell_prices.csv"
    prices_df = pd.read_csv(prices_path)
    enriched_df = join_data(melted_df, calendar_df, prices_df)
    print(f"  ✓ Joined shape: {enriched_df.shape} ({len(enriched_df):,} rows)")

    print("Step 6: Auditing data quality...")
    audit = audit_data_quality(enriched_df, config)
    metrics = audit["quality_metrics"]
    print(
        f"  ✓ Quality audit complete: {metrics['missing_prices_count']:,} missing prices "
        f"({metrics['missing_prices_percentage']}%), {metrics['zero_sales_count']:,} zero-sales days "
        f"({metrics['zero_sales_percentage']}%), {metrics['negative_sales_count']} negative values."
    )

    print("Step 7: Saving processed dataset to parquet...")
    parquet_path = save_processed(enriched_df, config)
    print(f"  ✓ Processed dataset saved to: {parquet_path}")

    return enriched_df


if __name__ == "__main__":
    cfg = load_config()
    run_data_preparation(cfg)
