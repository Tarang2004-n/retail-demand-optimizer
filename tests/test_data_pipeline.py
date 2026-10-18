"""Unit and Integration Tests for Stage 1 Data Preparation & EDA Pipeline.

Tests:
1. Row uniqueness: No duplicate (item_id, d) rows in processed data.
2. Join integrity: Calendar and price joins preserve exact row count.
3. Temporal splits: Windows are strictly non-overlapping, sequential, and 28 days each.
4. Product count: Exactly 100 products selected for CA_1.
5. No future leakage: Selection uses only training period sales (d_1 to d_1829).
6. Columns present: All required metadata, temporal, price, and target columns exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.config import ProjectConfig, load_config
from src.data_preparation import (
    audit_data_quality,
    define_temporal_splits,
    join_data,
    select_products,
    validate_input_files,
    wide_to_long,
)
from src.exploratory_analysis import category_summary, weekly_sales_summary, zero_sales_rates


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Session-scoped loaded project configuration."""
    return load_config()


@pytest.fixture(scope="session")
def processed_df(config: ProjectConfig) -> pd.DataFrame:
    """Session-scoped processed parquet DataFrame."""
    parquet_path = config.processed_data_path / "sales_processed.parquet"
    if not parquet_path.exists():
        pytest.fail(f"Processed file {parquet_path} does not exist. Run pipeline first.")
    return pd.read_parquet(parquet_path)


def test_input_files_validation(config: ProjectConfig):
    """Verify input file validation detects existing raw files and required columns."""
    report = validate_input_files(config)
    assert report["all_valid"] is True
    assert "calendar" in report["files_checked"]
    assert "sales" in report["files_checked"]
    assert "prices" in report["files_checked"]


def test_row_uniqueness(processed_df: pd.DataFrame, config: ProjectConfig):
    """Verify no duplicate (item_id, d) rows in processed data."""
    # Check duplicate (item_id, d)
    duplicates = processed_df.duplicated(subset=["item_id", "d"]).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate (item_id, d) rows"

    # Check duplicate (item_id, date)
    date_duplicates = processed_df.duplicated(subset=["item_id", "date"]).sum()
    assert date_duplicates == 0, f"Found {date_duplicates} duplicate (item_id, date) rows"

    # Verify expected total row count: 100 items * 1941 days
    expected_rows = config.data.n_products * config.temporal.total_days
    assert len(processed_df) == expected_rows, (
        f"Expected {expected_rows} rows, found {len(processed_df)}"
    )

    # Verify unique days and items
    assert processed_df["item_id"].nunique() == config.data.n_products
    assert processed_df["day_num"].nunique() == config.temporal.total_days


def test_join_integrity():
    """Verify join logic does not create duplicate rows or alter row counts."""
    # Create minimal synthetic data to verify join mechanics
    melted_sample = pd.DataFrame({
        "id": ["ITEM_1_CA_1", "ITEM_1_CA_1", "ITEM_2_CA_1", "ITEM_2_CA_1"],
        "item_id": ["ITEM_1", "ITEM_1", "ITEM_2", "ITEM_2"],
        "dept_id": ["FOODS_1", "FOODS_1", "FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS", "FOODS", "FOODS"],
        "store_id": ["CA_1", "CA_1", "CA_1", "CA_1"],
        "state_id": ["CA", "CA", "CA", "CA"],
        "d": ["d_1", "d_2", "d_1", "d_2"],
        "sales": [5, 10, 0, 3],
    })

    calendar_sample = pd.DataFrame({
        "d": ["d_1", "d_2"],
        "date": ["2011-01-29", "2011-01-30"],
        "wm_yr_wk": [11101, 11101],
        "weekday": ["Saturday", "Sunday"],
        "wday": [1, 2],
        "month": [1, 1],
        "year": [2011, 2011],
    })

    prices_sample = pd.DataFrame({
        "store_id": ["CA_1", "CA_1"],
        "item_id": ["ITEM_1", "ITEM_2"],
        "wm_yr_wk": [11101, 11101],
        "sell_price": [1.99, 2.49],
    })

    joined = join_data(melted_sample, calendar_sample, prices_sample)

    assert len(joined) == len(melted_sample), "Join produced unexpected number of rows"
    assert "sell_price" in joined.columns
    assert "day_num" in joined.columns
    assert joined["sell_price"].isna().sum() == 0
    assert list(joined["day_num"]) == [1, 2, 1, 2]


def test_temporal_splits(config: ProjectConfig):
    """Verify temporal splits are strictly sequential, non-overlapping, and 28 days each."""
    splits = define_temporal_splits(config)

    # 1. Selection period check
    train_selection = splits["train_selection_period"]
    assert train_selection["start_day_num"] == 1
    assert train_selection["end_day_num"] == config.temporal.train_end_d
    assert train_selection["num_days"] == config.temporal.train_end_d

    # 2. Validation windows check
    val_windows = splits["validation_windows"]
    assert len(val_windows) == 3, "Expected exactly 3 validation windows"

    prev_end = train_selection["end_day_num"]
    for w in val_windows:
        assert w["num_days"] == config.temporal.forecast_horizon, (
            f"Window {w['name']} horizon is {w['num_days']}, expected {config.temporal.forecast_horizon}"
        )
        assert w["start_day_num"] == prev_end + 1, (
            f"Window {w['name']} does not immediately follow previous window (starts at {w['start_day_num']}, previous ended at {prev_end})"
        )
        assert w["end_day_num"] == w["start_day_num"] + w["num_days"] - 1
        prev_end = w["end_day_num"]

    # 3. Test period check
    test_period = splits["test_period"]
    assert test_period["start_day_num"] == prev_end + 1, "Test period does not immediately follow val3"
    assert test_period["start_day_num"] == config.temporal.test_start_d
    assert test_period["end_day_num"] == config.temporal.test_end_d
    assert test_period["end_day_num"] == config.temporal.total_days
    assert test_period["num_days"] == config.temporal.forecast_horizon

    # Ensure temporal splits JSON file exists and matches
    splits_file = config.artifacts_path / "temporal_splits.json"
    assert splits_file.exists(), f"Missing {splits_file}"
    with open(splits_file, "r") as f:
        saved_splits = json.load(f)
    assert saved_splits["total_days"] == config.temporal.total_days


def test_product_count(config: ProjectConfig):
    """Verify exactly 100 products selected and saved to artifacts."""
    products_file = config.artifacts_path / "selected_products.json"
    assert products_file.exists(), f"Missing {products_file}"

    with open(products_file, "r") as f:
        data = json.load(f)

    assert data["store_id"] == config.data.store_id
    assert data["n_products"] == config.data.n_products
    assert len(data["selected_products"]) == config.data.n_products

    # Check uniqueness of items
    items = [p["item_id"] for p in data["selected_products"]]
    assert len(set(items)) == config.data.n_products


def test_no_future_leakage_in_selection(config: ProjectConfig):
    """Verify product selection uses ONLY training period (d_1 to d_1829).

    Simulate a scenario where an item has low training sales but massive future sales
    in validation/test periods. Verify it is NOT selected over an item with higher training sales.
    """
    train_end = config.temporal.train_end_d
    total_days = config.temporal.total_days

    # Construct synthetic wide dataframe
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(1, total_days + 1)]

    # Item A: High training sales (1000), 0 future sales
    row_a = ["A_CA_1", "ITEM_A", "FOODS_1", "FOODS", "CA_1", "CA"] + [10] * train_end + [0] * (total_days - train_end)
    # Item B: Low training sales (500), massive future sales (99999)
    row_b = ["B_CA_1", "ITEM_B", "FOODS_1", "FOODS", "CA_1", "CA"] + [5] * train_end + [500] * (total_days - train_end)
    # Item C: Lowest training sales (100)
    row_c = ["C_CA_1", "ITEM_C", "FOODS_1", "FOODS", "CA_1", "CA"] + [1] * train_end + [1000] * (total_days - train_end)

    mock_df = pd.DataFrame([row_a, row_b, row_c], columns=cols + day_cols)

    # If leakage existed (using full sum), order would be C > B > A.
    # Because NO leakage is allowed, order MUST be A > B > C.
    selected = select_products(config, sales_df=mock_df, save_artifact=False)

    # We requested n_products=100 in config, but mock only has 3. It will return all 3 sorted.
    assert selected[0] == "ITEM_A", "ITEM_A should be ranked #1 based on training sales"
    assert selected[1] == "ITEM_B", "ITEM_B should be ranked #2 based on training sales"
    assert selected[2] == "ITEM_C", "ITEM_C should be ranked #3 based on training sales"


def test_columns_present(processed_df: pd.DataFrame):
    """Verify all expected columns in processed data."""
    expected_cols = {
        # Identifiers & categories
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        # Time variables
        "d", "day_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        # Events & calendar flags
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
        # Pricing & Target
        "sell_price", "sales",
    }
    missing = expected_cols - set(processed_df.columns)
    assert not missing, f"Processed DataFrame missing expected columns: {missing}"


def test_data_quality_report_metrics(config: ProjectConfig):
    """Verify data quality report artifact contains required counts and integrity notices."""
    report_file = config.artifacts_path / "data_quality_report.json"
    assert report_file.exists(), f"Missing {report_file}"

    with open(report_file, "r") as f:
        report = json.load(f)

    metrics = report["quality_metrics"]
    assert metrics["missing_dates_count"] == 0
    assert metrics["negative_sales_count"] == 0
    assert metrics["negative_prices_count"] == 0
    assert metrics["duplicate_product_day_rows"] == 0
    assert metrics["date_continuity_verified"] is True

    # Missing prices and zero sales should be reported as positive counts
    assert metrics["missing_prices_count"] > 0
    assert metrics["zero_sales_count"] > 0

    # Verify integrity notices are present
    notes = report["scientific_integrity_notes"]
    assert "missing_prices_interpretation" in notes
    assert "zero_sales_interpretation" in notes
    assert "no_leakage_guarantee" in notes


def test_duckdb_queries(config: ProjectConfig):
    """Verify DuckDB SQL analytical queries run without error on processed parquet."""
    parquet_path = config.processed_data_path / "sales_processed.parquet"

    # Test weekly sales summary
    weekly_df = weekly_sales_summary(parquet_path)
    assert not weekly_df.empty
    assert "total_weekly_sales" in weekly_df.columns
    assert len(weekly_df) == 278  # 1941 days / 7 days ~ 277-278 weeks

    # Test category summary (pre-test)
    cat_df = category_summary(parquet_path, test_start_d=config.temporal.test_start_d)
    assert not cat_df.empty
    assert set(cat_df["cat_id"]) == {"FOODS", "HOBBIES", "HOUSEHOLD"}

    # Test zero sales rates (pre-test)
    zero_df = zero_sales_rates(parquet_path, test_start_d=config.temporal.test_start_d, limit=10)
    assert len(zero_df) == 10
    assert (zero_df["zero_sales_pct"] >= 0).all()
    assert (zero_df["zero_sales_pct"] <= 100).all()
