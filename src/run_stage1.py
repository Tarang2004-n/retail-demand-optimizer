"""CLI Entry Point for Stage 1: Data Preparation & Exploratory Analysis.

Usage:
    python src/run_stage1.py
    python src/run_stage1.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_preparation import run_data_preparation
from src.exploratory_analysis import run_exploratory_analysis


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1: Data Preparation and Exploratory Analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: config.yaml at project root).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute Stage 1 pipeline end-to-end."""
    args = parse_args()

    start_time = time.time()
    print("=" * 70)
    print("STAGE 1: RETAIL DEMAND FORECASTING - DATA PREPARATION & EDA")
    print("=" * 70)

    # 1. Load Configuration
    config = load_config(args.config)
    print(f"Configuration loaded from: {args.config or 'config.yaml'}")
    print(f"Target Store: {config.data.store_id}")
    print(f"Product Count: {config.data.n_products}")
    print(f"Total Days: {config.temporal.total_days} (d_1 to d_{config.temporal.total_days})")
    print(f"Train Selection Period: d_1 to d_{config.temporal.train_end_d}")
    print(f"Test Set Window: d_{config.temporal.test_start_d} to d_{config.temporal.test_end_d} (Untouched)")
    print("-" * 70)

    # 2. Run Data Preparation Pipeline
    print("\n>>> EXECUTING DATA PREPARATION PIPELINE...")
    processed_df = run_data_preparation(config)
    print(f"Data preparation completed: {len(processed_df):,} rows processed.")
    print("-" * 70)

    # 3. Run Exploratory Data Analysis & Chart Generation
    print("\n>>> EXECUTING EXPLORATORY ANALYSIS (DUCKDB + PLOTLY)...")
    eda_results = run_exploratory_analysis(config)
    print("Exploratory analysis completed successfully.")
    print("-" * 70)

    # 4. Print Summary Report
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("STAGE 1 PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("\n[Artifacts Created]")
    print(f"  • Temporal Splits:      {config.artifacts_path / 'temporal_splits.json'}")
    print(f"  • Selected Products:    {config.artifacts_path / 'selected_products.json'}")
    print(f"  • Data Quality Report:  {config.artifacts_path / 'data_quality_report.json'}")
    print(f"  • Processed Parquet:    {config.processed_data_path / 'sales_processed.parquet'}")

    print("\n[Figures Generated]")
    for name, paths in eda_results["figures"].items():
        print(f"  • {name}:")
        print(f"      HTML: {paths['html']}")
        print(f"      PNG:  {paths['png']}")

    cat_summary = eda_results["category_summary"]
    print("\n[Category Distribution in Top 100 Products (Pre-Test Period)]")
    for _, row in cat_summary.iterrows():
        print(
            f"  • {row['cat_id']:<10}: {int(row['n_items']):>2} items | "
            f"Total Units: {int(row['total_units_sold']):>8,} | "
            f"Avg Daily/Item: {row['avg_daily_sales']:>5.2f} | "
            f"Zero Sales: {row['zero_sales_pct']:>5.1f}%"
        )

    print("=" * 70)
    print("Stage 1 complete! Ready for forecasting & feature engineering (Stage 2).")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
