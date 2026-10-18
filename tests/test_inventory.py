"""Unit and Integration Tests for Inventory Policy Simulation.

Tests:
1. Inventory conservation: on_hand_end = on_hand_start + arrivals - min(demand, on_hand_start + arrivals).
2. Arrival timing: Orders placed on day t arrive exactly on day t + lead_time.
3. Outstanding orders tracked: In-transit orders are strictly subtracted when computing order quantity.
4. Non-negative inventory: on_hand is strictly >= 0 under all conditions.
5. Deterministic results: Running the same simulation twice produces identical results.
6. Identical initial conditions: Both policies start with the identical state without cross-contamination.
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
from src.inventory_simulator import InventorySimulator


@pytest.fixture
def inventory_test_setup():
    """Create minimal synthetic simulation fixtures."""
    config = load_config()
    items = ["TEST_ITEM_A", "TEST_ITEM_B"]
    days = list(range(1, 15))  # 14 days

    # Forecast and actuals
    fc_rows = []
    act_rows = []
    for item in items:
        base_demand = 10.0 if item == "TEST_ITEM_A" else 20.0
        for d in range(1, 30):  # 28+ days of forecast
            fc_rows.append({"item_id": item, "day_num": d, "predicted_sales": base_demand})
            if d in days:
                act_rows.append({"item_id": item, "day_num": d, "sales": base_demand + (d % 3)})

    forecast_df = pd.DataFrame(fc_rows)
    actual_df = pd.DataFrame(act_rows)

    initial_inventory = {"TEST_ITEM_A": 30.0, "TEST_ITEM_B": 60.0}
    safety_stock_stds = {"TEST_ITEM_A": 2.0, "TEST_ITEM_B": 4.0}

    return {
        "config": config,
        "forecast_df": forecast_df,
        "actual_df": actual_df,
        "initial_inventory": initial_inventory,
        "safety_stock_stds": safety_stock_stds,
        "simulation_days": days,
    }


def test_inventory_conservation(inventory_test_setup):
    """Verify units conservation: ending_stock = starting_stock + arrivals - served."""
    setup = inventory_test_setup
    sim = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=setup["actual_df"],
        model_name="TestPolicy",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory=setup["initial_inventory"],
        lead_time_days=3,
    )

    df_res = sim.run_simulation(setup["simulation_days"])

    for _, row in df_res.iterrows():
        start = row["starting_stock"]
        arr = row["arrivals"]
        demand = row["demand"]
        served = row["units_served"]
        lost = row["lost_sales"]
        end = row["ending_stock"]

        # 1. Units served must be min(demand, stock before demand)
        stock_available = start + arr
        assert np.isclose(served, min(demand, stock_available), atol=1e-5)

        # 2. Demand conservation: served + lost = demand
        assert np.isclose(served + lost, demand, atol=1e-5)

        # 3. Stock conservation: ending = starting + arrivals - served
        assert np.isclose(end, stock_available - served, atol=1e-5)


def test_arrival_timing(inventory_test_setup):
    """Verify orders placed on day t arrive exactly on day t + lead_time."""
    setup = inventory_test_setup
    lead_time = 4
    sim = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=setup["actual_df"],
        model_name="TestTiming",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory=setup["initial_inventory"],
        lead_time_days=lead_time,
    )

    df_res = sim.run_simulation(setup["simulation_days"])

    # Track order placements
    item_a_res = df_res[df_res["item_id"] == "TEST_ITEM_A"].sort_values("day_num").reset_index(drop=True)

    for idx, row in item_a_res.iterrows():
        order_day = row["day_num"]
        order_qty = row["order_quantity"]

        if order_qty > 0:
            arrival_day = order_day + lead_time
            # Find the row for arrival_day
            arrival_rows = item_a_res[item_a_res["day_num"] == arrival_day]
            if len(arrival_rows) > 0:
                # Arrival day must receive at least this order's quantity
                assert arrival_rows.iloc[0]["arrivals"] >= order_qty - 1e-5


def test_outstanding_orders_tracked(inventory_test_setup):
    """Verify outstanding orders are always subtracted when calculating order quantity."""
    setup = inventory_test_setup
    sim = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=setup["actual_df"],
        model_name="TestOutstanding",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory=setup["initial_inventory"],
        lead_time_days=5,
    )

    df_res = sim.run_simulation(setup["simulation_days"])

    for _, row in df_res.iterrows():
        order_up_to = row["order_up_to_level"]
        end_stock = row["ending_stock"]
        pipeline = row["pipeline_qty"]
        order_qty = row["order_quantity"]

        expected_qty = max(0.0, order_up_to - end_stock - pipeline)
        assert np.isclose(order_qty, expected_qty, atol=1e-5), (
            f"Expected order {expected_qty}, got {order_qty} (pipeline {pipeline} not subtracted properly)"
        )


def test_nonnegative_inventory(inventory_test_setup):
    """Verify inventory on_hand is strictly non-negative under extreme demand shocks."""
    setup = inventory_test_setup
    # Create extreme demand shock
    extreme_actuals = setup["actual_df"].copy()
    extreme_actuals["sales"] = 99999.0  # Massive stockout demand

    sim = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=extreme_actuals,
        model_name="TestStockout",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory={"TEST_ITEM_A": 5.0, "TEST_ITEM_B": 10.0},
        lead_time_days=7,
    )

    df_res = sim.run_simulation(setup["simulation_days"])

    assert (df_res["ending_stock"] >= 0.0).all(), "Negative inventory detected!"
    assert (df_res["starting_stock"] >= 0.0).all(), "Negative starting stock detected!"
    assert (df_res["lost_sales"] > 0).all(), "Lost sales should be recorded under extreme demand"


def test_deterministic_results(inventory_test_setup):
    """Verify running the same simulation twice produces bitwise identical results."""
    setup = inventory_test_setup

    def run_one():
        sim = InventorySimulator(
            config=setup["config"],
            forecast_df=setup["forecast_df"],
            actual_sales_df=setup["actual_df"],
            model_name="TestDeterminism",
            safety_stock_stds=setup["safety_stock_stds"],
            initial_inventory=setup["initial_inventory"],
            lead_time_days=3,
        )
        sim.run_simulation(setup["simulation_days"])
        return sim.get_summary()

    run1 = run_one()
    run2 = run_one()

    assert run1 == run2, "Simulation is not deterministic!"


def test_identical_initial_conditions(inventory_test_setup):
    """Verify both policies start with identical initial conditions without mutation bleed."""
    setup = inventory_test_setup
    initial_inv = {"TEST_ITEM_A": 25.0, "TEST_ITEM_B": 50.0}

    sim_naive = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=setup["actual_df"],
        model_name="Naive",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory=initial_inv,
        lead_time_days=4,
    )

    sim_lgb = InventorySimulator(
        config=setup["config"],
        forecast_df=setup["forecast_df"],
        actual_sales_df=setup["actual_df"],
        model_name="LightGBM",
        safety_stock_stds=setup["safety_stock_stds"],
        initial_inventory=initial_inv,
        lead_time_days=4,
    )

    # Initial state check
    assert sim_naive.on_hand == sim_lgb.on_hand
    assert sim_naive.outstanding_orders == sim_lgb.outstanding_orders

    # Run policy Naive
    sim_naive.run_simulation(setup["simulation_days"])

    # Verify policy LGBM was NOT mutated by policy Naive running
    assert sim_lgb.on_hand == initial_inv
    assert sim_lgb.outstanding_orders == {"TEST_ITEM_A": [], "TEST_ITEM_B": []}
