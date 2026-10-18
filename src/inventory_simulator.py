"""Inventory Policy Simulation Engine for Retail Demand Optimization.

Implements an Order-Up-To (R, S) daily review replenishment policy with lost sales.

SCIENTIFIC INTEGRITY & CONSTRAINTS:
- Simulation results are synthetic scenario evaluations under stated illustrative cost parameters.
- Costs ($0.05/unit/day holding, $2.00/unit lost-sale penalty) are illustrative assumptions, NOT actual retailer costs.
- Demand proxy: historical sales are used as a proxy for demand (true unconstrained demand is unobserved).
- Outstanding orders tracking: in-transit orders are strictly accounted for in the inventory position
  to prevent artificial double-ordering.
- Safety stock is estimated strictly from forecast error standard deviations on EARLIER validation windows.
- Lost sales model: stockouts result in lost sales (no backorders).
- Identical initial conditions: both policies start with identical initial inventory and pipeline orders.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config


class InventorySimulator:
    """Simulates an Order-Up-To (R, S) replenishment policy under lost sales.

    Daily sequence of events:
    1. Receive arrivals: in-transit orders with arrival_day == t arrive and enter on-hand inventory.
    2. Fulfill demand: actual demand is served from on-hand stock. Shortfalls become lost sales.
    3. Record holding units: end-of-day on-hand stock incurs holding costs.
    4. Calculate order: Order-up-to level S = forecast(protection_period) + safety_stock.
       Order quantity Q = max(0, S - on_hand - sum(outstanding_orders)).
    5. Place order: new order placed for delivery at day t + lead_time.
    """

    def __init__(
        self,
        config: ProjectConfig,
        forecast_df: pd.DataFrame,
        actual_sales_df: pd.DataFrame,
        model_name: str,
        safety_stock_stds: Dict[str, float],
        initial_inventory: Dict[str, float],
        initial_outstanding_orders: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        lead_time_days: Optional[int] = None,
        safety_stock_multiplier: Optional[float] = None,
        holding_cost_per_unit_per_day: Optional[float] = None,
        lost_sale_penalty_per_unit: Optional[float] = None,
        scale_protection_error: bool = True,
    ) -> None:
        """Initialize the inventory simulation engine.

        Args:
            config: ProjectConfig instance.
            forecast_df: DataFrame with ['item_id', 'day_num', 'predicted_sales'].
            actual_sales_df: DataFrame with ['item_id', 'day_num', 'sales'].
            model_name: Name of forecasting policy (e.g., 'Seasonal_Naive', 'LightGBM').
            safety_stock_stds: Dict of per-item forecast error std from earlier windows.
            initial_inventory: Dict mapping item_id to initial on_hand inventory.
            initial_outstanding_orders: Dict mapping item_id to list of order dicts.
            lead_time_days: Supplier lead time in days (default: from config).
            safety_stock_multiplier: Multiplier for safety stock (default: from config).
            holding_cost_per_unit_per_day: Daily holding cost per unit (default: from config).
            lost_sale_penalty_per_unit: Penalty per lost sales unit (default: from config).
            scale_protection_error: If True, scale error std by sqrt(lead_time + review_period).
        """
        self.config = config
        self.model_name = model_name
        self.lead_time = int(
            lead_time_days
            if lead_time_days is not None
            else config.inventory.lead_time_days
        )
        self.review_period = int(config.inventory.review_period_days)
        self.protection_period = self.lead_time + self.review_period
        self.safety_stock_multiplier = float(
            safety_stock_multiplier
            if safety_stock_multiplier is not None
            else config.inventory.safety_stock_multiplier
        )
        self.holding_cost_per_unit = float(
            holding_cost_per_unit_per_day
            if holding_cost_per_unit_per_day is not None
            else config.inventory.holding_cost_per_unit_per_day
        )
        self.lost_sale_penalty_per_unit = float(
            lost_sale_penalty_per_unit
            if lost_sale_penalty_per_unit is not None
            else config.inventory.lost_sale_penalty_per_unit
        )
        self.scale_protection_error = scale_protection_error

        # Precompute forecast and actual lookups
        self.forecast_map: Dict[Tuple[str, int], float] = {
            (row["item_id"], int(row["day_num"])): float(row["predicted_sales"])
            for _, row in forecast_df.iterrows()
        }
        self.actual_map: Dict[Tuple[str, int], float] = {
            (row["item_id"], int(row["day_num"])): float(row["sales"])
            for _, row in actual_sales_df.iterrows()
        }

        # Per-item mean forecast fallback if query exceeds horizon
        self.item_mean_forecast: Dict[str, float] = (
            forecast_df.groupby("item_id", observed=False)["predicted_sales"]
            .mean()
            .to_dict()
        )

        self.safety_stock_stds = safety_stock_stds
        self.items = sorted(list(initial_inventory.keys()))

        # Deepcopy initial conditions so simulation runs do not cross-contaminate
        self.on_hand: Dict[str, float] = {
            item: max(0.0, float(qty)) for item, qty in initial_inventory.items()
        }

        if initial_outstanding_orders is not None:
            self.outstanding_orders: Dict[str, List[Dict[str, Any]]] = copy.deepcopy(
                initial_outstanding_orders
            )
        else:
            self.outstanding_orders = {item: [] for item in self.items}

        # Simulation history tracking
        self.history_records: List[Dict[str, Any]] = []
        self.simulation_run: bool = False

    def _get_protection_forecast(self, item_id: str, current_day: int) -> float:
        """Compute expected total demand over the protection period (days t+1 to t+L+R)."""
        total_fc = 0.0
        for step in range(1, self.protection_period + 1):
            target_day = current_day + step
            if (item_id, target_day) in self.forecast_map:
                total_fc += self.forecast_map[(item_id, target_day)]
            else:
                total_fc += self.item_mean_forecast.get(item_id, 0.0)
        return total_fc

    def _get_safety_stock(self, item_id: str) -> float:
        """Calculate safety stock buffer based on earlier validation window error variance."""
        err_std = self.safety_stock_stds.get(item_id, 1.0)
        if self.scale_protection_error:
            scaling = np.sqrt(self.protection_period)
        else:
            scaling = 1.0
        ss = self.safety_stock_multiplier * err_std * scaling
        return max(0.0, float(ss))

    def run_simulation(self, simulation_days: List[int]) -> pd.DataFrame:
        """Execute the daily simulation loop across the specified days.

        Args:
            simulation_days: Chronologically ordered list of integer day numbers.

        Returns:
            DataFrame with per-product, per-day simulation ledger.
        """
        self.history_records = []

        for current_day in simulation_days:
            for item in self.items:
                starting_stock = self.on_hand[item]

                # Step 1: Receive arrivals
                arriving_qty = 0.0
                remaining_orders = []
                for order in self.outstanding_orders[item]:
                    if order["arrival_day"] <= current_day:
                        arriving_qty += order["quantity"]
                    else:
                        remaining_orders.append(order)
                self.outstanding_orders[item] = remaining_orders

                # Stock available before fulfilling demand
                stock_available = starting_stock + arriving_qty

                # Step 2: Fulfill demand (lost sales model)
                demand = self.actual_map.get((item, current_day), 0.0)
                units_served = min(stock_available, demand)
                lost_sales = demand - units_served
                ending_stock = stock_available - units_served

                # Assert non-negativity constraint
                assert ending_stock >= -1e-6, f"Negative stock detected for item {item} on day {current_day}"
                ending_stock = max(0.0, ending_stock)
                self.on_hand[item] = ending_stock

                # Step 3: Record holding (units on-hand at end of day)
                holding_units = ending_stock

                # Step 4: Calculate replenishment order
                forecast_lead_demand = self._get_protection_forecast(item, current_day)
                safety_stock = self._get_safety_stock(item)
                order_up_to_level = forecast_lead_demand + safety_stock

                # CRITICAL: Sum pipeline orders in transit
                pipeline_qty = sum(
                    order["quantity"] for order in self.outstanding_orders[item]
                )

                # Order quantity: subtract both on_hand AND pipeline orders
                inventory_position = ending_stock + pipeline_qty
                order_quantity = max(0.0, order_up_to_level - inventory_position)

                # Step 5: Place order
                if order_quantity > 0:
                    arrival_day = current_day + self.lead_time
                    self.outstanding_orders[item].append({
                        "arrival_day": arrival_day,
                        "quantity": order_quantity,
                        "order_day": current_day,
                    })

                self.history_records.append({
                    "day_num": current_day,
                    "item_id": item,
                    "model_name": self.model_name,
                    "starting_stock": starting_stock,
                    "arrivals": arriving_qty,
                    "stock_before_demand": stock_available,
                    "demand": demand,
                    "units_served": units_served,
                    "lost_sales": lost_sales,
                    "ending_stock": ending_stock,
                    "holding_units": holding_units,
                    "forecast_lead_demand": forecast_lead_demand,
                    "safety_stock": safety_stock,
                    "order_up_to_level": order_up_to_level,
                    "pipeline_qty": pipeline_qty,
                    "order_quantity": order_quantity,
                    "outstanding_orders_count": len(self.outstanding_orders[item]),
                })

        self.simulation_run = True
        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        """Return full simulation results ledger as a DataFrame."""
        if not self.simulation_run:
            raise RuntimeError("Simulation has not been executed. Call run_simulation() first.")
        return pd.DataFrame(self.history_records)

    def compute_costs(self) -> Dict[str, float]:
        """Compute total holding, lost sales penalty, and combined costs."""
        df = self.get_results_df()
        total_holding_units = float(df["holding_units"].sum())
        total_lost_units = float(df["lost_sales"].sum())

        holding_cost = total_holding_units * self.holding_cost_per_unit
        lost_sales_penalty = total_lost_units * self.lost_sale_penalty_per_unit
        total_cost = holding_cost + lost_sales_penalty

        return {
            "holding_cost": round(holding_cost, 2),
            "lost_sales_penalty": round(lost_sales_penalty, 2),
            "total_cost": round(total_cost, 2),
            "total_holding_units": round(total_holding_units, 1),
            "total_lost_units": round(total_lost_units, 1),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive summary of simulation KPIs."""
        df = self.get_results_df()
        costs = self.compute_costs()

        total_demand = float(df["demand"].sum())
        total_served = float(df["units_served"].sum())
        total_lost = float(df["lost_sales"].sum())
        fill_rate = float(total_served / total_demand) if total_demand > 0 else 1.0

        # Horizon effects: inventory and orders remaining at simulation end
        last_day = int(df["day_num"].max())
        last_day_df = df[df["day_num"] == last_day]
        ending_inventory = float(last_day_df["ending_stock"].sum())

        ending_outstanding_orders = sum(
            sum(order["quantity"] for order in orders)
            for orders in self.outstanding_orders.values()
        )

        avg_on_hand = float(df["ending_stock"].mean())
        stockout_instances = int((df["lost_sales"] > 0).sum())
        total_product_days = len(df)
        stockout_rate = float(stockout_instances / total_product_days) if total_product_days > 0 else 0.0

        return {
            "model_name": self.model_name,
            "lead_time_days": self.lead_time,
            "safety_stock_multiplier": self.safety_stock_multiplier,
            "holding_cost_per_unit": self.holding_cost_per_unit,
            "lost_sale_penalty_per_unit": self.lost_sale_penalty_per_unit,
            "fill_rate": round(fill_rate, 4),
            "fill_rate_percentage": round(fill_rate * 100, 2),
            "total_demand": round(total_demand, 1),
            "total_units_served": round(total_served, 1),
            "total_lost_units": round(total_lost, 1),
            "total_holding_cost": costs["holding_cost"],
            "total_lost_sales_penalty": costs["lost_sales_penalty"],
            "total_cost": costs["total_cost"],
            "avg_on_hand_per_product": round(avg_on_hand, 2),
            "stockout_rate": round(stockout_rate, 4),
            "stockout_days_count": stockout_instances,
            "ending_inventory": round(ending_inventory, 1),
            "ending_outstanding_orders": round(ending_outstanding_orders, 1),
            "total_orders_placed_units": round(float(df["order_quantity"].sum()), 1),
        }


def compute_safety_stock_stds(
    validation_predictions_path: Path | str,
    actual_sales_path: Path | str,
    earlier_windows: List[str] = ("val1", "val2"),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-item forecast error standard deviation strictly on earlier validation windows.

    NO LEAKAGE: Window 3 error data is strictly excluded.

    Args:
        validation_predictions_path: Path to validation_predictions.parquet.
        actual_sales_path: Path to sales_processed.parquet.
        earlier_windows: List of window names to compute error standard deviation from.

    Returns:
        Tuple of (seasonal_naive_stds_dict, lightgbm_stds_dict).
    """
    preds_df = pd.read_parquet(validation_predictions_path)
    actuals_df = pd.read_parquet(actual_sales_path)

    # Filter to earlier validation windows
    earlier_preds = preds_df[preds_df["window"].isin(earlier_windows)].copy()
    valid_days = earlier_preds["day_num"].unique()
    earlier_actuals = actuals_df[actuals_df["day_num"].isin(valid_days)][
        ["item_id", "day_num", "sales"]
    ].copy()

    merged = pd.merge(earlier_preds, earlier_actuals, on=["item_id", "day_num"], how="inner")

    merged["naive_err"] = merged["sales"] - merged["pred_seasonal_naive"]
    merged["lgb_err"] = merged["sales"] - merged["pred_lightgbm"]

    naive_stds = (
        merged.groupby("item_id", observed=False)["naive_err"]
        .std()
        .fillna(1.0)
        .to_dict()
    )
    lgb_stds = (
        merged.groupby("item_id", observed=False)["lgb_err"]
        .std()
        .fillna(1.0)
        .to_dict()
    )

    return naive_stds, lgb_stds


def compute_initial_inventory(
    actual_sales_path: Path | str,
    start_sim_day: int,
    lead_time: int,
    review_period: int = 1,
    lookback_days: int = 28,
    safety_stock_buffer: float = 2.0,
) -> Dict[str, float]:
    """Compute deterministic initial stock based strictly on pre-simulation sales history.

    Initial stock = average_daily_sales * (lead_time + review_period + buffer).

    Args:
        actual_sales_path: Path to sales_processed.parquet.
        start_sim_day: First day of the simulation (e.g., 1886).
        lead_time: Supplier lead time.
        review_period: Review period days (default: 1).
        lookback_days: Days of pre-simulation history to average (default: 28).
        safety_stock_buffer: Extra days buffer for initial stock.

    Returns:
        Dict mapping item_id to initial on_hand stock.
    """
    actuals_df = pd.read_parquet(actual_sales_path)
    pre_sim_mask = actuals_df["day_num"].between(
        start_sim_day - lookback_days, start_sim_day - 1
    )
    pre_sim_df = actuals_df[pre_sim_mask]

    avg_sales = pre_sim_df.groupby("item_id", observed=False)["sales"].mean().to_dict()

    multiplier = lead_time + review_period + safety_stock_buffer
    initial_stock = {
        item: round(max(0.0, float(avg * multiplier)), 2)
        for item, avg in avg_sales.items()
    }
    return initial_stock
