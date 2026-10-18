"""Configuration management for Retail Demand Forecasting & Inventory Optimization.

Provides typed access to all settings defined in config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ValidationWindow:
    """Represents a validation window."""
    name: str
    start_d: int
    end_d: int


@dataclass
class DataConfig:
    """Data preparation settings."""
    store_id: str
    n_products: int
    random_seed: int
    raw_dir: str
    processed_dir: str


@dataclass
class TemporalConfig:
    """Temporal split definitions and horizons."""
    total_days: int
    forecast_horizon: int
    test_start_d: int
    test_end_d: int
    train_end_d: int
    validation_windows: List[ValidationWindow] = field(default_factory=list)


@dataclass
class LightGBMConfig:
    """LightGBM hyperparameter search space and features."""
    lag_features: List[int]
    rolling_windows: List[int]
    n_estimators_options: List[int]
    learning_rate_options: List[float]
    max_depth_options: List[int]
    num_leaves_options: List[int]


@dataclass
class ForecastingConfig:
    """Forecasting models and evaluation metrics."""
    models: List[str]
    metrics: List[str]
    lightgbm: LightGBMConfig


@dataclass
class InventoryConfig:
    """Inventory simulation parameters and illustrative cost assumptions."""
    lead_time_days: int
    review_period_days: int
    holding_cost_per_unit_per_day: float
    lost_sale_penalty_per_unit: float
    safety_stock_multiplier: float
    policy: str
    lost_sales_model: bool


@dataclass
class OutputConfig:
    """Output directories."""
    figures_dir: str
    artifacts_dir: str
    reports_dir: str


@dataclass
class ProjectConfig:
    """Root configuration object containing all typed settings."""
    project_root: Path
    data: DataConfig
    temporal: TemporalConfig
    forecasting: ForecastingConfig
    inventory: InventoryConfig
    output: OutputConfig
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    # Resolved absolute path helpers
    @property
    def raw_data_path(self) -> Path:
        """Absolute path to raw data directory."""
        return (self.project_root / self.data.raw_dir).resolve()

    @property
    def processed_data_path(self) -> Path:
        """Absolute path to processed data directory."""
        return (self.project_root / self.data.processed_dir).resolve()

    @property
    def artifacts_path(self) -> Path:
        """Absolute path to artifacts directory."""
        return (self.project_root / self.output.artifacts_dir).resolve()

    @property
    def figures_path(self) -> Path:
        """Absolute path to figures directory."""
        return (self.project_root / self.output.figures_dir).resolve()

    @property
    def reports_path(self) -> Path:
        """Absolute path to reports directory."""
        return (self.project_root / self.output.reports_dir).resolve()


def get_default_project_root() -> Path:
    """Get default project root directory."""
    # src/config.py is in project_root/src/
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Load configuration from a YAML file into a typed ProjectConfig instance.

    Args:
        config_path: Optional path to config YAML file. If not provided, defaults
                     to 'config.yaml' at the project root.

    Returns:
        Typed ProjectConfig object.
    """
    default_root = get_default_project_root()
    if config_path is None:
        cfg_file = default_root / "config.yaml"
    else:
        cfg_file = Path(config_path).resolve()
        if not cfg_file.is_file():
            # If relative path provided, try resolving relative to project root
            cfg_file = (default_root / config_path).resolve()

    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Project root is the directory containing config.yaml (or default_root)
    project_root = cfg_file.parent

    # Parse data config
    data_cfg = DataConfig(
        store_id=str(raw["data"]["store_id"]),
        n_products=int(raw["data"]["n_products"]),
        random_seed=int(raw["data"]["random_seed"]),
        raw_dir=str(raw["data"]["raw_dir"]),
        processed_dir=str(raw["data"]["processed_dir"]),
    )

    # Parse temporal config
    val_windows = [
        ValidationWindow(
            name=str(w["name"]),
            start_d=int(w["start_d"]),
            end_d=int(w["end_d"]),
        )
        for w in raw["temporal"].get("validation_windows", [])
    ]
    temporal_cfg = TemporalConfig(
        total_days=int(raw["temporal"]["total_days"]),
        forecast_horizon=int(raw["temporal"]["forecast_horizon"]),
        test_start_d=int(raw["temporal"]["test_start_d"]),
        test_end_d=int(raw["temporal"]["test_end_d"]),
        train_end_d=int(raw["temporal"]["train_end_d"]),
        validation_windows=val_windows,
    )

    # Parse forecasting config
    lgb_raw = raw.get("forecasting", {}).get("lightgbm", {})
    lgb_cfg = LightGBMConfig(
        lag_features=list(lgb_raw.get("lag_features", [])),
        rolling_windows=list(lgb_raw.get("rolling_windows", [])),
        n_estimators_options=list(lgb_raw.get("n_estimators_options", [])),
        learning_rate_options=list(lgb_raw.get("learning_rate_options", [])),
        max_depth_options=list(lgb_raw.get("max_depth_options", [])),
        num_leaves_options=list(lgb_raw.get("num_leaves_options", [])),
    )
    forecasting_cfg = ForecastingConfig(
        models=list(raw.get("forecasting", {}).get("models", [])),
        metrics=list(raw.get("forecasting", {}).get("metrics", [])),
        lightgbm=lgb_cfg,
    )

    # Parse inventory config
    inv_raw = raw.get("inventory", {})
    inventory_cfg = InventoryConfig(
        lead_time_days=int(inv_raw.get("lead_time_days", 7)),
        review_period_days=int(inv_raw.get("review_period_days", 1)),
        holding_cost_per_unit_per_day=float(inv_raw.get("holding_cost_per_unit_per_day", 0.05)),
        lost_sale_penalty_per_unit=float(inv_raw.get("lost_sale_penalty_per_unit", 2.00)),
        safety_stock_multiplier=float(inv_raw.get("safety_stock_multiplier", 1.5)),
        policy=str(inv_raw.get("policy", "order_up_to")),
        lost_sales_model=bool(inv_raw.get("lost_sales_model", True)),
    )

    # Parse output config
    out_raw = raw.get("output", {})
    output_cfg = OutputConfig(
        figures_dir=str(out_raw.get("figures_dir", "reports/figures")),
        artifacts_dir=str(out_raw.get("artifacts_dir", "artifacts")),
        reports_dir=str(out_raw.get("reports_dir", "reports")),
    )

    return ProjectConfig(
        project_root=project_root,
        data=data_cfg,
        temporal=temporal_cfg,
        forecasting=forecasting_cfg,
        inventory=inventory_cfg,
        output=output_cfg,
        raw_dict=raw,
    )
