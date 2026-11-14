"""Typed YAML configuration loader with strict quantitative invariant validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from core.contracts import SymbolSpec, Timeframe


@dataclass(frozen=True)
class SystemConfig:
    name: str
    version: str
    environment: str
    symbol: str
    poll_interval_seconds: float
    random_seed: int


@dataclass(frozen=True)
class DataPipelineConfig:
    base_timeframe: Timeframe
    higher_timeframes: Tuple[Timeframe, ...]
    history_bars_per_tf: int
    atr_period: int
    parkinson_window: int
    zscore_window: int
    hurst_window: int
    volume_profile_window: int
    volume_profile_bins: int
    value_area_pct: float
    annualization_bars_m5: int


@dataclass(frozen=True)
class RegimeAIConfig:
    onnx_dir: str
    primary_onnx_filename: str
    meta_onnx_filename: str
    hurst_trending_min: float
    hurst_mean_reverting_max: float
    abnormal_vol_zscore_ceiling: float
    abnormal_vol_ratio_ceiling: float
    min_directional_efficiency: float
    meta_confidence_threshold: float
    triple_barrier_pt_atr: float
    triple_barrier_sl_atr: float
    triple_barrier_max_bars: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float


@dataclass(frozen=True)
class RiskAndScalingConfig:
    min_risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    kelly_fraction: float
    target_annual_volatility: float
    atr_stop_multiplier: float
    atr_tp_multiplier: float
    spread_risk_weight: float
    daily_drawdown_limit_pct: float
    max_trailing_drawdown_limit_pct: float
    hard_lock_drawdown_limit_pct: float
    cooldown_minutes: int
    recovery_lot_multiplier: float
    recovery_vol_zscore_max: float
    recovery_drawdown_buffer_pct: float
    max_lot_per_order: float
    max_total_lot_per_signal: float
    max_liquidity_participation_rate: float
    twap_slice_interval_seconds: float
    iceberg_jitter_pct: float
    max_adverse_drift_atr: float


@dataclass(frozen=True)
class BacktestValidationConfig:
    initial_equity: float
    commission_per_lot_usd: float
    slippage_points: float
    purged_cv_splits: int
    embargo_pct: float
    wfo_train_bars: int
    wfo_test_bars: int
    min_sharpe_target: float
    monte_carlo_iterations: int
    monte_carlo_block_size: int
    monte_carlo_stress_slippage_mult: float


@dataclass(frozen=True)
class DatabaseConfig:
    backend: str
    sqlite_path: str
    postgres_dsn: str


@dataclass(frozen=True)
class ExecutionMonitoringConfig:
    mt5_login: int
    mt5_password: str
    mt5_server: str
    mt5_terminal_path: str
    magic_number: int
    max_deviation_points: int
    max_spread_points: float
    max_spread_atr_ratio: float
    max_order_retries: int
    retry_backoff_seconds: float
    reconnect_max_attempts: int
    reconnect_base_delay_seconds: float
    rollover_start_utc: str
    rollover_end_utc: str
    friday_stop_hour_utc: int
    news_blackout_before_minutes: int
    news_blackout_after_minutes: int
    database: DatabaseConfig


@dataclass(frozen=True)
class AppConfig:
    system: SystemConfig
    symbol_spec: SymbolSpec
    data_pipeline: DataPipelineConfig
    regime_ai: RegimeAIConfig
    risk_and_scaling: RiskAndScalingConfig
    backtest_and_validation: BacktestValidationConfig
    execution_and_monitoring: ExecutionMonitoringConfig


def _validate_config(cfg: AppConfig) -> None:
    """Enforce First-Principles risk & statistical invariants at startup."""
    r = cfg.risk_and_scaling
    if not (0.0 < r.min_risk_per_trade_pct <= r.max_risk_per_trade_pct <= 0.02):
        raise ValueError(
            "Risk per trade must satisfy 0 < min_risk <= max_risk <= 2.0% (Martingale is forbidden)."
        )
    if not (0.0 < r.kelly_fraction <= 1.0):
        raise ValueError("kelly_fraction must be within (0, 1].")
    if not (0.0 < r.daily_drawdown_limit_pct < r.max_trailing_drawdown_limit_pct):
        raise ValueError("daily_drawdown_limit_pct must be positive and below max_trailing_drawdown_limit_pct.")
    if not (0.0 < r.recovery_lot_multiplier < 1.0):
        raise ValueError("recovery_lot_multiplier must be strictly in (0, 1) to reduce risk during recovery.")
    if r.max_lot_per_order <= 0 or r.max_total_lot_per_signal < r.max_lot_per_order:
        raise ValueError("Invalid liquidity lot ceilings: require 0 < max_lot_per_order <= max_total_lot_per_signal.")

    ai = cfg.regime_ai
    if not (0.0 < ai.hurst_mean_reverting_max < ai.hurst_trending_min < 1.0):
        raise ValueError("Hurst regime bounds must satisfy 0 < mean_reverting_max < trending_min < 1.")
    if not (0.50 <= ai.meta_confidence_threshold < 1.0):
        raise ValueError("meta_confidence_threshold must be within [0.50, 1.0).")


def load_config(config_path: str | Path) -> AppConfig:
    """Load and validate AppConfig from YAML file with environment overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    sys_raw = raw["system"]
    spec_raw = raw["symbol_spec"]
    dp_raw = raw["data_pipeline"]
    ai_raw = raw["regime_ai"]
    rk_raw = raw["risk_and_scaling"]
    bt_raw = raw["backtest_and_validation"]
    ex_raw = raw["execution_and_monitoring"]
    db_raw = ex_raw["database"]

    app_cfg = AppConfig(
        system=SystemConfig(
            name=str(sys_raw["name"]),
            version=str(sys_raw["version"]),
            environment=str(sys_raw["environment"]),
            symbol=str(sys_raw["symbol"]),
            poll_interval_seconds=float(sys_raw["poll_interval_seconds"]),
            random_seed=int(sys_raw["random_seed"]),
        ),
        symbol_spec=SymbolSpec(
            symbol=str(spec_raw["symbol"]),
            digits=int(spec_raw["digits"]),
            point=float(spec_raw["point"]),
            tick_size=float(spec_raw["tick_size"]),
            tick_value=float(spec_raw["tick_value"]),
            contract_size=float(spec_raw["contract_size"]),
            volume_min=float(spec_raw["volume_min"]),
            volume_max=float(spec_raw["volume_max"]),
            volume_step=float(spec_raw["volume_step"]),
        ),
        data_pipeline=DataPipelineConfig(
            base_timeframe=Timeframe(dp_raw["base_timeframe"]),
            higher_timeframes=tuple(Timeframe(tf) for tf in dp_raw["higher_timeframes"]),
            history_bars_per_tf=int(dp_raw["history_bars_per_tf"]),
            atr_period=int(dp_raw["atr_period"]),
            parkinson_window=int(dp_raw["parkinson_window"]),
            zscore_window=int(dp_raw["zscore_window"]),
            hurst_window=int(dp_raw["hurst_window"]),
            volume_profile_window=int(dp_raw["volume_profile_window"]),
            volume_profile_bins=int(dp_raw["volume_profile_bins"]),
            value_area_pct=float(dp_raw["value_area_pct"]),
            annualization_bars_m5=int(dp_raw["annualization_bars_m5"]),
        ),
        regime_ai=RegimeAIConfig(
            onnx_dir=str(ai_raw["onnx_dir"]),
            primary_onnx_filename=str(ai_raw["primary_onnx_filename"]),
            meta_onnx_filename=str(ai_raw["meta_onnx_filename"]),
            hurst_trending_min=float(ai_raw["hurst_trending_min"]),
            hurst_mean_reverting_max=float(ai_raw["hurst_mean_reverting_max"]),
            abnormal_vol_zscore_ceiling=float(ai_raw["abnormal_vol_zscore_ceiling"]),
            abnormal_vol_ratio_ceiling=float(ai_raw["abnormal_vol_ratio_ceiling"]),
            min_directional_efficiency=float(ai_raw["min_directional_efficiency"]),
            meta_confidence_threshold=float(ai_raw["meta_confidence_threshold"]),
            triple_barrier_pt_atr=float(ai_raw["triple_barrier_pt_atr"]),
            triple_barrier_sl_atr=float(ai_raw["triple_barrier_sl_atr"]),
            triple_barrier_max_bars=int(ai_raw["triple_barrier_max_bars"]),
            n_estimators=int(ai_raw["n_estimators"]),
            learning_rate=float(ai_raw["learning_rate"]),
            max_depth=int(ai_raw["max_depth"]),
            num_leaves=int(ai_raw["num_leaves"]),
            min_child_samples=int(ai_raw["min_child_samples"]),
            subsample=float(ai_raw["subsample"]),
            colsample_bytree=float(ai_raw["colsample_bytree"]),
        ),
        risk_and_scaling=RiskAndScalingConfig(
            min_risk_per_trade_pct=float(rk_raw["min_risk_per_trade_pct"]),
            max_risk_per_trade_pct=float(rk_raw["max_risk_per_trade_pct"]),
            kelly_fraction=float(rk_raw["kelly_fraction"]),
            target_annual_volatility=float(rk_raw["target_annual_volatility"]),
            atr_stop_multiplier=float(rk_raw["atr_stop_multiplier"]),
            atr_tp_multiplier=float(rk_raw["atr_tp_multiplier"]),
            spread_risk_weight=float(rk_raw["spread_risk_weight"]),
            daily_drawdown_limit_pct=float(rk_raw["daily_drawdown_limit_pct"]),
            max_trailing_drawdown_limit_pct=float(rk_raw["max_trailing_drawdown_limit_pct"]),
            hard_lock_drawdown_limit_pct=float(rk_raw["hard_lock_drawdown_limit_pct"]),
            cooldown_minutes=int(rk_raw["cooldown_minutes"]),
            recovery_lot_multiplier=float(rk_raw["recovery_lot_multiplier"]),
            recovery_vol_zscore_max=float(rk_raw["recovery_vol_zscore_max"]),
            recovery_drawdown_buffer_pct=float(rk_raw["recovery_drawdown_buffer_pct"]),
            max_lot_per_order=float(rk_raw["max_lot_per_order"]),
            max_total_lot_per_signal=float(rk_raw["max_total_lot_per_signal"]),
            max_liquidity_participation_rate=float(rk_raw["max_liquidity_participation_rate"]),
            twap_slice_interval_seconds=float(rk_raw["twap_slice_interval_seconds"]),
            iceberg_jitter_pct=float(rk_raw["iceberg_jitter_pct"]),
            max_adverse_drift_atr=float(rk_raw["max_adverse_drift_atr"]),
        ),
        backtest_and_validation=BacktestValidationConfig(
            initial_equity=float(bt_raw["initial_equity"]),
            commission_per_lot_usd=float(bt_raw["commission_per_lot_usd"]),
            slippage_points=float(bt_raw["slippage_points"]),
            purged_cv_splits=int(bt_raw["purged_cv_splits"]),
            embargo_pct=float(bt_raw["embargo_pct"]),
            wfo_train_bars=int(bt_raw["wfo_train_bars"]),
            wfo_test_bars=int(bt_raw["wfo_test_bars"]),
            min_sharpe_target=float(bt_raw["min_sharpe_target"]),
            monte_carlo_iterations=int(bt_raw["monte_carlo_iterations"]),
            monte_carlo_block_size=int(bt_raw["monte_carlo_block_size"]),
            monte_carlo_stress_slippage_mult=float(bt_raw["monte_carlo_stress_slippage_mult"]),
        ),
        execution_and_monitoring=ExecutionMonitoringConfig(
            mt5_login=int(os.environ.get("MT5_LOGIN", ex_raw["mt5_login"])),
            mt5_password=str(os.environ.get("MT5_PASSWORD", "")),
            mt5_server=str(os.environ.get("MT5_SERVER", ex_raw["mt5_server"])),
            mt5_terminal_path=str(os.environ.get("MT5_TERMINAL_PATH", ex_raw["mt5_terminal_path"])),
            magic_number=int(ex_raw["magic_number"]),
            max_deviation_points=int(ex_raw["max_deviation_points"]),
            max_spread_points=float(ex_raw["max_spread_points"]),
            max_spread_atr_ratio=float(ex_raw["max_spread_atr_ratio"]),
            max_order_retries=int(ex_raw["max_order_retries"]),
            retry_backoff_seconds=float(ex_raw["retry_backoff_seconds"]),
            reconnect_max_attempts=int(ex_raw["reconnect_max_attempts"]),
            reconnect_base_delay_seconds=float(ex_raw["reconnect_base_delay_seconds"]),
            rollover_start_utc=str(ex_raw["rollover_start_utc"]),
            rollover_end_utc=str(ex_raw["rollover_end_utc"]),
            friday_stop_hour_utc=int(ex_raw["friday_stop_hour_utc"]),
            news_blackout_before_minutes=int(ex_raw["news_blackout_before_minutes"]),
            news_blackout_after_minutes=int(ex_raw["news_blackout_after_minutes"]),
            database=DatabaseConfig(
                backend=str(os.environ.get("TRADING_DB_BACKEND", db_raw["backend"])).lower(),
                sqlite_path=str(db_raw["sqlite_path"]),
                postgres_dsn=str(os.environ.get("TRADING_DB_DSN", db_raw["postgres_dsn"])),
            ),
        ),
    )
    _validate_config(app_cfg)
    return app_cfg
