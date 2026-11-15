"""
24/5 VPS Production Orchestrator & End-to-End Pipeline CLI for Antigravity AI Trading System.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from backtest_and_validation.monte_carlo import MonteCarloTailRiskSimulator
from backtest_and_validation.walk_forward import WalkForwardOptimizer
from core.config_loader import AppConfig, load_config
from core.contracts import MT5GatewayProtocol, OrderExecutionReport, SignalSide
from data_pipeline.mt5_data_client import (
    MultiTimeframeDataFetcher,
    NativeMT5Gateway,
    SimulatedMT5Gateway,
)
from data_pipeline.mtf_aligner import MultiTimeframeFeaturePipeline
from execution_and_monitoring.audit_database import DecisionAuditDatabase
from execution_and_monitoring.mt5_executor import MT5ExecutionEngine
from execution_and_monitoring.time_and_spread_filters import (
    MarketScheduleAndNewsGuard,
    SpreadGuard,
)
from regime_ai_engine.meta_labeling_models import MetaLabelingTrainer
from regime_ai_engine.onnx_inference import ONNXMetaLabelingEngine
from risk_and_scaling_engine.circuit_breaker import EquityCircuitBreaker
from risk_and_scaling_engine.liquidity_scaler import LiquidityAwareExecutionPlanner
from risk_and_scaling_engine.position_sizer import InverseVolatilityPositionSizer

logger = logging.getLogger("antigravity.orchestrator")


class TradingSystemOrchestrator:
    """Coordinates Data Pipeline, ONNX AI Engine, Risk/Scaling, Execution, and Audit DB 24/5."""

    def __init__(
        self,
        config: AppConfig,
        gateway: MT5GatewayProtocol,
        onnx_engine: ONNXMetaLabelingEngine,
        audit_db: DecisionAuditDatabase,
        sleep_fn=time.sleep,
    ) -> None:
        self._cfg = config
        self._gateway = gateway
        self._onnx_engine = onnx_engine
        self._audit_db = audit_db
        self._sleep = sleep_fn

        self._fetcher = MultiTimeframeDataFetcher(gateway)
        self._pipeline = MultiTimeframeFeaturePipeline(config.data_pipeline)

        acc = gateway.get_account_snapshot()
        self._circuit_breaker = EquityCircuitBreaker(
            config=config.risk_and_scaling,
            initial_equity=acc.equity,
        )
        self._sizer = InverseVolatilityPositionSizer(
            config=config.risk_and_scaling,
            symbol_spec=config.symbol_spec,
        )
        self._planner = LiquidityAwareExecutionPlanner(
            risk_config=config.risk_and_scaling,
            symbol_spec=config.symbol_spec,
            max_deviation_points=config.execution_and_monitoring.max_deviation_points,
            seed=config.system.random_seed,
        )
        self._spread_guard = SpreadGuard(
            config=config.execution_and_monitoring,
            symbol_spec=config.symbol_spec,
        )
        self._schedule_guard = MarketScheduleAndNewsGuard(config.execution_and_monitoring)
        self._executor = MT5ExecutionEngine(
            gateway=gateway,
            config=config.execution_and_monitoring,
            symbol_spec=config.symbol_spec,
            spread_guard=self._spread_guard,
            schedule_guard=self._schedule_guard,
            sleep_fn=sleep_fn,
        )
        self._last_processed_close_utc: Optional[datetime] = None
        self._stop_requested = False

    @property
    def schedule_guard(self) -> MarketScheduleAndNewsGuard:
        return self._schedule_guard

    def request_stop(self) -> None:
        self._stop_requested = True

    def process_single_cycle(
        self,
        as_of_utc: Optional[datetime] = None,
    ) -> Optional[OrderExecutionReport]:
        """Run a single closed-bar evaluation, risk check, order execution, and DB audit."""
        if not self._executor.ensure_connected():
            logger.error("Skipping cycle: MT5 terminal unavailable.")
            return None

        symbol = self._cfg.system.symbol
        bars_by_tf = self._fetcher.fetch_multi_timeframe_bars(
            symbol=symbol,
            base_tf=self._cfg.data_pipeline.base_timeframe,
            higher_tfs=self._cfg.data_pipeline.higher_timeframes,
            num_bars=self._cfg.data_pipeline.history_bars_per_tf,
            as_of_utc=as_of_utc,
        )
        aligned_df = self._pipeline.build_aligned_dataset(bars_by_tf=bars_by_tf, drop_warmup_nans=True)
        if aligned_df.empty:
            logger.warning("Aligned feature dataset empty (insufficient warmup bars).")
            return None

        latest_row = aligned_df.iloc[-1]
        bar_close_ts = latest_row["close_time"].to_pydatetime()
        if self._last_processed_close_utc is not None and bar_close_ts <= self._last_processed_close_utc:
            return None

        live_tick = self._gateway.get_tick(symbol)
        snapshot = self._pipeline.row_to_market_snapshot(
            row=latest_row,
            symbol_spec=self._cfg.symbol_spec,
            live_tick=live_tick,
        )
        account = self._gateway.get_account_snapshot()

        ai_decision = self._onnx_engine.evaluate_snapshot(snapshot)
        cb_status = self._circuit_breaker.update(
            current_equity=account.equity,
            timestamp_utc=snapshot.timestamp_utc,
            regime=ai_decision.regime,
            vol_zscore=snapshot.vol_zscore,
        )

        self._last_processed_close_utc = bar_close_ts

        if not ai_decision.approved:
            self._audit_db.log_decision(ai_decision=ai_decision, cb_status=cb_status)
            return None

        entry_px = live_tick.ask if ai_decision.primary_side == SignalSide.LONG else live_tick.bid
        rk_cfg = self._cfg.risk_and_scaling
        port_vol_proxy = (
            0.75
            * rk_cfg.target_annual_volatility
            * min(2.5, max(0.6, snapshot.vol_ratio))
        )
        dd_taper_start = 0.45 * rk_cfg.max_trailing_drawdown_limit_pct
        if cb_status.trailing_drawdown_pct > dd_taper_start:
            excess_frac = (cb_status.trailing_drawdown_pct - dd_taper_start) / max(
                rk_cfg.max_trailing_drawdown_limit_pct - dd_taper_start,
                1e-6,
            )
            stress_mult = 1.0 + 1.25 * excess_frac
            port_vol_proxy = max(port_vol_proxy, rk_cfg.target_annual_volatility * stress_mult)
        size_res = self._sizer.calculate_position_size(
            equity=account.equity,
            side=ai_decision.primary_side,
            entry_price=entry_px,
            atr=snapshot.atr,
            spread_price=live_tick.spread_price,
            realized_annual_vol=port_vol_proxy,
            win_probability=ai_decision.meta_win_probability,
            reward_to_risk_ratio=ai_decision.reward_to_risk_ratio,
            circuit_status=cb_status,
        )
        if not size_res.approved:
            self._audit_db.log_decision(
                ai_decision=ai_decision,
                cb_status=cb_status,
                size_result=size_res,
            )
            return None

        plan = self._planner.build_execution_plan(
            parent_signal_id=ai_decision.signal_id,
            side=ai_decision.primary_side,
            requested_lots=size_res.final_lots,
            reference_entry_price=entry_px,
            stop_loss_price=size_res.stop_loss_price,
            take_profit_price=size_res.take_profit_price,
            atr=snapshot.atr,
            bar_tick_volume=snapshot.bar_tick_volume,
        )
        exec_report = self._executor.execute_plan(
            plan=plan,
            atr=snapshot.atr,
            decision_timestamp_utc=snapshot.timestamp_utc,
        )
        self._audit_db.log_decision(
            ai_decision=ai_decision,
            cb_status=cb_status,
            size_result=size_res,
            exec_report=exec_report,
        )
        return exec_report

    def run_vps_24_5_loop(self, max_cycles: Optional[int] = None) -> None:
        """Continuous 24/5 VPS event loop with graceful signal handling and fault isolation."""
        cycles = 0
        while not self._stop_requested:
            try:
                self.process_single_cycle()
            except Exception as exc:  # pragma: no cover - defensive daemon boundary
                logger.exception("Unhandled error in 24/5 trading loop (isolated): %s", exc)

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self._sleep(self._cfg.system.poll_interval_seconds)


def run_demo_pipeline(config_path: Path, total_m5_bars: int = 30000) -> None:
    """Execute complete end-to-end architecture verification on multi-regime market data."""
    cfg = load_config(config_path)
    base_dir = config_path.parent
    onnx_dir = base_dir / cfg.regime_ai.onnx_dir

    bars_per_day = 288.0  # 24h * 12 M5 bars/hour
    sim_gateway = SimulatedMT5Gateway(
        symbol_spec=cfg.symbol_spec,
        initial_equity=cfg.backtest_and_validation.initial_equity,
        total_m5_bars=total_m5_bars,
        seed=cfg.system.random_seed,
    )
    fetcher = MultiTimeframeDataFetcher(sim_gateway)
    pipeline = MultiTimeframeFeaturePipeline(cfg.data_pipeline)

    bars_by_tf = fetcher.fetch_multi_timeframe_bars(
        symbol=cfg.system.symbol,
        base_tf=cfg.data_pipeline.base_timeframe,
        higher_tfs=cfg.data_pipeline.higher_timeframes,
        num_bars=total_m5_bars,
    )
    aligned_df = pipeline.build_aligned_dataset(bars_by_tf=bars_by_tf, drop_warmup_nans=True)
    oos_bars = max(0, len(aligned_df) - cfg.backtest_and_validation.wfo_train_bars)
    logger.info(
        "Dataset Summary | Raw M5: %d bars (%.2f days) | Post-Warmup Aligned: %d bars (%.2f days) | "
        "WFO In-Sample Window: %d bars (%.2f days) | Strictly Out-Of-Sample Tested: %d bars (%.2f days)",
        total_m5_bars,
        total_m5_bars / bars_per_day,
        len(aligned_df),
        len(aligned_df) / bars_per_day,
        cfg.backtest_and_validation.wfo_train_bars,
        cfg.backtest_and_validation.wfo_train_bars / bars_per_day,
        oos_bars,
        oos_bars / bars_per_day,
    )

    # 1. Walk-Forward Optimization + ONNX Export
    wfo = WalkForwardOptimizer(config=cfg, onnx_artifact_dir=onnx_dir)
    wfo_res = wfo.run(aligned_df)
    m = wfo_res.metrics
    oos_start = wfo_res.equity_curve.index[0]
    oos_end = wfo_res.equity_curve.index[-1]
    calendar_span_days = (oos_end - oos_start).total_seconds() / 86400.0
    logger.info(
        "WFO OOS Summary (%d Folds | %.2f OOS Days: %s -> %s) | "
        "Initial: $%.2f -> Final: $%.2f (%+.2f%% | CAGR: %.2f%%) | "
        "Ann Vol: %.2f%% | Sharpe: %.2f | Sortino: %.2f | Calmar: %.2f | "
        "MaxDD: %.2f%% | WinRate: %.2f%% | ProfitFactor: %.2f | Trades: %d",
        len(wfo_res.fold_summaries),
        calendar_span_days,
        oos_start.strftime("%Y-%m-%d %H:%M"),
        oos_end.strftime("%Y-%m-%d %H:%M"),
        m.initial_equity,
        m.final_equity,
        m.total_return_pct * 100.0,
        m.cagr_pct * 100.0,
        m.annualized_volatility_pct * 100.0,
        m.sharpe_ratio,
        m.sortino_ratio,
        m.calmar_ratio,
        m.max_drawdown_pct * 100.0,
        m.win_rate_pct * 100.0,
        m.profit_factor,
        m.total_trades,
    )

    # 2. 1,000-Iteration Monte Carlo Tail-Risk Simulation
    eq_arr = wfo_res.equity_curve.to_numpy(dtype=np.float64)
    bar_rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    mc = MonteCarloTailRiskSimulator(
        iterations=cfg.backtest_and_validation.monte_carlo_iterations,
        block_size=cfg.backtest_and_validation.monte_carlo_block_size,
        stress_friction_bps=0.5,
        seed=cfg.system.random_seed,
    )
    mc_rep = mc.run_from_returns(
        period_returns=bar_rets,
        initial_equity=cfg.backtest_and_validation.initial_equity,
        trailing_dd_limit_pct=cfg.risk_and_scaling.max_trailing_drawdown_limit_pct,
        hard_lock_dd_limit_pct=cfg.risk_and_scaling.hard_lock_drawdown_limit_pct,
    )
    logger.info(
        "Monte Carlo (%d runs) | P05 Eq: $%.2f | Median Eq: $%.2f | P95 Eq: $%.2f | "
        "Median MaxDD: %.2f%% | P95 MaxDD: %.2f%% | P99 Worst DD: %.2f%% | P(DD>CB %.1f%%): %.2f%%",
        mc_rep.iterations,
        mc_rep.p05_terminal_equity,
        mc_rep.median_terminal_equity,
        mc_rep.p95_terminal_equity,
        mc_rep.median_max_drawdown_pct * 100.0,
        mc_rep.p95_max_drawdown_pct * 100.0,
        mc_rep.p99_worst_case_drawdown_pct * 100.0,
        cfg.risk_and_scaling.max_trailing_drawdown_limit_pct * 100.0,
        mc_rep.prob_breach_4pct_dd * 100.0,
    )

    # 3. Train Production ONNX Artifacts & Run Live Orchestrator Cycle + SQLite Audit
    trainer = MetaLabelingTrainer(
        ai_config=cfg.regime_ai,
        symbol_spec=cfg.symbol_spec,
        random_seed=cfg.system.random_seed,
    )
    artifacts = trainer.train_and_export(train_df=aligned_df.iloc[-2400:], output_dir=onnx_dir)
    onnx_engine = ONNXMetaLabelingEngine(
        ai_config=cfg.regime_ai,
        risk_config=cfg.risk_and_scaling,
        symbol_spec=cfg.symbol_spec,
        primary_model_source=artifacts.primary_onnx_path,
        meta_model_source=artifacts.meta_onnx_path,
    )

    db_cfg = cfg.execution_and_monitoring.database
    resolved_sqlite = str(base_dir / db_cfg.sqlite_path)
    from core.config_loader import DatabaseConfig

    audit_db = DecisionAuditDatabase(
        DatabaseConfig(backend="sqlite", sqlite_path=resolved_sqlite, postgres_dsn="")
    )
    orchestrator = TradingSystemOrchestrator(
        config=cfg,
        gateway=sim_gateway,
        onnx_engine=onnx_engine,
        audit_db=audit_db,
        sleep_fn=lambda _: None,
    )
    orchestrator.process_single_cycle(as_of_utc=aligned_df["close_time"].iloc[-2].to_pydatetime())
    logger.info(
        "Live Orchestrator cycle complete | Total SQLite audit rows: %d",
        audit_db.count_logged_decisions(),
    )
    audit_db.close()


def run_mt5_training_pipeline(
    config_path: Path,
    total_m5_bars: int = 30000,
    gateway: Optional[MT5GatewayProtocol] = None,
) -> None:
    """Fetch historical MTF bars from MetaTrader 5, validate via WFO, and export ONNX models."""
    cfg = load_config(config_path)
    base_dir = config_path.parent
    onnx_dir = base_dir / cfg.regime_ai.onnx_dir

    owns_gateway = gateway is None
    if gateway is None:
        gateway = NativeMT5Gateway(
            login=cfg.execution_and_monitoring.mt5_login,
            password=cfg.execution_and_monitoring.mt5_password,
            server=cfg.execution_and_monitoring.mt5_server,
            terminal_path=cfg.execution_and_monitoring.mt5_terminal_path,
        )
        if not gateway.connect():
            raise RuntimeError("Initial connection to MetaTrader 5 terminal failed during train-mt5.")

    try:
        fetcher = MultiTimeframeDataFetcher(gateway)
        pipeline = MultiTimeframeFeaturePipeline(cfg.data_pipeline)
        bars_by_tf = fetcher.fetch_multi_timeframe_bars(
            symbol=cfg.system.symbol,
            base_tf=cfg.data_pipeline.base_timeframe,
            higher_tfs=cfg.data_pipeline.higher_timeframes,
            num_bars=total_m5_bars,
        )
        aligned_df = pipeline.build_aligned_dataset(bars_by_tf=bars_by_tf, drop_warmup_nans=True)
        train_window = min(len(aligned_df), cfg.backtest_and_validation.wfo_train_bars)
        if train_window < 300:
            raise ValueError(
                f"Insufficient aligned bars ({len(aligned_df)}) from MT5 to train ONNX models."
            )

        if len(aligned_df) >= cfg.backtest_and_validation.wfo_train_bars + cfg.backtest_and_validation.wfo_test_bars:
            wfo = WalkForwardOptimizer(config=cfg, onnx_artifact_dir=onnx_dir)
            wfo_res = wfo.run(aligned_df)
            m = wfo_res.metrics
            logger.info(
                "MT5 Broker WFO Validation | Return: %+.2f%% | Sharpe: %.2f | Sortino: %.2f | "
                "MaxDD: %.2f%% | WinRate: %.2f%% | Trades: %d",
                m.total_return_pct * 100.0,
                m.sharpe_ratio,
                m.sortino_ratio,
                m.max_drawdown_pct * 100.0,
                m.win_rate_pct * 100.0,
                m.total_trades,
            )

        trainer = MetaLabelingTrainer(
            ai_config=cfg.regime_ai,
            symbol_spec=cfg.symbol_spec,
            random_seed=cfg.system.random_seed,
        )
        artifacts = trainer.train_and_export(
            train_df=aligned_df.iloc[-train_window:],
            output_dir=onnx_dir,
        )
        logger.info(
            "Production ONNX models exported successfully -> Primary: %s | Meta: %s",
            artifacts.primary_onnx_path,
            artifacts.meta_onnx_path,
        )
    finally:
        if owns_gateway:
            gateway.disconnect()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Antigravity Ultra-Low Volatility AI Trading System")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["demo-pipeline", "train-mt5", "live-mt5"],
        default="demo-pipeline",
        help="Execution mode: demo-pipeline (simulation), train-mt5 (train ONNX on MT5 data), live-mt5 (24/5 VPS)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=30000,
        help="Number of M5 bars to fetch/simulate in demo-pipeline or train-mt5 mode (288 M5 bars = 1 day)",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()

    if args.mode == "demo-pipeline":
        run_demo_pipeline(cfg_path, total_m5_bars=args.bars)
        return

    if args.mode == "train-mt5":
        run_mt5_training_pipeline(cfg_path, total_m5_bars=args.bars)
        return

    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent
    onnx_dir = base_dir / cfg.regime_ai.onnx_dir
    primary_path = onnx_dir / cfg.regime_ai.primary_onnx_filename
    meta_path = onnx_dir / cfg.regime_ai.meta_onnx_filename
    if not (primary_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"ONNX model files not found in {onnx_dir}. Train and export models before starting live-mt5."
        )

    gateway = NativeMT5Gateway(
        login=cfg.execution_and_monitoring.mt5_login,
        password=cfg.execution_and_monitoring.mt5_password,
        server=cfg.execution_and_monitoring.mt5_server,
        terminal_path=cfg.execution_and_monitoring.mt5_terminal_path,
    )
    if not gateway.connect():
        raise RuntimeError("Initial connection to MetaTrader 5 terminal failed.")

    onnx_engine = ONNXMetaLabelingEngine(
        ai_config=cfg.regime_ai,
        risk_config=cfg.risk_and_scaling,
        symbol_spec=cfg.symbol_spec,
        primary_model_source=primary_path,
        meta_model_source=meta_path,
    )
    audit_db = DecisionAuditDatabase(cfg.execution_and_monitoring.database)
    orchestrator = TradingSystemOrchestrator(
        config=cfg,
        gateway=gateway,
        onnx_engine=onnx_engine,
        audit_db=audit_db,
    )

    def _handle_sig(_signum, _frame):
        logger.info("Shutdown signal received; stopping 24/5 orchestrator cleanly...")
        orchestrator.request_stop()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        orchestrator.run_vps_24_5_loop()
    finally:
        audit_db.close()
        gateway.disconnect()


if __name__ == "__main__":
    main()
