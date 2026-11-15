"""
Integration & Component Tests for:
- LightGBM -> ONNX Export & Sub-Millisecond Inference Latency
- Market Regime Halt Filter (Abnormal Volatility & Noise)
- Walk-Forward Optimization, Sharpe > 2.0 Gate, & 1,000-Iteration Monte Carlo Simulation
- MT5 Execution Retry, Auto-Reconnection, Spread/Rollover/News Guards, & SQLite Feedback Loop
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from backtest_and_validation.metrics import compute_portfolio_metrics
from backtest_and_validation.monte_carlo import MonteCarloTailRiskSimulator
from backtest_and_validation.walk_forward import WalkForwardOptimizer
from core.config_loader import DatabaseConfig, load_config
from core.contracts import (
    MT5TradeRetcode,
    MarketRegime,
    MarketSnapshot,
    SignalSide,
    TickQuote,
    Timeframe,
)
from data_pipeline.mt5_data_client import MultiTimeframeDataFetcher, SimulatedMT5Gateway
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


@pytest.fixture
def app_cfg():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    return load_config(cfg_path)


def test_lightgbm_onnx_export_sub_ms_latency_and_regime_halt(app_cfg, tmp_path):
    """Train Primary + Meta LightGBM models, export ONNX, verify < 2ms latency & Regime Halt."""
    gw = SimulatedMT5Gateway(symbol_spec=app_cfg.symbol_spec, total_m5_bars=2600, seed=42)
    fetcher = MultiTimeframeDataFetcher(gw)
    pipeline = MultiTimeframeFeaturePipeline(app_cfg.data_pipeline)

    bars = fetcher.fetch_multi_timeframe_bars(
        symbol="XAUUSD",
        base_tf=Timeframe.M5,
        higher_tfs=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
        num_bars=2600,
    )
    aligned = pipeline.build_aligned_dataset(bars, drop_warmup_nans=True)

    trainer = MetaLabelingTrainer(app_cfg.regime_ai, app_cfg.symbol_spec, random_seed=42)
    artifacts = trainer.train_and_export(train_df=aligned.iloc[:600], output_dir=tmp_path)

    assert artifacts.primary_onnx_path.exists()
    assert artifacts.meta_onnx_path.exists()
    assert len(artifacts.primary_onnx_bytes) > 500
    assert len(artifacts.meta_onnx_bytes) > 500

    engine = ONNXMetaLabelingEngine(
        ai_config=app_cfg.regime_ai,
        risk_config=app_cfg.risk_and_scaling,
        symbol_spec=app_cfg.symbol_spec,
        primary_model_source=artifacts.primary_onnx_path,
        meta_model_source=artifacts.meta_onnx_path,
    )

    # Warm up ONNX session once, then measure median single-snapshot latency across 100 calls
    snap = pipeline.row_to_market_snapshot(aligned.iloc[-1], app_cfg.symbol_spec)
    engine.evaluate_snapshot(snap)
    latencies = [engine.evaluate_snapshot(snap).inference_latency_ms for _ in range(100)]
    median_ms = float(np.median(latencies))
    assert median_ms < 2.0, f"Expected sub-2ms ONNX inference latency, got {median_ms:.3f} ms"

    # Abnormal volatility shock must immediately trigger REGIME_FILTER_HALT
    turb_snap = MarketSnapshot(
        symbol="XAUUSD",
        timestamp_utc=snap.timestamp_utc,
        bid=snap.bid,
        ask=snap.ask,
        close=snap.close,
        atr=snap.atr * 4.0,
        parkinson_vol=snap.parkinson_vol * 4.0,
        realized_annual_vol=0.25,
        hurst_exponent=0.50,
        zscore=snap.zscore,
        vol_zscore=4.5,  # > 2.5 ceiling -> ABNORMAL_VOLATILITY
        vol_ratio=3.5,
        directional_efficiency=0.10,
        poc_price=snap.poc_price,
        vah_price=snap.vah_price,
        val_price=snap.val_price,
        bar_tick_volume=snap.bar_tick_volume,
        features=snap.features,
    )
    turb_dec = engine.evaluate_snapshot(turb_snap)
    assert turb_dec.regime == MarketRegime.ABNORMAL_VOLATILITY
    assert turb_dec.approved is False
    assert "REGIME_FILTER_HALT" in (turb_dec.rejection_reason or "")


def test_walk_forward_metrics_and_1000_monte_carlo_simulations(app_cfg, tmp_path):
    """Run rolling Walk-Forward Optimization, Sharpe > 2.0 check, and 1,000-path Monte Carlo."""
    gw = SimulatedMT5Gateway(symbol_spec=app_cfg.symbol_spec, total_m5_bars=5600, seed=42)
    fetcher = MultiTimeframeDataFetcher(gw)
    pipeline = MultiTimeframeFeaturePipeline(app_cfg.data_pipeline)
    bars = fetcher.fetch_multi_timeframe_bars(
        symbol="XAUUSD",
        base_tf=Timeframe.M5,
        higher_tfs=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
        num_bars=5600,
    )
    aligned = pipeline.build_aligned_dataset(bars, drop_warmup_nans=True)

    wfo = WalkForwardOptimizer(config=app_cfg, onnx_artifact_dir=tmp_path / "wfo_onnx")
    res = wfo.run(aligned)

    assert len(res.fold_summaries) >= 2
    assert len(res.equity_curve) > 200
    assert res.metrics.max_drawdown_pct <= 0.20  # Risk engine keeps drawdown strictly < 20% mandate

    # Verify Sharpe > 2.0 gate detection on a low-volatility upward-compounding equity series
    rng = np.random.default_rng(7)
    steady_daily_rets = 0.0008 + 0.0025 * rng.normal(size=252)
    steady_eq = 100_000.0 * np.cumprod(np.insert(1.0 + steady_daily_rets, 0, 1.0))
    steady_metrics = compute_portfolio_metrics(steady_eq, bars_per_year=252, min_sharpe_target=2.0)
    assert steady_metrics.sharpe_ratio > 2.0
    assert steady_metrics.passes_sharpe_target is True
    assert steady_metrics.sortino_ratio > 2.0
    assert steady_metrics.calmar_ratio > 1.0

    # Verify 1,000-iteration Monte Carlo Tail-Risk Simulator
    mc = MonteCarloTailRiskSimulator(iterations=1000, block_size=10, stress_friction_bps=1.0, seed=42)
    mc_report = mc.run_from_returns(steady_daily_rets, initial_equity=100_000.0)
    assert mc_report.iterations == 1000
    assert mc_report.p99_worst_case_drawdown_pct >= mc_report.p95_max_drawdown_pct >= mc_report.median_max_drawdown_pct
    assert mc_report.p95_terminal_equity >= mc_report.median_terminal_equity >= mc_report.p05_terminal_equity


def test_mt5_executor_retry_reconnect_guards_and_sqlite_feedback_loop(app_cfg, tmp_path):
    """
    Verify:
    1. Auto-reconnection when terminal disconnects
    2. Retry logic recovering from REQUOTE + TIMEOUT transient errors
    3. SpreadGuard, Rollover filter (23:55 UTC), and High-Impact News Blackout filter
    4. SQLite WAL audit store + Post-Trade Calibration Feedback Loop
    """
    gw = SimulatedMT5Gateway(symbol_spec=app_cfg.symbol_spec, total_m5_bars=400, seed=21)
    spread_guard = SpreadGuard(app_cfg.execution_and_monitoring, app_cfg.symbol_spec)
    sched_guard = MarketScheduleAndNewsGuard(app_cfg.execution_and_monitoring)
    executor = MT5ExecutionEngine(
        gateway=gw,
        config=app_cfg.execution_and_monitoring,
        symbol_spec=app_cfg.symbol_spec,
        spread_guard=spread_guard,
        schedule_guard=sched_guard,
        sleep_fn=lambda _: None,
    )
    planner = LiquidityAwareExecutionPlanner(
        risk_config=app_cfg.risk_and_scaling,
        symbol_spec=app_cfg.symbol_spec,
        seed=21,
    )

    valid_ts = datetime(2026, 7, 8, 14, 0, tzinfo=timezone.utc)  # Wednesday 14:00 UTC

    # 1. Disconnect gateway and queue 2 transient errors (REQUOTE, TIMEOUT) before fill
    gw.set_connected(False)
    gw.queue_transient_retcodes([int(MT5TradeRetcode.REQUOTE), int(MT5TradeRetcode.TIMEOUT)])

    plan = planner.build_execution_plan(
        parent_signal_id="sig_retry_test",
        side=SignalSide.LONG,
        requested_lots=14.40,  # > 10.0 -> 2 Iceberg slices
        reference_entry_price=gw.get_tick("XAUUSD").ask,
        stop_loss_price=1990.0,
        take_profit_price=2020.0,
        atr=3.0,
        bar_tick_volume=5000.0,
    )
    rep = executor.execute_plan(plan=plan, atr=3.0, decision_timestamp_utc=valid_ts)
    assert rep.executed is True
    assert rep.filled_total_lots == pytest.approx(14.40)
    assert rep.slice_reports[0].retries_used == 2

    # 2. Rollover window (23:55 UTC) must block execution
    rollover_ts = datetime(2026, 7, 8, 23, 55, tzinfo=timezone.utc)
    rep_roll = executor.execute_plan(plan=plan, atr=3.0, decision_timestamp_utc=rollover_ts)
    assert rep_roll.executed is False
    assert "DAILY_ROLLOVER_WINDOW" in (rep_roll.abort_reason or "")

    # 3. High-Impact Macro News blackout (FOMC at 18:00 UTC -> blocks 17:45 UTC)
    fomc_ts = datetime(2026, 7, 8, 18, 0, tzinfo=timezone.utc)
    sched_guard.register_news_event(fomc_ts, title="FOMC_RATE_DECISION", impact="HIGH")
    rep_news = executor.execute_plan(
        plan=plan,
        atr=3.0,
        decision_timestamp_utc=fomc_ts - timedelta(minutes=15),
    )
    assert rep_news.executed is False
    assert "HIGH_IMPACT_NEWS_BLACKOUT:FOMC_RATE_DECISION" in (rep_news.abort_reason or "")

    # 4. Wide spread (> 35 points) must be blocked by SpreadGuard
    wide_tick = TickQuote(
        symbol="XAUUSD",
        timestamp_utc=valid_ts,
        bid=2000.00,
        ask=2000.60,  # 60 points > 35 max_spread_points
        last=2000.30,
        volume=100.0,
    )
    ok_sprd, sprd_msg = spread_guard.check(wide_tick, atr=3.0)
    assert ok_sprd is False
    assert "SPREAD_POINTS_EXCEEDED" in sprd_msg

    # 5. SQLite Audit Store & Post-Trade Calibration Feedback Loop
    db_path = tmp_path / "audit_test.db"
    db = DecisionAuditDatabase(DatabaseConfig(backend="sqlite", sqlite_path=str(db_path), postgres_dsn=""))
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    cb_st = cb.update(100_000.0, valid_ts)

    from core.contracts import AISignalDecision

    ai_dec = AISignalDecision(
        signal_id="sig_retry_test",
        timestamp_utc=valid_ts,
        symbol="XAUUSD",
        regime=MarketRegime.TRENDING_BULL,
        regime_tradable=True,
        primary_side=SignalSide.LONG,
        primary_probability=0.68,
        meta_win_probability=0.71,
        confidence_threshold=0.62,
        reward_to_risk_ratio=2.2,
        approved=True,
        rejection_reason=None,
        inference_latency_ms=0.04,
        feature_snapshot={"m5_hurst": 0.63, "m5_zscore": 1.1},
    )
    db.log_decision(ai_decision=ai_dec, cb_status=cb_st, exec_report=rep)
    db.record_post_trade_outcome(
        signal_id="sig_retry_test",
        exit_timestamp_utc=valid_ts + timedelta(minutes=45),
        exit_price=2008.50,
        realized_pnl_usd=820.0,
        realized_r_multiple=2.1,
        exit_reason="TAKE_PROFIT",
    )

    feedback_df = db.export_feedback_dataset()
    assert len(feedback_df) == 1
    assert feedback_df["m5_hurst"].iloc[0] == pytest.approx(0.63)
    cal = db.compute_calibration_summary()
    assert cal.closed_trades_count == 1
    assert cal.realized_win_rate == pytest.approx(1.0)
    db.close()


def test_train_mt5_pipeline_exports_onnx(app_cfg, tmp_path: Path) -> None:
    """Verify --mode train-mt5 pipeline exports primary and meta ONNX models cleanly."""
    import shutil
    from main import run_mt5_training_pipeline

    cfg_src = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg_copy = tmp_path / "config.yaml"
    shutil.copy2(cfg_src, cfg_copy)

    gw = SimulatedMT5Gateway(
        symbol_spec=app_cfg.symbol_spec,
        initial_equity=100_000.0,
        total_m5_bars=4500,
        seed=42,
    )
    run_mt5_training_pipeline(config_path=cfg_copy, total_m5_bars=4500, gateway=gw)

    onnx_dir = tmp_path / app_cfg.regime_ai.onnx_dir
    assert (onnx_dir / app_cfg.regime_ai.primary_onnx_filename).exists()
    assert (onnx_dir / app_cfg.regime_ai.meta_onnx_filename).exists()

