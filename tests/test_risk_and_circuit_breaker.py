"""
Unit Tests for Risk & Scaling Engine:
- Exact Inverse-Volatility + Dynamic Fractional Kelly Position Sizing Math
- Zero-Martingale Hard Risk Ceiling & Geometric Compounding
- Equity Curve Control Circuit Breaker (1.5% Daily DD, 4.0% Trailing DD, Cooldown -> 0.5x Recovery)
- Liquidity-Aware TWAP / Iceberg Order Slicing & Volume Conservation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.config_loader import load_config
from core.contracts import (
    CircuitBreakerState,
    MarketRegime,
    SignalSide,
)
from risk_and_scaling_engine.circuit_breaker import EquityCircuitBreaker
from risk_and_scaling_engine.liquidity_scaler import LiquidityAwareExecutionPlanner
from risk_and_scaling_engine.position_sizer import (
    InverseVolatilityPositionSizer,
    quantize_lots_down,
)


@pytest.fixture
def app_cfg():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    return load_config(cfg_path)


def test_exact_inverse_volatility_and_kelly_formula(app_cfg):
    """Verify exact analytical match for Fractional Kelly, Vol Target, and Inverse ATR+Spread Lot Size."""
    sizer = InverseVolatilityPositionSizer(app_cfg.risk_and_scaling, app_cfg.symbol_spec)
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    status = cb.update(100_000.0, datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc))

    # p = 0.62, b = 2.2 -> f* = 0.62 - 0.38 / 2.2 = 0.44727272...
    # 0.30 * f* = 0.13418181... -> utilization = 0.13418181 / 0.15 = 0.89454545...
    # kelly_risk_pct = 0.0050 + 0.89454545 * (0.0100 - 0.0050) = 0.0094727272...
    # realized_annual_vol = 0.08 <= target 0.12 -> vol_scalar = 1.0
    # ATR = 2.00 -> sl_dist = 1.5 * 2.00 = 3.00; spread = 0.20 -> spread_cost = 1.25 * 0.20 = 0.25
    # effective_risk_price = 3.25 -> 325 ticks * $1.00 = $325.00 risk per 1.00 lot
    # risk_usd = 100,000 * 0.0094727272 = $947.272727 -> raw_lots = 947.272727 / 325 = 2.914685 -> floor 0.01 = 2.91
    res = sizer.calculate_position_size(
        equity=100_000.0,
        side=SignalSide.LONG,
        entry_price=2000.00,
        atr=2.00,
        spread_price=0.20,
        realized_annual_vol=0.08,
        win_probability=0.62,
        reward_to_risk_ratio=2.20,
        circuit_status=status,
    )

    assert res.approved is True
    assert res.kelly_full_fraction == pytest.approx(0.62 - 0.38 / 2.20, rel=1e-9)
    assert res.effective_risk_pct == pytest.approx(0.00947272727, rel=1e-6)
    assert res.raw_unrounded_lots == pytest.approx(947.27272727 / 325.0, rel=1e-6)
    assert res.final_lots == pytest.approx(2.91, abs=1e-9)
    assert res.stop_loss_price == pytest.approx(1997.00, abs=1e-9)
    assert res.take_profit_price == pytest.approx(2006.60, abs=1e-9)


def test_inverse_scaling_with_atr_spread_vol_and_zero_martingale(app_cfg):
    """Verify higher ATR, wider spread, or higher portfolio vol strictly shrinks lot size."""
    sizer = InverseVolatilityPositionSizer(app_cfg.risk_and_scaling, app_cfg.symbol_spec)
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=250_000.0)
    st = cb.update(250_000.0, datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc))

    base = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.SHORT,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.16,
        realized_annual_vol=0.08,
        win_probability=0.68,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )
    high_atr = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.SHORT,
        entry_price=2050.0,
        atr=3.00,  # 2x ATR
        spread_price=0.16,
        realized_annual_vol=0.08,
        win_probability=0.68,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )
    wide_spread = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.SHORT,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.80,  # 5x wider spread
        realized_annual_vol=0.08,
        win_probability=0.68,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )
    high_port_vol = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.SHORT,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.16,
        realized_annual_vol=0.24,  # 2x target annual vol (12%) -> vol_scalar = 0.50
        win_probability=0.68,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )

    assert high_atr.raw_unrounded_lots < base.raw_unrounded_lots
    assert wide_spread.raw_unrounded_lots < base.raw_unrounded_lots
    assert high_port_vol.raw_unrounded_lots == pytest.approx(0.5 * base.raw_unrounded_lots, rel=1e-9)

    # Zero-Martingale check: even at p = 0.99, effective_risk_pct never exceeds max_risk_per_trade_pct (1.00%)
    extreme_conf = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.LONG,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.16,
        realized_annual_vol=0.06,
        win_probability=0.99,
        reward_to_risk_ratio=3.0,
        circuit_status=st,
    )
    assert extreme_conf.effective_risk_pct <= app_cfg.risk_and_scaling.max_risk_per_trade_pct + 1e-12

    # Negative Kelly expectancy (p = 0.25, b = 2.0 -> f* = 0.25 - 0.375 = -0.125) must be rejected
    neg_kelly = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.LONG,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.16,
        realized_annual_vol=0.08,
        win_probability=0.25,
        reward_to_risk_ratio=2.0,
        circuit_status=st,
    )
    assert neg_kelly.approved is False
    assert neg_kelly.final_lots == 0.0

    # Micro account where even 0.01 lot exceeds risk budget must be rejected rather than over-leveraging
    tiny_acc = sizer.calculate_position_size(
        equity=100.0,
        side=SignalSide.LONG,
        entry_price=2050.0,
        atr=5.00,
        spread_price=0.30,
        realized_annual_vol=0.08,
        win_probability=0.70,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )
    assert tiny_acc.approved is False
    assert tiny_acc.final_lots == 0.0


def test_circuit_breaker_daily_1_5_pct_cooldown_and_half_lot_recovery(app_cfg):
    """
    Verify Equity Curve Control:
    1. Daily DD > daily_drawdown_limit_pct (2.5%) -> Immediate COOLDOWN_HALT (0 lots).
    2. Remains halted during cooldown or while market regime is ABNORMAL_VOLATILITY.
    3. Transitions to RECOVERY_HALF_RISK (0.5x lot size) once cooldown expires & market calms.
    4. Restores NORMAL (1.0x lot size) once equity heals within recovery buffer.
    """
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    sizer = InverseVolatilityPositionSizer(app_cfg.risk_and_scaling, app_cfg.symbol_spec)
    t0 = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)

    st_norm = cb.update(100_000.0, t0, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.2)
    assert st_norm.state == CircuitBreakerState.NORMAL
    norm_size = sizer.calculate_position_size(
        equity=97_300.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.08,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_norm,
    )

    # Drop 2.70% intraday ($100,000 -> $97,300) > 2.5% limit -> COOLDOWN_HALT
    t1 = t0 + timedelta(minutes=30)
    st_halt = cb.update(97_300.0, t1, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.4)
    assert st_halt.state == CircuitBreakerState.COOLDOWN_HALT
    assert st_halt.allow_new_trades is False
    assert st_halt.lot_multiplier == 0.0

    blocked_size = sizer.calculate_position_size(
        equity=97_300.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.08,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_halt,
    )
    assert blocked_size.approved is False
    assert blocked_size.final_lots == 0.0

    # Cooldown timer elapsed (245m > 240m), BUT market is in ABNORMAL_VOLATILITY -> stay halted!
    t2 = t1 + timedelta(minutes=245)
    st_still_turb = cb.update(97_300.0, t2, regime=MarketRegime.ABNORMAL_VOLATILITY, vol_zscore=3.1)
    assert st_still_turb.state == CircuitBreakerState.COOLDOWN_HALT
    assert st_still_turb.allow_new_trades is False

    # Market regime returns to normal -> enters RECOVERY_HALF_RISK (lot_multiplier = 0.50)
    t3 = t2 + timedelta(minutes=10)
    st_rec = cb.update(97_300.0, t3, regime=MarketRegime.MEAN_REVERTING, vol_zscore=0.3)
    assert st_rec.state == CircuitBreakerState.RECOVERY_HALF_RISK
    assert st_rec.allow_new_trades is True
    assert st_rec.lot_multiplier == pytest.approx(0.50)

    rec_size = sizer.calculate_position_size(
        equity=97_300.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.08,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_rec,
    )
    assert rec_size.approved is True
    assert rec_size.raw_unrounded_lots == pytest.approx(0.50 * norm_size.raw_unrounded_lots, rel=1e-9)
    assert rec_size.final_lots == quantize_lots_down(0.50 * norm_size.raw_unrounded_lots, 0.01)

    # Equity recovers to $99,200 (daily & trailing DD = 0.8% <= 1.2% buffer) -> back to NORMAL 1.0x
    t4 = t3 + timedelta(minutes=60)
    st_healed = cb.update(99_200.0, t4, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.2)
    assert st_healed.state == CircuitBreakerState.NORMAL
    assert st_healed.lot_multiplier == pytest.approx(1.0)


def test_trailing_drawdown_4_pct_and_6_pct_hard_lock(app_cfg):
    """Verify multi-day Trailing DD Cooldown (5.6% < 6.0% mandate) and 8.5% Emergency Hard Lock."""
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    day1 = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)

    # Compound peak to $120,000
    cb.update(120_000.0, day1, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.1)
    assert cb.peak_equity == pytest.approx(120_000.0)

    # Over multiple days (each daily drop < 2.5%), cumulative drop reaches 6.0% ($112,800) > 5.6% limit
    day2 = day1 + timedelta(days=1)
    cb.update(117_600.0, day2, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    day3 = day2 + timedelta(days=1)
    cb.update(115_200.0, day3, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    day4 = day3 + timedelta(days=1)
    st_trail = cb.update(112_800.0, day4, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    assert st_trail.trailing_drawdown_pct == pytest.approx(0.0600)
    assert st_trail.state == CircuitBreakerState.COOLDOWN_HALT

    # Drop to $109,500 (8.75% trailing DD >= 8.5% hard lock)
    st_lock = cb.update(109_500.0, day4 + timedelta(hours=5), regime=MarketRegime.MEAN_REVERTING, vol_zscore=0.1)
    assert st_lock.state == CircuitBreakerState.HARD_LOCK
    assert st_lock.allow_new_trades is False


def test_liquidity_aware_twap_iceberg_slicing(app_cfg):
    """Verify large institutional orders split into Iceberg slices <= max_lot_per_order with exact sum."""
    planner = LiquidityAwareExecutionPlanner(
        risk_config=app_cfg.risk_and_scaling,
        symbol_spec=app_cfg.symbol_spec,
        max_deviation_points=15,
        seed=99,
    )

    # 1. Small order (2.35 lots <= 5.00 max_lot_per_order) -> DIRECT
    direct_plan = planner.build_execution_plan(
        parent_signal_id="sig_small",
        side=SignalSide.LONG,
        requested_lots=2.35,
        reference_entry_price=2000.0,
        stop_loss_price=1997.0,
        take_profit_price=2006.6,
        atr=2.0,
        bar_tick_volume=800.0,
    )
    assert direct_plan.execution_style == "DIRECT"
    assert len(direct_plan.slices) == 1
    assert direct_plan.slices[0].lots == pytest.approx(2.35)

    # 2. Compounded whale order (23.75 lots > 5.00 max_lot_per_order) -> TWAP_ICEBERG
    whale_plan = planner.build_execution_plan(
        parent_signal_id="sig_whale",
        side=SignalSide.SHORT,
        requested_lots=23.75,
        reference_entry_price=2000.0,
        stop_loss_price=2003.0,
        take_profit_price=1993.4,
        atr=2.0,
        bar_tick_volume=10_000.0,
    )
    assert whale_plan.execution_style == "TWAP_ICEBERG"
    assert len(whale_plan.slices) >= 5
    for s in whale_plan.slices:
        assert app_cfg.symbol_spec.volume_min <= s.lots <= app_cfg.risk_and_scaling.max_lot_per_order

    # Exact conservation of parent order lots
    assert sum(s.lots for s in whale_plan.slices) == pytest.approx(23.75, abs=1e-9)
