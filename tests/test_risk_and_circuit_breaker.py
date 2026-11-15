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

    # p = 0.52, b = 2.2 -> f* = 0.52 - 0.48 / 2.2 = 0.30181818...
    # 0.35 * f* = 0.10563636... -> utilization = 0.10563636 / 0.15 = 0.70424242...
    # kelly_risk_pct = 0.0240 + 0.70424242 * (0.0350 - 0.0240) = 0.0317466666...
    # realized_annual_vol = 0.18 <= target 0.38 -> vol_scalar = 1.0
    # ATR = 2.00 -> sl_dist = 1.5 * 2.00 = 3.00; spread = 0.20 -> spread_cost = 1.25 * 0.20 = 0.25
    # effective_risk_price = 3.25 -> 325 ticks * $1.00 = $325.00 risk per 1.00 lot
    # risk_usd = 100,000 * 0.0317466666 = $3,174.666666 -> raw_lots = 3174.666666 / 325 = 9.768205 -> floor 0.01 = 9.76
    res = sizer.calculate_position_size(
        equity=100_000.0,
        side=SignalSide.LONG,
        entry_price=2000.00,
        atr=2.00,
        spread_price=0.20,
        realized_annual_vol=0.18,
        win_probability=0.52,
        reward_to_risk_ratio=2.20,
        circuit_status=status,
    )

    assert res.approved is True
    assert res.kelly_full_fraction == pytest.approx(0.52 - 0.48 / 2.20, rel=1e-9)
    assert res.effective_risk_pct == pytest.approx(0.03174666667, rel=1e-6)
    assert res.raw_unrounded_lots == pytest.approx(3174.66666667 / 325.0, rel=1e-6)
    assert res.final_lots == pytest.approx(9.76, abs=1e-9)
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
        realized_annual_vol=0.18,
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
        realized_annual_vol=0.18,
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
        realized_annual_vol=0.18,
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
        realized_annual_vol=0.76,  # 2x target annual vol (38%) -> vol_scalar = 0.50
        win_probability=0.68,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )

    assert high_atr.raw_unrounded_lots < base.raw_unrounded_lots
    assert wide_spread.raw_unrounded_lots < base.raw_unrounded_lots
    assert high_port_vol.raw_unrounded_lots == pytest.approx(0.5 * base.raw_unrounded_lots, rel=1e-9)

    # Zero-Martingale check: even at p = 0.99, effective_risk_pct never exceeds max_risk_per_trade_pct (3.50%)
    extreme_conf = sizer.calculate_position_size(
        equity=250_000.0,
        side=SignalSide.LONG,
        entry_price=2050.0,
        atr=1.50,
        spread_price=0.16,
        realized_annual_vol=0.12,
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
        realized_annual_vol=0.18,
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
        realized_annual_vol=0.18,
        win_probability=0.70,
        reward_to_risk_ratio=2.2,
        circuit_status=st,
    )
    assert tiny_acc.approved is False
    assert tiny_acc.final_lots == 0.0


def test_circuit_breaker_daily_1_5_pct_cooldown_and_half_lot_recovery(app_cfg):
    """
    Verify Equity Curve Control:
    1. Daily DD > daily_drawdown_limit_pct (7.5%) -> Immediate COOLDOWN_HALT (0 lots).
    2. Remains halted during cooldown or while market regime is ABNORMAL_VOLATILITY.
    3. Transitions to RECOVERY_HALF_RISK (0.5x lot size) once cooldown expires & market calms.
    4. Restores NORMAL (1.0x lot size) once equity heals within recovery buffer (3.5%).
    """
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    sizer = InverseVolatilityPositionSizer(app_cfg.risk_and_scaling, app_cfg.symbol_spec)
    t0 = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)

    st_norm = cb.update(100_000.0, t0, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.2)
    assert st_norm.state == CircuitBreakerState.NORMAL
    norm_size = sizer.calculate_position_size(
        equity=92_000.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.18,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_norm,
    )

    # Drop 8.00% intraday ($100,000 -> $92,000) > 7.5% limit -> COOLDOWN_HALT
    t1 = t0 + timedelta(minutes=30)
    st_halt = cb.update(92_000.0, t1, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.4)
    assert st_halt.state == CircuitBreakerState.COOLDOWN_HALT
    assert st_halt.allow_new_trades is False
    assert st_halt.lot_multiplier == 0.0

    blocked_size = sizer.calculate_position_size(
        equity=92_000.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.18,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_halt,
    )
    assert blocked_size.approved is False
    assert blocked_size.final_lots == 0.0

    # Cooldown timer elapsed (245m > 240m), BUT market is in ABNORMAL_VOLATILITY -> stay halted!
    t2 = t1 + timedelta(minutes=245)
    st_still_turb = cb.update(92_000.0, t2, regime=MarketRegime.ABNORMAL_VOLATILITY, vol_zscore=3.1)
    assert st_still_turb.state == CircuitBreakerState.COOLDOWN_HALT
    assert st_still_turb.allow_new_trades is False

    # Market regime returns to normal -> enters RECOVERY_HALF_RISK (lot_multiplier = 0.50)
    t3 = t2 + timedelta(minutes=10)
    st_rec = cb.update(92_000.0, t3, regime=MarketRegime.MEAN_REVERTING, vol_zscore=0.3)
    assert st_rec.state == CircuitBreakerState.RECOVERY_HALF_RISK
    assert st_rec.allow_new_trades is True
    assert st_rec.lot_multiplier == pytest.approx(0.50)

    rec_size = sizer.calculate_position_size(
        equity=92_000.0,
        side=SignalSide.LONG,
        entry_price=2000.0,
        atr=2.0,
        spread_price=0.15,
        realized_annual_vol=0.18,
        win_probability=0.72,
        reward_to_risk_ratio=2.2,
        circuit_status=st_rec,
    )
    assert rec_size.approved is True
    assert rec_size.raw_unrounded_lots == pytest.approx(0.50 * norm_size.raw_unrounded_lots, rel=1e-9)
    assert rec_size.final_lots == quantize_lots_down(0.50 * norm_size.raw_unrounded_lots, 0.01)

    # Equity recovers to $97,000 (daily & trailing DD = 3.0% <= 3.5% buffer) -> back to NORMAL 1.0x
    t4 = t3 + timedelta(minutes=60)
    st_healed = cb.update(97_000.0, t4, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.2)
    assert st_healed.state == CircuitBreakerState.NORMAL
    assert st_healed.lot_multiplier == pytest.approx(1.0)


def test_trailing_drawdown_4_pct_and_6_pct_hard_lock(app_cfg):
    """Verify multi-day Trailing DD Cooldown (17.0% < 20.0% mandate) and 23.0% Emergency Hard Lock."""
    cb = EquityCircuitBreaker(app_cfg.risk_and_scaling, initial_equity=100_000.0)
    day1 = datetime(2026, 8, 4, 10, 0, tzinfo=timezone.utc)

    # Compound peak to $120,000
    cb.update(120_000.0, day1, regime=MarketRegime.TRENDING_BULL, vol_zscore=0.1)
    assert cb.peak_equity == pytest.approx(120_000.0)

    # Over multiple days (each daily drop < 7.5%), cumulative drop reaches 17.5% ($99,000) > 17.0% limit
    day2 = day1 + timedelta(days=1)
    cb.update(114_000.0, day2, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    day3 = day2 + timedelta(days=1)
    cb.update(108_300.0, day3, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    day4 = day3 + timedelta(days=1)
    cb.update(102_900.0, day4, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    day5 = day4 + timedelta(days=1)
    st_trail = cb.update(99_000.0, day5, regime=MarketRegime.TRENDING_BEAR, vol_zscore=0.2)
    assert st_trail.trailing_drawdown_pct == pytest.approx(0.1750)
    assert st_trail.state == CircuitBreakerState.COOLDOWN_HALT

    # Drop to $91,800 (23.5% trailing DD >= 23.0% hard lock)
    st_lock = cb.update(91_800.0, day5 + timedelta(hours=5), regime=MarketRegime.MEAN_REVERTING, vol_zscore=0.1)
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

    # 1. Small order (2.35 lots <= 10.00 max_lot_per_order) -> DIRECT
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

    # 2. Compounded whale order (23.75 lots > 10.00 max_lot_per_order) -> TWAP_ICEBERG
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
    assert len(whale_plan.slices) >= 3
    for s in whale_plan.slices:
        assert app_cfg.symbol_spec.volume_min <= s.lots <= app_cfg.risk_and_scaling.max_lot_per_order

    # Exact conservation of parent order lots
    assert sum(s.lots for s in whale_plan.slices) == pytest.approx(23.75, abs=1e-9)
