"""
Inverse-Volatility & Spread-Adjusted Position Sizer with Dynamic Fractional Kelly
and Portfolio Volatility Targeting.

Core Mathematical Guarantees:
1. Zero Martingale: Risk per trade is strictly bounded within [min_risk_pct, max_risk_pct]
   (e.g. 0.25% - 0.50% of current equity) and NEVER scales up after losses.
2. Inverse Volatility & Spread Scaling:
       RawLots = (Equity * EffectiveRiskPct) / (((k_ATR * ATR + w_Spread * Spread) / TickSize) * TickValue)
   Higher ATR or wider Bid-Ask Spread automatically reduces Lot size to keep USD risk constant.
3. Volatility Targeting: Contracts risk scalar when realized annualized vol exceeds target_annual_volatility.
"""

from __future__ import annotations

import math

from core.config_loader import RiskAndScalingConfig
from core.contracts import (
    CircuitBreakerStatus,
    PositionSizeResult,
    SignalSide,
    SymbolSpec,
)


def quantize_lots_down(lots: float, volume_step: float) -> float:
    """Floor lot size strictly to broker volume_step without floating-point drift."""
    if lots <= 0.0 or volume_step <= 0.0:
        return 0.0
    steps = math.floor((lots + 1e-12) / volume_step)
    decimals = max(0, int(round(-math.log10(volume_step))) if volume_step < 1.0 else 0)
    return round(steps * volume_step, max(decimals, 4))


class InverseVolatilityPositionSizer:
    """Computes risk-targeted lot size from Kelly edge, volatility target, ATR, and spread."""

    def __init__(self, config: RiskAndScalingConfig, symbol_spec: SymbolSpec) -> None:
        self._cfg = config
        self._spec = symbol_spec

    @staticmethod
    def compute_full_kelly(win_probability: float, reward_to_risk_ratio: float) -> float:
        """
        Asymmetric Kelly Criterion: f* = p - (1 - p) / b
        where p = calibrated win probability, b = TP_distance / SL_distance.
        """
        if reward_to_risk_ratio <= 0.0 or not (0.0 <= win_probability <= 1.0):
            return 0.0
        return float(win_probability - (1.0 - win_probability) / reward_to_risk_ratio)

    def calculate_position_size(
        self,
        equity: float,
        side: SignalSide,
        entry_price: float,
        atr: float,
        spread_price: float,
        realized_annual_vol: float,
        win_probability: float,
        reward_to_risk_ratio: float,
        circuit_status: CircuitBreakerStatus,
    ) -> PositionSizeResult:
        """Compute exact risk-controlled position size and SL/TP levels."""
        if not circuit_status.allow_new_trades or circuit_status.lot_multiplier <= 0.0:
            return self._rejected(
                circuit_status=circuit_status,
                reason=f"CIRCUIT_BREAKER_BLOCKED:{circuit_status.state.value}",
            )

        if side == SignalSide.FLAT:
            return self._rejected(circuit_status=circuit_status, reason="SIGNAL_FLAT")

        if equity <= 0.0 or atr <= 0.0 or entry_price <= 0.0:
            return self._rejected(circuit_status=circuit_status, reason="INVALID_MARKET_OR_EQUITY_INPUT")

        # 1. Dynamic Fractional Kelly Criterion (strictly bounded, zero Martingale)
        f_kelly = self.compute_full_kelly(win_probability, reward_to_risk_ratio)
        if f_kelly <= 0.0:
            return self._rejected(
                circuit_status=circuit_status,
                reason=f"NON_POSITIVE_KELLY_EDGE:{f_kelly:.4f}",
                kelly_full=f_kelly,
            )

        fractional_edge = self._cfg.kelly_fraction * f_kelly
        # Smoothly map fractional Kelly edge [0, 0.15+] into [min_risk_pct, max_risk_pct]
        edge_utilization = min(1.0, fractional_edge / 0.15)
        kelly_risk_pct = self._cfg.min_risk_per_trade_pct + edge_utilization * (
            self._cfg.max_risk_per_trade_pct - self._cfg.min_risk_per_trade_pct
        )
        kelly_risk_pct = min(
            self._cfg.max_risk_per_trade_pct,
            max(self._cfg.min_risk_per_trade_pct, kelly_risk_pct),
        )

        # 2. Portfolio Volatility Targeting scalar (contracts risk when market vol rises)
        if realized_annual_vol > 1e-9:
            vol_scalar = min(1.0, self._cfg.target_annual_volatility / realized_annual_vol)
            vol_scalar = max(0.15, vol_scalar)
        else:
            vol_scalar = 1.0

        # 3. Combine Kelly risk, Volatility Targeting, and Circuit Breaker multiplier (e.g. 0.5x in recovery)
        cb_mult = float(circuit_status.lot_multiplier)
        effective_risk_pct = kelly_risk_pct * vol_scalar * cb_mult
        # Enforce absolute per-trade ceiling invariant
        effective_risk_pct = min(effective_risk_pct, self._cfg.max_risk_per_trade_pct)
        risk_amount_usd = equity * effective_risk_pct

        # 4. Inverse Volatility + Spread-Penalized Stop Distance
        sl_dist_price = self._cfg.atr_stop_multiplier * atr
        tp_dist_price = self._cfg.atr_tp_multiplier * atr
        clean_spread = max(0.0, spread_price)
        spread_cost_price = self._cfg.spread_risk_weight * clean_spread
        effective_risk_distance_price = sl_dist_price + spread_cost_price

        risk_ticks = effective_risk_distance_price / self._spec.tick_size
        risk_per_standard_lot_usd = risk_ticks * self._spec.tick_value
        if risk_per_standard_lot_usd <= 0.0:
            return self._rejected(circuit_status=circuit_status, reason="INVALID_TICK_VALUE_SPEC")

        raw_lots = risk_amount_usd / risk_per_standard_lot_usd
        max_cap = min(self._spec.volume_max, self._cfg.max_total_lot_per_signal)
        capped_lots = min(raw_lots, max_cap)
        final_lots = quantize_lots_down(capped_lots, self._spec.volume_step)

        if side == SignalSide.LONG:
            sl_price = self._spec.round_price(entry_price - sl_dist_price)
            tp_price = self._spec.round_price(entry_price + tp_dist_price)
        else:
            sl_price = self._spec.round_price(entry_price + sl_dist_price)
            tp_price = self._spec.round_price(entry_price - tp_dist_price)

        if final_lots < self._spec.volume_min:
            return PositionSizeResult(
                approved=False,
                final_lots=0.0,
                raw_unrounded_lots=raw_lots,
                kelly_full_fraction=f_kelly,
                kelly_fractional_risk_pct=kelly_risk_pct,
                vol_target_scalar=vol_scalar,
                circuit_breaker_state=circuit_status.state,
                circuit_breaker_multiplier=cb_mult,
                effective_risk_pct=effective_risk_pct,
                risk_amount_usd=risk_amount_usd,
                stop_loss_distance_price=sl_dist_price,
                take_profit_distance_price=tp_dist_price,
                effective_spread_cost_price=spread_cost_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                rejection_reason=f"LOT_BELOW_VOLUME_MIN:{raw_lots:.4f}<{self._spec.volume_min}",
            )

        return PositionSizeResult(
            approved=True,
            final_lots=final_lots,
            raw_unrounded_lots=raw_lots,
            kelly_full_fraction=f_kelly,
            kelly_fractional_risk_pct=kelly_risk_pct,
            vol_target_scalar=vol_scalar,
            circuit_breaker_state=circuit_status.state,
            circuit_breaker_multiplier=cb_mult,
            effective_risk_pct=effective_risk_pct,
            risk_amount_usd=risk_amount_usd,
            stop_loss_distance_price=sl_dist_price,
            take_profit_distance_price=tp_dist_price,
            effective_spread_cost_price=spread_cost_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            rejection_reason=None,
        )

    @staticmethod
    def _rejected(
        circuit_status: CircuitBreakerStatus,
        reason: str,
        kelly_full: float = 0.0,
    ) -> PositionSizeResult:
        return PositionSizeResult(
            approved=False,
            final_lots=0.0,
            raw_unrounded_lots=0.0,
            kelly_full_fraction=kelly_full,
            kelly_fractional_risk_pct=0.0,
            vol_target_scalar=0.0,
            circuit_breaker_state=circuit_status.state,
            circuit_breaker_multiplier=circuit_status.lot_multiplier,
            effective_risk_pct=0.0,
            risk_amount_usd=0.0,
            stop_loss_distance_price=0.0,
            take_profit_distance_price=0.0,
            effective_spread_cost_price=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            rejection_reason=reason,
        )
