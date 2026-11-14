"""
Liquidity-Aware Order Scaling & TWAP / Iceberg Execution Planner.

Prevents market impact and slippage as compounded equity scales into 8-9 figure AUM:
1. Caps aggregate order exposure by `max_total_lot_per_signal` and optional market tick-liquidity cap.
2. Automatically decomposes large parent orders (`total_lots > max_lot_per_order`) into
   TWAP / Iceberg child slices where every slice <= `max_lot_per_order` and exact lot sum is conserved.
3. Attaches `max_adverse_drift_price` abort threshold to halt remaining slices if price runs away.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from core.config_loader import RiskAndScalingConfig
from core.contracts import (
    ChildOrderSlice,
    ExecutionPlan,
    SignalSide,
    SymbolSpec,
)
from risk_and_scaling_engine.position_sizer import quantize_lots_down


class LiquidityAwareExecutionPlanner:
    """Plans direct vs TWAP/Iceberg sliced execution schedules."""

    def __init__(
        self,
        risk_config: RiskAndScalingConfig,
        symbol_spec: SymbolSpec,
        max_deviation_points: int = 15,
        seed: int = 42,
    ) -> None:
        self._cfg = risk_config
        self._spec = symbol_spec
        self._max_dev = max_deviation_points
        self._rng = np.random.default_rng(seed)

    def build_execution_plan(
        self,
        parent_signal_id: str,
        side: SignalSide,
        requested_lots: float,
        reference_entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        atr: float,
        bar_tick_volume: Optional[float] = None,
    ) -> ExecutionPlan:
        """Create a liquidity-capped Direct or TWAP/Iceberg ExecutionPlan."""
        step = self._spec.volume_step
        min_lot = self._spec.volume_min
        max_slice_lot = quantize_lots_down(
            min(self._cfg.max_lot_per_order, self._spec.volume_max),
            step,
        )

        # Apply aggregate parent ceiling + optional tick-liquidity participation ceiling
        effective_ceiling = self._cfg.max_total_lot_per_signal
        if bar_tick_volume is not None and bar_tick_volume > 0:
            liquidity_cap = bar_tick_volume * self._cfg.max_liquidity_participation_rate
            # Ensure normal institutional liquidity floor of at least max_slice_lot when active
            effective_ceiling = min(effective_ceiling, max(max_slice_lot, liquidity_cap))

        total_lots = quantize_lots_down(min(requested_lots, effective_ceiling), step)
        if total_lots < min_lot:
            raise ValueError(f"Cannot build ExecutionPlan for {total_lots} lots (< {min_lot}).")

        max_drift_price = self._cfg.max_adverse_drift_atr * max(atr, self._spec.point)

        # Direct single-slice execution when within max_lot_per_order
        if total_lots <= max_slice_lot + 1e-12:
            single = ChildOrderSlice(slice_index=0, total_slices=1, lots=total_lots, delay_seconds=0.0)
            return ExecutionPlan(
                parent_signal_id=parent_signal_id,
                symbol=self._spec.symbol,
                side=side,
                total_lots=total_lots,
                execution_style="DIRECT",
                slices=(single,),
                reference_entry_price=reference_entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                max_deviation_points=self._max_dev,
                max_adverse_drift_price=max_drift_price,
            )

        # TWAP / Iceberg multi-slice decomposition in exact integer step units
        total_units = int(round(total_lots / step))
        max_slice_units = int(round(max_slice_lot / step))
        min_slice_units = max(1, int(round(min_lot / step)))

        # Choose slice count with subtle headroom so iceberg jitter can randomize slice footprints
        min_slices_needed = int(math.ceil(total_units / max_slice_units))
        target_slices = min_slices_needed
        if (
            self._cfg.iceberg_jitter_pct > 0.0
            and total_units >= (min_slices_needed + 1) * min_slice_units
            and total_units / min_slices_needed > 0.92 * max_slice_units
        ):
            target_slices = min_slices_needed + 1

        base_units = total_units // target_slices
        remainder = total_units % target_slices
        slice_units: List[int] = [
            base_units + (1 if idx < remainder else 0) for idx in range(target_slices)
        ]

        # Apply zero-sum Iceberg jitter across slice pairs while strictly respecting [min, max] bounds
        if self._cfg.iceberg_jitter_pct > 0.0 and target_slices >= 2:
            max_jitter_u = max(1, int(round(base_units * self._cfg.iceberg_jitter_pct)))
            for i in range(0, target_slices - 1, 2):
                j = i + 1
                delta = int(self._rng.integers(-max_jitter_u, max_jitter_u + 1))
                cand_i = slice_units[i] + delta
                cand_j = slice_units[j] - delta
                if (
                    min_slice_units <= cand_i <= max_slice_units
                    and min_slice_units <= cand_j <= max_slice_units
                ):
                    slice_units[i] = cand_i
                    slice_units[j] = cand_j

        decimals = max(0, int(round(-math.log10(step))) if step < 1.0 else 0)
        slices = tuple(
            ChildOrderSlice(
                slice_index=idx,
                total_slices=target_slices,
                lots=round(u * step, max(decimals, 4)),
                delay_seconds=0.0 if idx == 0 else float(self._cfg.twap_slice_interval_seconds),
            )
            for idx, u in enumerate(slice_units)
        )

        return ExecutionPlan(
            parent_signal_id=parent_signal_id,
            symbol=self._spec.symbol,
            side=side,
            total_lots=total_lots,
            execution_style="TWAP_ICEBERG",
            slices=slices,
            reference_entry_price=reference_entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            max_deviation_points=self._max_dev,
            max_adverse_drift_price=max_drift_price,
        )
