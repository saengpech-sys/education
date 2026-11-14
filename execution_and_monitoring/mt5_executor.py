"""
Resilient MetaTrader 5 Order Execution Engine (24/5 VPS Ready).

Features:
- Automatic Terminal Reconnection with exponential backoff
- Pre-trade SpreadGuard & MarketScheduleAndNewsGuard verification
- Smart MT5 Retcode Retry Loop (retries transient Requotes/Timeouts, aborts fatal errors)
- Sequential TWAP / Iceberg Slice Runner with live price refresh and Adverse-Drift Abort
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, List, Optional

from core.config_loader import ExecutionMonitoringConfig
from core.contracts import (
    ChildOrderSlice,
    ExecutionPlan,
    MT5GatewayProtocol,
    MT5TradeRetcode,
    OrderExecutionReport,
    OrderSendRequest,
    SignalSide,
    SliceFillReport,
    SymbolSpec,
    TRANSIENT_MT5_RETCODES,
)
from execution_and_monitoring.time_and_spread_filters import (
    MarketScheduleAndNewsGuard,
    SpreadGuard,
)

logger = logging.getLogger(__name__)


class MT5ExecutionEngine:
    """Executes Direct and TWAP/Iceberg plans over MT5GatewayProtocol with full fault tolerance."""

    def __init__(
        self,
        gateway: MT5GatewayProtocol,
        config: ExecutionMonitoringConfig,
        symbol_spec: SymbolSpec,
        spread_guard: SpreadGuard,
        schedule_guard: MarketScheduleAndNewsGuard,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._gateway = gateway
        self._cfg = config
        self._spec = symbol_spec
        self._spread_guard = spread_guard
        self._schedule_guard = schedule_guard
        self._sleep = sleep_fn

    def ensure_connected(self) -> bool:
        """Verify live MT5 connection; auto-reconnect with exponential backoff if dropped."""
        if self._gateway.is_connected():
            return True

        for attempt in range(1, self._cfg.reconnect_max_attempts + 1):
            logger.warning(
                "MT5 connection lost; attempting reconnect %d/%d...",
                attempt,
                self._cfg.reconnect_max_attempts,
            )
            if self._gateway.connect() and self._gateway.is_connected():
                logger.info("MT5 reconnected on attempt %d.", attempt)
                return True
            delay = self._cfg.reconnect_base_delay_seconds * (2 ** (attempt - 1))
            self._sleep(delay)

        logger.error("Failed to reconnect to MT5 after %d attempts.", self._cfg.reconnect_max_attempts)
        return False

    def _send_slice_with_retry(
        self,
        plan: ExecutionPlan,
        child: ChildOrderSlice,
    ) -> SliceFillReport:
        retries_used = 0
        last_retcode = int(MT5TradeRetcode.ERROR)
        last_msg = "UNINITIALIZED"
        req_price = plan.reference_entry_price

        for attempt in range(self._cfg.max_order_retries + 1):
            if not self.ensure_connected():
                return SliceFillReport(
                    slice_index=child.slice_index,
                    requested_lots=child.lots,
                    filled_lots=0.0,
                    requested_price=req_price,
                    fill_price=0.0,
                    slippage_points=0.0,
                    ticket=0,
                    retcode=int(MT5TradeRetcode.CONNECTION),
                    retries_used=retries_used,
                    success=False,
                    message="MT5_DISCONNECTED_ABORT",
                )

            tick = self._gateway.get_tick(plan.symbol)
            req_price = tick.ask if plan.side == SignalSide.LONG else tick.bid

            # Adverse price drift check on child slices > 0
            if child.slice_index > 0:
                adverse_move = (
                    (req_price - plan.reference_entry_price)
                    if plan.side == SignalSide.LONG
                    else (plan.reference_entry_price - req_price)
                )
                if adverse_move > plan.max_adverse_drift_price + 1e-12:
                    return SliceFillReport(
                        slice_index=child.slice_index,
                        requested_lots=child.lots,
                        filled_lots=0.0,
                        requested_price=req_price,
                        fill_price=0.0,
                        slippage_points=0.0,
                        ticket=0,
                        retcode=int(MT5TradeRetcode.PRICE_CHANGED),
                        retries_used=retries_used,
                        success=False,
                        message=(
                            f"ADVERSE_DRIFT_ABORT:{adverse_move:.4f}>"
                            f"{plan.max_adverse_drift_price:.4f}"
                        ),
                    )

            order_req = OrderSendRequest(
                symbol=plan.symbol,
                side=plan.side,
                lots=child.lots,
                price=req_price,
                sl=plan.stop_loss_price,
                tp=plan.take_profit_price,
                deviation_points=plan.max_deviation_points,
                magic=self._cfg.magic_number,
                comment=f"AGY_{plan.parent_signal_id[:8]}_{child.slice_index + 1}of{child.total_slices}",
            )
            res = self._gateway.send_market_order(order_req)
            last_retcode = res.retcode
            last_msg = res.comment

            if res.retcode in (int(MT5TradeRetcode.DONE), int(MT5TradeRetcode.DONE_PARTIAL)):
                slip_px = (
                    (res.fill_price - req_price)
                    if plan.side == SignalSide.LONG
                    else (req_price - res.fill_price)
                )
                slip_pts = self._spec.price_to_points(slip_px)
                return SliceFillReport(
                    slice_index=child.slice_index,
                    requested_lots=child.lots,
                    filled_lots=res.filled_lots,
                    requested_price=req_price,
                    fill_price=res.fill_price,
                    slippage_points=slip_pts,
                    ticket=res.ticket,
                    retcode=res.retcode,
                    retries_used=retries_used,
                    success=True,
                    message=res.comment or "FILLED",
                )

            # Abort immediately on non-transient broker errors (e.g. NO_MONEY, MARKET_CLOSED)
            if res.retcode not in TRANSIENT_MT5_RETCODES:
                break

            if attempt < self._cfg.max_order_retries:
                retries_used += 1
                if res.retcode == int(MT5TradeRetcode.CONNECTION):
                    self._gateway.disconnect()
                backoff = self._cfg.retry_backoff_seconds * (2 ** attempt)
                self._sleep(backoff)

        return SliceFillReport(
            slice_index=child.slice_index,
            requested_lots=child.lots,
            filled_lots=0.0,
            requested_price=req_price,
            fill_price=0.0,
            slippage_points=0.0,
            ticket=0,
            retcode=last_retcode,
            retries_used=retries_used,
            success=False,
            message=last_msg,
        )

    def execute_plan(
        self,
        plan: ExecutionPlan,
        atr: float,
        decision_timestamp_utc: Optional[datetime] = None,
    ) -> OrderExecutionReport:
        """Validate live filters and execute all child slices of an ExecutionPlan."""
        if not self.ensure_connected():
            return OrderExecutionReport(
                parent_signal_id=plan.parent_signal_id,
                symbol=plan.symbol,
                side=plan.side,
                executed=False,
                requested_total_lots=plan.total_lots,
                filled_total_lots=0.0,
                vwap_fill_price=0.0,
                average_slippage_points=0.0,
                slice_reports=(),
                abort_reason="MT5_CONNECTION_UNAVAILABLE",
            )

        live_tick = self._gateway.get_tick(plan.symbol)
        check_ts = decision_timestamp_utc or live_tick.timestamp_utc

        sched_ok, sched_reason = self._schedule_guard.check(check_ts)
        if not sched_ok:
            return OrderExecutionReport(
                parent_signal_id=plan.parent_signal_id,
                symbol=plan.symbol,
                side=plan.side,
                executed=False,
                requested_total_lots=plan.total_lots,
                filled_total_lots=0.0,
                vwap_fill_price=0.0,
                average_slippage_points=0.0,
                slice_reports=(),
                abort_reason=sched_reason,
            )

        sprd_ok, sprd_reason = self._spread_guard.check(live_tick, atr=atr)
        if not sprd_ok:
            return OrderExecutionReport(
                parent_signal_id=plan.parent_signal_id,
                symbol=plan.symbol,
                side=plan.side,
                executed=False,
                requested_total_lots=plan.total_lots,
                filled_total_lots=0.0,
                vwap_fill_price=0.0,
                average_slippage_points=0.0,
                slice_reports=(),
                abort_reason=sprd_reason,
            )

        reports: List[SliceFillReport] = []
        abort_reason: Optional[str] = None

        for child in plan.slices:
            if child.delay_seconds > 0.0 and child.slice_index > 0:
                self._sleep(child.delay_seconds)

            rep = self._send_slice_with_retry(plan=plan, child=child)
            reports.append(rep)
            if not rep.success:
                abort_reason = rep.message
                break

        filled_slices = [r for r in reports if r.success and r.filled_lots > 0]
        filled_total = round(sum(r.filled_lots for r in filled_slices), 6)

        if filled_total > 0.0:
            vwap_px = sum(r.fill_price * r.filled_lots for r in filled_slices) / filled_total
            avg_slip = sum(r.slippage_points * r.filled_lots for r in filled_slices) / filled_total
            executed = True
        else:
            vwap_px = 0.0
            avg_slip = 0.0
            executed = False

        return OrderExecutionReport(
            parent_signal_id=plan.parent_signal_id,
            symbol=plan.symbol,
            side=plan.side,
            executed=executed,
            requested_total_lots=plan.total_lots,
            filled_total_lots=filled_total,
            vwap_fill_price=round(vwap_px, self._spec.digits) if executed else 0.0,
            average_slippage_points=avg_slip,
            slice_reports=tuple(reports),
            abort_reason=abort_reason,
        )
