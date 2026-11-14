"""
Pre-Trade Microstructure & Market-Calendar Guards:
1. SpreadGuard: Rejects entries when Bid-Ask spread exceeds absolute point ceiling or ATR fraction.
2. MarketScheduleAndNewsGuard: Blocks entries during daily broker rollover (widened spreads/swaps),
   Friday evening weekend close buffer, and scheduled Tier-1 Macro News windows (FOMC, NFP, CPI).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

from core.config_loader import ExecutionMonitoringConfig
from core.contracts import SymbolSpec, TickQuote


@dataclass(frozen=True)
class ScheduledNewsEvent:
    timestamp_utc: datetime
    title: str
    impact: str = "HIGH"


def _parse_hhmm(hhmm: str) -> dt_time:
    parts = hhmm.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM time string, got {hhmm!r}")
    return dt_time(hour=int(parts[0]), minute=int(parts[1]))


class SpreadGuard:
    """Enforces absolute and volatility-relative spread ceilings."""

    def __init__(self, config: ExecutionMonitoringConfig, symbol_spec: SymbolSpec) -> None:
        self._cfg = config
        self._spec = symbol_spec

    def check(self, tick: TickQuote, atr: float) -> Tuple[bool, str]:
        spread_px = tick.spread_price
        spread_pts = self._spec.price_to_points(spread_px)
        if spread_pts > self._cfg.max_spread_points + 1e-9:
            return (
                False,
                f"SPREAD_POINTS_EXCEEDED:{spread_pts:.1f}>{self._cfg.max_spread_points:.1f}",
            )
        if atr > 0:
            ratio = spread_px / atr
            if ratio > self._cfg.max_spread_atr_ratio + 1e-9:
                return (
                    False,
                    f"SPREAD_ATR_RATIO_EXCEEDED:{ratio:.3f}>{self._cfg.max_spread_atr_ratio:.3f}",
                )
        return True, "OK"


class MarketScheduleAndNewsGuard:
    """Blocks execution during rollover liquidity droughts, weekends, and macro news releases."""

    def __init__(
        self,
        config: ExecutionMonitoringConfig,
        initial_news_events: Optional[Sequence[ScheduledNewsEvent]] = None,
    ) -> None:
        self._cfg = config
        self._rollover_start = _parse_hhmm(config.rollover_start_utc)
        self._rollover_end = _parse_hhmm(config.rollover_end_utc)
        self._news_events: List[ScheduledNewsEvent] = list(initial_news_events or [])

    def register_news_event(self, timestamp_utc: datetime, title: str, impact: str = "HIGH") -> None:
        ts = (
            timestamp_utc.astimezone(timezone.utc)
            if timestamp_utc.tzinfo
            else timestamp_utc.replace(tzinfo=timezone.utc)
        )
        self._news_events.append(ScheduledNewsEvent(timestamp_utc=ts, title=title, impact=impact.upper()))

    def _in_rollover_window(self, current_time: dt_time) -> bool:
        if self._rollover_start <= self._rollover_end:
            return self._rollover_start <= current_time <= self._rollover_end
        # Overnight wrap-around (e.g. 23:50 -> 00:15 UTC)
        return current_time >= self._rollover_start or current_time <= self._rollover_end

    def check(self, timestamp_utc: datetime) -> Tuple[bool, str]:
        ts = (
            timestamp_utc.astimezone(timezone.utc)
            if timestamp_utc.tzinfo
            else timestamp_utc.replace(tzinfo=timezone.utc)
        )

        weekday = ts.weekday()  # Monday=0 .. Friday=4, Saturday=5, Sunday=6
        if weekday in (5, 6):
            return False, "WEEKEND_MARKET_CLOSED"
        if weekday == 4 and ts.hour >= self._cfg.friday_stop_hour_utc:
            return False, f"FRIDAY_WEEKEND_BUFFER:hour={ts.hour}>={self._cfg.friday_stop_hour_utc}"

        t_only = ts.timetz().replace(tzinfo=None)
        if self._in_rollover_window(t_only):
            return (
                False,
                f"DAILY_ROLLOVER_WINDOW:{self._cfg.rollover_start_utc}-{self._cfg.rollover_end_utc}",
            )

        before_delta = timedelta(minutes=self._cfg.news_blackout_before_minutes)
        after_delta = timedelta(minutes=self._cfg.news_blackout_after_minutes)
        for ev in self._news_events:
            if ev.impact != "HIGH":
                continue
            if (ev.timestamp_utc - before_delta) <= ts <= (ev.timestamp_utc + after_delta):
                return False, f"HIGH_IMPACT_NEWS_BLACKOUT:{ev.title}"

        return True, "OK"
