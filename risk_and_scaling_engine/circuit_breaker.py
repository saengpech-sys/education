"""
Equity Curve Control & Hard Drawdown Circuit Breaker.

Operational Rules:
1. Daily Drawdown > 1.5% OR Max Trailing Drawdown > 4.0%:
   Immediately trips into COOLDOWN_HALT (`allow_new_trades = False`, `lot_multiplier = 0.0`).
2. After `cooldown_minutes` elapses AND market regime/volatility normalize:
   Transitions into RECOVERY_HALF_RISK (`allow_new_trades = True`, `lot_multiplier = 0.50`)
   cutting position sizing in half until equity and market stability recover.
3. Trailing Drawdown >= `hard_lock_drawdown_limit_pct` (e.g. 6.0%):
   Locks system into HARD_LOCK (`allow_new_trades = False`) until explicit manual review.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from core.config_loader import RiskAndScalingConfig
from core.contracts import CircuitBreakerState, CircuitBreakerStatus, MarketRegime


class EquityCircuitBreaker:
    """Stateful Equity Curve Controller enforcing Daily & Trailing Drawdown protection."""

    def __init__(self, config: RiskAndScalingConfig, initial_equity: float) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")
        self._cfg = config
        self._peak_equity = float(initial_equity)
        self._daily_reference_equity = float(initial_equity)
        self._current_day: Optional[date] = None
        self._state = CircuitBreakerState.NORMAL
        self._cooldown_until_utc: Optional[datetime] = None
        self._recovery_anchor_equity: float = float(initial_equity)
        self._reason: str = "OK"

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def manual_reset(self, new_reference_equity: float, timestamp_utc: datetime) -> None:
        """Explicit administrative reset after reviewing a HARD_LOCK event."""
        ts = timestamp_utc.astimezone(timezone.utc) if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
        self._peak_equity = float(new_reference_equity)
        self._daily_reference_equity = float(new_reference_equity)
        self._recovery_anchor_equity = float(new_reference_equity)
        self._current_day = ts.date()
        self._state = CircuitBreakerState.NORMAL
        self._cooldown_until_utc = None
        self._reason = "MANUAL_RESET"

    def update(
        self,
        current_equity: float,
        timestamp_utc: datetime,
        regime: MarketRegime = MarketRegime.MEAN_REVERTING,
        vol_zscore: float = 0.0,
    ) -> CircuitBreakerStatus:
        """Evaluate current equity & market state and return updated CircuitBreakerStatus."""
        if current_equity <= 0:
            self._state = CircuitBreakerState.HARD_LOCK
            self._reason = "DEPLETED_EQUITY"
            return CircuitBreakerStatus(
                state=self._state,
                allow_new_trades=False,
                lot_multiplier=0.0,
                current_equity=current_equity,
                peak_equity=self._peak_equity,
                daily_reference_equity=self._daily_reference_equity,
                daily_drawdown_pct=1.0,
                trailing_drawdown_pct=1.0,
                cooldown_until_utc=self._cooldown_until_utc,
                reason=self._reason,
            )

        ts = timestamp_utc.astimezone(timezone.utc) if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
        utc_day = ts.date()

        # Roll daily reference equity at start of each new UTC trading day
        if self._current_day is None or utc_day > self._current_day:
            self._current_day = utc_day
            self._daily_reference_equity = float(current_equity)

        # Update high-water mark for geometric compounding
        if current_equity > self._peak_equity:
            self._peak_equity = float(current_equity)
        if current_equity > self._daily_reference_equity:
            self._daily_reference_equity = float(current_equity)

        daily_dd = max(0.0, (self._daily_reference_equity - current_equity) / self._daily_reference_equity)
        trailing_dd = max(0.0, (self._peak_equity - current_equity) / self._peak_equity)

        # 1. Check catastrophic hard lock first
        if self._state == CircuitBreakerState.HARD_LOCK or trailing_dd >= self._cfg.hard_lock_drawdown_limit_pct:
            self._state = CircuitBreakerState.HARD_LOCK
            self._reason = (
                f"HARD_LOCK:trailing_dd={trailing_dd:.4%}>={self._cfg.hard_lock_drawdown_limit_pct:.4%}"
            )
            return self._snapshot(current_equity, daily_dd, trailing_dd)

        market_is_normal = regime.is_tradable and (vol_zscore <= self._cfg.recovery_vol_zscore_max)

        # 2. State-specific transitions
        if self._state == CircuitBreakerState.NORMAL:
            if daily_dd > self._cfg.daily_drawdown_limit_pct:
                self._trip_cooldown(ts, f"DAILY_DD_BREACH:{daily_dd:.4%}>{self._cfg.daily_drawdown_limit_pct:.4%}")
            elif trailing_dd > self._cfg.max_trailing_drawdown_limit_pct:
                self._trip_cooldown(
                    ts,
                    f"TRAILING_DD_BREACH:{trailing_dd:.4%}>{self._cfg.max_trailing_drawdown_limit_pct:.4%}",
                )
            else:
                self._reason = "NORMAL_OPERATIONS"

        elif self._state == CircuitBreakerState.COOLDOWN_HALT:
            timer_elapsed = self._cooldown_until_utc is not None and ts >= self._cooldown_until_utc
            if timer_elapsed and market_is_normal:
                self._state = CircuitBreakerState.RECOVERY_HALF_RISK
                self._recovery_anchor_equity = float(current_equity)
                self._cooldown_until_utc = None
                self._reason = "RECOVERY_HALF_LOT_PROBATION"

        elif self._state == CircuitBreakerState.RECOVERY_HALF_RISK:
            # Re-trip cooldown if equity drops another daily limit from recovery anchor
            incremental_loss = max(
                0.0,
                (self._recovery_anchor_equity - current_equity) / max(self._recovery_anchor_equity, 1e-9),
            )
            if incremental_loss > self._cfg.daily_drawdown_limit_pct:
                self._trip_cooldown(
                    ts,
                    f"RECOVERY_RELAPSE_DD:{incremental_loss:.4%}>{self._cfg.daily_drawdown_limit_pct:.4%}",
                )
            else:
                # Restore full 1.0x sizing when drawdown heals within recovery buffer and market is calm
                healed_daily = daily_dd <= self._cfg.recovery_drawdown_buffer_pct
                healed_trailing = trailing_dd <= (
                    self._cfg.max_trailing_drawdown_limit_pct - self._cfg.recovery_drawdown_buffer_pct
                )
                if market_is_normal and healed_daily and healed_trailing:
                    self._state = CircuitBreakerState.NORMAL
                    self._reason = "EQUITY_AND_REGIME_RECOVERED"

        return self._snapshot(current_equity, daily_dd, trailing_dd)

    def _trip_cooldown(self, ts: datetime, reason: str) -> None:
        self._state = CircuitBreakerState.COOLDOWN_HALT
        self._cooldown_until_utc = ts + timedelta(minutes=self._cfg.cooldown_minutes)
        self._reason = reason

    def _snapshot(self, current_equity: float, daily_dd: float, trailing_dd: float) -> CircuitBreakerStatus:
        if self._state == CircuitBreakerState.NORMAL:
            allow = True
            mult = 1.0
        elif self._state == CircuitBreakerState.RECOVERY_HALF_RISK:
            allow = True
            mult = float(self._cfg.recovery_lot_multiplier)
        else:
            allow = False
            mult = 0.0

        return CircuitBreakerStatus(
            state=self._state,
            allow_new_trades=allow,
            lot_multiplier=mult,
            current_equity=float(current_equity),
            peak_equity=self._peak_equity,
            daily_reference_equity=self._daily_reference_equity,
            daily_drawdown_pct=daily_dd,
            trailing_drawdown_pct=trailing_dd,
            cooldown_until_utc=self._cooldown_until_utc,
            reason=self._reason,
        )
