"""Strict type contracts, enums, MT5 constants, and gateway protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd


class Timeframe(str, Enum):
    """Supported MetaTrader 5 timeframes with exact bar duration semantics."""

    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"

    @property
    def minutes(self) -> int:
        mapping = {
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
        }
        return mapping[self]

    @property
    def delta(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    @property
    def pandas_freq(self) -> str:
        mapping = {
            Timeframe.M5: "5min",
            Timeframe.M15: "15min",
            Timeframe.H1: "1h",
            Timeframe.H4: "4h",
        }
        return mapping[self]

    @property
    def mt5_constant(self) -> int:
        """Exact MetaTrader 5 integer timeframe constants."""
        mapping = {
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 16385,
            Timeframe.H4: 16388,
        }
        return mapping[self]


class MarketRegime(str, Enum):
    """First-principles market state classification."""

    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    MEAN_REVERTING = "MEAN_REVERTING"
    DIRECTIONLESS_NOISE = "DIRECTIONLESS_NOISE"
    ABNORMAL_VOLATILITY = "ABNORMAL_VOLATILITY"

    @property
    def is_tradable(self) -> bool:
        return self in {
            MarketRegime.TRENDING_BULL,
            MarketRegime.TRENDING_BEAR,
            MarketRegime.MEAN_REVERTING,
        }


class SignalSide(IntEnum):
    """Directional signal side."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class CircuitBreakerState(str, Enum):
    """Equity Curve Control operational states."""

    NORMAL = "NORMAL"
    COOLDOWN_HALT = "COOLDOWN_HALT"
    RECOVERY_HALF_RISK = "RECOVERY_HALF_RISK"
    HARD_LOCK = "HARD_LOCK"


class MT5TradeRetcode(IntEnum):
    """Standard MetaTrader 5 trade server return codes."""

    REQUOTE = 10004
    REJECT = 10006
    CANCEL = 10007
    PLACED = 10008
    DONE = 10009
    DONE_PARTIAL = 10010
    ERROR = 10011
    TIMEOUT = 10012
    INVALID = 10013
    INVALID_VOLUME = 10014
    INVALID_PRICE = 10015
    INVALID_STOPS = 10016
    TRADE_DISABLED = 10017
    MARKET_CLOSED = 10018
    NO_MONEY = 10019
    PRICE_CHANGED = 10020
    PRICE_OFF = 10021
    TOO_MANY_REQUESTS = 10024
    CONNECTION = 10031


TRANSIENT_MT5_RETCODES = frozenset(
    {
        MT5TradeRetcode.REQUOTE,
        MT5TradeRetcode.TIMEOUT,
        MT5TradeRetcode.PRICE_CHANGED,
        MT5TradeRetcode.PRICE_OFF,
        MT5TradeRetcode.TOO_MANY_REQUESTS,
        MT5TradeRetcode.CONNECTION,
    }
)


@dataclass(frozen=True)
class SymbolSpec:
    """Broker instrument specification for risk and order normalization."""

    symbol: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float

    def price_to_points(self, price_distance: float) -> float:
        if self.point <= 0:
            raise ValueError("Symbol point must be strictly positive.")
        return price_distance / self.point

    def round_price(self, price: float) -> float:
        return round(price, self.digits)


@dataclass(frozen=True)
class TickQuote:
    """Real-time Level-1 tick quote from MT5."""

    symbol: str
    timestamp_utc: datetime
    bid: float
    ask: float
    last: float
    volume: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_price(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass(frozen=True)
class AccountSnapshot:
    """Real-time MT5 account equity and margin snapshot."""

    timestamp_utc: datetime
    balance: float
    equity: float
    margin_free: float
    open_positions_count: int = 0


@dataclass(frozen=True)
class MarketSnapshot:
    """Aggregated leak-free market & microstructure snapshot at bar close."""

    symbol: str
    timestamp_utc: datetime
    bid: float
    ask: float
    close: float
    atr: float
    parkinson_vol: float
    realized_annual_vol: float
    hurst_exponent: float
    zscore: float
    vol_zscore: float
    vol_ratio: float
    directional_efficiency: float
    poc_price: float
    vah_price: float
    val_price: float
    bar_tick_volume: float
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def spread_price(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass(frozen=True)
class AISignalDecision:
    """Output of Market Regime Detector + Primary Model + Secondary Meta-Model."""

    signal_id: str
    timestamp_utc: datetime
    symbol: str
    regime: MarketRegime
    regime_tradable: bool
    primary_side: SignalSide
    primary_probability: float
    meta_win_probability: float
    confidence_threshold: float
    reward_to_risk_ratio: float
    approved: bool
    rejection_reason: Optional[str]
    inference_latency_ms: float
    feature_snapshot: Dict[str, float]


@dataclass(frozen=True)
class CircuitBreakerStatus:
    """Current Equity Curve Control state and drawdown telemetry."""

    state: CircuitBreakerState
    allow_new_trades: bool
    lot_multiplier: float
    current_equity: float
    peak_equity: float
    daily_reference_equity: float
    daily_drawdown_pct: float
    trailing_drawdown_pct: float
    cooldown_until_utc: Optional[datetime]
    reason: str


@dataclass(frozen=True)
class PositionSizeResult:
    """Output of Inverse-Volatility + Dynamic Fractional Kelly position sizer."""

    approved: bool
    final_lots: float
    raw_unrounded_lots: float
    kelly_full_fraction: float
    kelly_fractional_risk_pct: float
    vol_target_scalar: float
    circuit_breaker_state: CircuitBreakerState
    circuit_breaker_multiplier: float
    effective_risk_pct: float
    risk_amount_usd: float
    stop_loss_distance_price: float
    take_profit_distance_price: float
    effective_spread_cost_price: float
    stop_loss_price: float
    take_profit_price: float
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class ChildOrderSlice:
    """Single child order slice within a TWAP / Iceberg execution schedule."""

    slice_index: int
    total_slices: int
    lots: float
    delay_seconds: float


@dataclass(frozen=True)
class ExecutionPlan:
    """Liquidity-aware parent execution schedule."""

    parent_signal_id: str
    symbol: str
    side: SignalSide
    total_lots: float
    execution_style: str
    slices: Tuple[ChildOrderSlice, ...]
    reference_entry_price: float
    stop_loss_price: float
    take_profit_price: float
    max_deviation_points: int
    max_adverse_drift_price: float


@dataclass(frozen=True)
class OrderSendRequest:
    """Normalized MT5 market order request."""

    symbol: str
    side: SignalSide
    lots: float
    price: float
    sl: float
    tp: float
    deviation_points: int
    magic: int
    comment: str


@dataclass(frozen=True)
class OrderSendResult:
    """Normalized MT5 order_send response."""

    retcode: int
    ticket: int
    filled_lots: float
    fill_price: float
    comment: str


@dataclass(frozen=True)
class SliceFillReport:
    """Execution telemetry for an individual TWAP/Iceberg child slice."""

    slice_index: int
    requested_lots: float
    filled_lots: float
    requested_price: float
    fill_price: float
    slippage_points: float
    ticket: int
    retcode: int
    retries_used: int
    success: bool
    message: str


@dataclass(frozen=True)
class OrderExecutionReport:
    """Aggregate execution outcome across all slices of an ExecutionPlan."""

    parent_signal_id: str
    symbol: str
    side: SignalSide
    executed: bool
    requested_total_lots: float
    filled_total_lots: float
    vwap_fill_price: float
    average_slippage_points: float
    slice_reports: Tuple[SliceFillReport, ...]
    abort_reason: Optional[str] = None


class MT5GatewayProtocol(Protocol):
    """Protocol abstracting Windows MetaTrader5 C-API and deterministic simulators."""

    def connect(self) -> bool:
        ...

    def disconnect(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        ...

    def get_tick(self, symbol: str) -> TickQuote:
        ...

    def get_account_snapshot(self) -> AccountSnapshot:
        ...

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        num_bars: int,
        as_of_utc: Optional[datetime] = None,
    ) -> pd.DataFrame:
        ...

    def fetch_ticks(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        ...

    def send_market_order(self, request: OrderSendRequest) -> OrderSendResult:
        ...
