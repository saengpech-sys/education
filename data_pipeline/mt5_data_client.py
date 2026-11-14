"""MetaTrader 5 historical OHLCV, tick data, and live gateway implementations."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.contracts import (
    AccountSnapshot,
    MT5GatewayProtocol,
    MT5TradeRetcode,
    OrderSendRequest,
    OrderSendResult,
    SignalSide,
    SymbolSpec,
    TickQuote,
    Timeframe,
)

logger = logging.getLogger(__name__)


class NativeMT5Gateway:
    """Production MetaTrader 5 terminal gateway wrapping the official MetaTrader5 package."""

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        terminal_path: str = "",
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._mt5 = None
        self._connected = False

    def _import_mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Native MetaTrader5 Python package is unavailable on this OS. "
                    "Run on a Windows VPS with MetaTrader 5 installed or use SimulatedMT5Gateway."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> bool:
        mt5 = self._import_mt5()
        init_kwargs: Dict[str, object] = {}
        if self._terminal_path:
            init_kwargs["path"] = self._terminal_path
        if self._login > 0:
            init_kwargs["login"] = self._login
        if self._password:
            init_kwargs["password"] = self._password
        if self._server:
            init_kwargs["server"] = self._server

        ok = bool(mt5.initialize(**init_kwargs)) if init_kwargs else bool(mt5.initialize())
        if not ok:
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            self._connected = False
            return False

        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        if not self._connected or self._mt5 is None:
            return False
        term_info = self._mt5.terminal_info()
        return term_info is not None and bool(getattr(term_info, "connected", False))

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        mt5 = self._import_mt5()
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol} in MT5 MarketWatch: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info returned None for {symbol}: {mt5.last_error()}")
        return SymbolSpec(
            symbol=symbol,
            digits=int(info.digits),
            point=float(info.point),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
        )

    def get_tick(self, symbol: str) -> TickQuote:
        mt5 = self._import_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed for {symbol}: {mt5.last_error()}")
        ts = datetime.fromtimestamp(float(tick.time_msc) / 1000.0, tz=timezone.utc)
        last_price = float(tick.last) if float(tick.last) > 0 else 0.5 * (float(tick.bid) + float(tick.ask))
        vol = float(getattr(tick, "volume_real", 0.0) or getattr(tick, "volume", 1.0))
        return TickQuote(
            symbol=symbol,
            timestamp_utc=ts,
            bid=float(tick.bid),
            ask=float(tick.ask),
            last=last_price,
            volume=max(vol, 1.0),
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        mt5 = self._import_mt5()
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")
        positions = mt5.positions_get()
        pos_count = len(positions) if positions is not None else 0
        return AccountSnapshot(
            timestamp_utc=datetime.now(timezone.utc),
            balance=float(acc.balance),
            equity=float(acc.equity),
            margin_free=float(acc.margin_free),
            open_positions_count=pos_count,
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        num_bars: int,
        as_of_utc: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch strictly CLOSED historical bars (pos=1 in live mode to exclude forming candle)."""
        mt5 = self._import_mt5()
        if as_of_utc is None:
            # Start from pos=1 so the currently forming (unclosed) bar at pos=0 is NEVER leaked
            rates = mt5.copy_rates_from_pos(symbol, timeframe.mt5_constant, 1, num_bars)
        else:
            cutoff = as_of_utc.astimezone(timezone.utc)
            rates = mt5.copy_rates_from(symbol, timeframe.mt5_constant, cutoff, num_bars + 2)

        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Empty OHLCV returned for {symbol} {timeframe.value}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["close_time"] = df["open_time"] + timeframe.delta
        if as_of_utc is not None:
            cutoff_ts = pd.Timestamp(as_of_utc).tz_convert("UTC") if pd.Timestamp(as_of_utc).tzinfo else pd.Timestamp(as_of_utc, tz="UTC")
            df = df.loc[df["close_time"] <= cutoff_ts].tail(num_bars)

        df = df.rename(columns={"real_volume": "real_volume", "tick_volume": "tick_volume"})
        cols = ["open_time", "close_time", "open", "high", "low", "close", "tick_volume", "spread"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[cols].sort_values("close_time").reset_index(drop=True)

    def fetch_ticks(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        mt5 = self._import_mt5()
        ticks = mt5.copy_ticks_range(
            symbol,
            start_utc.astimezone(timezone.utc),
            end_utc.astimezone(timezone.utc),
            mt5.COPY_TICKS_ALL,
        )
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(columns=["timestamp_utc", "bid", "ask", "last", "volume"])
        df = pd.DataFrame(ticks)
        df["timestamp_utc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        if "volume_real" in df.columns and (df["volume_real"] > 0).any():
            df["volume"] = df["volume_real"].astype(float)
        elif "volume" in df.columns:
            df["volume"] = df["volume"].astype(float).clip(lower=1.0)
        else:
            df["volume"] = 1.0
        df["last"] = np.where(df["last"] > 0, df["last"], 0.5 * (df["bid"] + df["ask"]))
        return df[["timestamp_utc", "bid", "ask", "last", "volume"]].sort_values("timestamp_utc").reset_index(drop=True)

    def send_market_order(self, request: OrderSendRequest) -> OrderSendResult:
        mt5 = self._import_mt5()
        order_type = mt5.ORDER_TYPE_BUY if request.side == SignalSide.LONG else mt5.ORDER_TYPE_SELL
        payload = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": float(request.lots),
            "type": order_type,
            "price": float(request.price),
            "sl": float(request.sl),
            "tp": float(request.tp),
            "deviation": int(request.deviation_points),
            "magic": int(request.magic),
            "comment": request.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(payload)
        if res is None:
            return OrderSendResult(
                retcode=int(MT5TradeRetcode.CONNECTION),
                ticket=0,
                filled_lots=0.0,
                fill_price=0.0,
                comment=f"order_send returned None: {mt5.last_error()}",
            )
        return OrderSendResult(
            retcode=int(res.retcode),
            ticket=int(getattr(res, "order", 0) or getattr(res, "deal", 0)),
            filled_lots=float(getattr(res, "volume", 0.0)),
            fill_price=float(getattr(res, "price", 0.0)),
            comment=str(getattr(res, "comment", "")),
        )


class SimulatedMT5Gateway:
    """
    Deterministic multi-regime market & broker simulator implementing MT5GatewayProtocol.
    Generates realistic regime-switching M5 bars (Trending, Mean-Reverting OU, Random Walk,
    and Volatility Shocks) and exact aggregated M15, H1, and H4 candles without look-ahead bias.
    """

    def __init__(
        self,
        symbol_spec: SymbolSpec,
        initial_equity: float = 100_000.0,
        total_m5_bars: int = 3600,
        seed: int = 42,
        base_price: float = 2000.0,
    ) -> None:
        self._spec = symbol_spec
        self._balance = initial_equity
        self._equity = initial_equity
        self._connected = True
        self._seed = seed
        self._next_ticket = 100_001
        self._open_positions = 0
        self._queued_retcodes: List[int] = []
        self._slippage_points: float = 1.0

        self._m5_df = self._generate_multi_regime_m5(
            total_bars=total_m5_bars,
            base_price=base_price,
            seed=seed,
        )
        self._htf_cache: Dict[Timeframe, pd.DataFrame] = {
            Timeframe.M5: self._m5_df,
            Timeframe.M15: self._resample_exact(self._m5_df, Timeframe.M15),
            Timeframe.H1: self._resample_exact(self._m5_df, Timeframe.H1),
            Timeframe.H4: self._resample_exact(self._m5_df, Timeframe.H4),
        }

    def _generate_multi_regime_m5(
        self,
        total_bars: int,
        base_price: float,
        seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        start_ts = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
        open_times = pd.date_range(start=start_ts, periods=total_bars, freq="5min")
        close_times = open_times + pd.Timedelta(minutes=5)

        closes = np.empty(total_bars, dtype=np.float64)
        opens = np.empty(total_bars, dtype=np.float64)
        highs = np.empty(total_bars, dtype=np.float64)
        lows = np.empty(total_bars, dtype=np.float64)
        volumes = np.empty(total_bars, dtype=np.float64)
        spreads = np.empty(total_bars, dtype=np.float64)

        price = base_price
        anchor = base_price
        momentum = 0.0
        block_len = 600

        for i in range(total_bars):
            regime_idx = (i // block_len) % 4
            prev_price = price

            if regime_idx == 0:
                # Persistent multi-day bullish trend with autocorrelated order-flow momentum
                noise = rng.normal(0.0, 0.55)
                momentum = 0.65 * momentum + 0.35 * (0.28 + noise)
                shock = momentum
                vol = rng.uniform(420, 820)
                sprd = rng.uniform(10, 16)
            elif regime_idx == 1:
                # Stationary Ornstein-Uhlenbeck mean-reverting range around equilibrium anchor
                if i % block_len == 0:
                    anchor = price
                    momentum = 0.0
                kappa = 0.24
                shock = -kappa * (price - anchor) + rng.normal(0.0, 0.58)
                vol = rng.uniform(260, 520)
                sprd = rng.uniform(10, 15)
            elif regime_idx == 2:
                # Persistent multi-day bearish trend with autocorrelated downward momentum
                noise = rng.normal(0.0, 0.55)
                momentum = 0.65 * momentum + 0.35 * (-0.26 + noise)
                shock = momentum
                vol = rng.uniform(420, 850)
                sprd = rng.uniform(10, 17)
            else:
                # Choppy directionless noise with intermittent abnormal volatility spikes
                is_spike = (i % block_len) > int(block_len * 0.75)
                sigma = 3.40 if is_spike else 0.90
                momentum = 0.0
                shock = rng.normal(0.0, sigma)
                vol = rng.uniform(950, 1950) if is_spike else rng.uniform(180, 360)
                sprd = rng.uniform(38, 65) if is_spike else rng.uniform(15, 24)

            price = max(100.0, prev_price + shock)
            intrabar_wick = abs(rng.normal(0.45, 0.18)) * (
                2.8 if (regime_idx == 3 and (i % block_len) > int(block_len * 0.75)) else 1.0
            )
            bar_high = max(prev_price, price) + intrabar_wick
            bar_low = min(prev_price, price) - intrabar_wick

            opens[i] = round(prev_price, self._spec.digits)
            closes[i] = round(price, self._spec.digits)
            highs[i] = round(bar_high, self._spec.digits)
            lows[i] = round(max(1.0, bar_low), self._spec.digits)
            volumes[i] = round(vol, 1)
            spreads[i] = round(sprd, 1)

        return pd.DataFrame(
            {
                "open_time": open_times,
                "close_time": close_times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "tick_volume": volumes,
                "spread": spreads,
            }
        )

    @staticmethod
    def _resample_exact(m5_df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
        indexed = m5_df.set_index("open_time")
        grouped = indexed.resample(tf.pandas_freq, label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "mean",
            }
        ).dropna()
        grouped = grouped.reset_index()
        grouped["close_time"] = grouped["open_time"] + tf.delta
        return grouped[
            ["open_time", "close_time", "open", "high", "low", "close", "tick_volume", "spread"]
        ].reset_index(drop=True)

    def queue_transient_retcodes(self, retcodes: List[int]) -> None:
        """Test helper to inject simulated broker requotes/timeouts before fill."""
        self._queued_retcodes.extend(retcodes)

    def set_connected(self, state: bool) -> None:
        self._connected = state

    def set_equity(self, equity: float, balance: Optional[float] = None) -> None:
        self._equity = equity
        if balance is not None:
            self._balance = balance

    def set_slippage_points(self, slippage_points: float) -> None:
        self._slippage_points = slippage_points

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol != self._spec.symbol:
            raise ValueError(f"Unknown symbol {symbol}, expected {self._spec.symbol}")
        return self._spec

    def get_tick(self, symbol: str, as_of_utc: Optional[datetime] = None) -> TickQuote:
        df = self._m5_df
        if as_of_utc is not None:
            ts = pd.Timestamp(as_of_utc)
            ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            sub = df.loc[df["close_time"] <= ts]
            row = sub.iloc[-1] if not sub.empty else df.iloc[0]
        else:
            row = df.iloc[-1]

        mid = float(row["close"])
        half_spread = 0.5 * float(row["spread"]) * self._spec.point
        return TickQuote(
            symbol=symbol,
            timestamp_utc=row["close_time"].to_pydatetime(),
            bid=round(mid - half_spread, self._spec.digits),
            ask=round(mid + half_spread, self._spec.digits),
            last=mid,
            volume=float(row["tick_volume"]),
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp_utc=self._m5_df["close_time"].iloc[-1].to_pydatetime(),
            balance=self._balance,
            equity=self._equity,
            margin_free=self._equity * 0.90,
            open_positions_count=self._open_positions,
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        num_bars: int,
        as_of_utc: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if symbol != self._spec.symbol:
            raise ValueError(f"Unknown symbol {symbol}")
        df = self._htf_cache[timeframe]
        if as_of_utc is not None:
            cutoff = pd.Timestamp(as_of_utc)
            cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
            # Strictly include only bars whose close_time <= cutoff (100% leak-free!)
            df = df.loc[df["close_time"] <= cutoff]
        return df.tail(num_bars).copy().reset_index(drop=True)

    def fetch_ticks(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        t0 = pd.Timestamp(start_utc)
        t0 = t0.tz_convert("UTC") if t0.tzinfo else t0.tz_localize("UTC")
        t1 = pd.Timestamp(end_utc)
        t1 = t1.tz_convert("UTC") if t1.tzinfo else t1.tz_localize("UTC")

        sub = self._m5_df.loc[(self._m5_df["close_time"] >= t0) & (self._m5_df["close_time"] <= t1)]
        if sub.empty:
            return pd.DataFrame(columns=["timestamp_utc", "bid", "ask", "last", "volume"])

        records = []
        for _, row in sub.iterrows():
            half_sprd = 0.5 * float(row["spread"]) * self._spec.point
            base_ts = row["open_time"]
            prices = [float(row["open"]), float(row["low"]), float(row["high"]), float(row["close"])]
            vol_quarter = float(row["tick_volume"]) / 4.0
            for k, px in enumerate(prices):
                records.append(
                    {
                        "timestamp_utc": base_ts + pd.Timedelta(seconds=75 * (k + 1) - 1),
                        "bid": round(px - half_sprd, self._spec.digits),
                        "ask": round(px + half_sprd, self._spec.digits),
                        "last": px,
                        "volume": vol_quarter,
                    }
                )
        return pd.DataFrame(records)

    def send_market_order(self, request: OrderSendRequest) -> OrderSendResult:
        if not self._connected:
            return OrderSendResult(
                retcode=int(MT5TradeRetcode.CONNECTION),
                ticket=0,
                filled_lots=0.0,
                fill_price=0.0,
                comment="Simulated terminal disconnected",
            )
        if self._queued_retcodes:
            rc = self._queued_retcodes.pop(0)
            return OrderSendResult(
                retcode=int(rc),
                ticket=0,
                filled_lots=0.0,
                fill_price=0.0,
                comment=f"Simulated retcode {rc}",
            )

        slip_price = self._slippage_points * self._spec.point
        fill_px = (
            request.price + slip_price
            if request.side == SignalSide.LONG
            else request.price - slip_price
        )
        ticket = self._next_ticket
        self._next_ticket += 1
        self._open_positions += 1
        return OrderSendResult(
            retcode=int(MT5TradeRetcode.DONE),
            ticket=ticket,
            filled_lots=request.lots,
            fill_price=round(fill_px, self._spec.digits),
            comment="Simulated IOC fill",
        )


class MultiTimeframeDataFetcher:
    """Pulls synchronized, strictly closed multi-timeframe OHLCV and tick streams."""

    def __init__(self, gateway: MT5GatewayProtocol) -> None:
        self._gateway = gateway

    def fetch_multi_timeframe_bars(
        self,
        symbol: str,
        base_tf: Timeframe,
        higher_tfs: tuple[Timeframe, ...],
        num_bars: int,
        as_of_utc: Optional[datetime] = None,
    ) -> Dict[Timeframe, pd.DataFrame]:
        result: Dict[Timeframe, pd.DataFrame] = {}
        all_tfs = (base_tf, *higher_tfs)
        for tf in all_tfs:
            bars_needed = num_bars if tf == base_tf else max(250, num_bars // max(1, tf.minutes // base_tf.minutes) + 150)
            df = self._gateway.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                num_bars=bars_needed,
                as_of_utc=as_of_utc,
            )
            if as_of_utc is not None:
                cutoff = pd.Timestamp(as_of_utc)
                cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
                if (df["close_time"] > cutoff).any():
                    raise AssertionError(f"Look-ahead violation detected in {tf.value} OHLCV fetch!")
            result[tf] = df
        return result
