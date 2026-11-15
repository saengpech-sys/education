"""
Strictly causal (zero look-ahead) Microstructure & Volatility Feature Engineering:
- Wilder's ATR
- Parkinson High-Low Range Volatility
- Rolling Z-Score (Price, Return, & Volatility Z-Scores)
- Multi-Scale Rolling Hurst Exponent (Trending vs Mean-Reverting vs Random Walk)
- Institutional Rolling Volume Profile (POC, VAH, VAL & ATR-Normalized Distance)
- Tick-Level Order Flow Imbalance (OFI) & Effective Spread Microstructure
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute strictly causal Wilder's Average True Range (ATR)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's causal exponential smoothing (alpha = 1 / period, adjust=False)
    atr = true_range.ewm(alpha=1.0 / float(period), min_periods=period, adjust=False).mean()
    return atr


def compute_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_bars: Optional[int] = None,
) -> pd.Series:
    """
    Compute strictly causal Parkinson extreme-value volatility estimator:
        sigma_P = sqrt( (1 / (4 * N * ln(2))) * sum_{i=0}^{N-1} (ln(H_{t-i} / L_{t-i}))^2 )
    """
    high = df["high"].astype(float).clip(lower=EPS)
    low = df["low"].astype(float).clip(lower=EPS)
    log_hl_sq = np.log(high / low) ** 2

    factor = 1.0 / (4.0 * math.log(2.0))
    mean_sq = log_hl_sq.rolling(window=window, min_periods=window).mean()
    parkinson = np.sqrt((factor * mean_sq).clip(lower=0.0))

    if annualization_bars is not None and annualization_bars > 0:
        parkinson = parkinson * math.sqrt(float(annualization_bars))
    return parkinson


def compute_rolling_zscore(series: pd.Series, window: int = 48) -> pd.Series:
    """Compute strictly backward-looking rolling Z-score over [t - window + 1, t]."""
    s = series.astype(float)
    roll_mean = s.rolling(window=window, min_periods=window).mean()
    roll_std = s.rolling(window=window, min_periods=window).std(ddof=1)
    z = (s - roll_mean) / roll_std.where(roll_std > EPS, np.nan)
    return z.fillna(0.0).where(roll_mean.notna(), np.nan)


def compute_directional_efficiency(close: pd.Series, window: int = 24) -> pd.Series:
    """
    Kaufman Efficiency Ratio (causal): |C_t - C_{t-W}| / sum_{k=0}^{W-1} |C_{t-k} - C_{t-k-1}|.
    Separates clean directional displacement (near 1.0) from choppy noise (near 0.0).
    """
    c = close.astype(float)
    net_change = (c - c.shift(window)).abs()
    path_length = c.diff().abs().rolling(window=window, min_periods=window).sum()
    er = net_change / path_length.where(path_length > EPS, np.nan)
    return er.clip(0.0, 1.0)


def compute_rolling_hurst_exponent(close: pd.Series, window: int = 96) -> pd.Series:
    """
    Vectorized strictly causal multi-scale variance-scaling Hurst Exponent over [t - window + 1, t].
      - H < 0.45 : Mean-Reverting (anti-persistent)
      - H ~ 0.50 : Geometric Brownian Motion / Directionless Random Walk
      - H > 0.55 : Trending (persistent memory)
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 32:
        return pd.Series(out, index=close.index, name="hurst")

    log_p = np.log(np.maximum(close.to_numpy(dtype=np.float64), EPS))
    windows = sliding_window_view(log_p, window_shape=window)  # shape: (n - window + 1, window)

    candidate_lags = [lag for lag in (2, 4, 8, 16, 24) if lag <= window // 3]
    lags = np.array(candidate_lags, dtype=np.float64)
    log_lags = np.log(lags)
    x_centered = log_lags - log_lags.mean()
    denom = float(np.sum(x_centered ** 2))

    log_rms = np.empty((windows.shape[0], len(candidate_lags)), dtype=np.float64)
    for idx, lag in enumerate(candidate_lags):
        diffs = windows[:, lag:] - windows[:, :-lag]
        rms_vals = np.sqrt(np.mean(diffs ** 2, axis=1))
        log_rms[:, idx] = np.log(np.maximum(rms_vals, EPS))

    y_mean = log_rms.mean(axis=1, keepdims=True)
    slopes = np.sum((log_rms - y_mean) * x_centered[None, :], axis=1) / max(denom, EPS)
    out[window - 1 :] = np.clip(slopes, 0.05, 0.95)
    return pd.Series(out, index=close.index, name="hurst")


def compute_rolling_volume_profile(
    df: pd.DataFrame,
    window: int = 96,
    num_bins: int = 24,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    """
    Strictly causal CME-style Rolling Volume Profile over [t - window + 1, t].
    Computes Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    """
    n = len(df)
    poc = np.full(n, np.nan, dtype=np.float64)
    vah = np.full(n, np.nan, dtype=np.float64)
    val = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return pd.DataFrame({"vp_poc": poc, "vp_vah": vah, "vp_val": val}, index=df.index)

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    typical = (high + low + close) / 3.0
    vol = np.maximum(df["tick_volume"].to_numpy(dtype=np.float64), 1.0)

    tp_wins = sliding_window_view(typical, window_shape=window)
    hi_wins = sliding_window_view(high, window_shape=window)
    lo_wins = sliding_window_view(low, window_shape=window)
    vol_wins = sliding_window_view(vol, window_shape=window)

    num_windows = tp_wins.shape[0]
    for w_idx in range(num_windows):
        t_pos = w_idx + window - 1
        p_min = float(np.min(lo_wins[w_idx]))
        p_max = float(np.max(hi_wins[w_idx]))
        if p_max - p_min <= EPS:
            poc[t_pos] = p_min
            vah[t_pos] = p_max
            val[t_pos] = p_min
            continue

        bin_width = (p_max - p_min) / float(num_bins)
        bin_indices = np.clip(
            ((tp_wins[w_idx] - p_min) / bin_width).astype(np.int64),
            0,
            num_bins - 1,
        )
        hist = np.bincount(bin_indices, weights=vol_wins[w_idx], minlength=num_bins)
        poc_bin = int(np.argmax(hist))
        total_vol = float(np.sum(hist))
        target_vol = value_area_pct * total_vol

        cum_vol = float(hist[poc_bin])
        left_bin = poc_bin
        right_bin = poc_bin

        while cum_vol < target_vol and (left_bin > 0 or right_bin < num_bins - 1):
            next_left_vol = float(hist[left_bin - 1]) if left_bin > 0 else -1.0
            next_right_vol = float(hist[right_bin + 1]) if right_bin < num_bins - 1 else -1.0
            if next_right_vol >= next_left_vol:
                right_bin += 1
                cum_vol += float(hist[right_bin])
            else:
                left_bin -= 1
                cum_vol += float(hist[left_bin])

        poc[t_pos] = p_min + (poc_bin + 0.5) * bin_width
        val[t_pos] = p_min + left_bin * bin_width
        vah[t_pos] = p_min + (right_bin + 1.0) * bin_width

    return pd.DataFrame({"vp_poc": poc, "vp_vah": vah, "vp_val": val}, index=df.index)


def compute_tick_microstructure_features(
    bars_df: pd.DataFrame,
    ticks_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate strictly causal tick-level Order Flow Imbalance (Lee-Ready / Tick Rule)
    and Effective Spread into each bar [open_time, close_time].
    """
    out = pd.DataFrame(index=bars_df.index)
    if ticks_df is None or ticks_df.empty:
        # Causal bar-level fallback microstructure proxies when historical ticks are omitted
        bar_range = (bars_df["high"] - bars_df["low"]).replace(0.0, np.nan)
        signed_body = (bars_df["close"] - bars_df["open"]) / bar_range
        out["tick_ofi"] = signed_body.fillna(0.0).rolling(window=6, min_periods=1).mean()
        out["effective_spread"] = bars_df["spread"].astype(float)
        return out

    tdf = ticks_df.sort_values("timestamp_utc").copy()
    mid = 0.5 * (tdf["bid"].astype(float) + tdf["ask"].astype(float))
    mid_diff = mid.diff().fillna(0.0)
    tick_sign = np.sign(mid_diff).replace(0.0, np.nan).ffill().fillna(0.0)
    tdf["signed_vol"] = tick_sign * tdf["volume"].astype(float)
    tdf["eff_spread"] = 2.0 * (tdf["last"].astype(float) - mid).abs()

    # Bin ticks strictly into closed bar intervals (open_time, close_time]
    bar_edges = pd.Index([bars_df["open_time"].iloc[0]]).append(pd.Index(bars_df["close_time"]))
    bin_codes = pd.cut(tdf["timestamp_utc"], bins=bar_edges, right=True, labels=False)
    tdf["bar_idx"] = bin_codes
    grouped = tdf.dropna(subset=["bar_idx"]).groupby("bar_idx")

    vol_sum = grouped["volume"].sum()
    signed_sum = grouped["signed_vol"].sum()
    ofi_series = (signed_sum / vol_sum.replace(0.0, np.nan)).reindex(range(len(bars_df))).fillna(0.0)
    eff_sprd = grouped["eff_spread"].mean().reindex(range(len(bars_df))).ffill().fillna(0.0)

    out["tick_ofi"] = ofi_series.to_numpy()
    out["effective_spread"] = eff_sprd.to_numpy()
    return out


def build_single_timeframe_features(
    df: pd.DataFrame,
    prefix: str,
    atr_period: int = 14,
    parkinson_window: int = 20,
    zscore_window: int = 48,
    hurst_window: int = 96,
    vp_window: int = 96,
    vp_bins: int = 24,
    value_area_pct: float = 0.70,
    annualization_bars: Optional[int] = None,
    ticks_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build complete leak-free feature frame for a single timeframe."""
    feat = pd.DataFrame(index=df.index)
    feat["open_time"] = df["open_time"]
    feat["close_time"] = df["close_time"]

    close = df["close"].astype(float)
    log_ret = np.log(close.clip(lower=EPS) / close.shift(1).clip(lower=EPS))

    atr = compute_atr(df, period=atr_period)
    parkinson = compute_parkinson_volatility(df, window=parkinson_window)
    realized_vol = log_ret.rolling(window=parkinson_window, min_periods=parkinson_window).std(ddof=1)
    if annualization_bars is not None and annualization_bars > 0:
        annual_vol = realized_vol * math.sqrt(float(annualization_bars))
    else:
        annual_vol = realized_vol

    high = df["high"].astype(float)
    low = df["low"].astype(float)

    price_zscore = compute_rolling_zscore(close, window=zscore_window)
    # Use full 24h cycle (288 M5 bars) for M5 volatility regime baseline so normal London/NY opens
    # are not misclassified as abnormal shocks relative to quiet Asian hours.
    vol_base_win = max(zscore_window, 288) if prefix == "m5" else zscore_window
    vol_mean = parkinson.rolling(window=vol_base_win, min_periods=zscore_window).mean()
    vol_std = parkinson.rolling(window=vol_base_win, min_periods=zscore_window).std(ddof=1)
    vol_zscore = ((parkinson - vol_mean) / vol_std.where(vol_std > EPS, np.nan)).fillna(0.0).where(
        vol_mean.notna(), np.nan
    )
    long_vol = parkinson.rolling(window=vol_base_win, min_periods=parkinson_window).median()
    vol_ratio = (parkinson / long_vol.where(long_vol > EPS, np.nan)).fillna(1.0)

    hurst = compute_rolling_hurst_exponent(close, window=hurst_window)
    efficiency = compute_directional_efficiency(close, window=min(24, zscore_window))
    vp_df = compute_rolling_volume_profile(
        df,
        window=vp_window,
        num_bins=vp_bins,
        value_area_pct=value_area_pct,
    )

    safe_atr = atr.where(atr > EPS, np.nan)
    va_width = (vp_df["vp_vah"] - vp_df["vp_val"]).where((vp_df["vp_vah"] - vp_df["vp_val"]) > EPS, np.nan)

    # Causal institutional EMA trend structure, Wilder RSI, and Donchian channel position
    ema_fast = close.ewm(span=20, min_periods=10, adjust=False).mean()
    ema_slow = close.ewm(span=55, min_periods=20, adjust=False).mean()
    ema_spread_atr = (ema_fast - ema_slow) / safe_atr
    dist_ema_fast_atr = (close - ema_fast) / safe_atr

    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = gain / loss.clip(lower=EPS)
    rsi_centered = (1.0 - (1.0 / (1.0 + rs))) - 0.5  # [-0.5, +0.5]

    donchian_hi = high.rolling(window=20, min_periods=14).max()
    donchian_lo = low.rolling(window=20, min_periods=14).min()
    donchian_span = (donchian_hi - donchian_lo).where((donchian_hi - donchian_lo) > EPS, np.nan)
    donchian_pos = ((close - donchian_lo) / donchian_span).fillna(0.5)

    feat[f"{prefix}_ret_1"] = log_ret
    feat[f"{prefix}_ret_4"] = np.log(close.clip(lower=EPS) / close.shift(4).clip(lower=EPS))
    feat[f"{prefix}_ret_12"] = np.log(close.clip(lower=EPS) / close.shift(12).clip(lower=EPS))
    feat[f"{prefix}_atr"] = atr
    feat[f"{prefix}_atr_pct"] = atr / close.clip(lower=EPS)
    feat[f"{prefix}_parkinson_vol"] = parkinson
    feat[f"{prefix}_annual_vol"] = annual_vol
    feat[f"{prefix}_zscore"] = price_zscore
    feat[f"{prefix}_vol_zscore"] = vol_zscore
    feat[f"{prefix}_vol_ratio"] = vol_ratio
    feat[f"{prefix}_hurst"] = hurst
    feat[f"{prefix}_efficiency"] = efficiency
    feat[f"{prefix}_ema_spread_atr"] = ema_spread_atr
    feat[f"{prefix}_dist_ema_fast_atr"] = dist_ema_fast_atr
    feat[f"{prefix}_rsi"] = rsi_centered
    feat[f"{prefix}_donchian_pos"] = donchian_pos
    feat[f"{prefix}_vp_poc"] = vp_df["vp_poc"]
    feat[f"{prefix}_vp_vah"] = vp_df["vp_vah"]
    feat[f"{prefix}_vp_val"] = vp_df["vp_val"]
    feat[f"{prefix}_vp_dist_poc_atr"] = (close - vp_df["vp_poc"]) / safe_atr
    feat[f"{prefix}_vp_pos_in_va"] = (close - vp_df["vp_val"]) / va_width

    if prefix == "m5":
        micro = compute_tick_microstructure_features(df, ticks_df=ticks_df)
        feat["m5_tick_ofi"] = micro["tick_ofi"]
        feat["m5_effective_spread"] = micro["effective_spread"]
        feat["m5_atr_24h"] = atr.rolling(window=288, min_periods=1).mean()
        feat["m5_ret_72"] = ((close / close.shift(72).clip(lower=EPS)) - 1.0).fillna(0.0)
        feat["m5_ret_144"] = ((close / close.shift(144).clip(lower=EPS)) - 1.0).fillna(0.0)

        # Causal 6-hour (72 M5 bars) prior liquidity pool high/low for institutional sweep detection
        prev_hi_72 = high.shift(1).rolling(window=72, min_periods=24).max()
        prev_lo_72 = low.shift(1).rolling(window=72, min_periods=24).min()
        bull_sweep = (low < prev_lo_72) & (close > prev_lo_72)
        bear_sweep = (high > prev_hi_72) & (close < prev_hi_72)
        sweep_sign = np.where(bull_sweep, 1.0, np.where(bear_sweep, -1.0, 0.0))
        bull_depth = ((prev_lo_72 - low) / safe_atr).clip(0.0, 3.0).fillna(0.0)
        bear_depth = ((high - prev_hi_72) / safe_atr).clip(0.0, 3.0).fillna(0.0)
        sweep_depth = np.where(bull_sweep, bull_depth, np.where(bear_sweep, -bear_depth, 0.0))

        feat["m5_sweep_6h"] = sweep_sign
        feat["m5_sweep_depth_atr"] = sweep_depth

        ts_utc = pd.to_datetime(df["close_time"], utc=True)
        dow = ts_utc.dt.day_of_week.to_numpy()
        hours = ts_utc.dt.hour.to_numpy()
        mins = ts_utc.dt.minute.to_numpy()
        hour_frac = hours.astype(float) + mins.astype(float) / 60.0
        no_fri_night = (dow <= 4) & (~((dow == 4) & (hours >= 19)))

        feat["m5_session_sin"] = np.sin(2.0 * math.pi * hour_frac / 24.0)
        feat["m5_session_cos"] = np.cos(2.0 * math.pi * hour_frac / 24.0)
        feat["m5_is_prime_session"] = ((hour_frac >= 7.0) & (hour_frac <= 20.5)).astype(float)
        # Institutional Bullion Market-Clock Handoff Windows (UTC)
        feat["m5_is_sge_win"] = (
            no_fri_night & (((hours == 20) & (mins >= 45)) | ((hours == 21) & (mins <= 30)))
        ).astype(float)
        feat["m5_is_comex_win"] = (no_fri_night & (hours >= 12) & (hours <= 15)).astype(float)
        feat["m5_is_ldn_close_win"] = (
            no_fri_night & (((hours == 16) & (mins >= 30)) | ((hours >= 17) & (hours <= 19)))
        ).astype(float)

    if prefix == "h4":
        # Causal ~5-day (30 H4 bars) and ~20-day (120 H4 bars) macro regime return filters
        feat["h4_ret_30"] = ((close / close.shift(30).clip(lower=EPS)) - 1.0).fillna(0.0)
        feat["h4_ret_120"] = ((close / close.shift(120).clip(lower=EPS)) - 1.0).fillna(0.0)

    return feat
