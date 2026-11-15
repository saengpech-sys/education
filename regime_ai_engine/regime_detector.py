"""
First-Principles Market Regime Detection Engine.

Enforces Principle #1:
Trade ONLY in statistically advantageous regimes (Persistent Trend or Stationary Mean-Reversion).
Immediately halt trading during Abnormal Volatility Shocks or Directionless Random-Walk Noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config_loader import RegimeAIConfig
from core.contracts import MarketRegime, MarketSnapshot


REGIME_TO_NUMERIC_CODE = {
    MarketRegime.TRENDING_BULL: 1.0,
    MarketRegime.TRENDING_BEAR: -1.0,
    MarketRegime.MEAN_REVERTING: 0.5,
    MarketRegime.DIRECTIONLESS_NOISE: 0.0,
    MarketRegime.ABNORMAL_VOLATILITY: -2.0,
}


class MarketRegimeDetector:
    """Classifies real-time snapshots and historical feature frames into MarketRegime states."""

    def __init__(self, config: RegimeAIConfig) -> None:
        self._cfg = config

    def classify_metrics(
        self,
        m5_hurst: float,
        h1_hurst: float,
        vol_zscore: float,
        vol_ratio: float,
        directional_efficiency: float,
        trend_alignment: float,
        m5_ret_12: float,
        h1_ema_spread_atr: float = 0.0,
        h4_ema_spread_atr: float = 0.0,
        h1_efficiency: float = 0.25,
        in_handoff_window: bool = False,
    ) -> MarketRegime:
        # 1. Hard volatility shock gate: protect equity curve from tail-risk turbulence
        if (
            vol_zscore > self._cfg.abnormal_vol_zscore_ceiling
            or vol_ratio > self._cfg.abnormal_vol_ratio_ceiling
        ):
            return MarketRegime.ABNORMAL_VOLATILITY

        # Dead illiquid chop gate: skip ultra-compressed aimless drift outside handoffs
        if not in_handoff_window and vol_ratio < 0.42 and directional_efficiency < 0.12:
            return MarketRegime.DIRECTIONLESS_NOISE

        composite_hurst = 0.45 * m5_hurst + 0.55 * h1_hurst

        # 2. Confirmed Macro Bear Regime (protects Long sleeves during multi-week bear trends)
        if h4_ema_spread_atr <= -0.20:
            return MarketRegime.TRENDING_BEAR

        # 3. Confirmed Macro Bull Breakout Regime (protects Short sleeves during bull surges)
        if h4_ema_spread_atr >= 0.65:
            return MarketRegime.TRENDING_BULL

        # 4. Institutional Bullion Handoff Windows or Stationary Anti-Persistent Ranges
        if in_handoff_window or composite_hurst <= self._cfg.hurst_mean_reverting_max:
            return MarketRegime.MEAN_REVERTING

        # 5. Mid-session Persistent Directional Trend
        if (
            composite_hurst >= self._cfg.hurst_trending_min
            and directional_efficiency >= self._cfg.min_directional_efficiency
        ):
            direction_score = 0.5 * trend_alignment + 0.5 * np.sign(m5_ret_12)
            if h4_ema_spread_atr >= 0.0 and direction_score >= 0.0:
                return MarketRegime.TRENDING_BULL
            if h4_ema_spread_atr < 0.0 and direction_score < 0.0:
                return MarketRegime.TRENDING_BEAR

        # 6. Random-Walk / Choppy Mid-Session Noise -> Pause Trading
        return MarketRegime.DIRECTIONLESS_NOISE

    def classify_snapshot(self, snapshot: MarketSnapshot) -> MarketRegime:
        h1_hurst = float(snapshot.features.get("h1_hurst", snapshot.hurst_exponent))
        trend_align = float(snapshot.features.get("mtf_trend_alignment", 0.0))
        ret_12 = float(snapshot.features.get("m5_ret_12", 0.0))
        h1_ema = float(snapshot.features.get("h1_ema_spread_atr", 0.0))
        h4_ema = float(snapshot.features.get("h4_ema_spread_atr", 0.0))
        h1_eff = float(snapshot.features.get("h1_efficiency", snapshot.directional_efficiency))
        in_win = (
            float(snapshot.features.get("m5_is_sge_win", 0.0)) > 0.5
            or float(snapshot.features.get("m5_is_comex_win", 0.0)) > 0.5
            or float(snapshot.features.get("m5_is_ldn_close_win", 0.0)) > 0.5
        )
        return self.classify_metrics(
            m5_hurst=snapshot.hurst_exponent,
            h1_hurst=h1_hurst,
            vol_zscore=snapshot.vol_zscore,
            vol_ratio=snapshot.vol_ratio,
            directional_efficiency=snapshot.directional_efficiency,
            trend_alignment=trend_align,
            m5_ret_12=ret_12,
            h1_ema_spread_atr=h1_ema,
            h4_ema_spread_atr=h4_ema,
            h1_efficiency=h1_eff,
            in_handoff_window=in_win,
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized/row-consistent regime classification over an aligned feature DataFrame."""
        n = len(df)
        m5_h = df["m5_hurst"].to_numpy(dtype=np.float64)
        h1_h = df["h1_hurst"].to_numpy(dtype=np.float64) if "h1_hurst" in df.columns else m5_h
        vol_z = df["m5_vol_zscore"].to_numpy(dtype=np.float64)
        vol_r = df["m5_vol_ratio"].to_numpy(dtype=np.float64)
        eff = df["m5_efficiency"].to_numpy(dtype=np.float64)
        align = df["mtf_trend_alignment"].to_numpy(dtype=np.float64)
        ret12 = df["m5_ret_12"].to_numpy(dtype=np.float64)
        h1_ema = (
            df["h1_ema_spread_atr"].to_numpy(dtype=np.float64)
            if "h1_ema_spread_atr" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        h4_ema = (
            df["h4_ema_spread_atr"].to_numpy(dtype=np.float64)
            if "h4_ema_spread_atr" in df.columns
            else np.zeros(n, dtype=np.float64)
        )
        h1_eff = (
            df["h1_efficiency"].to_numpy(dtype=np.float64)
            if "h1_efficiency" in df.columns
            else eff
        )
        if "m5_is_sge_win" in df.columns:
            in_win_arr = (
                (df["m5_is_sge_win"].to_numpy(dtype=np.float64) > 0.5)
                | (df["m5_is_comex_win"].to_numpy(dtype=np.float64) > 0.5)
                | (df["m5_is_ldn_close_win"].to_numpy(dtype=np.float64) > 0.5)
            )
        else:
            in_win_arr = np.zeros(n, dtype=bool)

        regimes = [
            self.classify_metrics(
                m5_hurst=float(m5_h[i]),
                h1_hurst=float(h1_h[i]),
                vol_zscore=float(vol_z[i]),
                vol_ratio=float(vol_r[i]),
                directional_efficiency=float(eff[i]),
                trend_alignment=float(align[i]),
                m5_ret_12=float(ret12[i]),
                h1_ema_spread_atr=float(h1_ema[i]),
                h4_ema_spread_atr=float(h4_ema[i]),
                h1_efficiency=float(h1_eff[i]),
                in_handoff_window=bool(in_win_arr[i]),
            )
            for i in range(n)
        ]
        return pd.Series(regimes, index=df.index, name="market_regime")
