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
    ) -> MarketRegime:
        # 1. Hard volatility shock gate: protect equity curve from tail-risk turbulence
        if (
            vol_zscore > self._cfg.abnormal_vol_zscore_ceiling
            or vol_ratio > self._cfg.abnormal_vol_ratio_ceiling
        ):
            return MarketRegime.ABNORMAL_VOLATILITY

        composite_hurst = 0.60 * m5_hurst + 0.40 * h1_hurst

        # 2. Persistent Trending Regime (H > hurst_trending_min & clean directional path)
        if (
            composite_hurst >= self._cfg.hurst_trending_min
            and directional_efficiency >= self._cfg.min_directional_efficiency
        ):
            direction_score = 0.5 * trend_alignment + 0.5 * np.sign(m5_ret_12)
            if direction_score >= 0.0:
                return MarketRegime.TRENDING_BULL
            return MarketRegime.TRENDING_BEAR

        # 3. Anti-Persistent Mean-Reverting Regime (H <= hurst_mean_reverting_max)
        if composite_hurst <= self._cfg.hurst_mean_reverting_max:
            return MarketRegime.MEAN_REVERTING

        # 4. Random-Walk / Choppy Noise (0.45 < H < 0.55 or low efficiency) -> Pause Trading
        return MarketRegime.DIRECTIONLESS_NOISE

    def classify_snapshot(self, snapshot: MarketSnapshot) -> MarketRegime:
        h1_hurst = float(snapshot.features.get("h1_hurst", snapshot.hurst_exponent))
        trend_align = float(snapshot.features.get("mtf_trend_alignment", 0.0))
        ret_12 = float(snapshot.features.get("m5_ret_12", 0.0))
        return self.classify_metrics(
            m5_hurst=snapshot.hurst_exponent,
            h1_hurst=h1_hurst,
            vol_zscore=snapshot.vol_zscore,
            vol_ratio=snapshot.vol_ratio,
            directional_efficiency=snapshot.directional_efficiency,
            trend_alignment=trend_align,
            m5_ret_12=ret_12,
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized/row-consistent regime classification over an aligned feature DataFrame."""
        m5_h = df["m5_hurst"].to_numpy(dtype=np.float64)
        h1_h = df["h1_hurst"].to_numpy(dtype=np.float64) if "h1_hurst" in df.columns else m5_h
        vol_z = df["m5_vol_zscore"].to_numpy(dtype=np.float64)
        vol_r = df["m5_vol_ratio"].to_numpy(dtype=np.float64)
        eff = df["m5_efficiency"].to_numpy(dtype=np.float64)
        align = df["mtf_trend_alignment"].to_numpy(dtype=np.float64)
        ret12 = df["m5_ret_12"].to_numpy(dtype=np.float64)

        regimes = [
            self.classify_metrics(
                m5_hurst=float(m5_h[i]),
                h1_hurst=float(h1_h[i]),
                vol_zscore=float(vol_z[i]),
                vol_ratio=float(vol_r[i]),
                directional_efficiency=float(eff[i]),
                trend_alignment=float(align[i]),
                m5_ret_12=float(ret12[i]),
            )
            for i in range(len(df))
        ]
        return pd.Series(regimes, index=df.index, name="market_regime")
