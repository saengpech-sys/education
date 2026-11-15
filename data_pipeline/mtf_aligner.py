"""
Multi-Timeframe (M5, M15, H1, H4) Feature Aligner with 100% Zero Look-Ahead Guarantee.

CRITICAL INVARIANT:
Higher-timeframe candles (M15, H1, H4) are merged onto M5 decision timestamps
strictly using `close_time` with backward `merge_asof` (`htf.close_time <= m5.close_time`).
An H1 bar opening at 10:00 closes at 11:00 and is NEVER visible to M5 bars at 10:05..10:55.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config_loader import DataPipelineConfig
from core.contracts import MarketSnapshot, SymbolSpec, TickQuote, Timeframe
from data_pipeline.feature_engineering import build_single_timeframe_features

PRIMARY_FEATURE_COLUMNS: Tuple[str, ...] = (
    # Base M5 features
    "m5_ret_1",
    "m5_ret_4",
    "m5_ret_12",
    "m5_ret_72",
    "m5_ret_144",
    "m5_atr_pct",
    "m5_parkinson_vol",
    "m5_annual_vol",
    "m5_zscore",
    "m5_vol_zscore",
    "m5_vol_ratio",
    "m5_hurst",
    "m5_efficiency",
    "m5_ema_spread_atr",
    "m5_dist_ema_fast_atr",
    "m5_rsi",
    "m5_donchian_pos",
    "m5_vp_dist_poc_atr",
    "m5_vp_pos_in_va",
    "m5_tick_ofi",
    "m5_sweep_6h",
    "m5_sweep_depth_atr",
    "m5_session_sin",
    "m5_session_cos",
    "m5_is_prime_session",
    "m5_is_sge_win",
    "m5_is_comex_win",
    "m5_is_ldn_close_win",
    # M15 context features
    "m15_ret_1",
    "m15_ret_4",
    "m15_zscore",
    "m15_hurst",
    "m15_efficiency",
    "m15_ema_spread_atr",
    "m15_rsi",
    "m15_donchian_pos",
    "m15_vp_dist_poc_atr",
    # H1 structural features
    "h1_ret_1",
    "h1_ret_4",
    "h1_zscore",
    "h1_hurst",
    "h1_efficiency",
    "h1_ema_spread_atr",
    "h1_dist_ema_fast_atr",
    "h1_rsi",
    "h1_vp_dist_poc_atr",
    # H4 macro-regime features
    "h4_ret_1",
    "h4_ret_30",
    "h4_ret_120",
    "h4_zscore",
    "h4_hurst",
    "h4_efficiency",
    "h4_ema_spread_atr",
    "h4_rsi",
    "h4_vp_dist_poc_atr",
    # Cross-timeframe confluence features
    "mtf_trend_alignment",
    "mtf_ema_alignment",
    "mtf_hurst_mean",
    "mtf_zscore_divergence",
)


class MultiTimeframeFeaturePipeline:
    """Builds leak-free multi-timeframe feature matrices and live MarketSnapshots."""

    def __init__(self, config: DataPipelineConfig) -> None:
        self._cfg = config

    def _tf_windows(self, tf: Timeframe) -> Tuple[int, int, int]:
        """Return (zscore_window, hurst_window, vp_window) scaled sensibly for higher TFs."""
        if tf == Timeframe.M5:
            return self._cfg.zscore_window, self._cfg.hurst_window, self._cfg.volume_profile_window
        if tf == Timeframe.M15:
            return (
                min(self._cfg.zscore_window, 48),
                min(self._cfg.hurst_window, 64),
                min(self._cfg.volume_profile_window, 64),
            )
        if tf == Timeframe.H1:
            return (
                min(self._cfg.zscore_window, 36),
                max(32, min(self._cfg.hurst_window, 48)),
                min(self._cfg.volume_profile_window, 48),
            )
        # H4
        return (
            min(self._cfg.zscore_window, 24),
            max(32, min(self._cfg.hurst_window, 36)),
            min(self._cfg.volume_profile_window, 36),
        )

    def build_aligned_dataset(
        self,
        bars_by_tf: Dict[Timeframe, pd.DataFrame],
        ticks_df: Optional[pd.DataFrame] = None,
        drop_warmup_nans: bool = True,
    ) -> pd.DataFrame:
        """
        Compute causal features per timeframe and align strictly on `close_time`.
        """
        base_tf = self._cfg.base_timeframe
        if base_tf not in bars_by_tf:
            raise KeyError(f"Base timeframe {base_tf.value} missing from bars_by_tf.")

        base_bars = bars_by_tf[base_tf].sort_values("close_time").reset_index(drop=True)
        z_w, h_w, vp_w = self._tf_windows(base_tf)
        base_feat = build_single_timeframe_features(
            df=base_bars,
            prefix=base_tf.value.lower(),
            atr_period=self._cfg.atr_period,
            parkinson_window=self._cfg.parkinson_window,
            zscore_window=z_w,
            hurst_window=h_w,
            vp_window=vp_w,
            vp_bins=self._cfg.volume_profile_bins,
            value_area_pct=self._cfg.value_area_pct,
            annualization_bars=self._cfg.annualization_bars_m5,
            ticks_df=ticks_df,
        )

        merged = pd.concat(
            [
                base_bars[["open_time", "close_time", "open", "high", "low", "close", "tick_volume", "spread"]],
                base_feat.drop(columns=["open_time", "close_time"]),
            ],
            axis=1,
        ).sort_values("close_time")

        for htf in self._cfg.higher_timeframes:
            if htf not in bars_by_tf:
                raise KeyError(f"Higher timeframe {htf.value} missing from bars_by_tf.")
            htf_bars = bars_by_tf[htf].sort_values("close_time").reset_index(drop=True)
            hz_w, hh_w, hvp_w = self._tf_windows(htf)
            htf_prefix = htf.value.lower()
            htf_feat = build_single_timeframe_features(
                df=htf_bars,
                prefix=htf_prefix,
                atr_period=self._cfg.atr_period,
                parkinson_window=min(self._cfg.parkinson_window, hz_w),
                zscore_window=hz_w,
                hurst_window=hh_w,
                vp_window=hvp_w,
                vp_bins=self._cfg.volume_profile_bins,
                value_area_pct=self._cfg.value_area_pct,
                annualization_bars=None,
                ticks_df=None,
            )
            # Keep an explicit audit column of HTF close_time to verify zero look-ahead in tests
            htf_feat = htf_feat.rename(columns={"open_time": f"{htf_prefix}_open_time"})
            htf_feat[f"{htf_prefix}_closed_at"] = htf_feat["close_time"]

            merged = pd.merge_asof(
                merged,
                htf_feat.drop(columns=[f"{htf_prefix}_open_time"]),
                on="close_time",
                direction="backward",
            )

            # Hard runtime assertion: HTF bar close_time must NEVER exceed base close_time
            valid_mask = merged[f"{htf_prefix}_closed_at"].notna()
            if (merged.loc[valid_mask, f"{htf_prefix}_closed_at"] > merged.loc[valid_mask, "close_time"]).any():
                raise RuntimeError(f"CRITICAL: Look-ahead bias detected when aligning {htf.value}!")

        # Cross-timeframe confluence features (100% causal since inputs are already close-aligned)
        merged = merged.copy()
        sign_m5 = np.sign(merged["m5_ret_4"].fillna(0.0))
        sign_m15 = np.sign(merged["m15_ret_4"].fillna(0.0))
        sign_h1 = np.sign(merged["h1_ret_4"].fillna(0.0))
        sign_h4 = np.sign(merged["h4_ret_1"].fillna(0.0))

        merged["mtf_trend_alignment"] = (sign_m5 + sign_m15 + sign_h1 + sign_h4) / 4.0
        merged["mtf_ema_alignment"] = (
            0.15 * np.tanh(merged["m5_ema_spread_atr"])
            + 0.20 * np.tanh(merged["m15_ema_spread_atr"])
            + 0.35 * np.tanh(merged["h1_ema_spread_atr"])
            + 0.30 * np.tanh(merged["h4_ema_spread_atr"])
        )
        merged["mtf_hurst_mean"] = (
            merged["m5_hurst"] + merged["m15_hurst"] + merged["h1_hurst"] + merged["h4_hurst"]
        ) / 4.0
        merged["mtf_zscore_divergence"] = merged["m5_zscore"] - merged["h1_zscore"]
        m5_atr_base = np.maximum(merged["m5_atr"], 0.95 * merged["m5_atr_24h"])
        merged["structural_atr"] = 2.8 * m5_atr_base

        if drop_warmup_nans:
            merged = merged.dropna(subset=list(PRIMARY_FEATURE_COLUMNS)).reset_index(drop=True)
        return merged

    @staticmethod
    def row_to_market_snapshot(
        row: pd.Series,
        symbol_spec: SymbolSpec,
        live_tick: Optional[TickQuote] = None,
    ) -> MarketSnapshot:
        """Convert the latest aligned feature row into an immutable MarketSnapshot."""
        close_px = float(row["close"])
        if live_tick is not None:
            bid = live_tick.bid
            ask = live_tick.ask
        else:
            half_spread = 0.5 * float(row["spread"]) * symbol_spec.point
            bid = round(close_px - half_spread, symbol_spec.digits)
            ask = round(close_px + half_spread, symbol_spec.digits)

        feature_dict: Dict[str, float] = {
            col: float(row[col]) for col in PRIMARY_FEATURE_COLUMNS if col in row and pd.notna(row[col])
        }

        ts = row["close_time"]
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        eff_atr = float(row["structural_atr"]) if "structural_atr" in row and pd.notna(row["structural_atr"]) else float(row["m5_atr"])

        return MarketSnapshot(
            symbol=symbol_spec.symbol,
            timestamp_utc=ts_dt,
            bid=bid,
            ask=ask,
            close=close_px,
            atr=eff_atr,
            parkinson_vol=float(row["m5_parkinson_vol"]),
            realized_annual_vol=float(row["m5_annual_vol"]),
            hurst_exponent=float(row["m5_hurst"]),
            zscore=float(row["m5_zscore"]),
            vol_zscore=float(row["m5_vol_zscore"]),
            vol_ratio=float(row["m5_vol_ratio"]),
            directional_efficiency=float(row["m5_efficiency"]),
            poc_price=float(row["m5_vp_poc"]),
            vah_price=float(row["m5_vp_vah"]),
            val_price=float(row["m5_vp_val"]),
            bar_tick_volume=float(row["tick_volume"]),
            features=feature_dict,
        )
