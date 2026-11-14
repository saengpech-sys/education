"""
Unit Tests for 100% Data Leakage & Look-Ahead Bias Prevention:
- Causal Feature Invariance under Future Price/Volume Shocks
- Multi-Timeframe (M5, M15, H1, H4) Closed-Bar Alignment Proof
- Hurst Exponent Trending vs Mean-Reverting Discrimination
- Purged K-Fold Horizon Overlap & Embargo Verification
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_and_validation.purged_cv import PurgedKFoldCV
from core.config_loader import load_config
from core.contracts import Timeframe
from data_pipeline.feature_engineering import (
    build_single_timeframe_features,
    compute_rolling_hurst_exponent,
)
from data_pipeline.mt5_data_client import MultiTimeframeDataFetcher, SimulatedMT5Gateway
from data_pipeline.mtf_aligner import MultiTimeframeFeaturePipeline, PRIMARY_FEATURE_COLUMNS


@pytest.fixture
def app_cfg():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    return load_config(cfg_path)


def test_causal_features_invariance_to_future_shocks(app_cfg):
    """
    Mutating future bars t > cutoff with massive price/volume jumps must have
    ZERO effect (atol=1e-12) on all computed features at t <= cutoff.
    """
    gw = SimulatedMT5Gateway(symbol_spec=app_cfg.symbol_spec, total_m5_bars=450, seed=7)
    m5_clean = gw.fetch_ohlcv("XAUUSD", Timeframe.M5, num_bars=400)
    cutoff_idx = 260

    m5_shocked = m5_clean.copy()
    m5_shocked.loc[cutoff_idx + 1 :, "open"] *= 4.5
    m5_shocked.loc[cutoff_idx + 1 :, "high"] *= 9.0
    m5_shocked.loc[cutoff_idx + 1 :, "low"] *= 0.2
    m5_shocked.loc[cutoff_idx + 1 :, "close"] *= 6.0
    m5_shocked.loc[cutoff_idx + 1 :, "tick_volume"] *= 50.0
    m5_shocked.loc[cutoff_idx + 1 :, "spread"] *= 20.0

    feat_clean = build_single_timeframe_features(
        df=m5_clean,
        prefix="m5",
        atr_period=14,
        parkinson_window=20,
        zscore_window=48,
        hurst_window=96,
        vp_window=96,
        vp_bins=24,
        annualization_bars=72576,
    )
    feat_shocked = build_single_timeframe_features(
        df=m5_shocked,
        prefix="m5",
        atr_period=14,
        parkinson_window=20,
        zscore_window=48,
        hurst_window=96,
        vp_window=96,
        vp_bins=24,
        annualization_bars=72576,
    )

    numeric_cols = [c for c in feat_clean.columns if c not in ("open_time", "close_time")]
    for col in numeric_cols:
        a = feat_clean.loc[:cutoff_idx, col].to_numpy(dtype=np.float64)
        b = feat_shocked.loc[:cutoff_idx, col].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            a,
            b,
            atol=1e-12,
            rtol=0.0,
            err_msg=f"Look-ahead leakage detected in single-TF feature column {col}!",
        )


def test_mtf_alignment_never_leaks_unclosed_higher_timeframe_bars(app_cfg):
    """
    Prove that corrupting a forming H1 candle (e.g. open 10:00, close 11:00)
    does NOT affect M5 aligned features at 10:05..10:55, and only appears at >= 11:00.
    """
    gw = SimulatedMT5Gateway(symbol_spec=app_cfg.symbol_spec, total_m5_bars=2600, seed=11)
    fetcher = MultiTimeframeDataFetcher(gw)
    pipeline = MultiTimeframeFeaturePipeline(app_cfg.data_pipeline)

    bars_clean = fetcher.fetch_multi_timeframe_bars(
        symbol="XAUUSD",
        base_tf=Timeframe.M5,
        higher_tfs=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
        num_bars=2600,
    )
    aligned_clean = pipeline.build_aligned_dataset(bars_clean, drop_warmup_nans=True)

    # Every HTF bar merged onto M5 must have closed at or before M5 close_time
    assert (aligned_clean["m15_closed_at"] <= aligned_clean["close_time"]).all()
    assert (aligned_clean["h1_closed_at"] <= aligned_clean["close_time"]).all()
    assert (aligned_clean["h4_closed_at"] <= aligned_clean["close_time"]).all()

    # Pick a specific H1 bar near the tail and spike its OHLC massively
    bars_mutated = {tf: df.copy() for tf, df in bars_clean.items()}
    target_h1_idx = len(bars_mutated[Timeframe.H1]) - 5
    h1_open_ts = bars_mutated[Timeframe.H1].loc[target_h1_idx, "open_time"]
    h1_close_ts = bars_mutated[Timeframe.H1].loc[target_h1_idx, "close_time"]

    bars_mutated[Timeframe.H1].loc[target_h1_idx:, "high"] = 99999.0
    bars_mutated[Timeframe.H1].loc[target_h1_idx:, "close"] = 88888.0

    aligned_mutated = pipeline.build_aligned_dataset(bars_mutated, drop_warmup_nans=True)

    # M5 bars strictly inside (h1_open_ts, h1_close_ts) must be 100% identical!
    before_h1_close_clean = aligned_clean.loc[aligned_clean["close_time"] < h1_close_ts]
    before_h1_close_mut = aligned_mutated.loc[aligned_mutated["close_time"] < h1_close_ts]
    assert len(before_h1_close_clean) == len(before_h1_close_mut)

    intrabar_mask = before_h1_close_clean["close_time"] > h1_open_ts
    assert int(intrabar_mask.sum()) == 11  # 10:05, 10:10, ..., 10:55 (11 M5 bars)

    for col in PRIMARY_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            before_h1_close_clean[col].to_numpy(),
            before_h1_close_mut[col].to_numpy(),
            atol=1e-12,
            rtol=0.0,
            err_msg=f"HTF look-ahead leakage in column {col} before H1 bar closed!",
        )

    # Once H1 closes at h1_close_ts, the updated H1 feature becomes visible as expected
    at_close_clean = aligned_clean.loc[aligned_clean["close_time"] == h1_close_ts, "h1_ret_1"].iloc[0]
    at_close_mut = aligned_mutated.loc[aligned_mutated["close_time"] == h1_close_ts, "h1_ret_1"].iloc[0]
    assert abs(at_close_mut - at_close_clean) > 1.0


def test_hurst_regime_discrimination():
    """Verify Hurst estimator separates persistent trends (H > 0.55) from mean reversion (H < 0.45)."""
    rng = np.random.default_rng(123)
    # 1. Smooth persistent trend
    trend_steps = 0.5 + 0.08 * rng.normal(size=200)
    trend_series = pd.Series(1000.0 + np.cumsum(trend_steps))
    h_trend = compute_rolling_hurst_exponent(trend_series, window=96).iloc[-1]
    assert h_trend > 0.65

    # 2. Strong Ornstein-Uhlenbeck / negative-autocorrelation mean-reverting process
    mr_vals = [1000.0]
    for _ in range(199):
        mr_vals.append(1000.0 - 0.75 * (mr_vals[-1] - 1000.0) + 0.5 * rng.normal())
    mr_series = pd.Series(mr_vals)
    h_mr = compute_rolling_hurst_exponent(mr_series, window=96).iloc[-1]
    assert h_mr < 0.35


def test_purged_kfold_zero_horizon_overlap_and_embargo():
    """Verify PurgedKFoldCV purges all overlapping event horizons and applies post-test embargo."""
    n = 250
    starts = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # Every label spans 12 bars forward into the future
    ends = starts + pd.Timedelta(minutes=5 * 12)

    cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.04)  # embargo = ceil(250 * 0.04) = 10 bars
    splits = list(cv.split(starts, ends))
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        test_start = starts[test_idx[0]]
        test_max_end = ends[test_idx].max()

        for i in train_idx:
            # Interval [starts[i], ends[i]] must NEVER overlap [test_start, test_max_end]
            overlaps = (starts[i] <= test_max_end) and (ends[i] >= test_start)
            assert not overlaps, f"Training index {i} overlaps test horizon!"

            # Embargo check after test block
            if i > test_idx[-1]:
                assert i >= test_idx[-1] + 1 + 10
