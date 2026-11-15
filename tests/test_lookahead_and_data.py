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


def test_dynamic_breakeven_and_trailing_stop_zero_intrabar_lookahead():
    """
    Prove simulate_dynamic_barrier_trade:
    1. Checks initial SL pessimistically BEFORE applying any same-bar favorable excursion ratchet.
    2. Locks +0.15R Breakeven only AFTER a bar closes with >= +1.0R favorable excursion, protecting bar j+1.
    """
    from regime_ai_engine.triple_barrier import simulate_dynamic_barrier_trade

    # Bar 0: entry close = 2000.0, ATR = 2.0 -> SL dist = 1.5 * 2.0 = 3.0 (initial SL = 1997.0), TP = 2006.6
    # Case A: Bar 1 spikes up +1.2R (high=2003.8) AND wicked down to 1996.9 in the SAME bar -> must hit initial SL (-1.0R)!
    closes_a = np.array([2000.0, 2002.0, 2002.0])
    highs_a = np.array([2000.5, 2003.8, 2002.5])
    lows_a = np.array([1999.5, 1996.9, 2001.5])
    atrs = np.array([2.0, 2.0, 2.0])
    spreads = np.zeros(3)

    exit_idx_a, _, _, reason_a, r_a = simulate_dynamic_barrier_trade(
        entry_idx=0,
        side=1,
        closes=closes_a,
        highs=highs_a,
        lows=lows_a,
        atrs=atrs,
        spreads_price=spreads,
        sl_atr_mult=1.5,
        pt_atr_mult=3.3,
        max_holding_bars=10,
    )
    assert exit_idx_a == 1
    assert reason_a == "STOP_LOSS"
    assert r_a == pytest.approx(-1.0)

    # Case B: Bar 1 rallies cleanly to +1.8R (high=2005.4, close=2004.8, low=1999.8 -> survives!),
    # then Bar 2 reverses sharply down to 1998.0 -> exits at late-stage Breakeven+buffer (2000.0 + 0.35*3.0 = 2001.05 -> +0.35R)!
    closes_b = np.array([2000.0, 2004.8, 1998.5])
    highs_b = np.array([2000.5, 2005.4, 2004.9])
    lows_b = np.array([1999.5, 1999.8, 1998.0])

    exit_idx_b, _, exit_px_b, reason_b, r_b = simulate_dynamic_barrier_trade(
        entry_idx=0,
        side=1,
        closes=closes_b,
        highs=highs_b,
        lows=lows_b,
        atrs=atrs,
        spreads_price=spreads,
        sl_atr_mult=1.5,
        pt_atr_mult=3.3,
        max_holding_bars=10,
    )
    assert exit_idx_b == 2
    assert reason_b == "TRAILING_STOP"
    assert exit_px_b == pytest.approx(2001.05)
    assert r_b == pytest.approx(0.35)


def test_regime_aligned_candidate_never_fights_macro_trend():
    """Verify build_meta_feature_matrix never shorts TRENDING_BULL or longs TRENDING_BEAR."""
    from regime_ai_engine.meta_labeling_models import build_meta_feature_matrix

    rng = np.random.default_rng(99)
    n = 200
    X_prim = rng.normal(0.0, 0.5, size=(n, len(PRIMARY_FEATURE_COLUMNS))).astype(np.float32)
    raw_probs = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    raw_probs /= raw_probs.sum(axis=1, keepdims=True)
    spread_to_atr = np.full(n, 0.05, dtype=np.float32)

    bull_regimes = np.full(n, 1.0, dtype=np.float32)
    _, bull_sides, _ = build_meta_feature_matrix(X_prim, raw_probs, bull_regimes, spread_to_atr)
    assert np.all(bull_sides >= 0), "Counter-trend SHORT emitted during TRENDING_BULL!"

    bear_regimes = np.full(n, -1.0, dtype=np.float32)
    _, bear_sides, _ = build_meta_feature_matrix(X_prim, raw_probs, bear_regimes, spread_to_atr)
    assert np.all(bear_sides <= 0), "Counter-trend LONG emitted during TRENDING_BEAR!"


def test_institutional_sleeve_signals_and_horizons():
    """Verify sleeve IDs and session-specific holding horizons (SGE=66, COMEX=156, LondonClose=84)."""
    from regime_ai_engine.meta_labeling_models import extract_institutional_sleeve_signals

    idx_map = {col: i for i, col in enumerate(PRIMARY_FEATURE_COLUMNS)}
    X = np.zeros((3, len(PRIMARY_FEATURE_COLUMNS)), dtype=np.float32)
    regimes = np.full(3, 0.5, dtype=np.float32)  # MEAN_REVERTING

    # Row 0: SGE window (20:45-21:30 UTC) in non-bear macro -> LONG (+1), Sleeve 1, Hold 66 bars
    X[0, idx_map["m5_is_sge_win"]] = 1.0
    X[0, idx_map["h4_ema_spread_atr"]] = 0.25

    # Row 1: COMEX window (12:00-15:55 UTC) + 6h rally exhaustion -> SHORT (-1), Sleeve 2, Hold 156 bars
    X[1, idx_map["m5_is_comex_win"]] = 1.0
    X[1, idx_map["m5_ret_72"]] = 0.0060
    X[1, idx_map["m5_ret_1"]] = -0.0005
    X[1, idx_map["h4_ema_spread_atr"]] = 0.10

    # Row 2: London-Close window (16:30-19:30 UTC) + 6h selloff exhaustion -> LONG (+1), Sleeve 3, Hold 84 bars
    X[2, idx_map["m5_is_ldn_close_win"]] = 1.0
    X[2, idx_map["m5_ret_72"]] = -0.0045
    X[2, idx_map["m5_ret_1"]] = 0.0004
    X[2, idx_map["h4_ema_spread_atr"]] = 0.20

    sides, sleeves, holds = extract_institutional_sleeve_signals(X, regimes)
    assert list(sides) == [1, -1, 1]
    assert list(sleeves) == [1, 2, 3]
    assert list(holds) == [66, 156, 84]


def test_chandelier_trend_runner_barrier_schedule():
    """Verify 4-Sleeve schedule: S1=66, S2=156, S3 Bull-Runner=108/6.0 Chandelier, S4 PMFix Breakout=96/6.0 Chandelier."""
    from regime_ai_engine.meta_labeling_models import extract_sleeve_barrier_schedule

    idx_map = {col: i for i, col in enumerate(PRIMARY_FEATURE_COLUMNS)}
    X = np.zeros((4, len(PRIMARY_FEATURE_COLUMNS)), dtype=np.float32)
    regimes = np.array([1.0, 0.5, 1.0, 1.0], dtype=np.float32)

    # Row 0: SGE AM Fix Accumulation -> LONG (+1), Sleeve 1, Hold 66 bars, PT 3.30 ATR, Chandelier False
    X[0, idx_map["m5_is_sge_win"]] = 1.0
    X[0, idx_map["h4_ema_spread_atr"]] = 0.55

    # Row 1: COMEX exhaustion in neutral H4 (< 0.30 gate) -> SHORT (-1), Sleeve 2, Hold 156, PT 3.30, Chandelier False
    X[1, idx_map["m5_is_comex_win"]] = 1.0
    X[1, idx_map["m5_ret_72"]] = 0.0060
    X[1, idx_map["m5_ret_1"]] = -0.0005
    X[1, idx_map["h4_ema_spread_atr"]] = 0.15

    # Row 2: London-Close in strong H4 bull (>= 0.50) -> LONG (+1), Sleeve 3, Hold 108, PT 6.00, Chandelier True
    X[2, idx_map["m5_is_ldn_close_win"]] = 1.0
    X[2, idx_map["m5_ret_72"]] = -0.0045
    X[2, idx_map["m5_ret_1"]] = 0.0004
    X[2, idx_map["h4_ema_spread_atr"]] = 0.55

    # Row 3: Post-LBMA PM Fix 12h Breakout in H4 bull (>= 0.40) -> LONG (+1), Sleeve 4, Hold 96, PT 6.00, Chandelier True
    X[3, idx_map["m5_is_pmfix_breakout"]] = 1.0
    X[3, idx_map["h4_ema_spread_atr"]] = 0.45

    sides, sleeves, holds, pt_mults, use_chand = extract_sleeve_barrier_schedule(X, regimes)
    assert list(sides) == [1, -1, 1, 1]
    assert list(sleeves) == [1, 2, 3, 4]
    assert list(holds) == [66, 156, 108, 96]
    assert list(np.round(pt_mults, 2)) == [3.3, 3.3, 6.0, 6.0]
    assert list(use_chand) == [False, False, True, True]
