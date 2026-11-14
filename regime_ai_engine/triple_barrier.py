"""
Marcos López de Prado Volatility-Scaled Triple-Barrier Labeling & Meta-Labeling.

IMPORTANT ANTI-LEAKAGE GUARANTEE:
1. Labels are computed strictly within the provided dataset slice (`max_idx = len(df) - 1`),
   so In-Sample training slices NEVER inspect prices beyond the In-Sample cutoff.
2. Every observation records `event_end_time` (`t1`) so Purged K-Fold Cross-Validation
   purges any training observation whose barrier horizon overlaps a validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    pt_atr_mult: float = 2.2,
    sl_atr_mult: float = 1.1,
    max_holding_bars: int = 24,
    point: float = 0.01,
) -> pd.DataFrame:
    """
    Generate Primary 3-class directional labels {-1, 0, +1} and exact `event_end_time` (t1).
    Uses symmetrical volatility-scaled barriers to discover genuine directional/mean-reversion edge.
    """
    n = len(df)
    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    atrs = df["m5_atr"].to_numpy(dtype=np.float64)
    spreads = df["spread"].to_numpy(dtype=np.float64) * point
    close_times = df["close_time"].to_numpy()

    primary_label = np.zeros(n, dtype=np.int64)
    event_end_idx = np.arange(n, dtype=np.int64)
    realized_ret_atr = np.zeros(n, dtype=np.float64)

    symmetric_barrier_mult = 0.5 * (pt_atr_mult + sl_atr_mult)

    for i in range(n - 1):
        atr_i = max(atrs[i], point * 5.0)
        entry_mid = closes[i]
        cost = spreads[i]
        up_level = entry_mid + symmetric_barrier_mult * atr_i + 0.5 * cost
        dn_level = entry_mid - symmetric_barrier_mult * atr_i - 0.5 * cost

        horizon_end = min(n - 1, i + max_holding_bars)
        end_j = horizon_end
        lbl = 0

        for j in range(i + 1, horizon_end + 1):
            hit_up = highs[j] >= up_level
            hit_dn = lows[j] <= dn_level
            if hit_up and not hit_dn:
                lbl = 1
                end_j = j
                break
            if hit_dn and not hit_up:
                lbl = -1
                end_j = j
                break
            if hit_up and hit_dn:
                # Conservative tie-break -> neutral
                lbl = 0
                end_j = j
                break

        event_end_idx[i] = end_j
        primary_label[i] = lbl
        realized_ret_atr[i] = (closes[end_j] - entry_mid) / atr_i

    return pd.DataFrame(
        {
            "event_start_time": df["close_time"].to_numpy(),
            "event_end_time": close_times[event_end_idx],
            "event_end_idx": event_end_idx,
            "primary_label": primary_label,
            "realized_ret_atr": realized_ret_atr,
        },
        index=df.index,
    )


def compute_meta_labels_for_sides(
    df: pd.DataFrame,
    predicted_sides: np.ndarray,
    pt_atr_mult: float = 2.2,
    sl_atr_mult: float = 1.1,
    max_holding_bars: int = 24,
    point: float = 0.01,
) -> pd.DataFrame:
    """
    Given Primary Model directional predictions `predicted_sides` in {-1, 0, +1},
    evaluate asymmetric Triple-Barrier outcome (TP = pt_atr_mult * ATR, SL = sl_atr_mult * ATR)
    including realistic bid-ask spread cost.

    Returns binary `meta_label` in {0, 1} (1 iff the directional trade achieves positive net PnL
    via TP touch or positive vertical-barrier exit) and realized R-multiple (`r_multiple`).
    """
    n = len(df)
    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    atrs = df["m5_atr"].to_numpy(dtype=np.float64)
    spreads = df["spread"].to_numpy(dtype=np.float64) * point
    close_times = df["close_time"].to_numpy()

    meta_y = np.zeros(n, dtype=np.int64)
    r_mult = np.zeros(n, dtype=np.float64)
    end_indices = np.arange(n, dtype=np.int64)

    for i in range(n - 1):
        side = int(predicted_sides[i])
        if side == 0:
            continue

        atr_i = max(atrs[i], point * 5.0)
        half_sprd = 0.5 * spreads[i]
        sl_dist = sl_atr_mult * atr_i
        tp_dist = pt_atr_mult * atr_i
        horizon_end = min(n - 1, i + max_holding_bars)
        end_j = horizon_end

        if side == 1:
            entry_px = closes[i] + half_sprd
            tp_px = entry_px + tp_dist
            sl_px = entry_px - sl_dist
            outcome_r = None
            for j in range(i + 1, horizon_end + 1):
                # Conservative intrabar check: check SL against bid (low - half_sprd) first
                bid_low = lows[j] - half_sprd
                bid_high = highs[j] - half_sprd
                if bid_low <= sl_px:
                    outcome_r = -1.0
                    end_j = j
                    break
                if bid_high >= tp_px:
                    outcome_r = tp_dist / sl_dist
                    end_j = j
                    break
            if outcome_r is None:
                exit_px = closes[horizon_end] - half_sprd
                outcome_r = (exit_px - entry_px) / sl_dist
        else:
            entry_px = closes[i] - half_sprd
            tp_px = entry_px - tp_dist
            sl_px = entry_px + sl_dist
            outcome_r = None
            for j in range(i + 1, horizon_end + 1):
                ask_high = highs[j] + half_sprd
                ask_low = lows[j] + half_sprd
                if ask_high >= sl_px:
                    outcome_r = -1.0
                    end_j = j
                    break
                if ask_low <= tp_px:
                    outcome_r = tp_dist / sl_dist
                    end_j = j
                    break
            if outcome_r is None:
                exit_px = closes[horizon_end] + half_sprd
                outcome_r = (entry_px - exit_px) / sl_dist

        end_indices[i] = end_j
        r_mult[i] = float(outcome_r)
        meta_y[i] = 1 if outcome_r > 0.15 else 0

    return pd.DataFrame(
        {
            "event_end_time": close_times[end_indices],
            "event_end_idx": end_indices,
            "meta_label": meta_y,
            "r_multiple": r_mult,
        },
        index=df.index,
    )
