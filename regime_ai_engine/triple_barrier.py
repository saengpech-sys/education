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
    atr_col = "structural_atr" if "structural_atr" in df.columns else "m5_atr"
    atrs = df[atr_col].to_numpy(dtype=np.float64)
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


def simulate_dynamic_barrier_trade(
    entry_idx: int,
    side: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    spreads_price: np.ndarray,
    sl_atr_mult: float,
    pt_atr_mult: float,
    max_holding_bars: int,
    point: float = 0.01,
    slippage_price: float = 0.0,
    enable_ratchet: bool = True,
) -> tuple[int, float, float, str, float]:
    """
    Simulate a single directional trade with 100% causal, pessimistic intrabar execution:
    1. On each bar `j > entry_idx`, check existing `current_sl_px` against adverse extreme FIRST.
    2. Only if SL survived bar `j`, check `tp_px` against favorable extreme.
    3. When `enable_ratchet=True`, apply end-of-bar late-stage profit protection strictly for bar `j + 1`.
       When `enable_ratchet=False` (Institutional Market-Clock sleeves), hold cleanly to TP, SL, or
       session benchmark Time-Barrier (e.g. Shanghai AM Fix 02:30 UTC).

    Returns `(exit_idx, entry_px, exit_px, exit_reason, r_multiple)`.
    """
    n = len(closes)
    atr_entry = max(float(atrs[entry_idx]), point * 5.0)
    sl_dist = sl_atr_mult * atr_entry
    tp_dist = pt_atr_mult * atr_entry
    horizon = min(n - 1, entry_idx + max_holding_bars)
    half_s0 = 0.5 * float(spreads_price[entry_idx])

    if side == 1:
        entry_px = float(closes[entry_idx]) + half_s0 + slippage_price
        current_sl_px = entry_px - sl_dist
        tp_px = entry_px + tp_dist
        max_fav_px = entry_px

        for j in range(entry_idx + 1, horizon + 1):
            half_s = 0.5 * float(spreads_price[j])
            bid_low = float(lows[j]) - half_s
            bid_high = float(highs[j]) - half_s
            bid_close = float(closes[j]) - half_s

            if bid_low <= current_sl_px:
                exit_px = current_sl_px - slippage_price
                reason = "TRAILING_STOP" if current_sl_px > entry_px else "STOP_LOSS"
                return j, entry_px, exit_px, reason, (exit_px - entry_px) / sl_dist

            if bid_high >= tp_px:
                exit_px = tp_px - slippage_price
                return j, entry_px, exit_px, "TAKE_PROFIT", (exit_px - entry_px) / sl_dist

            if enable_ratchet:
                # End-of-bar causal stop ratchet (applies strictly starting on bar j + 1)
                if bid_high > max_fav_px:
                    max_fav_px = bid_high
                fav_r = (max_fav_px - entry_px) / sl_dist
                atr_j = max(float(atrs[j]), point * 5.0)
                max_valid_sl = bid_close - 0.10 * atr_j

                if fav_r >= 1.70:
                    be_sl = min(entry_px + 0.35 * sl_dist, max_valid_sl)
                    if be_sl > current_sl_px:
                        current_sl_px = be_sl
                if fav_r >= 1.95:
                    trail_sl = min(max_fav_px - 0.85 * atr_j, max_valid_sl)
                    if trail_sl > current_sl_px:
                        current_sl_px = trail_sl

        half_s_end = 0.5 * float(spreads_price[horizon])
        exit_px = float(closes[horizon]) - half_s_end - slippage_price
        return horizon, entry_px, exit_px, "TIME_BARRIER", (exit_px - entry_px) / sl_dist

    # SHORT (side == -1)
    entry_px = float(closes[entry_idx]) - half_s0 - slippage_price
    current_sl_px = entry_px + sl_dist
    tp_px = entry_px - tp_dist
    min_fav_px = entry_px

    for j in range(entry_idx + 1, horizon + 1):
        half_s = 0.5 * float(spreads_price[j])
        ask_high = float(highs[j]) + half_s
        ask_low = float(lows[j]) + half_s
        ask_close = float(closes[j]) + half_s

        if ask_high >= current_sl_px:
            exit_px = current_sl_px + slippage_price
            reason = "TRAILING_STOP" if current_sl_px < entry_px else "STOP_LOSS"
            return j, entry_px, exit_px, reason, (entry_px - exit_px) / sl_dist

        if ask_low <= tp_px:
            exit_px = tp_px + slippage_price
            return j, entry_px, exit_px, "TAKE_PROFIT", (entry_px - exit_px) / sl_dist

        if enable_ratchet:
            # End-of-bar causal stop ratchet (applies strictly starting on bar j + 1)
            if ask_low < min_fav_px:
                min_fav_px = ask_low
            fav_r = (entry_px - min_fav_px) / sl_dist
            atr_j = max(float(atrs[j]), point * 5.0)
            min_valid_sl = ask_close + 0.10 * atr_j

            if fav_r >= 1.70:
                be_sl = max(entry_px - 0.35 * sl_dist, min_valid_sl)
                if be_sl < current_sl_px:
                    current_sl_px = be_sl
            if fav_r >= 1.95:
                trail_sl = max(min_fav_px + 0.85 * atr_j, min_valid_sl)
                if trail_sl < current_sl_px:
                    current_sl_px = trail_sl

    half_s_end = 0.5 * float(spreads_price[horizon])
    exit_px = float(closes[horizon]) + half_s_end + slippage_price
    return horizon, entry_px, exit_px, "TIME_BARRIER", (entry_px - exit_px) / sl_dist


def compute_meta_labels_for_sides(
    df: pd.DataFrame,
    predicted_sides: np.ndarray,
    pt_atr_mult: float = 3.3,
    sl_atr_mult: float = 1.5,
    max_holding_bars: int = 156,
    point: float = 0.01,
    slippage_points: float = 2.0,
    max_holding_bars_arr: np.ndarray | None = None,
    commission_per_lot_usd: float = 3.50,
    contract_size: float = 100.0,
    enable_ratchet: bool = False,
) -> pd.DataFrame:
    """
    Given Primary directional candidate predictions `predicted_sides` in {-1, 0, +1},
    evaluate asymmetric Triple-Barrier + sleeve-specific session horizon outcome including
    realistic bid-ask spread, slippage, and round-turn broker commission.

    Returns binary `meta_label` in {0, 1} (1 iff trade achieves net positive R > 0 after commission)
    and realized net `r_multiple`.
    """
    n = len(df)
    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    atr_col = "structural_atr" if "structural_atr" in df.columns else "m5_atr"
    atrs = df[atr_col].to_numpy(dtype=np.float64)
    spreads = df["spread"].to_numpy(dtype=np.float64) * point
    close_times = df["close_time"].to_numpy()
    slip_px = slippage_points * point

    meta_y = np.zeros(n, dtype=np.int64)
    r_mult = np.zeros(n, dtype=np.float64)
    end_indices = np.arange(n, dtype=np.int64)

    for i in range(n - 1):
        side = int(predicted_sides[i])
        if side == 0:
            continue

        hold_i = int(max_holding_bars_arr[i]) if max_holding_bars_arr is not None else max_holding_bars
        end_j, _, _, _, outcome_r = simulate_dynamic_barrier_trade(
            entry_idx=i,
            side=side,
            closes=closes,
            highs=highs,
            lows=lows,
            atrs=atrs,
            spreads_price=spreads,
            sl_atr_mult=sl_atr_mult,
            pt_atr_mult=pt_atr_mult,
            max_holding_bars=hold_i,
            point=point,
            slippage_price=slip_px,
            enable_ratchet=enable_ratchet,
        )
        comm_r = commission_per_lot_usd / max(sl_atr_mult * float(atrs[i]) * contract_size, 1e-6)
        net_r = float(outcome_r) - comm_r
        end_indices[i] = end_j
        r_mult[i] = net_r
        meta_y[i] = 1 if net_r > 0.0 else 0

    return pd.DataFrame(
        {
            "event_end_time": close_times[end_indices],
            "event_end_idx": end_indices,
            "meta_label": meta_y,
            "r_multiple": r_mult,
        },
        index=df.index,
    )
