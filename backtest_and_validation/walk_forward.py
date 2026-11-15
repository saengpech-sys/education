"""
Walk-Forward Optimization (WFO) Engine with Strictly Out-Of-Sample ONNX Execution.

Workflow per Rolling Window k:
1. Slice strictly historical In-Sample window `[train_start, train_end)`.
2. Train Primary + Secondary Meta-Models using Purged K-Fold CV & export ONNX binaries.
3. Select optimal Meta-Confidence Threshold on an In-Sample validation sub-window maximizing Sharpe.
4. Step chronologically through strictly unseen Out-Of-Sample window `[test_start, test_end)`
   using ONNXMetaLabelingEngine + EquityCircuitBreaker + InverseVolatilityPositionSizer,
   accounting for live bid-ask spread, execution slippage, and round-turn commissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest_and_validation.metrics import (
    PortfolioStabilityMetrics,
    compute_portfolio_metrics,
)
from core.config_loader import AppConfig
from core.contracts import MarketRegime, SignalSide
from regime_ai_engine.meta_labeling_models import MetaLabelingTrainer
from regime_ai_engine.onnx_inference import ONNXMetaLabelingEngine
from regime_ai_engine.triple_barrier import (
    compute_meta_labels_for_sides,
    simulate_dynamic_barrier_trade,
)
from risk_and_scaling_engine.circuit_breaker import EquityCircuitBreaker
from risk_and_scaling_engine.position_sizer import InverseVolatilityPositionSizer


@dataclass(frozen=True)
class WFOFoldSummary:
    fold_index: int
    train_start_utc: pd.Timestamp
    train_end_utc: pd.Timestamp
    test_start_utc: pd.Timestamp
    test_end_utc: pd.Timestamp
    selected_confidence_threshold: float
    oos_trades: int
    oos_return_pct: float


@dataclass(frozen=True)
class WalkForwardResult:
    equity_curve: pd.Series
    trades_df: pd.DataFrame
    fold_summaries: List[WFOFoldSummary]
    metrics: PortfolioStabilityMetrics


class WalkForwardOptimizer:
    """Executes leak-free Walk-Forward Optimization across multi-timeframe feature datasets."""

    def __init__(self, config: AppConfig, onnx_artifact_dir: Optional[Path] = None) -> None:
        self._cfg = config
        self._onnx_dir = onnx_artifact_dir or Path(config.regime_ai.onnx_dir)

    def _optimize_threshold_in_sample(
        self,
        val_df: pd.DataFrame,
        primary_bytes: bytes,
        meta_bytes: bytes,
    ) -> float:
        """Select confidence threshold maximizing risk-adjusted expectancy strictly inside IS."""
        candidates = (0.45, 0.47, 0.49)
        best_thr = self._cfg.regime_ai.meta_confidence_threshold
        best_score = -1e9

        base_engine = ONNXMetaLabelingEngine(
            ai_config=self._cfg.regime_ai,
            risk_config=self._cfg.risk_and_scaling,
            symbol_spec=self._cfg.symbol_spec,
            primary_model_source=primary_bytes,
            meta_model_source=meta_bytes,
            confidence_threshold_override=0.36,
        )
        (
            sides_base,
            slv_base,
            hold_bars,
            pt_mults,
            use_chand,
            _,
            meta_probs,
            regimes,
        ) = base_engine.evaluate_batch_with_barriers(val_df)

        # De-cluster consecutive M5 candidate bars per sleeve to match live execution
        eval_sides = np.zeros_like(sides_base)
        last_entry_slv: Dict[int, int] = {}
        for idx_i in range(len(sides_base)):
            s_id = int(slv_base[idx_i])
            if sides_base[idx_i] != 0 and s_id > 0 and (idx_i - last_entry_slv.get(s_id, -999)) >= 36:
                eval_sides[idx_i] = sides_base[idx_i]
                last_entry_slv[s_id] = idx_i

        outcomes = compute_meta_labels_for_sides(
            df=val_df,
            predicted_sides=eval_sides,
            pt_atr_mult=self._cfg.risk_and_scaling.atr_tp_multiplier,
            sl_atr_mult=self._cfg.risk_and_scaling.atr_stop_multiplier,
            max_holding_bars=self._cfg.regime_ai.triple_barrier_max_bars,
            point=self._cfg.symbol_spec.point,
            slippage_points=self._cfg.backtest_and_validation.slippage_points,
            max_holding_bars_arr=hold_bars,
            pt_atr_mult_arr=pt_mults,
            enable_ratchet_arr=use_chand,
            commission_per_lot_usd=self._cfg.backtest_and_validation.commission_per_lot_usd,
            contract_size=self._cfg.symbol_spec.contract_size,
            enable_ratchet=False,
        )
        r_mult = outcomes["r_multiple"].to_numpy(dtype=np.float64)
        tradable = np.array([MarketRegime(r).is_tradable for r in regimes], dtype=bool)

        for thr in candidates:
            mask = tradable & (eval_sides != 0) & (meta_probs >= thr)
            if int(np.sum(mask)) < 8:
                continue
            r_sel = r_mult[mask]
            mean_r = float(np.mean(r_sel))
            if mean_r <= 0.02:
                continue
            std_r = float(np.std(r_sel, ddof=1)) if len(r_sel) > 1 else 1.0
            sharpe_like = (mean_r / max(std_r, 0.10)) * np.sqrt(float(len(r_sel)))
            if sharpe_like > best_score:
                best_score = sharpe_like
                best_thr = thr

        return float(best_thr)

    def run(
        self,
        aligned_df: pd.DataFrame,
        initial_train_bars: Optional[int] = None,
        max_train_bars: int = 28800,
        retrain_every_bars: Optional[int] = None,
        max_concurrent_sleeves: int = 3,
        sleeve_cooldown_bars: int = 36,
        same_side_risk_scale: float = 0.70,
    ) -> WalkForwardResult:
        """Run rolling Walk-Forward Optimization and stitch compounded Out-Of-Sample equity."""
        bt_cfg = self._cfg.backtest_and_validation
        rk_cfg = self._cfg.risk_and_scaling
        ai_cfg = self._cfg.regime_ai
        spec = self._cfg.symbol_spec

        train_bars = int(initial_train_bars) if initial_train_bars is not None else bt_cfg.wfo_train_bars
        effective_max_train = max(train_bars, int(max_train_bars))
        test_bars = int(retrain_every_bars) if retrain_every_bars is not None else bt_cfg.wfo_test_bars
        n_total = len(aligned_df)
        if n_total < train_bars + test_bars:
            raise ValueError(
                f"Aligned dataset ({n_total} bars) shorter than train_bars + wfo_test_bars "
                f"({train_bars + test_bars})."
            )

        trainer = MetaLabelingTrainer(
            ai_config=ai_cfg,
            symbol_spec=spec,
            random_seed=self._cfg.system.random_seed,
        )
        circuit_breaker = EquityCircuitBreaker(
            config=rk_cfg,
            initial_equity=bt_cfg.initial_equity,
        )
        sizer = InverseVolatilityPositionSizer(config=rk_cfg, symbol_spec=spec)

        equity = float(bt_cfg.initial_equity)
        oos_times: List[pd.Timestamp] = []
        oos_equities: List[float] = []
        oos_bar_returns: List[float] = []
        trade_records: List[Dict[str, object]] = []
        fold_summaries: List[WFOFoldSummary] = []

        # Global price/ATR/spread arrays for exact chronological settlement across fold boundaries
        all_closes = aligned_df["close"].to_numpy(dtype=np.float64)
        all_highs = aligned_df["high"].to_numpy(dtype=np.float64)
        all_lows = aligned_df["low"].to_numpy(dtype=np.float64)
        atr_col = "structural_atr" if "structural_atr" in aligned_df.columns else "m5_atr"
        all_atrs = aligned_df[atr_col].to_numpy(dtype=np.float64)
        all_vol_zscores = aligned_df["m5_vol_zscore"].to_numpy(dtype=np.float64)
        all_spreads_pts = aligned_df["spread"].to_numpy(dtype=np.float64)
        all_spreads_px = all_spreads_pts * spec.point
        all_times = aligned_df["close_time"].tolist()

        # Active portfolio positions: (global_exit_idx, sleeve_id, side_int, net_pnl, trade_record_dict)
        open_positions: List[tuple[int, int, int, float, Dict[str, object]]] = []
        last_sleeve_entry_global: Dict[int, int] = {}

        ann_factor = np.sqrt(float(self._cfg.data_pipeline.annualization_bars_m5))
        cursor = train_bars
        fold_idx = 0

        while cursor + 20 <= n_total:
            train_start = max(0, cursor - effective_max_train)
            train_end = cursor
            test_end = min(n_total, cursor + test_bars)

            train_slice = aligned_df.iloc[train_start:train_end].reset_index(drop=True)
            test_slice = aligned_df.iloc[train_end:test_end].reset_index(drop=True)

            fold_dir = self._onnx_dir / f"fold_{fold_idx:02d}"
            artifacts = trainer.train_and_export(
                train_df=train_slice,
                output_dir=fold_dir,
                n_splits=bt_cfg.purged_cv_splits,
                embargo_pct=bt_cfg.embargo_pct,
            )

            # Tune confidence threshold strictly inside last 35% of In-Sample window
            val_tail = train_slice.iloc[int(len(train_slice) * 0.65) :].reset_index(drop=True)
            chosen_thr = self._optimize_threshold_in_sample(
                val_df=val_tail,
                primary_bytes=artifacts.primary_onnx_bytes,
                meta_bytes=artifacts.meta_onnx_bytes,
            )

            engine = ONNXMetaLabelingEngine(
                ai_config=ai_cfg,
                risk_config=rk_cfg,
                symbol_spec=spec,
                primary_model_source=artifacts.primary_onnx_bytes,
                meta_model_source=artifacts.meta_onnx_bytes,
                confidence_threshold_override=chosen_thr,
            )

            (
                sides,
                sleeve_ids,
                hold_bars,
                pt_mults,
                use_chand,
                _,
                meta_probs,
                regimes,
            ) = engine.evaluate_batch_with_barriers(test_slice)
            fold_start_eq = equity
            fold_trades = 0

            if fold_idx == 0:
                oos_times.append(train_slice["close_time"].iloc[-1])
                oos_equities.append(equity)

            for i in range(len(test_slice)):
                global_i = train_end + i
                ts = all_times[global_i]
                ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                prev_equity = equity

                # Settle any closed sleeve trades strictly at their chronological exit timestamp
                still_open: List[tuple[int, int, int, float, Dict[str, object]]] = []
                for exit_g_idx, slv_open, side_open, pnl_open, tr_dict in open_positions:
                    if global_i >= exit_g_idx:
                        equity += pnl_open
                        tr_dict["equity_after"] = round(equity, 2)
                        trade_records.append(tr_dict)
                    else:
                        still_open.append((exit_g_idx, slv_open, side_open, pnl_open, tr_dict))
                open_positions = still_open

                regime_i = MarketRegime(regimes.iloc[i])
                cb_status = circuit_breaker.update(
                    current_equity=equity,
                    timestamp_utc=ts_dt,
                    regime=regime_i,
                    vol_zscore=float(all_vol_zscores[global_i]),
                )

                # Causal trailing downside semi-volatility (1,440 M5 bars = 5 trading days)
                if len(oos_bar_returns) >= 288:
                    recent_rets = np.asarray(oos_bar_returns[-1440:], dtype=np.float64)
                    neg_rets = np.minimum(0.0, recent_rets)
                    realized_port_vol = float(np.sqrt(np.mean(neg_rets ** 2))) * ann_factor
                else:
                    realized_port_vol = 0.65 * rk_cfg.target_annual_volatility

                # Dimensionless pre-circuit-breaker portfolio drawdown risk taper & high-DD overlap shield
                dd_taper_start = 0.45 * rk_cfg.max_trailing_drawdown_limit_pct
                dd_overlap_cap = 0.65 * rk_cfg.max_trailing_drawdown_limit_pct
                if cb_status.trailing_drawdown_pct > dd_taper_start:
                    excess_frac = (cb_status.trailing_drawdown_pct - dd_taper_start) / max(
                        rk_cfg.max_trailing_drawdown_limit_pct - dd_taper_start,
                        1e-6,
                    )
                    stress_mult = 1.0 + 1.25 * excess_frac
                    realized_port_vol = max(
                        realized_port_vol,
                        rk_cfg.target_annual_volatility * stress_mult,
                    )

                slv_id = int(sleeve_ids[i])
                side_int = int(sides[i])
                active_sleeve_set = {s_open for _, s_open, _, _, _ in open_positions}
                same_side_open_count = sum(1 for _, _, sd_open, _, _ in open_positions if sd_open == side_int)
                last_g_entry = last_sleeve_entry_global.get(slv_id, -999999)
                allow_same_side = (same_side_open_count == 0) or (
                    same_side_risk_scale > 0.0
                    and same_side_open_count == 1
                    and cb_status.trailing_drawdown_pct <= dd_overlap_cap
                )

                if (
                    global_i < n_total - 5
                    and side_int != 0
                    and slv_id > 0
                    and slv_id not in active_sleeve_set
                    and allow_same_side
                    and (global_i - last_g_entry) >= sleeve_cooldown_bars
                    and len(open_positions) < max_concurrent_sleeves
                    and all_spreads_pts[global_i] <= self._cfg.execution_and_monitoring.max_spread_points
                ):
                    spread_px = float(all_spreads_px[global_i])
                    slip_px = bt_cfg.slippage_points * spec.point
                    side_enum = SignalSide(side_int)
                    entry_ref = (
                        all_closes[global_i] + 0.5 * spread_px + slip_px
                        if side_enum == SignalSide.LONG
                        else all_closes[global_i] - 0.5 * spread_px - slip_px
                    )
                    effective_vol = realized_port_vol
                    if same_side_open_count == 1 and 0.0 < same_side_risk_scale < 1.0:
                        effective_vol = max(
                            realized_port_vol / same_side_risk_scale,
                            rk_cfg.target_annual_volatility / same_side_risk_scale,
                        )
                    is_runner = bool(use_chand[i])
                    rr_for_kelly = 2.60 if is_runner else engine.reward_to_risk_ratio
                    size_res = sizer.calculate_position_size(
                        equity=equity,
                        side=side_enum,
                        entry_price=entry_ref,
                        atr=float(all_atrs[global_i]),
                        spread_price=spread_px,
                        realized_annual_vol=effective_vol,
                        win_probability=float(meta_probs[i]),
                        reward_to_risk_ratio=rr_for_kelly,
                        circuit_status=cb_status,
                    )
                    if size_res.approved and size_res.final_lots >= spec.volume_min:
                        exit_g_idx, entry_px, exit_px, exit_reason, _ = simulate_dynamic_barrier_trade(
                            entry_idx=global_i,
                            side=side_int,
                            closes=all_closes,
                            highs=all_highs,
                            lows=all_lows,
                            atrs=all_atrs,
                            spreads_price=all_spreads_px,
                            sl_atr_mult=rk_cfg.atr_stop_multiplier,
                            pt_atr_mult=float(pt_mults[i]),
                            max_holding_bars=int(hold_bars[i]),
                            point=spec.point,
                            slippage_price=slip_px,
                            enable_ratchet=is_runner,
                            be_trigger_r=1.55,
                            be_offset_r=0.25,
                            trail_trigger_r=1.95,
                            trail_atr_mult=1.50,
                        )

                        price_diff = (
                            (exit_px - entry_px)
                            if side_enum == SignalSide.LONG
                            else (entry_px - exit_px)
                        )
                        gross_pnl = (price_diff / spec.tick_size) * spec.tick_value * size_res.final_lots
                        commission = bt_cfg.commission_per_lot_usd * size_res.final_lots
                        net_pnl = gross_pnl - commission

                        tr_record: Dict[str, object] = {
                            "fold": fold_idx,
                            "sleeve": slv_id,
                            "trend_runner": is_runner,
                            "entry_time": ts,
                            "exit_time": all_times[exit_g_idx],
                            "side": side_enum.name,
                            "regime": regime_i.value,
                            "lots": size_res.final_lots,
                            "entry_price": round(entry_px, spec.digits),
                            "exit_price": round(exit_px, spec.digits),
                            "meta_prob": round(float(meta_probs[i]), 4),
                            "cb_state": cb_status.state.value,
                            "exit_reason": exit_reason,
                            "net_pnl_usd": round(net_pnl, 2),
                            "equity_after": round(equity + net_pnl, 2),
                        }
                        open_positions.append((exit_g_idx, slv_id, side_int, net_pnl, tr_record))
                        last_sleeve_entry_global[slv_id] = global_i
                        fold_trades += 1

                oos_bar_returns.append((equity - prev_equity) / max(prev_equity, 1e-9))
                oos_times.append(ts)
                oos_equities.append(equity)

            fold_ret = (equity / max(fold_start_eq, 1e-9)) - 1.0
            fold_summaries.append(
                WFOFoldSummary(
                    fold_index=fold_idx,
                    train_start_utc=train_slice["close_time"].iloc[0],
                    train_end_utc=train_slice["close_time"].iloc[-1],
                    test_start_utc=test_slice["close_time"].iloc[0],
                    test_end_utc=test_slice["close_time"].iloc[-1],
                    selected_confidence_threshold=chosen_thr,
                    oos_trades=fold_trades,
                    oos_return_pct=fold_ret,
                )
            )
            cursor = test_end
            fold_idx += 1

        # Settle any remaining open positions landing on the terminal bar of the dataset
        if open_positions and oos_equities:
            for _, _, _, pnl_open, tr_dict in open_positions:
                equity += pnl_open
                tr_dict["equity_after"] = round(equity, 2)
                trade_records.append(tr_dict)
            oos_equities[-1] = equity

        eq_series = pd.Series(oos_equities, index=pd.DatetimeIndex(oos_times), name="equity")
        trades_df = pd.DataFrame(trade_records)
        pnls = trades_df["net_pnl_usd"].to_numpy(dtype=np.float64) if not trades_df.empty else None

        metrics = compute_portfolio_metrics(
            equity_curve=eq_series.to_numpy(),
            bars_per_year=self._cfg.data_pipeline.annualization_bars_m5,
            trade_pnls=pnls,
            min_sharpe_target=bt_cfg.min_sharpe_target,
        )
        return WalkForwardResult(
            equity_curve=eq_series,
            trades_df=trades_df,
            fold_summaries=fold_summaries,
            metrics=metrics,
        )
