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
from regime_ai_engine.triple_barrier import compute_meta_labels_for_sides
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
        candidates = (0.52, 0.56, 0.60, 0.64, 0.68)
        best_thr = self._cfg.regime_ai.meta_confidence_threshold
        best_score = -1e9

        base_engine = ONNXMetaLabelingEngine(
            ai_config=self._cfg.regime_ai,
            risk_config=self._cfg.risk_and_scaling,
            symbol_spec=self._cfg.symbol_spec,
            primary_model_source=primary_bytes,
            meta_model_source=meta_bytes,
            confidence_threshold_override=0.50,
        )
        sides_50, _, meta_probs, regimes = base_engine.evaluate_batch_dataframe(val_df)
        outcomes = compute_meta_labels_for_sides(
            df=val_df,
            predicted_sides=sides_50,
            pt_atr_mult=self._cfg.regime_ai.triple_barrier_pt_atr,
            sl_atr_mult=self._cfg.regime_ai.triple_barrier_sl_atr,
            max_holding_bars=self._cfg.regime_ai.triple_barrier_max_bars,
            point=self._cfg.symbol_spec.point,
        )
        r_mult = outcomes["r_multiple"].to_numpy(dtype=np.float64)
        tradable = np.array([MarketRegime(r).is_tradable for r in regimes], dtype=bool)

        for thr in candidates:
            mask = tradable & (sides_50 != 0) & (meta_probs >= thr)
            if int(np.sum(mask)) < 5:
                continue
            r_sel = r_mult[mask]
            mean_r = float(np.mean(r_sel))
            std_r = float(np.std(r_sel, ddof=1)) if len(r_sel) > 1 else 1.0
            sharpe_like = (mean_r / max(std_r, 0.05)) * np.sqrt(float(len(r_sel)))
            if sharpe_like > best_score:
                best_score = sharpe_like
                best_thr = thr

        return float(best_thr)

    def run(self, aligned_df: pd.DataFrame) -> WalkForwardResult:
        """Run rolling Walk-Forward Optimization and stitch compounded Out-Of-Sample equity."""
        bt_cfg = self._cfg.backtest_and_validation
        rk_cfg = self._cfg.risk_and_scaling
        ai_cfg = self._cfg.regime_ai
        spec = self._cfg.symbol_spec

        train_bars = bt_cfg.wfo_train_bars
        test_bars = bt_cfg.wfo_test_bars
        n_total = len(aligned_df)
        if n_total < train_bars + test_bars:
            raise ValueError(
                f"Aligned dataset ({n_total} bars) shorter than wfo_train_bars + wfo_test_bars "
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
        trade_records: List[Dict[str, object]] = []
        fold_summaries: List[WFOFoldSummary] = []

        cursor = train_bars
        fold_idx = 0

        while cursor + 20 <= n_total:
            train_start = cursor - train_bars
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

            # Tune confidence threshold strictly inside last 25% of In-Sample window
            val_tail = train_slice.iloc[int(len(train_slice) * 0.75) :].reset_index(drop=True)
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

            sides, _, meta_probs, regimes = engine.evaluate_batch_dataframe(test_slice)
            fold_start_eq = equity
            fold_trades = 0

            closes = test_slice["close"].to_numpy(dtype=np.float64)
            highs = test_slice["high"].to_numpy(dtype=np.float64)
            lows = test_slice["low"].to_numpy(dtype=np.float64)
            atrs = test_slice["m5_atr"].to_numpy(dtype=np.float64)
            ann_vols = test_slice["m5_annual_vol"].to_numpy(dtype=np.float64)
            vol_zscores = test_slice["m5_vol_zscore"].to_numpy(dtype=np.float64)
            spreads_pts = test_slice["spread"].to_numpy(dtype=np.float64)
            times = test_slice["close_time"].tolist()

            if fold_idx == 0:
                oos_times.append(train_slice["close_time"].iloc[-1])
                oos_equities.append(equity)

            active_until_idx = -1
            pending_exit_idx = -1
            pending_net_pnl = 0.0
            pending_trade_dict: Optional[Dict[str, object]] = None

            for i in range(len(test_slice)):
                ts = times[i]
                ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

                # Settle closed trade strictly at its true chronological exit bar
                if i == pending_exit_idx:
                    equity += pending_net_pnl
                    if pending_trade_dict is not None:
                        pending_trade_dict["equity_after"] = round(equity, 2)
                        trade_records.append(pending_trade_dict)
                        pending_trade_dict = None
                    pending_exit_idx = -1
                    pending_net_pnl = 0.0

                regime_i = MarketRegime(regimes.iloc[i])
                cb_status = circuit_breaker.update(
                    current_equity=equity,
                    timestamp_utc=ts_dt,
                    regime=regime_i,
                    vol_zscore=float(vol_zscores[i]),
                )

                # Only evaluate new entry if flat and enough bars remain in OOS slice
                if i > active_until_idx and i < len(test_slice) - 2 and sides[i] != 0:
                    spread_px = float(spreads_pts[i]) * spec.point
                    slip_px = bt_cfg.slippage_points * spec.point
                    side_enum = SignalSide(int(sides[i]))
                    entry_px = (
                        closes[i] + 0.5 * spread_px + slip_px
                        if side_enum == SignalSide.LONG
                        else closes[i] - 0.5 * spread_px - slip_px
                    )
                    size_res = sizer.calculate_position_size(
                        equity=equity,
                        side=side_enum,
                        entry_price=entry_px,
                        atr=float(atrs[i]),
                        spread_price=spread_px,
                        realized_annual_vol=float(ann_vols[i]),
                        win_probability=float(meta_probs[i]),
                        reward_to_risk_ratio=engine.reward_to_risk_ratio,
                        circuit_status=cb_status,
                    )
                    if size_res.approved and size_res.final_lots >= spec.volume_min:
                        horizon = min(len(test_slice) - 1, i + ai_cfg.triple_barrier_max_bars)
                        exit_idx = horizon
                        exit_px = closes[horizon]
                        exit_reason = "TIME_BARRIER"

                        for j in range(i + 1, horizon + 1):
                            half_s = 0.5 * float(spreads_pts[j]) * spec.point
                            if side_enum == SignalSide.LONG:
                                if lows[j] - half_s <= size_res.stop_loss_price:
                                    exit_px = size_res.stop_loss_price - slip_px
                                    exit_idx = j
                                    exit_reason = "STOP_LOSS"
                                    break
                                if highs[j] - half_s >= size_res.take_profit_price:
                                    exit_px = size_res.take_profit_price - slip_px
                                    exit_idx = j
                                    exit_reason = "TAKE_PROFIT"
                                    break
                            else:
                                if highs[j] + half_s >= size_res.stop_loss_price:
                                    exit_px = size_res.stop_loss_price + slip_px
                                    exit_idx = j
                                    exit_reason = "STOP_LOSS"
                                    break
                                if lows[j] + half_s <= size_res.take_profit_price:
                                    exit_px = size_res.take_profit_price + slip_px
                                    exit_idx = j
                                    exit_reason = "TAKE_PROFIT"
                                    break

                        if exit_reason == "TIME_BARRIER":
                            half_s_end = 0.5 * float(spreads_pts[exit_idx]) * spec.point
                            exit_px = (
                                closes[exit_idx] - half_s_end - slip_px
                                if side_enum == SignalSide.LONG
                                else closes[exit_idx] + half_s_end + slip_px
                            )

                        price_diff = (
                            (exit_px - entry_px)
                            if side_enum == SignalSide.LONG
                            else (entry_px - exit_px)
                        )
                        gross_pnl = (price_diff / spec.tick_size) * spec.tick_value * size_res.final_lots
                        commission = bt_cfg.commission_per_lot_usd * size_res.final_lots
                        net_pnl = gross_pnl - commission

                        active_until_idx = exit_idx
                        pending_exit_idx = exit_idx
                        pending_net_pnl = net_pnl
                        fold_trades += 1

                        pending_trade_dict = {
                            "fold": fold_idx,
                            "entry_time": ts,
                            "exit_time": times[exit_idx],
                            "side": side_enum.name,
                            "lots": size_res.final_lots,
                            "entry_price": round(entry_px, spec.digits),
                            "exit_price": round(exit_px, spec.digits),
                            "meta_prob": round(float(meta_probs[i]), 4),
                            "cb_state": cb_status.state.value,
                            "exit_reason": exit_reason,
                            "net_pnl_usd": round(net_pnl, 2),
                            "equity_after": round(equity + net_pnl, 2),
                        }

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
