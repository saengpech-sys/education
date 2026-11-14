"""
Institutional Portfolio Stability & Risk-Adjusted Return Metrics:
- Annualized Sharpe Ratio (Target Gate > 2.0)
- Annualized Sortino Ratio (Downside Semi-Deviation)
- Calmar Ratio (CAGR / Max Drawdown)
- Maximum Drawdown (%) & Annualized Equity Volatility (%)
- Profit Factor & Tail Ratio
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class PortfolioStabilityMetrics:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    total_trades: int
    tail_ratio: float
    passes_sharpe_target: bool


def compute_max_drawdown_pct(equity_curve: np.ndarray) -> float:
    """Compute peak-to-trough Maximum Drawdown as a fraction in [0, 1]."""
    if len(equity_curve) == 0:
        return 0.0
    eq = np.asarray(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(eq)
    drawdowns = np.where(running_peak > EPS, (running_peak - eq) / running_peak, 0.0)
    return float(np.max(drawdowns))


def compute_portfolio_metrics(
    equity_curve: pd.Series | np.ndarray,
    bars_per_year: int = 72_576,
    trade_pnls: Optional[np.ndarray] = None,
    min_sharpe_target: float = 2.0,
    risk_free_rate_annual: float = 0.0,
) -> PortfolioStabilityMetrics:
    """
    Compute full institutional risk-adjusted performance metrics from an equity curve.
    Note: Active returns or full-bar returns are annualized consistently via `bars_per_year`.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        init_eq = float(eq[0]) if len(eq) == 1 else 0.0
        return PortfolioStabilityMetrics(
            initial_equity=init_eq,
            final_equity=init_eq,
            total_return_pct=0.0,
            cagr_pct=0.0,
            annualized_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            win_rate_pct=0.0,
            total_trades=0,
            tail_ratio=1.0,
            passes_sharpe_target=False,
        )

    init_eq = float(eq[0])
    final_eq = float(eq[-1])
    total_ret = (final_eq / max(init_eq, EPS)) - 1.0

    # Aggregate bar-level returns into daily-equivalent buckets (288 M5 bars/day) when series is long
    # to avoid zero-return flat-bar kurtosis distortion on high-frequency equity curves,
    # or evaluate directly if series is short / trade-indexed.
    bars_per_day = max(1, bars_per_year // 252)
    if len(eq) >= bars_per_day * 5:
        sampled_eq = eq[::bars_per_day]
        if sampled_eq[-1] != eq[-1]:
            sampled_eq = np.append(sampled_eq, eq[-1])
        rets = np.diff(sampled_eq) / np.maximum(sampled_eq[:-1], EPS)
        ann_factor = 252.0
    else:
        rets = np.diff(eq) / np.maximum(eq[:-1], EPS)
        ann_factor = float(bars_per_year)

    rf_per_period = risk_free_rate_annual / ann_factor
    excess_rets = rets - rf_per_period

    mean_excess = float(np.mean(excess_rets))
    std_ret = float(np.std(excess_rets, ddof=1)) if len(excess_rets) > 1 else 0.0
    ann_vol = std_ret * math.sqrt(ann_factor)

    sharpe = (mean_excess / std_ret) * math.sqrt(ann_factor) if std_ret > EPS else 0.0

    downside = np.minimum(0.0, excess_rets)
    downside_dev = float(math.sqrt(np.mean(downside ** 2)))
    sortino = (mean_excess / downside_dev) * math.sqrt(ann_factor) if downside_dev > EPS else (
        sharpe * 1.5 if mean_excess > 0 else 0.0
    )

    years = max((len(eq) - 1) / float(bars_per_year), 1.0 / 252.0)
    if final_eq > 0 and init_eq > 0:
        cagr = (final_eq / init_eq) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    max_dd = compute_max_drawdown_pct(eq)
    calmar = (cagr / max_dd) if max_dd > 1e-6 else (10.0 if cagr > 0 else 0.0)

    if trade_pnls is not None and len(trade_pnls) > 0:
        pnls = np.asarray(trade_pnls, dtype=np.float64)
        n_trades = int(len(pnls))
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = float(len(wins)) / float(n_trades)
        gross_profit = float(np.sum(wins))
        gross_loss = float(np.abs(np.sum(losses)))
        pf = gross_profit / gross_loss if gross_loss > EPS else (99.0 if gross_profit > 0 else 0.0)
    else:
        active_rets = rets[np.abs(rets) > EPS]
        n_trades = int(len(active_rets))
        if n_trades > 0:
            win_rate = float(np.mean(active_rets > 0))
            gp = float(np.sum(active_rets[active_rets > 0]))
            gl = float(np.abs(np.sum(active_rets[active_rets < 0])))
            pf = gp / gl if gl > EPS else (99.0 if gp > 0 else 0.0)
        else:
            win_rate = 0.0
            pf = 0.0

    active_for_tail = rets[np.abs(rets) > EPS]
    if len(active_for_tail) >= 10:
        p95 = abs(float(np.percentile(active_for_tail, 95)))
        p05 = abs(float(np.percentile(active_for_tail, 5)))
        tail_ratio = p95 / max(p05, EPS)
    else:
        tail_ratio = 1.0

    return PortfolioStabilityMetrics(
        initial_equity=init_eq,
        final_equity=final_eq,
        total_return_pct=total_ret,
        cagr_pct=cagr,
        annualized_volatility_pct=ann_vol,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
        max_drawdown_pct=max_dd,
        profit_factor=float(pf),
        win_rate_pct=win_rate,
        total_trades=n_trades,
        tail_ratio=float(tail_ratio),
        passes_sharpe_target=bool(sharpe >= min_sharpe_target),
    )
