"""
1,000-Iteration Circular Block-Bootstrap Monte Carlo Tail-Risk Simulator.

Quantifies worst-case equity drawdown and compounding survival under:
1. Random temporal reordering of return blocks (preserving volatility clustering & losing streaks).
2. Execution friction & tail-slippage stress noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloTailRiskReport:
    iterations: int
    initial_equity: float
    median_terminal_equity: float
    p05_terminal_equity: float
    p01_terminal_equity: float
    p95_terminal_equity: float
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    p99_worst_case_drawdown_pct: float
    prob_breach_4pct_dd: float
    prob_breach_6pct_hard_lock: float


class MonteCarloTailRiskSimulator:
    """Vectorized block-bootstrap Monte Carlo engine for portfolio tail-risk stress testing."""

    def __init__(
        self,
        iterations: int = 1000,
        block_size: int = 10,
        stress_friction_bps: float = 1.5,
        seed: int = 42,
    ) -> None:
        if iterations < 100:
            raise ValueError("Monte Carlo iterations should be at least 100.")
        self.iterations = iterations
        self.block_size = max(1, block_size)
        self.stress_friction_bps = max(0.0, stress_friction_bps)
        self._rng = np.random.default_rng(seed)

    def run_from_returns(
        self,
        period_returns: np.ndarray,
        initial_equity: float = 100_000.0,
        trailing_dd_limit_pct: float = 0.04,
        hard_lock_dd_limit_pct: float = 0.06,
    ) -> MonteCarloTailRiskReport:
        """
        Simulate `self.iterations` compounded equity paths via circular block bootstrap.
        `period_returns` may be trade percentage returns or daily/bar percentage returns.
        """
        rets = np.asarray(period_returns, dtype=np.float64)
        if len(rets) == 0:
            return MonteCarloTailRiskReport(
                iterations=self.iterations,
                initial_equity=initial_equity,
                median_terminal_equity=initial_equity,
                p05_terminal_equity=initial_equity,
                p01_terminal_equity=initial_equity,
                p95_terminal_equity=initial_equity,
                median_max_drawdown_pct=0.0,
                p95_max_drawdown_pct=0.0,
                p99_worst_case_drawdown_pct=0.0,
                prob_breach_4pct_dd=0.0,
                prob_breach_6pct_hard_lock=0.0,
            )

        n = len(rets)
        b = min(self.block_size, n)
        num_blocks = int(np.ceil(n / b))

        # Circular array extension so block slices wrap cleanly around the end
        extended = np.concatenate([rets, rets[:b]])
        block_starts = self._rng.integers(0, n, size=(self.iterations, num_blocks))
        offsets = np.arange(b, dtype=np.int64)[None, None, :]
        sampled_indices = (block_starts[:, :, None] + offsets).reshape(self.iterations, num_blocks * b)[:, :n]

        sim_rets = extended[sampled_indices].copy()

        # Apply subtle asymmetric execution-stress noise strictly to active non-zero return steps
        active_mask = np.abs(sim_rets) > 1e-10
        if self.stress_friction_bps > 0.0 and np.any(active_mask):
            friction = (self.stress_friction_bps * 1e-4) * self._rng.exponential(scale=1.0, size=sim_rets.shape)
            sim_rets = np.where(active_mask, sim_rets - friction, sim_rets)

        # Prevent unphysical <-99% single-step return
        sim_rets = np.clip(sim_rets, -0.95, 5.0)

        growth_factors = np.cumprod(1.0 + sim_rets, axis=1)
        eq_paths = np.hstack(
            [
                np.full((self.iterations, 1), initial_equity, dtype=np.float64),
                initial_equity * growth_factors,
            ]
        )

        running_peaks = np.maximum.accumulate(eq_paths, axis=1)
        dd_matrix = (running_peaks - eq_paths) / np.maximum(running_peaks, 1e-12)
        max_dds = np.max(dd_matrix, axis=1)
        terminal_eqs = eq_paths[:, -1]

        return MonteCarloTailRiskReport(
            iterations=self.iterations,
            initial_equity=float(initial_equity),
            median_terminal_equity=float(np.percentile(terminal_eqs, 50)),
            p05_terminal_equity=float(np.percentile(terminal_eqs, 5)),
            p01_terminal_equity=float(np.percentile(terminal_eqs, 1)),
            p95_terminal_equity=float(np.percentile(terminal_eqs, 95)),
            median_max_drawdown_pct=float(np.percentile(max_dds, 50)),
            p95_max_drawdown_pct=float(np.percentile(max_dds, 95)),
            p99_worst_case_drawdown_pct=float(np.percentile(max_dds, 99)),
            prob_breach_4pct_dd=float(np.mean(max_dds > trailing_dd_limit_pct)),
            prob_breach_6pct_hard_lock=float(np.mean(max_dds >= hard_lock_dd_limit_pct)),
        )
