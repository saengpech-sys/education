"""
Marcos López de Prado Purged K-Fold Cross-Validation with Embargo.

Eliminates serial label-horizon leakage in financial time-series ML by:
1. Purging all training samples whose [event_start_time, event_end_time] interval
   overlaps with the test fold's span [test_start_time, max(test_event_end_time)].
2. Embargoing `ceil(N * embargo_pct)` samples immediately after each test fold.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd


class PurgedKFoldCV:
    """Purged K-Fold splitter compatible with Triple-Barrier event horizons."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02) -> None:
        if n_splits < 2:
            raise ValueError("PurgedKFoldCV requires n_splits >= 2.")
        if not (0.0 <= embargo_pct < 0.25):
            raise ValueError("embargo_pct must be in [0.0, 0.25).")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        event_start_times: Sequence | pd.Series | np.ndarray,
        event_end_times: Sequence | pd.Series | np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        starts = pd.to_datetime(pd.Series(event_start_times), utc=True).to_numpy()
        ends = pd.to_datetime(pd.Series(event_end_times), utc=True).to_numpy()
        n = len(starts)
        if len(ends) != n:
            raise ValueError("event_start_times and event_end_times must have identical length.")
        if n < self.n_splits * 4:
            raise ValueError(f"Insufficient samples ({n}) for {self.n_splits}-fold Purged CV.")
        if (ends < starts).any():
            raise ValueError("Found event_end_time < event_start_time in PurgedKFoldCV.")

        indices = np.arange(n, dtype=np.int64)
        embargo_size = int(math.ceil(n * self.embargo_pct)) if self.embargo_pct > 0 else 0
        fold_slices = np.array_split(indices, self.n_splits)

        for test_idx in fold_slices:
            if len(test_idx) == 0:
                continue
            test_first = int(test_idx[0])
            test_last = int(test_idx[-1])

            test_start_ts = starts[test_first]
            test_max_end_ts = ends[test_idx].max()

            # Overlap check: interval [starts[i], ends[i]] overlaps [test_start_ts, test_max_end_ts]
            overlaps_test = (starts <= test_max_end_ts) & (ends >= test_start_ts)

            # Embargo check: block indices immediately after the test fold
            embargo_mask = np.zeros(n, dtype=bool)
            if embargo_size > 0 and test_last + 1 < n:
                embargo_end = min(n, test_last + 1 + embargo_size)
                embargo_mask[test_last + 1 : embargo_end] = True

            is_test = np.zeros(n, dtype=bool)
            is_test[test_idx] = True

            train_mask = ~(is_test | overlaps_test | embargo_mask)
            train_idx = indices[train_mask]
            yield train_idx, test_idx
