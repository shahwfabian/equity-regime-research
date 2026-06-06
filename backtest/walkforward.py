"""Walk-forward validation with embargo gap.

The splitter generates non-overlapping (train, test) folds such that:

    train_end + embargo_days <= test_start          [INVARIANT]

This prevents any model trained on the training set from incorporating
information inside the test window.  A unit test in :mod:`tests.test_backtest`
asserts the invariant for every fold.

Expanding vs rolling windows
-----------------------------
* ``expanding``: training window grows with each fold (all history used).
* ``rolling``:   fixed-length training window that slides forward.

Typical settings for monthly-rebalanced strategies:
    initial_train_days = 504   (2 years)
    test_days          = 63    (3 months)
    step_days          = 63
    embargo_days       = 21
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    """One walk-forward fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp        # last training date (inclusive)
    embargo_end: pd.Timestamp      # last embargoed date (inclusive)
    test_start: pd.Timestamp       # first test date
    test_end: pd.Timestamp         # last test date (inclusive)


@dataclass
class WalkForwardConfig:
    """Configuration for the walk-forward splitter."""

    mode: str = "expanding"          # "expanding" | "rolling"
    initial_train_days: int = 504
    test_days: int = 63
    step_days: int = 63
    embargo_days: int = 21

    @classmethod
    def from_dict(cls, d: dict) -> "WalkForwardConfig":
        return cls(
            mode=d.get("mode", "expanding"),
            initial_train_days=int(d.get("initial_train_days", 504)),
            test_days=int(d.get("test_days", 63)),
            step_days=int(d.get("step_days", 63)),
            embargo_days=int(d.get("embargo_days", 21)),
        )


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
class WalkForwardSplitter:
    """Generate walk-forward train/test folds with an embargo gap.

    Parameters
    ----------
    dates:
        Full sorted DatetimeIndex of available trading dates.
    cfg:
        Walk-forward configuration.
    """

    def __init__(self, dates: pd.DatetimeIndex, cfg: WalkForwardConfig):
        self.dates = dates.sort_values()
        self.cfg = cfg

    def folds(self) -> List[Fold]:
        """Return all valid folds."""
        return list(self._generate_folds())

    def _generate_folds(self) -> Iterator[Fold]:
        dates = self.dates
        n = len(dates)
        cfg = self.cfg

        train_size = cfg.initial_train_days
        step = cfg.step_days
        test_size = cfg.test_days
        embargo = cfg.embargo_days

        fold_id = 0
        train_start_idx = 0

        while True:
            train_end_idx = train_start_idx + train_size - 1
            if train_end_idx >= n:
                break

            # Embargo window
            test_start_idx = train_end_idx + 1 + embargo
            test_end_idx = test_start_idx + test_size - 1

            if test_start_idx >= n:
                break
            test_end_idx = min(test_end_idx, n - 1)

            train_end = dates[train_end_idx]
            embargo_end = dates[min(train_end_idx + embargo, n - 1)]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            # Verify embargo invariant
            gap_days = (test_start - train_end).days
            assert gap_days > 0, (
                f"Fold {fold_id}: test_start {test_start} <= train_end {train_end}"
            )

            yield Fold(
                fold_id=fold_id,
                train_start=dates[train_start_idx],
                train_end=train_end,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )

            fold_id += 1

            # Advance start
            if cfg.mode == "expanding":
                train_size += step      # grow training window
            else:
                train_start_idx += step  # slide window (rolling)
                train_size = cfg.initial_train_days

            if cfg.mode == "expanding":
                train_start_idx = 0    # always from beginning

    def validate_embargo(self) -> bool:
        """Assert the embargo invariant holds for every fold.

        Returns True if all folds pass; raises AssertionError on first failure.
        """
        for fold in self.folds():
            # Convert to trading-day count for robustness
            train_end_pos = self.dates.get_loc(fold.train_end)
            test_start_pos = self.dates.get_loc(fold.test_start)
            gap_td = test_start_pos - train_end_pos
            assert gap_td > self.cfg.embargo_days, (
                f"Fold {fold.fold_id}: trading-day gap {gap_td} <= "
                f"embargo {self.cfg.embargo_days}.  "
                f"train_end={fold.train_end.date()}, "
                f"test_start={fold.test_start.date()}"
            )
        log.info(
            "[walkforward] Embargo validated: %d folds, all gaps > %d trading days",
            len(self.folds()), self.cfg.embargo_days,
        )
        return True


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Results for one walk-forward fold."""

    fold: Fold
    strategy_name: str
    oos_returns: pd.Series      # daily net-of-cost returns in the test window
    is_returns: pd.Series       # daily net-of-cost returns in the train window


def run_walk_forward(
    engine_factory,
    factor_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    wf_cfg: WalkForwardConfig,
    strategy_name: str = "strategy",
) -> List[FoldResult]:
    """Run a strategy walk-forward across all folds.

    Parameters
    ----------
    engine_factory:
        Callable ``(factor_daily, market_daily) -> BacktestEngine``.
    factor_daily / market_daily:
        Full data tables.
    wf_cfg:
        Walk-forward configuration.
    strategy_name:
        Label for logging.

    Returns
    -------
    List of :class:`FoldResult`, one per fold.
    """
    dates = factor_daily.index.sort_values()
    splitter = WalkForwardSplitter(dates, wf_cfg)
    folds = splitter.folds()

    if len(folds) == 0:
        log.warning(
            "[walkforward] No folds generated for %d dates with initial_train=%d, "
            "test=%d, embargo=%d",
            len(dates), wf_cfg.initial_train_days, wf_cfg.test_days, wf_cfg.embargo_days,
        )
        return []

    log.info(
        "[walkforward] %s: %d folds (%s to %s)",
        strategy_name, len(folds),
        folds[0].test_start.date(), folds[-1].test_end.date(),
    )

    fold_results = []
    for fold in folds:
        engine = engine_factory(factor_daily, market_daily)

        is_run = engine.run(start=fold.train_start, end=fold.train_end)
        oos_run = engine.run(start=fold.test_start, end=fold.test_end)

        fold_results.append(FoldResult(
            fold=fold,
            strategy_name=strategy_name,
            oos_returns=oos_run.net_returns,
            is_returns=is_run.net_returns,
        ))
        log.debug(
            "[walkforward] Fold %d IS=%d days OOS=%d days",
            fold.fold_id, len(is_run.returns), len(oos_run.returns),
        )

    return fold_results


def concatenate_oos_returns(fold_results: List[FoldResult]) -> pd.Series:
    """Concatenate OOS returns from all folds into a single Series."""
    parts = [fr.oos_returns for fr in fold_results if len(fr.oos_returns) > 0]
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts).sort_index()
    # De-duplicate (can happen on boundary days)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined
