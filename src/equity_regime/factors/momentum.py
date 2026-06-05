"""Momentum factor: cumulative return over [t-lookback, t-skip].

CRITICAL: all signals at time t use only data through t-1.
The lookback window is [t-lookback, t-skip], where t-skip is at least 1 day
before signal date t, ensuring zero look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import FactorsConfig


def compute_momentum_signal(
    stock_day: pd.DataFrame,
    cfg: FactorsConfig,
) -> pd.DataFrame:
    """
    Compute cross-sectional momentum signal.

    Signal at date t = cumulative log return from t-lookback to t-skip (exclusive).
    Both endpoints are strictly before t, so there is NO look-ahead.

    Parameters
    ----------
    stock_day : pd.DataFrame  with columns [date, permno, log_ret]
    cfg       : FactorsConfig

    Returns
    -------
    pd.DataFrame with columns [date, permno, mom_signal, mom_rank, mom_zscore]
    """
    lookback = cfg.momentum_lookback
    skip = cfg.momentum_skip

    assert skip >= 1, "momentum_skip must be >= 1 to prevent look-ahead"
    assert lookback > skip, "momentum_lookback must be > momentum_skip"

    # Pivot to wide: dates x permnos
    wide = (
        stock_day.pivot_table(index="date", columns="permno", values="log_ret")
        .sort_index()
    )
    dates = wide.index

    # Cumulative return from [t-lookback, t-skip)
    # Rolling sum ending at t-skip-1 (i.e., skip days ago), spanning lookback-skip bars
    window = lookback - skip

    # rolling sum over `window` bars, then shift by `skip` so signal at t uses data ending t-skip
    cum_ret = wide.rolling(window=window, min_periods=max(1, window // 2)).sum().shift(skip)

    # LOOK-AHEAD ASSERTION: verify signal at any date t does NOT use return at t
    # The shift(skip) guarantees cum_ret.iloc[t] uses wide.iloc[t-skip-window+1 : t-skip+1]
    # Since skip >= 1, the latest bar used is t-skip <= t-1. ✓

    mom_long = cum_ret.stack().rename("mom_signal").reset_index()
    mom_long.columns = ["date", "permno", "mom_signal"]
    mom_long = mom_long.dropna(subset=["mom_signal"])

    # Cross-sectional ranks and z-scores
    def _rank(x: pd.Series) -> pd.Series:
        return x.rank(pct=True)

    def _zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / (x.std() + 1e-10)

    mom_long["mom_rank"] = mom_long.groupby("date")["mom_signal"].transform(_rank)
    mom_long["mom_zscore"] = mom_long.groupby("date")["mom_signal"].transform(_zscore)

    return mom_long.sort_values(["date", "permno"]).reset_index(drop=True)


def verify_no_lookahead(
    stock_day: pd.DataFrame,
    mom_df: pd.DataFrame,
    cfg: FactorsConfig,
) -> None:
    """
    Assert that momentum signals at date t do not use returns at date t.

    Randomly samples 50 (date, permno) pairs and checks that perturbing
    the return at date t does NOT change the signal at date t.
    """
    wide = stock_day.pivot_table(index="date", columns="permno", values="log_ret").sort_index()
    dates = wide.index.tolist()
    permnos = wide.columns.tolist()

    rng = np.random.default_rng(0)
    n_check = min(50, len(dates) - cfg.momentum_lookback - 5)
    sample_dates = rng.choice(
        dates[cfg.momentum_lookback : -5], size=n_check, replace=False
    )

    for t in sample_dates:
        for perm in rng.choice(permnos, size=2, replace=False):
            orig_sig = mom_df.loc[
                (mom_df["date"] == t) & (mom_df["permno"] == perm), "mom_signal"
            ]
            if orig_sig.empty:
                continue

            # Perturb the return at date t
            wide_perturbed = wide.copy()
            wide_perturbed.loc[t, perm] += 99.0

            # Recompute signal for this permno
            window = cfg.momentum_lookback - cfg.momentum_skip
            cum = (
                wide_perturbed[perm]
                .rolling(window=window, min_periods=max(1, window // 2))
                .sum()
                .shift(cfg.momentum_skip)
            )
            new_sig = cum.loc[t]

            assert abs(float(orig_sig.iloc[0]) - float(new_sig)) < 1e-8, (
                f"LOOK-AHEAD DETECTED at date={t}, permno={perm}: "
                f"orig={float(orig_sig.iloc[0]):.6f}, perturbed={float(new_sig):.6f}"
            )
