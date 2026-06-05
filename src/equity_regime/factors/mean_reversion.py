"""Mean-reversion (short-horizon reversal) factor.

Signal = negative of trailing cumulative return over reversal_lookback days.
Signal at t uses data through t-1 only (shift(1) on the rolling sum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import FactorsConfig


def compute_reversal_signal(
    stock_day: pd.DataFrame,
    cfg: FactorsConfig,
) -> pd.DataFrame:
    """
    Compute cross-sectional short-horizon reversal signal.

    Signal at date t = -1 * sum(log_ret[t-lookback : t-1])
    (negative of past short-horizon return — buy losers, sell winners)

    Parameters
    ----------
    stock_day : pd.DataFrame  with columns [date, permno, log_ret]
    cfg       : FactorsConfig

    Returns
    -------
    pd.DataFrame with columns [date, permno, rev_signal, rev_rank, rev_zscore]
    """
    lookback = cfg.reversal_lookback

    assert lookback >= 1, "reversal_lookback must be >= 1"

    wide = (
        stock_day.pivot_table(index="date", columns="permno", values="log_ret")
        .sort_index()
    )

    # Rolling sum over lookback bars, shifted by 1 so signal at t uses [t-lookback, t-1]
    past_ret = wide.rolling(window=lookback, min_periods=max(1, lookback // 2)).sum().shift(1)
    rev_signal = -past_ret  # contrarian: negative of past return

    rev_long = rev_signal.stack().rename("rev_signal").reset_index()
    rev_long.columns = ["date", "permno", "rev_signal"]
    rev_long = rev_long.dropna(subset=["rev_signal"])

    def _rank(x: pd.Series) -> pd.Series:
        return x.rank(pct=True)

    def _zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / (x.std() + 1e-10)

    rev_long["rev_rank"] = rev_long.groupby("date")["rev_signal"].transform(_rank)
    rev_long["rev_zscore"] = rev_long.groupby("date")["rev_signal"].transform(_zscore)

    return rev_long.sort_values(["date", "permno"]).reset_index(drop=True)


def verify_no_lookahead(
    stock_day: pd.DataFrame,
    rev_df: pd.DataFrame,
    cfg: FactorsConfig,
) -> None:
    """Assert reversal signals at date t do not use returns at date t."""
    wide = stock_day.pivot_table(index="date", columns="permno", values="log_ret").sort_index()
    dates = wide.index.tolist()
    permnos = wide.columns.tolist()

    rng = np.random.default_rng(1)
    n_check = min(50, len(dates) - cfg.reversal_lookback - 5)
    sample_dates = rng.choice(
        dates[cfg.reversal_lookback : -5], size=n_check, replace=False
    )

    for t in sample_dates:
        for perm in rng.choice(permnos, size=2, replace=False):
            orig = rev_df.loc[
                (rev_df["date"] == t) & (rev_df["permno"] == perm), "rev_signal"
            ]
            if orig.empty:
                continue

            wide_p = wide.copy()
            wide_p.loc[t, perm] += 99.0

            new_past = (
                wide_p[perm]
                .rolling(window=cfg.reversal_lookback, min_periods=max(1, cfg.reversal_lookback // 2))
                .sum()
                .shift(1)
            )
            new_sig = -new_past.loc[t]

            assert abs(float(orig.iloc[0]) - float(new_sig)) < 1e-8, (
                f"LOOK-AHEAD DETECTED in reversal at date={t}, permno={perm}"
            )
