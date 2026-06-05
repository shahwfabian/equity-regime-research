"""Decile portfolio sorts and long-short return series construction.

Fully vectorised: no Python-level loops over dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from equity_regime.config import FactorsConfig


def build_portfolios(
    stock_day: pd.DataFrame,
    signal_df: pd.DataFrame,
    signal_col: str,
    cfg: FactorsConfig,
    label: str = "factor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build value- or equal-weighted decile portfolios and long-short series.

    Fully vectorised — no Python for-loop over dates.

    Parameters
    ----------
    stock_day  : pd.DataFrame  with [date, permno, log_ret, mktcap]
    signal_df  : pd.DataFrame  with [date, permno, <signal_col>]
    signal_col : str           column name for the raw signal
    cfg        : FactorsConfig
    label      : str           prefix (unused, kept for API compat)

    Returns
    -------
    port_rets  : pd.DataFrame  [date, decile, port_ret]
    ls_series  : pd.DataFrame  [date, ls_ret]  (top decile − bottom decile)
    """
    n_p = cfg.n_portfolios

    base = stock_day[["date", "permno", "log_ret", "mktcap"]].copy()
    merged = base.merge(
        signal_df[["date", "permno", signal_col]],
        on=["date", "permno"],
        how="inner",
    ).dropna(subset=[signal_col, "log_ret", "mktcap"])

    if merged.empty:
        return (
            pd.DataFrame(columns=["date", "decile", "port_ret"]),
            pd.DataFrame(columns=["date", "ls_ret"]),
        )

    # --- Vectorised decile assignment via cross-sectional rank ---
    # pct rank within each date → multiply by n_p → ceil → clip to [1, n_p]
    merged["_rank"] = merged.groupby("date")[signal_col].rank(pct=True, method="first")
    merged["decile"] = np.ceil(merged["_rank"] * n_p).clip(1, n_p).astype(int)

    # --- Vectorised weighted return per (date, decile) ---
    if cfg.weighting == "value":
        # weight = mktcap / sum(mktcap) within each (date, decile)
        grp = merged.groupby(["date", "decile"])
        mktcap_sum = grp["mktcap"].transform("sum")
        merged["_w"] = merged["mktcap"] / mktcap_sum
        merged["_wret"] = merged["_w"] * merged["log_ret"]
        port_rets = (
            merged.groupby(["date", "decile"])["_wret"]
            .sum()
            .rename("port_ret")
            .reset_index()
        )
    else:
        port_rets = (
            merged.groupby(["date", "decile"])["log_ret"]
            .mean()
            .rename("port_ret")
            .reset_index()
        )

    # --- Long-short: top decile − bottom decile ---
    top = port_rets[port_rets["decile"] == n_p].set_index("date")["port_ret"]
    bot = port_rets[port_rets["decile"] == 1].set_index("date")["port_ret"]
    ls = (top - bot).dropna().rename("ls_ret").reset_index()
    ls.columns = ["date", "ls_ret"]

    return port_rets[["date", "decile", "port_ret"]], ls


def decile_monotonicity_check(port_rets: pd.DataFrame, n_portfolios: int = 10) -> dict:
    """
    Check whether average portfolio returns are monotone across deciles.

    Returns dict with mean_by_decile, is_monotone_increasing, spearman_corr, spearman_pval.
    """
    mean_by_dec = (
        port_rets.groupby("decile")["port_ret"].mean().sort_index()
    )
    deciles = mean_by_dec.index.values
    rets = mean_by_dec.values

    rho, pval = spearmanr(deciles, rets)
    diffs = np.diff(rets)
    is_mono = bool(np.all(diffs >= 0))

    return {
        "mean_by_decile": mean_by_dec,
        "is_monotone_increasing": is_mono,
        "spearman_corr": float(rho),
        "spearman_pval": float(pval),
    }
