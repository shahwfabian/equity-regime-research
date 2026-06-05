"""Decile portfolio sorts and long-short return series construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import FactorsConfig


def _assign_deciles(signal: pd.Series, n_portfolios: int = 10) -> pd.Series:
    """Assign cross-sectional decile ranks (1=bottom, n=top)."""
    return pd.qcut(signal, q=n_portfolios, labels=False, duplicates="drop") + 1


def build_portfolios(
    stock_day: pd.DataFrame,
    signal_df: pd.DataFrame,
    signal_col: str,
    cfg: FactorsConfig,
    label: str = "factor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build value- or equal-weighted decile portfolios and long-short series.

    Parameters
    ----------
    stock_day  : pd.DataFrame  with [date, permno, log_ret, mktcap]
    signal_df  : pd.DataFrame  with [date, permno, <signal_col>]
    signal_col : str           column name for the raw signal
    cfg        : FactorsConfig
    label      : str           prefix for output column names

    Returns
    -------
    port_rets  : pd.DataFrame  columns = [date, decile, port_ret]
    ls_series  : pd.DataFrame  columns = [date, ls_ret]  (decile 10 - decile 1)
    """
    n_p = cfg.n_portfolios

    # Merge signal with next-day returns (signal at t -> return at t+1)
    # Signal already uses data through t-1 (no look-ahead), so we use same-day return
    # i.e. signal at t predicts ret at t (which is the return on day t before close)
    base = stock_day[["date", "permno", "log_ret", "mktcap"]].copy()
    merged = base.merge(
        signal_df[["date", "permno", signal_col]],
        on=["date", "permno"],
        how="inner",
    )
    merged = merged.dropna(subset=[signal_col, "log_ret", "mktcap"])

    def _port_date(grp: pd.DataFrame) -> pd.DataFrame:
        date_val = grp["date"].iloc[0]
        if len(grp) < n_p:
            return pd.DataFrame()
        try:
            grp = grp.copy()
            grp["decile"] = _assign_deciles(grp[signal_col], n_p)
        except ValueError:
            return pd.DataFrame()

        rows = []
        for dec in range(1, n_p + 1):
            sub = grp[grp["decile"] == dec]
            if sub.empty:
                continue
            if cfg.weighting == "value":
                w = sub["mktcap"] / sub["mktcap"].sum()
                ret = (w * sub["log_ret"]).sum()
            else:  # equal
                ret = sub["log_ret"].mean()
            rows.append({"date": date_val, "decile": dec, "port_ret": ret})
        return pd.DataFrame(rows)

    result_frames = []
    for date_val, grp in merged.groupby("date"):
        res = _port_date(grp)
        if not res.empty:
            result_frames.append(res)

    if not result_frames:
        port_rets = pd.DataFrame(columns=["date", "decile", "port_ret"])
        ls_series = pd.DataFrame(columns=["date", "ls_ret"])
        return port_rets, ls_series

    grouped = pd.concat(result_frames, ignore_index=True)
    port_rets = grouped[["date", "decile", "port_ret"]].copy()

    # Long-short: top decile minus bottom decile
    top = port_rets[port_rets["decile"] == n_p].set_index("date")["port_ret"]
    bot = port_rets[port_rets["decile"] == 1].set_index("date")["port_ret"]
    ls = (top - bot).dropna().rename("ls_ret").reset_index()
    ls.columns = ["date", "ls_ret"]

    return port_rets, ls


def decile_monotonicity_check(port_rets: pd.DataFrame, n_portfolios: int = 10) -> dict:
    """
    Check whether average portfolio returns are monotone across deciles.

    Returns a dict with 'mean_by_decile', 'is_monotone_increasing', 'spearman_corr'.
    """
    mean_by_dec = (
        port_rets.groupby("decile")["port_ret"].mean().sort_index()
    )
    deciles = mean_by_dec.index.values
    rets = mean_by_dec.values

    from scipy.stats import spearmanr
    rho, pval = spearmanr(deciles, rets)

    diffs = np.diff(rets)
    is_mono = bool(np.all(diffs >= 0))

    return {
        "mean_by_decile": mean_by_dec,
        "is_monotone_increasing": is_mono,
        "spearman_corr": rho,
        "spearman_pval": pval,
    }
