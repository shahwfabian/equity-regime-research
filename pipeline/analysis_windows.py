"""Analysis-window definitions for the full-history equity-regime study.

Because the French factor series have different native start dates, each analysis
must select the date range where ALL its required inputs are non-NaN.  This
module provides:

  1. :func:`resolve_windows`  -- compute the valid date range per analysis.
  2. :func:`select_window`    -- filter a DataFrame to one analysis's valid rows.
  3. :func:`print_coverage`   -- print + save the required coverage table.

Analysis windows (columns required → approximate native start)
--------------------------------------------------------------
  umd_raw       : [umd]                              1926-11-03  (daily)
  reversal_raw  : [st_rev]                           1926-01-26
  momentum_core : [umd, mkt_rf, rf]                 1963-07-01
  reversal_core : [st_rev, mkt_rf, rf]               1963-07-01
  five_factor   : [mkt_rf,smb,hml,rmw,cma,rf]       1963-07-01
  full_factor   : [mkt_rf,smb,hml,rmw,cma,umd,st_rev,rf]  1963-07-01
  vix_regime    : [mkt_rf, rf, vix]                  1990-01-02

The VIX is joined into the market_daily table; to use it the caller should
pass the combined factor_daily + vix series, or market_daily directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default window definitions
# ---------------------------------------------------------------------------
DEFAULT_WINDOWS: Dict[str, List[str]] = {
    "umd_raw":       ["umd"],
    "reversal_raw":  ["st_rev"],
    "momentum_core": ["umd", "mkt_rf", "rf"],
    "reversal_core": ["st_rev", "mkt_rf", "rf"],
    "five_factor":   ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    "full_factor":   ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "st_rev", "rf"],
    "vix_regime":    ["mkt_rf", "rf", "vix"],
}


def resolve_windows(
    factor_daily: pd.DataFrame,
    market_daily: pd.DataFrame | None = None,
    extra_windows: Dict[str, List[str]] | None = None,
) -> Dict[str, Dict]:
    """Compute the valid date range for each analysis window.

    Parameters
    ----------
    factor_daily:
        Full-history factor DataFrame (ragged NaN for pre-1963 5-factor columns).
    market_daily:
        Optional market_daily table (has ``vix`` column).
    extra_windows:
        Additional analysis windows from the YAML config.  Merged with defaults.

    Returns
    -------
    dict mapping window_name -> {required, start, end, n_obs, df}
    """
    windows = dict(DEFAULT_WINDOWS)
    if extra_windows:
        windows.update(extra_windows)

    # Build a merged lookup table for column resolution
    if market_daily is not None and "vix" in market_daily.columns:
        # Add vix to factor_daily for window resolution
        merged = factor_daily.copy()
        merged["vix"] = market_daily["vix"].reindex(merged.index)
    else:
        merged = factor_daily.copy()

    results = {}
    for name, required in windows.items():
        avail = [c for c in required if c in merged.columns]
        missing_cols = [c for c in required if c not in merged.columns]

        if missing_cols:
            log.warning(
                "[analysis_windows] '%s' missing columns %s — window unavailable",
                name, missing_cols,
            )
            results[name] = {
                "required": required,
                "available": avail,
                "missing_cols": missing_cols,
                "start": None, "end": None, "n_obs": 0, "df": pd.DataFrame(),
            }
            continue

        # Valid rows: all required columns non-NaN
        mask = merged[avail].notna().all(axis=1)
        df_window = merged.loc[mask, avail].copy()

        if len(df_window) == 0:
            results[name] = {
                "required": required, "available": avail, "missing_cols": [],
                "start": None, "end": None, "n_obs": 0, "df": pd.DataFrame(),
            }
            continue

        results[name] = {
            "required": required,
            "available": avail,
            "missing_cols": [],
            "start": df_window.index.min(),
            "end":   df_window.index.max(),
            "n_obs": len(df_window),
            "df":    df_window,
        }

    return results


def select_window(
    factor_daily: pd.DataFrame,
    required_cols: List[str],
    market_daily: pd.DataFrame | None = None,
    extra_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Return the subset of *factor_daily* rows where all *required_cols* are non-NaN.

    Also appends *vix* from *market_daily* if ``"vix"`` is in required_cols.

    Parameters
    ----------
    factor_daily:
        Full-history factor DataFrame.
    required_cols:
        Columns that must all be non-NaN.
    market_daily:
        Used to attach ``vix`` if required.
    extra_cols:
        Additional columns to include in the output (beyond required_cols).

    Returns
    -------
    Filtered DataFrame indexed by date.
    """
    df = factor_daily.copy()

    if market_daily is not None and "vix" in required_cols:
        df["vix"] = market_daily["vix"].reindex(df.index)

    avail = [c for c in required_cols if c in df.columns]
    if avail:
        mask = df[avail].notna().all(axis=1)
        df = df.loc[mask]

    keep_cols = list(required_cols)
    if extra_cols:
        keep_cols += [c for c in extra_cols if c not in keep_cols]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def print_coverage(
    windows: Dict[str, Dict],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Print and optionally save the analysis coverage table.

    Parameters
    ----------
    windows:
        Output of :func:`resolve_windows`.
    save_path:
        CSV path; if None the table is only printed.

    Returns
    -------
    pd.DataFrame  with one row per analysis window.
    """
    rows = []
    for name, info in windows.items():
        rows.append({
            "analysis":    name,
            "required":    ", ".join(info["required"]),
            "start":       info["start"].date() if info["start"] else "N/A",
            "end":         info["end"].date()   if info["end"]   else "N/A",
            "n_obs":       info["n_obs"],
            "n_years":     round(info["n_obs"] / 252, 1) if info["n_obs"] else 0,
            "missing_cols": ", ".join(info.get("missing_cols", [])) or "—",
        })
    df = pd.DataFrame(rows)

    header = "\n" + "=" * 80
    header += "\nANALYSIS COVERAGE TABLE"
    header += "\n" + "=" * 80
    log.info(header)
    log.info("\n%s", df.to_string(index=False))
    log.info("=" * 80)
    print(header)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info("Coverage table saved -> %s", save_path)

    return df
