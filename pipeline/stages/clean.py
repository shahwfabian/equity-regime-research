"""Clean stage: missing-value rules, sentinel removal, calendar alignment, type coercion.

Rules (never silently impute returns):
  1. French sentinels (-99.99, -999) -> already replaced in source layer -> drop remaining NaN rows.
  2. Date dedup + sort + assert monotonic.
  3. Align VIX to factor trading calendar; ffill max 1 day.
  4. Returns: if missing after sentinel removal, DROP row and log count. No imputation.
  5. Log every modification with before/after row counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from pipeline.stages.extract import ExtractResult

log = logging.getLogger(__name__)

RETURN_COLS_DAILY = ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "st_rev", "lt_rev"]
RETURN_COLS_MONTHLY = ["mkt_rf", "smb", "hml", "rmw", "cma"]


@dataclass
class CleanResult:
    factor_daily: pd.DataFrame
    factor_monthly: pd.DataFrame
    market_daily: pd.DataFrame   # factors + vix aligned
    stocks: Optional[pd.DataFrame] = None


def _clean_factor_frame(
    df: pd.DataFrame,
    return_cols: list[str],
    label: str,
    sentinel_values: tuple[float, ...] = (-99.99, -999.0),
) -> pd.DataFrame:
    """
    Apply French-specific cleaning to a factor DataFrame.

    Steps:
      1. Replace any remaining sentinel values with NaN.
      2. Parse + sort + deduplicate date index.
      3. Assert monotonic dates.
      4. Drop any rows with NaN in return columns (never impute).
      5. Coerce all return columns to float64.
    """
    n_in = len(df)

    # 1. Sentinel -> NaN (belt-and-suspenders; source layer already did this)
    n_sentinel = 0
    for col in df.columns:
        for sv in sentinel_values:
            mask = df[col] == sv
            n_sentinel += int(mask.sum())
            df.loc[mask, col] = float("nan")
    if n_sentinel:
        log.warning("[clean] %s: replaced %d sentinel values with NaN", label, n_sentinel)

    # 2. Sort + deduplicate
    df = df.sort_index()
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        log.warning("[clean] %s: dropped %d duplicate dates", label, n_dupes)
        df = df[~df.index.duplicated(keep="first")]

    # 3. Assert monotonic
    assert df.index.is_monotonic_increasing, f"{label}: date index is not monotonic after dedup"

    # 4. Drop NaN return rows (no imputation)
    present_ret_cols = [c for c in return_cols if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=present_ret_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning(
            "[clean] %s: dropped %d rows with NaN returns (no imputation)", label, n_dropped
        )

    # 5. Coerce all numeric cols to float64
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].astype("float64")

    log.info(
        "[clean] %s: %d rows in -> %d rows out (dropped %d)",
        label, n_in, len(df), n_in - len(df),
    )
    return df


def _clean_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean yfinance stock data.
    Drop rows where adj_close or ret is NaN.
    """
    n_in = len(df)
    df = df.dropna(subset=["adj_close"])
    n_dropped = n_in - len(df)
    if n_dropped:
        log.warning("[clean] stocks: dropped %d rows with NaN adj_close", n_dropped)

    # Ensure correct dtypes
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce").astype("float64")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info("[clean] stocks: %d rows in -> %d rows out", n_in, len(df))
    return df


def clean(
    extracted: ExtractResult,
    cfg,
) -> CleanResult:
    """
    Full cleaning pass over all extracted data.

    Returns CleanResult with cleaned DataFrames.
    """
    from pipeline.sources.vix import align_vix_to_factor_dates

    # --- Factor daily ---
    factor_daily = _clean_factor_frame(
        extracted.factor_daily.copy(),
        RETURN_COLS_DAILY,
        label="factor_daily",
    )

    # --- Factor monthly ---
    factor_monthly = _clean_factor_frame(
        extracted.factor_monthly.copy(),
        RETURN_COLS_MONTHLY,
        label="factor_monthly",
    )

    # --- VIX alignment ---
    vix_aligned = align_vix_to_factor_dates(
        extracted.vix_raw,
        factor_daily.index,
        ffill_limit=cfg.validation.vix_ffill_limit,
    )

    # --- Build market_daily: factor_daily + aligned VIX ---
    market_daily = factor_daily.copy()
    market_daily["vix"] = vix_aligned

    # --- Stocks (optional) ---
    stocks = None
    if extracted.stocks is not None and not extracted.stocks.empty:
        stocks = _clean_stocks(extracted.stocks.copy())

    log.info(
        "[clean] Done. factor_daily=%d rows, factor_monthly=%d rows, market_daily=%d rows",
        len(factor_daily), len(factor_monthly), len(market_daily),
    )
    return CleanResult(
        factor_daily=factor_daily,
        factor_monthly=factor_monthly,
        market_daily=market_daily,
        stocks=stocks,
    )
