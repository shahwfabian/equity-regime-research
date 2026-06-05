"""Extract stage: pull from all configured sources, cache raw data.

Each source returns its raw DataFrame; no cleaning happens here.
Every call logs row counts and timing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    factor_daily: pd.DataFrame
    factor_monthly: pd.DataFrame
    vix_raw: pd.Series
    stocks: Optional[pd.DataFrame] = None   # None when yfinance disabled


def extract(cfg, force: bool = False) -> ExtractResult:
    """
    Run all enabled sources and return raw DataFrames.

    Parameters
    ----------
    cfg   : ETLConfig
    force : bool  re-download even if cache exists

    Returns
    -------
    ExtractResult
    """
    t_total = time.time()

    # --- French factors ---
    from pipeline.sources.french import fetch_daily_factors, fetch_monthly_factors

    t0 = time.time()
    factor_daily = fetch_daily_factors(cfg, force=force)
    log.info(
        "[extract] French daily: %d rows, %d cols in %.1fs",
        len(factor_daily), factor_daily.shape[1], time.time() - t0,
    )

    t0 = time.time()
    factor_monthly = fetch_monthly_factors(cfg, force=force)
    log.info(
        "[extract] French monthly: %d rows, %d cols in %.1fs",
        len(factor_monthly), factor_monthly.shape[1], time.time() - t0,
    )

    # --- VIX ---
    from pipeline.sources.vix import fetch_vix

    t0 = time.time()
    vix_raw = fetch_vix(cfg, force=force)
    log.info(
        "[extract] VIX: %d rows in %.1fs",
        len(vix_raw), time.time() - t0,
    )

    # --- Optional yfinance ---
    stocks = None
    if cfg.sources.yfinance.enabled:
        from pipeline.sources.yfinance_stocks import fetch_yfinance
        t0 = time.time()
        stocks = fetch_yfinance(cfg, force=force)
        log.info(
            "[extract] yfinance stocks: %d rows, %d tickers in %.1fs",
            len(stocks), stocks["ticker"].nunique() if not stocks.empty else 0,
            time.time() - t0,
        )

    log.info("[extract] Total extract time: %.1fs", time.time() - t_total)
    return ExtractResult(
        factor_daily=factor_daily,
        factor_monthly=factor_monthly,
        vix_raw=vix_raw,
        stocks=stocks,
    )
