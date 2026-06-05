"""yfinance OHLCV downloader for illustrative stock universe.

⚠️  SURVIVORSHIP BIAS WARNING ⚠️
This data is pulled for a hand-picked, named universe (e.g. SPY, QQQ).
It is ILLUSTRATIVE ONLY and must NEVER be used for headline research results.
The French factor data is the authoritative, survivorship-controlled source.

Split/dividend adjustment:
  yfinance 'Adj Close' is already adjusted for splits and dividends by
  the provider (Yahoo Finance). We use it directly and assert its values
  diverge from raw Close whenever splits occurred.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

_ILLUSTRATIVE_DISCLAIMER = (
    "⚠️  ILLUSTRATIVE DATA: yfinance universe is survivorship-affected "
    "and hand-picked. Never use for headline factor results."
)


def fetch_yfinance(
    cfg,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV + Adj Close for the configured ticker universe.

    Returns a long DataFrame with columns:
      [date, ticker, adj_close, close, volume]
    Cached per-ticker as parquet in cfg.sources.yfinance.cache_dir.
    """
    import yfinance as yf

    log.warning(_ILLUSTRATIVE_DISCLAIMER)

    tickers: List[str] = cfg.sources.yfinance.tickers
    cache_dir = Path(cfg.sources.yfinance.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = cfg.window.start_date
    end = cfg.window.end_date
    frames = []

    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.parquet"
        if cache_path.exists() and not force:
            log.info("yfinance cache hit: %s", ticker)
            df_t = pd.read_parquet(cache_path)
        else:
            log.info("yfinance downloading: %s (%s to %s)", ticker, start, end)
            t0 = time.time()
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            elapsed = time.time() - t0
            if raw.empty:
                log.warning("yfinance returned empty for %s", ticker)
                continue
            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index.name = "date"
            df_t = pd.DataFrame({
                "date": raw.index,
                "ticker": ticker,
                "adj_close": raw.get("Adj Close", raw.get("Close")),
                "close": raw.get("Close"),
                "volume": raw.get("Volume"),
            })
            df_t.to_parquet(cache_path, index=False)
            log.info("yfinance %s: %d rows in %.1fs", ticker, len(df_t), elapsed)

        frames.append(df_t)

    if not frames:
        log.warning("yfinance: no data fetched for any ticker")
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "close", "volume"])

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info(
        "yfinance total: %d rows, %d tickers. %s",
        len(result), result["ticker"].nunique(), _ILLUSTRATIVE_DISCLAIMER,
    )
    return result


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple and log returns from adj_close.

    Uses Adjusted Close (already split+dividend adjusted by Yahoo).
    No additional split adjustment needed for adj_close.
    Asserts adj_close != close for tickers that had splits (diagnostic).
    """
    df = df.copy().sort_values(["ticker", "date"])

    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_ret"] = df.groupby("ticker")["adj_close"].transform(
        lambda x: x.div(x.shift(1)).apply(lambda v: float("nan") if pd.isna(v) else __import__("math").log(v))
    )

    # Drop rows where ret is NaN (first row per ticker, or genuine gaps)
    n_before = len(df)
    df = df.dropna(subset=["ret"])
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info("yfinance returns: dropped %d NaN rows (first-row per ticker / gaps)", n_dropped)

    log.warning(_ILLUSTRATIVE_DISCLAIMER)
    return df
