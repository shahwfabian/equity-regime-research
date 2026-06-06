"""VIX data fetcher: FRED primary, Stooq fallback, local cache.

VIX alignment rules:
- Forward-fill at most 1 business day (e.g., holidays), log every fill.
- If a factor trading day has no VIX value after fill, it is left NaN and logged.
- VIX is NOT a return series; no sentinel cleanup needed (FRED is clean).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
_STOOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"


def _fetch_fred(series: str, cache_path: Path) -> pd.Series | None:
    """Download from FRED (no API key required for simple CSV endpoint)."""
    url = _FRED_URL.format(series=series)
    try:
        log.info("Fetching VIX from FRED: %s", url)
        t0 = time.time()
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        elapsed = time.time() - t0
        s = df.iloc[:, 0].rename("vix")
        # FRED uses '.' for missing
        s = pd.to_numeric(s, errors="coerce")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(cache_path)
        log.info("FRED VIX: %d rows in %.1fs, cached to %s", len(s), elapsed, cache_path)
        return s
    except Exception as e:
        log.warning("FRED fetch failed: %s", e)
        return None


def _fetch_stooq(ticker: str, cache_path: Path) -> pd.Series | None:
    """Download VIX from Stooq as fallback."""
    url = _STOOQ_URL.format(ticker=ticker)
    try:
        log.info("Fetching VIX from Stooq: %s", url)
        t0 = time.time()
        df = pd.read_csv(url, index_col="Date", parse_dates=True)
        elapsed = time.time() - t0
        if "Close" not in df.columns:
            log.warning("Stooq response missing 'Close' column: %s", list(df.columns))
            return None
        s = df["Close"].rename("vix")
        s = pd.to_numeric(s, errors="coerce")
        s = s.sort_index()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(cache_path)
        log.info("Stooq VIX: %d rows in %.1fs, cached to %s", len(s), elapsed, cache_path)
        return s
    except Exception as e:
        log.warning("Stooq fetch failed: %s", e)
        return None


def _load_cache(cache_path: Path) -> pd.Series | None:
    if not cache_path.exists():
        return None
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        s = df.iloc[:, 0].rename("vix")
        s = pd.to_numeric(s, errors="coerce")
        log.info("VIX cache hit: %s (%d rows)", cache_path.name, len(s))
        return s
    except Exception as e:
        log.warning("VIX cache read failed: %s", e)
        return None


def _fetch_yfinance_vix(cache_path: Path) -> pd.Series | None:
    """Fetch VIX from yfinance (^VIX) as third-tier fallback."""
    try:
        import yfinance as yf
        log.info("Fetching VIX from yfinance (^VIX)")
        raw = yf.download("^VIX", period="max", progress=False, auto_adjust=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].rename("vix")
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(cache_path)
        log.info("yfinance VIX: %d rows, cached to %s", len(s), cache_path)
        return s
    except Exception as e:
        log.warning("yfinance VIX fetch failed: %s", e)
        return None


def fetch_vix(cfg, force: bool = False) -> pd.Series:
    """
    Fetch VIX index. Order: cache -> FRED -> Stooq -> yfinance(^VIX).

    Returns a pd.Series with DatetimeIndex named 'vix',
    windowed to config dates, NOT yet aligned to factor calendar
    (alignment happens in the clean stage).
    """
    cache_dir = Path(cfg.sources.vix.cache_dir)
    cache_path = cache_dir / "vix.csv"

    vix = None
    if not force:
        vix = _load_cache(cache_path)

    if vix is None:
        vix = _fetch_fred(cfg.sources.vix.fred_series, cache_path)
    if vix is None:
        vix = _fetch_stooq(cfg.sources.vix.stooq_ticker, cache_path)
    if vix is None:
        vix = _fetch_yfinance_vix(cache_path)

    if vix is None:
        raise RuntimeError("Could not fetch VIX from FRED, Stooq, or yfinance")

    # Optional windowing (None = full available history)
    if cfg.window.start_date:
        vix = vix[vix.index >= pd.Timestamp(cfg.window.start_date)]
    if cfg.window.end_date:
        vix = vix[vix.index <= pd.Timestamp(cfg.window.end_date)]
    vix = vix.sort_index()

    log.info(
        "VIX ready: %d observations, %s to %s, NaN count=%d",
        len(vix), vix.index.min().date(), vix.index.max().date(), vix.isna().sum(),
    )
    return vix


def align_vix_to_factor_dates(
    vix: pd.Series,
    factor_dates: pd.DatetimeIndex,
    ffill_limit: int = 1,
) -> pd.Series:
    """
    Align VIX to the factor trading calendar.

    Forward-fill at most *ffill_limit* day(s) (handles holidays where VIX
    is not published). Every filled cell is logged. Returns a Series
    indexed by factor_dates.
    """
    # Reindex to factor dates
    vix_aligned = vix.reindex(factor_dates)

    # Count and log forward fills
    missing_before = vix_aligned.isna().sum()
    vix_filled = vix_aligned.ffill(limit=ffill_limit)
    missing_after = vix_filled.isna().sum()
    n_filled = int(missing_before - missing_after)

    if n_filled > 0:
        filled_dates = vix_aligned.index[vix_aligned.isna() & vix_filled.notna()]
        log.info(
            "VIX alignment: forward-filled %d day(s) (limit=%d). Dates: %s",
            n_filled, ffill_limit,
            [str(d.date()) for d in filled_dates[:10]],
        )

    if missing_after > 0:
        unmatched = vix_filled.index[vix_filled.isna()]
        log.warning(
            "VIX: %d factor trading day(s) have no VIX value after ffill. "
            "First 5: %s",
            missing_after,
            [str(d.date()) for d in unmatched[:5]],
        )

    coverage_pct = (vix_filled.notna().sum() / len(factor_dates)) * 100
    log.info("VIX coverage: %.1f%% of factor trading days", coverage_pct)

    return vix_filled
