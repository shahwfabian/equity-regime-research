"""Kenneth French Data Library downloader with local ZIP cache.

Downloads CSV-ZIP files directly from Dartmouth, caches them in data/raw/french/.
Parses the factor CSV inside each ZIP into a clean DataFrame.

French data notes
-----------------
- Returns are already in PERCENT (e.g. 0.21 means 0.21%). We divide by 100.
- Sentinel values: -99.99 and -999 -> NaN -> rows dropped with count logged.
- No split adjustment needed: these are factor returns, not individual stock prices.
- Survivorship bias: controlled at the factor-construction level (French documents
  the full CRSP universe). See V9 in the validation report.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request

import pandas as pd

log = logging.getLogger(__name__)

# Sentinel values used by Ken French for missing data
FRENCH_SENTINELS = {-99.99, -999.0, -99.99000}

# Column name normalisation maps
_DAILY_5F_COLS = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": "rf",
}
_MOM_COLS   = {"Mom   ": "umd", "Mom": "umd", "Mom    ": "umd"}
_STREV_COLS = {"ST_Rev": "st_rev", "ST Rev": "st_rev"}
_LTREV_COLS = {"LT_Rev": "lt_rev", "LT Rev": "lt_rev"}


def _fetch_zip(url: str, cache_path: Path, force: bool = False) -> bytes:
    """Download a ZIP from *url*, caching to *cache_path*. Returns raw bytes."""
    if cache_path.exists() and not force:
        log.info("French cache hit: %s", cache_path.name)
        return cache_path.read_bytes()
    log.info("Downloading French data: %s", url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    elapsed = time.time() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    log.info("Downloaded %d bytes in %.1fs -> %s", len(data), elapsed, cache_path)
    return data


def _parse_french_csv(
    raw_bytes: bytes,
    col_map: dict[str, str],
    freq: str = "D",
) -> pd.DataFrame:
    """
    Parse the first data table from a French factor ZIP.

    French ZIPs contain one CSV file. The CSV has a header block
    (copyright text) then a column-header line that starts with ','
    (because the date index column has no label), followed by data rows.
    Returns are in percent; we divide by 100.
    """
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        raw_text = z.read(csv_name).decode("latin-1")

    lines = raw_text.splitlines()

    # The column header line starts with ',' (no label for the date index column)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(",") and len(line.strip()) > 1:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find column header line in French CSV")

    # Data rows start immediately after the header line
    # Stop at the first blank line (separates tables in annual files)
    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if stripped == "":
            break
        # Data rows begin with a date (digit)
        if stripped[0].isdigit():
            data_lines.append(line)

    header_line = lines[header_idx]
    col_names = [c.strip() for c in header_line.split(",")]
    # col_names[0] is '' (the date index); rest are factor names

    csv_block = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_block), header=0, index_col=0)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names using the provided map
    rename = {}
    for col in df.columns:
        for key, val in col_map.items():
            if col.strip() == key.strip():
                rename[col] = val
                break
    df = df.rename(columns=rename)

    # Parse index as date
    df.index = df.index.astype(str).str.strip()
    if freq == "D":
        df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
    else:
        df.index = pd.to_datetime(df.index, format="%Y%m", errors="coerce")
    df.index.name = "date"
    df = df[~df.index.isna()].sort_index()

    # Convert every column to numeric float64 (handles Arrow-backed string dtypes in pandas 3.x)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Replace sentinels with NaN
    for sentinel in FRENCH_SENTINELS:
        df = df.replace(float(sentinel), float("nan"))

    # Divide by 100 (French stores returns as percentages)
    df = df.astype("float64") / 100.0

    return df


def fetch_daily_factors(
    cfg,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch French daily 5-factor + momentum + ST_Rev + LT_Rev data.

    Returns a DataFrame with columns:
      [mkt_rf, smb, hml, rmw, cma, umd, st_rev, lt_rev, rf]
    """
    cache_dir = Path(cfg.sources.french.cache_dir)
    base_url = cfg.sources.french.base_url

    datasets = [
        (cfg.sources.french.daily_dataset, _DAILY_5F_COLS),
        (cfg.sources.french.momentum_dataset, _MOM_COLS),
        (cfg.sources.french.st_rev_dataset, _STREV_COLS),
        (cfg.sources.french.lt_rev_dataset, _LTREV_COLS),
    ]

    frames = {}
    for dataset, col_map in datasets:
        url = f"{base_url}/{dataset}_CSV.zip"
        cache_path = cache_dir / f"{dataset}.zip"
        raw = _fetch_zip(url, cache_path, force=force)
        df = _parse_french_csv(raw, col_map, freq="D")
        frames[dataset] = df
        log.info(
            "Parsed %s: %d rows (%s to %s), cols=%s",
            dataset, len(df),
            df.index.min().date() if len(df) else "N/A",
            df.index.max().date() if len(df) else "N/A",
            list(df.columns),
        )

    # Build master index = UNION of all series (preserves ragged native starts)
    master_idx = frames[cfg.sources.french.daily_dataset].index
    for dataset, _ in datasets[1:]:
        master_idx = master_idx.union(frames[dataset].index)
    master_idx = master_idx.sort_values()

    # Reindex each frame to master index, then assemble column-wise
    base_key = cfg.sources.french.daily_dataset
    base = frames[base_key].reindex(master_idx)
    for dataset, _ in datasets[1:]:
        other = frames[dataset].reindex(master_idx)
        cols = [c for c in other.columns if c not in base.columns]
        if cols:
            base[cols] = other[cols]
        log.info("Merged %s: added cols %s", dataset, cols)

    # Optional date windowing (None = keep full history)
    if cfg.window.start_date:
        start = pd.Timestamp(cfg.window.start_date)
        base = base[base.index >= start]
    if cfg.window.end_date:
        end = pd.Timestamp(cfg.window.end_date)
        base = base[base.index <= end]

    # Log native start dates per column
    for col in base.columns:
        first_valid = base[col].first_valid_index()
        last_valid  = base[col].last_valid_index()
        nan_count   = base[col].isna().sum()
        log.info(
            "  col %-10s: %s to %s  (%d NaN of %d rows)",
            col,
            first_valid.date() if first_valid else "all-NaN",
            last_valid.date()  if last_valid  else "all-NaN",
            nan_count, len(base),
        )

    log.info(
        "French daily factors: %d rows total (%s to %s), cols=%s",
        len(base),
        base.index.min().date() if len(base) else "N/A",
        base.index.max().date() if len(base) else "N/A",
        list(base.columns),
    )
    return base


def fetch_monthly_factors(
    cfg,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch French monthly 5-factor data."""
    cache_dir = Path(cfg.sources.french.cache_dir)
    base_url = cfg.sources.french.base_url
    dataset = cfg.sources.french.monthly_dataset
    url = f"{base_url}/{dataset}_CSV.zip"
    cache_path = cache_dir / f"{dataset}.zip"

    raw = _fetch_zip(url, cache_path, force=force)
    df = _parse_french_csv(raw, _DAILY_5F_COLS, freq="M")
    log.info("French monthly factors: %d rows, %s to %s, cols=%s",
             len(df),
             df.index.min().date() if len(df) else "N/A",
             df.index.max().date() if len(df) else "N/A",
             list(df.columns))

    if cfg.window.start_date:
        df = df[df.index >= pd.Timestamp(cfg.window.start_date)]
    if cfg.window.end_date:
        df = df[df.index <= pd.Timestamp(cfg.window.end_date)]
    return df
