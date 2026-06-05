"""Data ingestion: load parquet panels or dispatch to synthetic generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from equity_regime.config import Config
from equity_regime.data import synthetic as syn_mod

REQUIRED_STOCK_COLS = {"date", "permno", "ret", "prc", "mktcap"}
REQUIRED_MARKET_COLS = {"date", "mkt_ret", "rf"}


def _validate_stock_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_STOCK_COLS - set(df.columns)
    if missing:
        raise ValueError(f"stock_day panel missing columns: {missing}")
    if df["date"].isnull().any():
        raise ValueError("stock_day has null dates")
    if df["permno"].isnull().any():
        raise ValueError("stock_day has null permno values")


def _validate_market_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_MARKET_COLS - set(df.columns)
    if missing:
        raise ValueError(f"market_day panel missing columns: {missing}")
    if df["date"].isnull().any():
        raise ValueError("market_day has null dates")


def load_stock_panel(path: str | Path) -> pd.DataFrame:
    """Load stock×day parquet panel and validate schema."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    _validate_stock_schema(df)
    return df


def load_market_panel(path: str | Path) -> pd.DataFrame:
    """Load market×day parquet panel and validate schema."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    _validate_market_schema(df)
    return df


def ingest(
    cfg: Config,
    use_synthetic: bool = False,
    stock_panel_path: Optional[str] = None,
    market_panel_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Main entry point for data ingestion.

    Returns
    -------
    stock_day  : pd.DataFrame
    market_day : pd.DataFrame
    true_regime: pd.Series or None  (only for synthetic data)
    """
    if use_synthetic:
        stock_day, market_day, true_regime = syn_mod.generate(
            cfg.synthetic, seed=cfg.run.seed
        )
        return stock_day, market_day, true_regime

    if stock_panel_path is None or market_panel_path is None:
        raise ValueError(
            "Must provide stock_panel_path and market_panel_path "
            "when not using synthetic data"
        )

    stock_day = load_stock_panel(stock_panel_path)
    market_day = load_market_panel(market_panel_path)

    # Filter to config date range
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date)
    stock_day = stock_day[(stock_day["date"] >= start) & (stock_day["date"] <= end)].copy()
    market_day = market_day[(market_day["date"] >= start) & (market_day["date"] <= end)].copy()

    return stock_day, market_day, None
