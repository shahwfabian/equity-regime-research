"""Data cleaning: universe filters, log returns, winsorization, excess returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import DataConfig


def filter_universe(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Apply price floor and minimum history filters."""
    # Price filter
    df = df[df["prc"] >= cfg.min_price].copy()

    # Minimum history filter: keep only stocks with >= min_history_days observations
    counts = df.groupby("permno")["date"].count()
    valid = counts[counts >= cfg.min_history_days].index
    df = df[df["permno"].isin(valid)].copy()

    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_ret column; prefer price-derived if ret is available, else compute from prc."""
    df = df.copy()
    if "ret" in df.columns:
        # Clip extreme raw returns before log transform
        ret_clipped = df["ret"].clip(-0.99, 10.0)
        df["log_ret"] = np.log1p(ret_clipped)
    else:
        df = df.sort_values(["permno", "date"])
        df["log_ret"] = (
            df.groupby("permno")["prc"]
            .transform(lambda x: np.log(x / x.shift(1)))
        )
    return df


def winsorize_cross_section(
    df: pd.DataFrame,
    col: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    """Cross-sectional winsorization at each date."""
    df = df.copy()

    def _winsor(x: pd.Series) -> pd.Series:
        lo, hi = x.quantile([low, high])
        return x.clip(lo, hi)

    df[col] = df.groupby("date")[col].transform(_winsor)
    return df


def add_excess_returns(df: pd.DataFrame, market_day: pd.DataFrame) -> pd.DataFrame:
    """Add excess_ret = log_ret - rf, joined from market_day."""
    rf_map = market_day.set_index("date")["rf"]
    df = df.copy()
    df["rf"] = df["date"].map(rf_map).fillna(0.0)
    df["excess_ret"] = df["log_ret"] - df["rf"]
    return df


def align_trading_calendar(
    df: pd.DataFrame, market_day: pd.DataFrame
) -> pd.DataFrame:
    """Keep only dates that appear in market_day (trading calendar)."""
    valid_dates = set(market_day["date"])
    return df[df["date"].isin(valid_dates)].copy()


def clean(
    stock_day: pd.DataFrame,
    market_day: pd.DataFrame,
    cfg: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full cleaning pipeline.

    Returns
    -------
    stock_day_clean  : pd.DataFrame  with log_ret, excess_ret, mktcap
    market_day_clean : pd.DataFrame  unchanged but sorted/typed
    """
    # Ensure datetime
    stock_day = stock_day.copy()
    market_day = market_day.copy()
    stock_day["date"] = pd.to_datetime(stock_day["date"])
    market_day["date"] = pd.to_datetime(market_day["date"])

    # Sort
    stock_day = stock_day.sort_values(["permno", "date"]).reset_index(drop=True)
    market_day = market_day.sort_values("date").reset_index(drop=True)

    # Align to trading calendar
    stock_day = align_trading_calendar(stock_day, market_day)

    # Universe filters
    stock_day = filter_universe(stock_day, cfg)

    # Log returns
    stock_day = compute_log_returns(stock_day)

    # Winsorize log returns cross-sectionally
    stock_day = winsorize_cross_section(
        stock_day, "log_ret", cfg.winsor_low, cfg.winsor_high
    )

    # Excess returns
    stock_day = add_excess_returns(stock_day, market_day)

    # Null audit
    n_null = stock_day["log_ret"].isnull().sum()
    if n_null > 0:
        pct = n_null / len(stock_day) * 100
        if pct > 5:
            raise ValueError(f"Too many null log_ret after cleaning: {pct:.1f}%")
        stock_day = stock_day.dropna(subset=["log_ret"])

    return stock_day, market_day
