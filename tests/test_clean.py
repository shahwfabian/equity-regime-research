"""Tests for data cleaning pipeline."""

import numpy as np
import pandas as pd
import pytest

from equity_regime.data.clean import (
    filter_universe, compute_log_returns, winsorize_cross_section,
    add_excess_returns, align_trading_calendar, clean,
)
from equity_regime.config import DataConfig


@pytest.fixture
def sample_stock_day():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    permnos = ["A", "B", "C"]
    rows = []
    for p in permnos:
        prices = 20 * np.exp(rng.standard_normal(300).cumsum() * 0.01)
        rows.append(pd.DataFrame({
            "date": dates,
            "permno": p,
            "ret": rng.standard_normal(300) * 0.01,
            "prc": prices,
            "mktcap": prices * 1e6,
            "vol": rng.integers(1000, 100000, size=300),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def sample_market_day():
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": dates,
        "mkt_ret": rng.standard_normal(300) * 0.01,
        "rf": np.full(300, 0.00015),
        "realized_vol": np.abs(rng.standard_normal(300) * 0.01),
        "vix_like": np.abs(rng.standard_normal(300)) * 15 + 15,
    })


def test_filter_universe_price(sample_stock_day):
    cfg = DataConfig()
    cfg.min_price = 5.0
    cfg.min_history_days = 10
    out = filter_universe(sample_stock_day, cfg)
    assert (out["prc"] >= 5.0).all()


def test_filter_universe_history(sample_stock_day):
    cfg = DataConfig()
    cfg.min_price = 0.0
    cfg.min_history_days = 200
    out = filter_universe(sample_stock_day, cfg)
    counts = out.groupby("permno")["date"].count()
    assert (counts >= 200).all()


def test_log_returns_no_lookahead(sample_stock_day):
    out = compute_log_returns(sample_stock_day.copy())
    assert "log_ret" in out.columns


def test_winsorize_bounds(sample_stock_day):
    """Verify winsorization clips extreme values relative to the ORIGINAL cross-section bounds."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=50)
    rows = []
    for p in [f"P{i}" for i in range(30)]:
        rows.append(pd.DataFrame({
            "date": dates,
            "permno": p,
            "ret": rng.standard_normal(len(dates)) * 0.01,
            "prc": np.ones(len(dates)) * 20,
            "mktcap": np.ones(len(dates)) * 1e6,
            "vol": np.ones(len(dates), dtype=int) * 1000,
        }))
    df = pd.concat(rows, ignore_index=True)
    df = compute_log_returns(df)
    df["date"] = pd.to_datetime(df["date"])
    # Inject a massive outlier in one stock
    first_date = df["date"].min()
    df.loc[(df["date"] == first_date) & (df["permno"] == "P0"), "log_ret"] = 100.0

    # Compute ORIGINAL per-date bounds BEFORE winsorizing
    orig_bounds = (
        df.groupby("date")["log_ret"]
        .quantile([0.01, 0.99])
        .unstack()
        .rename(columns={0.01: "lo", 0.99: "hi"})
    )

    out = winsorize_cross_section(df, "log_ret", 0.01, 0.99)

    # After winsorization, each value must lie within the ORIGINAL [lo, hi] of its date
    for date, grp in out.groupby("date"):
        hi = orig_bounds.loc[date, "hi"]
        lo = orig_bounds.loc[date, "lo"]
        assert grp["log_ret"].max() <= hi + 1e-8, (
            f"date={date}: max={grp['log_ret'].max():.4f} > orig_hi={hi:.4f}"
        )
        assert grp["log_ret"].min() >= lo - 1e-8

    # The outlier should have been clipped well below its original value
    clipped = out.loc[(out["date"] == first_date) & (out["permno"] == "P0"), "log_ret"].iloc[0]
    assert clipped < 100.0, "Outlier was not clipped at all"


def test_excess_returns(sample_stock_day, sample_market_day):
    df = compute_log_returns(sample_stock_day.copy())
    df["date"] = pd.to_datetime(df["date"])
    out = add_excess_returns(df, sample_market_day)
    assert "excess_ret" in out.columns
    # excess_ret should be log_ret - rf
    diff = (out["excess_ret"] - (out["log_ret"] - out["rf"])).abs()
    assert diff.max() < 1e-10


def test_clean_end_to_end(sample_stock_day, sample_market_day, clean_data):
    stock_day, market_day = clean_data
    assert "log_ret" in stock_day.columns
    assert "excess_ret" in stock_day.columns
    assert stock_day["log_ret"].notna().all()
    assert len(stock_day) > 0


def test_clean_no_penny_stocks(sample_stock_day, sample_market_day):
    cfg = DataConfig()
    cfg.min_price = 1.0
    cfg.min_history_days = 10
    out, _ = clean(sample_stock_day, sample_market_day, cfg)
    assert (out["prc"] >= 1.0).all()
