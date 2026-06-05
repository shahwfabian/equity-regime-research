"""Tests for synthetic data generation."""

import numpy as np
import pandas as pd
import pytest

from equity_regime.data.synthetic import generate
from equity_regime.config import SyntheticConfig


@pytest.fixture
def tiny_cfg():
    cfg = SyntheticConfig()
    cfg.n_stocks = 10
    cfg.n_days = 500
    return cfg


def test_generate_shapes(tiny_cfg):
    stock_day, market_day, regime = generate(tiny_cfg, seed=0)
    assert len(stock_day) == tiny_cfg.n_stocks * tiny_cfg.n_days
    assert len(market_day) == tiny_cfg.n_days
    assert len(regime) == tiny_cfg.n_days


def test_generate_columns(tiny_cfg):
    stock_day, market_day, regime = generate(tiny_cfg, seed=0)
    for col in ["date", "permno", "ret", "prc", "mktcap", "vol"]:
        assert col in stock_day.columns, f"Missing column: {col}"
    for col in ["date", "mkt_ret", "rf", "realized_vol", "vix_like"]:
        assert col in market_day.columns, f"Missing column: {col}"


def test_no_null_prices(tiny_cfg):
    stock_day, _, _ = generate(tiny_cfg, seed=0)
    assert stock_day["prc"].notna().all()
    assert (stock_day["prc"] > 0).all()


def test_regime_path_binary(tiny_cfg):
    _, _, regime = generate(tiny_cfg, seed=0)
    assert set(regime.unique()).issubset({0, 1})


def test_reproducible_seed(tiny_cfg):
    s1, m1, r1 = generate(tiny_cfg, seed=42)
    s2, m2, r2 = generate(tiny_cfg, seed=42)
    pd.testing.assert_series_equal(m1["mkt_ret"], m2["mkt_ret"])
    pd.testing.assert_series_equal(r1, r2)


def test_different_seeds_differ(tiny_cfg):
    _, m1, _ = generate(tiny_cfg, seed=1)
    _, m2, _ = generate(tiny_cfg, seed=2)
    assert not m1["mkt_ret"].equals(m2["mkt_ret"])


def test_survivorship_bias_absent(tiny_cfg):
    """All stocks survive for the full panel (synthetic has no survivorship bias)."""
    stock_day, _, _ = generate(tiny_cfg, seed=0)
    counts = stock_day.groupby("permno")["date"].count()
    # All stocks should have exactly n_days rows
    assert (counts == tiny_cfg.n_days).all()


def test_regime_both_states_present(tiny_cfg):
    """Both regimes should be visited with long enough panel."""
    cfg = SyntheticConfig()
    cfg.n_stocks = 5
    cfg.n_days = 2000
    _, _, regime = generate(cfg, seed=0)
    assert 0 in regime.values
    assert 1 in regime.values
