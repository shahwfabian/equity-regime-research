"""Tests for factor computation including no-look-ahead validation."""

import numpy as np
import pandas as pd
import pytest

from equity_regime.factors.momentum import compute_momentum_signal, verify_no_lookahead as mom_nla
from equity_regime.factors.mean_reversion import compute_reversal_signal, verify_no_lookahead as rev_nla
from equity_regime.factors.portfolios import build_portfolios, decile_monotonicity_check
from equity_regime.config import FactorsConfig


@pytest.fixture(scope="module")
def factor_stock_day():
    """Minimal stock panel for factor testing."""
    from equity_regime.data.synthetic import generate
    from equity_regime.config import SyntheticConfig
    cfg = SyntheticConfig()
    cfg.n_stocks = 50
    cfg.n_days = 800
    stock_day, _, _ = generate(cfg, seed=99)
    # Add log_ret and mktcap
    stock_day["log_ret"] = np.log1p(stock_day["ret"].clip(-0.99, 10.0))
    return stock_day


@pytest.fixture(scope="module")
def default_factors_cfg():
    cfg = FactorsConfig()
    cfg.momentum_lookback = 120
    cfg.momentum_skip = 21
    cfg.reversal_lookback = 21
    cfg.n_portfolios = 5
    cfg.weighting = "equal"
    return cfg


def test_momentum_signal_shape(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    assert "mom_signal" in mom.columns
    assert "mom_rank" in mom.columns
    assert "mom_zscore" in mom.columns
    assert len(mom) > 0


def test_momentum_rank_range(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    assert mom["mom_rank"].between(0, 1).all()


def test_momentum_zscore_mean(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    # Cross-sectional z-scores should have mean ~0 per date
    date_means = mom.groupby("date")["mom_zscore"].mean().abs()
    assert date_means.median() < 0.1


def test_momentum_no_lookahead(factor_stock_day, default_factors_cfg):
    """CRITICAL: Perturbation test proves signals at t do not use data at t."""
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    mom_nla(factor_stock_day, mom, default_factors_cfg)  # raises AssertionError if look-ahead found


def test_reversal_signal_shape(factor_stock_day, default_factors_cfg):
    rev = compute_reversal_signal(factor_stock_day, default_factors_cfg)
    assert "rev_signal" in rev.columns
    assert "rev_rank" in rev.columns
    assert len(rev) > 0


def test_reversal_contrarian_sign(factor_stock_day, default_factors_cfg):
    """Reversal signal should be negatively correlated with past return."""
    rev = compute_reversal_signal(factor_stock_day, default_factors_cfg)
    # rev_signal = -past_ret => high rev_signal means low past return
    # Correlation of signal with past_ret should be negative
    past_wide = (
        factor_stock_day.pivot_table(index="date", columns="permno", values="log_ret")
        .sort_index()
        .rolling(default_factors_cfg.reversal_lookback, min_periods=1)
        .sum()
        .shift(1)
        .stack()
        .rename("past_ret")
        .reset_index()
    )
    past_wide.columns = ["date", "permno", "past_ret"]
    merged = rev.merge(past_wide, on=["date", "permno"])
    corr = merged["rev_signal"].corr(merged["past_ret"])
    assert corr < -0.9, f"Expected strong negative corr, got {corr:.3f}"


def test_reversal_no_lookahead(factor_stock_day, default_factors_cfg):
    rev = compute_reversal_signal(factor_stock_day, default_factors_cfg)
    rev_nla(factor_stock_day, rev, default_factors_cfg)


def test_portfolios_ls_series(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    port, ls = build_portfolios(factor_stock_day, mom, "mom_signal", default_factors_cfg, "momentum")
    assert not ls.empty
    assert "ls_ret" in ls.columns


def test_portfolios_decile_count(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    port, ls = build_portfolios(factor_stock_day, mom, "mom_signal", default_factors_cfg, "momentum")
    n = default_factors_cfg.n_portfolios
    assert port["decile"].nunique() <= n


def test_decile_monotonicity_output(factor_stock_day, default_factors_cfg):
    mom = compute_momentum_signal(factor_stock_day, default_factors_cfg)
    port, _ = build_portfolios(factor_stock_day, mom, "mom_signal", default_factors_cfg, "momentum")
    result = decile_monotonicity_check(port, n_portfolios=default_factors_cfg.n_portfolios)
    assert "spearman_corr" in result
    assert "mean_by_decile" in result
    # With injected momentum signal, Spearman corr should be positive
    assert result["spearman_corr"] > 0.0, f"Expected positive Spearman, got {result['spearman_corr']:.3f}"
