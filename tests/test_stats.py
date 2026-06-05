"""Tests for statistical test modules."""

import numpy as np
import pandas as pd
import pytest

from equity_regime.stats.tests import (
    compute_acf, variance_ratio_test, stationarity_tests,
    predictive_regression, regime_interaction_regression,
)
from equity_regime.stats.performance import (
    full_performance_table, train_test_split_with_embargo,
    oos_r_squared, diebold_mariano_test,
    sharpe_ratio, max_drawdown, calmar_ratio,
)


@pytest.fixture
def iid_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)


@pytest.fixture
def trending_returns():
    """Returns with positive autocorrelation (trend)."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0005, 0.01, 1000)
    autocorr = 0.3 * np.concatenate([[0], base[:-1]])
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    return pd.Series(base + autocorr, index=idx)


def test_acf_shape(iid_returns):
    result = compute_acf(iid_returns, nlags=10)
    assert len(result) == 11  # lag 0..10
    assert "acf" in result.columns
    assert "lb_pvalue" in result.columns


def test_acf_lag0_is_one(iid_returns):
    result = compute_acf(iid_returns, nlags=5)
    assert abs(result.iloc[0]["acf"] - 1.0) < 1e-10


def test_variance_ratio_rw(iid_returns):
    """IID returns should have VR close to 1."""
    result = variance_ratio_test(iid_returns, q_values=[2, 5, 10])
    assert not result.empty
    # VR should be close to 1 for IID
    assert (result["vr"] - 1).abs().max() < 0.3


def test_variance_ratio_trending(trending_returns):
    """Trending returns should have VR > 1."""
    result = variance_ratio_test(trending_returns, q_values=[2, 5, 10])
    assert (result["vr"] > 1).any(), "Expected VR > 1 for autocorrelated returns"


def test_stationarity_stationary(iid_returns):
    res = stationarity_tests(iid_returns)
    assert res["is_stationary_adf"], f"Expected stationary, ADF p={res['adf_pvalue']:.4f}"


def test_predictive_regression_output(iid_returns):
    X = pd.DataFrame({"x": iid_returns.shift(1)}, index=iid_returns.index)
    result = predictive_regression(iid_returns, X, nw_lags=3)
    assert "coef" in result.columns
    assert "se_nw" in result.columns
    assert "pvalue" in result.columns
    assert len(result) >= 2  # const + x


def test_regime_interaction_output(iid_returns):
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.standard_normal(len(iid_returns)), index=iid_returns.index)
    regime = pd.Series(rng.integers(0, 2, len(iid_returns)), index=iid_returns.index)
    result = regime_interaction_regression(iid_returns, signal, regime, nw_lags=3)
    assert "coef_table" in result
    assert "wald_stat" in result
    assert "wald_pvalue" in result
    assert isinstance(result["regime_invariance_rejected"], bool)


def test_sharpe_positive(trending_returns):
    sr = sharpe_ratio(trending_returns)
    assert np.isfinite(sr)


def test_max_drawdown_negative(iid_returns):
    mdd = max_drawdown(iid_returns)
    assert mdd <= 0


def test_train_test_split_embargo():
    dates = pd.bdate_range("2010-01-01", "2023-12-31")
    train, test = train_test_split_with_embargo(dates, "2018-01-01", embargo_days=21)
    assert train.max() < pd.Timestamp("2018-01-01")
    assert len(test) > 0
    assert (test.min() - train.max()).days >= 21


def test_oos_r_squared_perfect(iid_returns):
    oos = oos_r_squared(iid_returns, iid_returns)
    assert oos == pytest.approx(1.0, abs=1e-8)


def test_oos_r_squared_random_predictor(iid_returns):
    rng = np.random.default_rng(5)
    noise = pd.Series(rng.standard_normal(len(iid_returns)), index=iid_returns.index)
    oos = oos_r_squared(iid_returns, noise)
    assert oos < 0, "Random predictor should have negative OOS R²"


def test_diebold_mariano(iid_returns):
    rng = np.random.default_rng(7)
    p1 = iid_returns + rng.normal(0, 0.001, len(iid_returns))
    p2 = rng.normal(0, 0.1, len(iid_returns))
    p2 = pd.Series(p2, index=iid_returns.index)
    result = diebold_mariano_test(iid_returns, p1, p2)
    assert "dm_stat" in result
    assert "pvalue" in result
    assert np.isfinite(result["dm_stat"])


def test_full_performance_table(iid_returns, trending_returns):
    strategies = {"iid": iid_returns, "trend": trending_returns}
    tbl = full_performance_table(strategies)
    assert set(["ann_return", "ann_vol", "sharpe", "max_drawdown"]).issubset(tbl.columns)
    assert len(tbl) == 2
