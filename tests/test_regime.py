"""Tests for regime detection modules."""

import numpy as np
import pandas as pd
import pytest

from equity_regime.regime.rule_based import compute_rule_based_regimes, label_vol_regime, label_trend_regime
from equity_regime.config import RegimeConfig


@pytest.fixture(scope="module")
def sample_market():
    rng = np.random.default_rng(42)
    n = 1000
    dates = pd.bdate_range("2015-01-01", periods=n)
    # Two-state returns: calm first half, turbulent second half
    rets = np.concatenate([
        rng.normal(0.0005, 0.006, n // 2),
        rng.normal(-0.0003, 0.018, n - n // 2),
    ])
    vol = pd.Series(rets).rolling(21).std().bfill().values
    return pd.DataFrame({
        "date": dates,
        "mkt_ret": rets,
        "rf": np.full(n, 0.00015),
        "realized_vol": vol,
        "vix_like": vol * np.sqrt(252) * 100,
    })


def test_vol_regime_binary(sample_market):
    cfg = RegimeConfig()
    out = label_vol_regime(sample_market, cfg)
    assert set(out["vol_regime"].unique()).issubset({0, 1})


def test_trend_regime_binary(sample_market):
    out = label_trend_regime(sample_market)
    assert set(out["trend_regime"].unique()).issubset({0, 1})


def test_rule_based_combined(sample_market):
    cfg = RegimeConfig()
    out = compute_rule_based_regimes(sample_market, cfg)
    assert "combined_regime" in out.columns
    assert set(out["combined_regime"].unique()).issubset({0, 1})


def test_vol_regime_detects_high_vol(sample_market):
    """Second half (high vol) should have more vol_regime=1 observations."""
    cfg = RegimeConfig()
    out = label_vol_regime(sample_market, cfg)
    n = len(out)
    first = out.iloc[: n // 2]["vol_regime"].mean()
    second = out.iloc[n // 2 :]["vol_regime"].mean()
    assert second > first, f"Expected second half more turbulent, got first={first:.2f}, second={second:.2f}"


def test_markov_filtered_smoothed_differ(sample_market):
    """Filtered and smoothed probabilities must differ (test for correct separation)."""
    from equity_regime.regime.markov import fit_markov_model
    cfg = RegimeConfig()
    result = fit_markov_model(sample_market, cfg, seed=42)
    df = result.market_day
    assert "filtered_prob_state0" in df.columns
    assert "smoothed_prob_state0" in df.columns
    # They must differ (smoothed uses future data, filtered does not)
    diff = (df["filtered_prob_state0"] - df["smoothed_prob_state0"]).abs().mean()
    assert diff > 1e-6, "Filtered and smoothed probs are identical — check implementation"


def test_markov_probabilities_sum_to_one(sample_market):
    from equity_regime.regime.markov import fit_markov_model
    cfg = RegimeConfig()
    result = fit_markov_model(sample_market, cfg, seed=42)
    df = result.market_day
    n_states = cfg.n_states
    filt_sum = sum(df[f"filtered_prob_state{s}"] for s in range(n_states))
    np.testing.assert_allclose(filt_sum.values, 1.0, atol=1e-6,
                               err_msg="Filtered probs don't sum to 1")


def test_markov_transition_matrix_rows(sample_market):
    from equity_regime.regime.markov import fit_markov_model
    cfg = RegimeConfig()
    result = fit_markov_model(sample_market, cfg, seed=42)
    row_sums = result.transition_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)
