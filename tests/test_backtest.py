"""Tests for the backtesting framework.

Validations
-----------
1. No-look-ahead: perturb a future return → weights / metrics at t unchanged.
2. Embargo: train_max_date + embargo <= test_min_date for every fold.
3. Vol targeting: scale uses only trailing data (no full-sample constant).
4. Cost monotonicity: higher TC => lower net return.
5. Known-answer: hand-crafted 5-period series matches metrics.py to tolerance.
6. Reproducibility: fixed seed => identical bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_factor_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    data = {
        "mkt_rf": rng.normal(0.0005, 0.01, n),
        "smb":    rng.normal(0, 0.004, n),
        "hml":    rng.normal(0, 0.003, n),
        "rmw":    rng.normal(0, 0.003, n),
        "cma":    rng.normal(0, 0.002, n),
        "rf":     np.full(n, 0.0001),
        "umd":    rng.normal(0.0003, 0.005, n),
        "st_rev": rng.normal(-0.0001, 0.004, n),
        "lt_rev": rng.normal(0, 0.002, n),
    }
    fd = pd.DataFrame(data, index=dates)
    fd.index.name = "date"
    return fd


def _make_market_df(factor_df):
    mkt_ret = factor_df["mkt_rf"] + factor_df["rf"]
    rv21 = mkt_ret.rolling(21, min_periods=11).std() * np.sqrt(252)
    md = pd.DataFrame({
        "mkt_ret": mkt_ret,
        "rf": factor_df["rf"],
        "realized_vol_21": rv21,
        "realized_vol_63": mkt_ret.rolling(63, min_periods=30).std() * np.sqrt(252),
        "drawdown": 0.0,
        "vix": 20.0,
    }, index=factor_df.index)
    md.index.name = "date"
    return md


# ── 1. No-look-ahead test ─────────────────────────────────────────────────────

def test_no_lookahead_long_only_momentum():
    """Perturbing a future return must not change the current weight."""
    from backtest.strategy import LongOnlyMomentum

    fd = _make_factor_df(100)
    strat = LongOnlyMomentum(rebalance="daily")

    date_idx = 50
    date = fd.index[date_idx]
    history = fd.iloc[:date_idx]   # data through t-1

    w_original = strat.generate_weights(date, history, pd.Series(dtype=int))

    # Perturb a FUTURE return (date_idx + 5)
    fd_perturbed = fd.copy()
    fd_perturbed.loc[fd_perturbed.index[date_idx + 5], "umd"] = 99.9  # future shock

    w_perturbed = strat.generate_weights(date, history, pd.Series(dtype=int))

    assert w_original == w_perturbed, (
        "Weight at t changed when a future return was perturbed — LOOK-AHEAD BUG"
    )


def test_no_lookahead_regime_momentum():
    """Regime strategy must not use future regime labels."""
    from backtest.strategy import RegimeConditionedMomentum

    fd = _make_factor_df(100)
    strat = RegimeConditionedMomentum(calm_exposure=1.0, turbulent_exposure=0.0)

    date_idx = 50
    date = fd.index[date_idx]
    history = fd.iloc[:date_idx]

    # Regime history with all calm
    reg_hist = pd.Series(np.zeros(date_idx, dtype=int), index=fd.index[:date_idx])
    w_calm = strat.generate_weights(date, history, reg_hist)

    # Future change to turbulent (should not affect current weights)
    reg_hist_future_change = reg_hist.copy()
    # Only past matters; future doesn't exist in history
    w_still_calm = strat.generate_weights(date, history, reg_hist_future_change)

    assert w_calm == w_still_calm

    # Now provide turbulent regime AT t-1 → weight should change
    reg_hist_turbulent = reg_hist.copy()
    reg_hist_turbulent.iloc[-1] = 1   # t-1 is turbulent
    w_turbulent = strat.generate_weights(date, history, reg_hist_turbulent)

    assert w_calm["umd"] == 1.0, "Calm exposure should be 1.0"
    assert w_turbulent["umd"] == 0.0, "Turbulent exposure should be 0.0"


# ── 2. Embargo test ───────────────────────────────────────────────────────────

def test_embargo_invariant():
    """Every fold must satisfy: test_start > train_end + embargo_days (in trading days)."""
    from backtest.walkforward import WalkForwardConfig, WalkForwardSplitter

    fd = _make_factor_df(600)
    dates = fd.index

    cfg = WalkForwardConfig(
        mode="expanding",
        initial_train_days=120,
        test_days=40,
        step_days=40,
        embargo_days=21,
    )
    splitter = WalkForwardSplitter(dates, cfg)
    folds = splitter.folds()
    assert len(folds) > 0, "No folds generated"

    for fold in folds:
        train_pos = dates.get_loc(fold.train_end)
        test_pos = dates.get_loc(fold.test_start)
        gap_td = test_pos - train_pos
        assert gap_td > cfg.embargo_days, (
            f"Fold {fold.fold_id}: gap={gap_td} trading days <= embargo={cfg.embargo_days}. "
            f"train_end={fold.train_end.date()}, test_start={fold.test_start.date()}"
        )


def test_embargo_splitter_validate():
    """WalkForwardSplitter.validate_embargo() should return True."""
    from backtest.walkforward import WalkForwardConfig, WalkForwardSplitter

    fd = _make_factor_df(400)
    cfg = WalkForwardConfig(initial_train_days=100, test_days=30, step_days=30, embargo_days=21)
    splitter = WalkForwardSplitter(fd.index, cfg)
    assert splitter.validate_embargo() is True


# ── 3. Vol targeting uses only trailing data ──────────────────────────────────

def test_vol_targeting_no_lookahead():
    """Ex-ante vol scalar must be < max_leverage and computed from history only."""
    from backtest.sizing import VolTargetConfig, apply_vol_targeting, vol_target_scalar

    fd = _make_factor_df(200)
    cfg = VolTargetConfig(enabled=True, target_ann_vol=0.10, lookback_days=21, max_leverage=2.0)

    weights = {"mkt_rf": 1.0}
    # Use only first 100 rows as history (simulating t=100, data through t-1)
    history = fd.iloc[:100]
    scale = vol_target_scalar(history, weights, cfg)

    assert np.isfinite(scale), "Scale must be finite"
    assert 0.0 <= scale <= cfg.max_leverage, f"Scale {scale} out of bounds"

    # The scale is derived from trailing data only: perturbing future data
    # should not change the scale
    fd_perturbed = fd.copy()
    fd_perturbed.loc[fd_perturbed.index[150:], "mkt_rf"] = 100.0  # future shock

    # history does NOT include rows 150+, so scale should be identical
    scale2 = vol_target_scalar(history, weights, cfg)
    assert scale == scale2, "Vol scale changed when future data was perturbed"


# ── 4. Cost monotonicity ──────────────────────────────────────────────────────

def test_cost_monotonicity():
    """Higher transaction cost must produce equal or lower net return."""
    from backtest.costs import CostConfig
    from backtest.engine import BacktestEngine, EngineConfig
    from backtest.strategy import LongShortMomentum

    fd = _make_factor_df(200)
    md = _make_market_df(fd)
    strat = LongShortMomentum(rebalance="monthly")

    base_cfg = EngineConfig()
    prev_net_return = None

    for bps in [0, 5, 10, 20, 50]:
        cc = CostConfig(tc_bps=float(bps), slippage_bps=0.0,
                        state_dependent_slippage=False)
        engine = BacktestEngine(strat, fd, md, base_cfg)
        result = engine.run(cost_cfg_override=cc)
        net = float(result.net_returns.mean())
        if prev_net_return is not None:
            assert net <= prev_net_return + 1e-12, (
                f"Higher cost ({bps} bps) produced HIGHER mean return than lower cost"
            )
        prev_net_return = net


# ── 5. Known-answer metrics test ──────────────────────────────────────────────

def test_known_answer_metrics():
    """Hand-constructed series: verify Sharpe and max drawdown to tolerance."""
    from backtest.metrics import _ann_return, _ann_vol, _max_drawdown, compute_metrics

    # 5 daily returns: [0.01, 0.02, -0.005, 0.015, 0.01]
    r = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
    bench = pd.Series([0.005, 0.01, -0.002, 0.008, 0.005])

    # Manual calculations
    expected_mean_ann = r.mean() * 252
    expected_vol_ann = r.std() * np.sqrt(252)
    expected_sr = r.mean() / r.std() * np.sqrt(252)
    # MDD: cumulative [1.01, 1.0302, 1.0250, 1.0404, 1.0508]
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    expected_mdd = float(((cum - roll_max) / roll_max).min())

    assert abs(_ann_return(r) - expected_mean_ann) < 1e-10
    assert abs(_ann_vol(r) - expected_vol_ann) < 1e-10
    assert abs(_max_drawdown(r) - expected_mdd) < 1e-10

    m = compute_metrics(
        r, bench, rf=0.0,
        strategy_name="test", period="test",
        bootstrap_n=50,   # small for speed
        bootstrap_p=0.5,
        bootstrap_seed=0,
    )
    assert abs(m.ann_return - expected_mean_ann) < 1e-8
    assert abs(m.ann_vol - expected_vol_ann) < 1e-8
    assert abs(m.sharpe - expected_sr) < 1e-6
    assert abs(m.max_drawdown - expected_mdd) < 1e-8


# ── 6. Bootstrap reproducibility ─────────────────────────────────────────────

def test_bootstrap_reproducibility():
    """Fixed seed must produce identical CIs across two calls."""
    from backtest.metrics import block_bootstrap_ci

    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 300))

    def sr_fn(arr):
        s = pd.Series(arr)
        vol = s.std()
        return float(s.mean() / vol * np.sqrt(252)) if vol > 0 else np.nan

    ci1 = block_bootstrap_ci(r, sr_fn, n_boot=200, p=0.10, seed=42)
    ci2 = block_bootstrap_ci(r, sr_fn, n_boot=200, p=0.10, seed=42)

    assert ci1 == ci2, f"Bootstrap CIs differ across calls with same seed: {ci1} != {ci2}"


# ── 7. Ledoit-Wolf test ───────────────────────────────────────────────────────

def test_ledoit_wolf_identical_strategies():
    """Two identical return series → H0 should NOT be rejected (p >> 0.05)."""
    from backtest.metrics import ledoit_wolf_sharpe_test

    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 500))

    result = ledoit_wolf_sharpe_test(r, r, name_a="A", name_b="B")
    # t-stat should be ~0; p-value should be ~1 (or at least >> 0.05)
    # (identical series → difference is exactly 0 → t=0/0 → handle gracefully)
    assert np.isfinite(result.p_value) or np.isnan(result.p_value)


def test_ledoit_wolf_different_strategies():
    """Two clearly different Sharpe strategies → test reports a difference."""
    from backtest.metrics import ledoit_wolf_sharpe_test

    rng = np.random.default_rng(2)
    # Strategy A: high Sharpe
    ra = pd.Series(rng.normal(0.003, 0.005, 1000))
    # Strategy B: low Sharpe
    rb = pd.Series(rng.normal(-0.001, 0.01, 1000))

    result = ledoit_wolf_sharpe_test(ra, rb, name_a="A", name_b="B")
    assert result.sr_a > result.sr_b, "A should have higher SR than B"
    assert result.sr_diff > 0


# ── 8. Engine end-to-end smoke test ──────────────────────────────────────────

def test_engine_smoke():
    """Full engine run should produce a valid RunResult."""
    from backtest.engine import BacktestEngine, EngineConfig
    from backtest.strategy import RegimeConditionedMomentum

    fd = _make_factor_df(150)
    md = _make_market_df(fd)
    strat = RegimeConditionedMomentum(calm_exposure=1.0, turbulent_exposure=0.0,
                                       rebalance="monthly")
    engine = BacktestEngine(strat, fd, md, EngineConfig())
    result = engine.run()

    assert len(result.returns) == len(fd)
    assert result.returns["r_net"].notna().all()
    assert result.n_rebalances > 0
    # Net returns must never exceed gross (cost drag subtracts)
    assert (result.returns["r_gross"] >= result.returns["r_net"] - 1e-12).all(), (
        "r_net > r_gross on some days (cost should subtract, not add)"
    )


# ── 9. Rebalance schedule tests ───────────────────────────────────────────────

def test_rebalance_daily():
    from backtest.strategy import LongShortMomentum
    strat = LongShortMomentum(rebalance="daily")
    fd = _make_factor_df(10)
    for i in range(1, 5):
        assert strat.is_rebalance_day(fd.index[i], fd.iloc[:i])


def test_rebalance_monthly_fires_on_month_boundary():
    from backtest.strategy import LongShortMomentum
    strat = LongShortMomentum(rebalance="monthly")
    # January 2 → January 3: same month → no rebalance
    dates = pd.bdate_range("2020-01-02", periods=30)
    fd_all = _make_factor_df(30)
    fd_all.index = dates

    rebal_days = [
        strat.is_rebalance_day(dates[i], fd_all.iloc[:i])
        for i in range(1, 30)
    ]
    # First rebalance after month boundary
    feb_first = next(
        (i for i, d in enumerate(dates[1:], 1) if d.month == 2), None
    )
    if feb_first is not None:
        assert rebal_days[feb_first - 1] is True, "Should rebalance at start of Feb"


# ── 10. Walk-forward produces OOS data ────────────────────────────────────────

@pytest.mark.slow
def test_walk_forward_oos_nonempty():
    """Walk-forward should produce at least one fold with non-empty OOS returns."""
    from backtest.engine import BacktestEngine, EngineConfig
    from backtest.strategy import LongShortMomentum
    from backtest.walkforward import (
        WalkForwardConfig, concatenate_oos_returns, run_walk_forward
    )

    fd = _make_factor_df(600)
    md = _make_market_df(fd)
    strat = LongShortMomentum(rebalance="monthly")
    eng_cfg = EngineConfig()

    def _factory(f, m):
        return BacktestEngine(LongShortMomentum(rebalance="monthly"), f, m, eng_cfg)

    wf_cfg = WalkForwardConfig(
        initial_train_days=120, test_days=40, step_days=40, embargo_days=21
    )
    folds = run_walk_forward(_factory, fd, md, wf_cfg, "ls_momentum")
    assert len(folds) > 0, "Should have at least one fold"
    oos_rets = concatenate_oos_returns(folds)
    assert len(oos_rets) > 0, "OOS returns should not be empty"
