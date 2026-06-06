"""Backtest orchestrator: load data, run strategies, compute metrics, emit report.

Pipeline
--------
1. Load config (YAML → BacktestConfig dataclass).
2. Load data (parquet ETL tables or synthetic fallback).
3. Compute regime series (expanding vol-percentile, inherently causal).
4. Build and run each enabled strategy on the full data window.
5. Compute in-sample and OOS (walk-forward) metrics.
6. Run cost sweep for the regime-conditioned strategy.
7. Run Ledoit-Wolf Sharpe-difference test (regime vs unconditional momentum).
8. Save per-strategy CSV tables to ``outputs/tables/``.
9. Render self-contained HTML report.
"""

from __future__ import annotations

import base64
import datetime
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Top-level backtest configuration loaded from YAML."""

    name: str = "equity_backtest"
    version: str = "1.0.0"

    data_source: str = "parquet"
    parquet_dir: str = "data/processed/etl_full"
    synthetic_n_days: int = 1260
    synthetic_seed: int = 42
    analysis_window: str = "momentum_core"  # which factor window to use

    strategies_cfg: dict = field(default_factory=dict)
    costs_cfg: dict = field(default_factory=dict)
    sizing_cfg: dict = field(default_factory=dict)
    walkforward_cfg: dict = field(default_factory=dict)

    cost_sweep_enabled: bool = True
    bps_grid: list = field(default_factory=lambda: [0, 5, 10, 20, 30, 50])

    # Subperiod breakdown
    subperiod_breaks: list = field(default_factory=lambda: ["1990-01-01", "2010-01-01"])
    crash_windows: dict = field(default_factory=dict)

    bootstrap_n: int = 1000
    bootstrap_p: float = 0.10
    bootstrap_seed: int = 42

    report_path: str = "outputs/reports/backtest_report.html"
    tables_dir: str = "outputs/tables"
    log_file: str = "outputs/logs/backtest.log"
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | Path) -> "BacktestConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        bt = raw.get("backtest", {})
        data = raw.get("data", {})
        strats = raw.get("strategies", {})
        costs = raw.get("costs", {})
        sizing = raw.get("sizing", {})
        wf = raw.get("walkforward", {})
        sweep = raw.get("cost_sweep", {})
        metrics_cfg = raw.get("metrics", {})
        report = raw.get("report", {})
        log_cfg = raw.get("logging", {})

        return cls(
            name=bt.get("name", "equity_backtest"),
            version=bt.get("version", "1.0.0"),
            data_source=data.get("source", "parquet"),
            parquet_dir=data.get("parquet_dir", "data/processed/etl_full"),
            synthetic_n_days=int(data.get("synthetic_n_days", 1260)),
            synthetic_seed=int(data.get("synthetic_seed", 42)),
            analysis_window=data.get("analysis_window", "momentum_core"),
            strategies_cfg=strats,
            costs_cfg=costs,
            sizing_cfg=sizing,
            walkforward_cfg=wf,
            cost_sweep_enabled=bool(sweep.get("enabled", True)),
            bps_grid=list(sweep.get("bps_grid", [0, 5, 10, 20, 30, 50])),
            subperiod_breaks=raw.get("subperiod_breaks", ["1990-01-01", "2010-01-01"]),
            crash_windows=raw.get("crash_windows", {}),
            bootstrap_n=int(metrics_cfg.get("bootstrap_n", 1000)),
            bootstrap_p=float(metrics_cfg.get("bootstrap_p", 0.10)),
            bootstrap_seed=int(metrics_cfg.get("bootstrap_seed", 42)),
            report_path=report.get("out_path", "outputs/reports/backtest_report.html"),
            tables_dir=report.get("tables_dir", "outputs/tables"),
            log_file=log_cfg.get("log_file", "outputs/logs/backtest.log"),
            log_level=log_cfg.get("level", "INFO"),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (factor_daily, market_daily) filtered to the analysis window.

    The analysis window selects rows where all required columns are non-NaN,
    enforcing the native start date of each series without truncating the
    other series unnecessarily.
    """
    if cfg.data_source == "parquet":
        pdir = Path(cfg.parquet_dir)
        fd_path = pdir / "factor_daily.parquet"
        md_path = pdir / "market_daily.parquet"
        if fd_path.exists() and md_path.exists():
            fd_full = pd.read_parquet(fd_path)
            md_full = pd.read_parquet(md_path)
            log.info(
                "[data] Loaded parquet: factor_daily=%d rows (%s to %s), "
                "market_daily=%d rows",
                len(fd_full),
                fd_full.index.min().date(), fd_full.index.max().date(),
                len(md_full),
            )

            # Apply analysis-window filtering
            from pipeline.analysis_windows import resolve_windows
            windows = resolve_windows(fd_full, md_full)
            win_info = windows.get(cfg.analysis_window)
            if win_info and win_info["n_obs"] > 0:
                fd = win_info["df"].copy()
                # Add any other factor columns not in required but present in full data
                for col in fd_full.columns:
                    if col not in fd.columns:
                        fd[col] = fd_full[col].reindex(fd.index)
                # market_daily: align to the analysis window dates
                md = md_full.reindex(fd.index)
                log.info(
                    "[data] Analysis window '%s': %d rows (%s to %s)",
                    cfg.analysis_window, len(fd),
                    fd.index.min().date(), fd.index.max().date(),
                )
            else:
                log.warning(
                    "[data] Analysis window '%s' not found or empty — using full data",
                    cfg.analysis_window,
                )
                fd, md = fd_full, md_full

            return fd, md
        else:
            log.warning("[data] Parquet not found at %s — falling back to synthetic", pdir)

    # Synthetic fallback
    log.info("[data] Generating synthetic data (%d days)", cfg.synthetic_n_days)
    return _generate_synthetic(cfg.synthetic_n_days, cfg.synthetic_seed)


def _generate_synthetic(n_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic factor returns for testing without real data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    # Markov regime
    states = np.zeros(n_days, dtype=int)
    P = np.array([[0.96, 0.04], [0.10, 0.90]])
    for t in range(1, n_days):
        states[t] = rng.choice(2, p=P[states[t - 1]])

    mu = np.where(states == 0, 0.0003, -0.0001)
    sigma = np.where(states == 0, 0.008, 0.018)

    mkt_rf = rng.normal(mu, sigma)
    rf = np.full(n_days, 0.0001)

    # UMD: positive autocorr in calm, negative in turbulent
    umd = np.zeros(n_days)
    for t in range(1, n_days):
        ac = 0.05 if states[t] == 0 else -0.05
        umd[t] = ac * mkt_rf[t - 1] + rng.normal(0, 0.005)

    st_rev = -0.03 * np.roll(mkt_rf, 1) + rng.normal(0, 0.004, n_days)

    factor_daily = pd.DataFrame({
        "mkt_rf": mkt_rf, "smb": rng.normal(0, 0.004, n_days),
        "hml": rng.normal(0, 0.003, n_days), "rmw": rng.normal(0, 0.003, n_days),
        "cma": rng.normal(0, 0.002, n_days), "rf": rf,
        "umd": umd, "st_rev": st_rev, "lt_rev": rng.normal(0, 0.002, n_days),
    }, index=dates)
    factor_daily.index.name = "date"

    mkt_ret = mkt_rf + rf
    import math
    realized_vol = (
        pd.Series(mkt_ret, index=dates)
        .rolling(21, min_periods=11)
        .std()
        .mul(math.sqrt(TRADING_DAYS))
    )
    vix = 15 + realized_vol.mul(100).fillna(20)

    market_daily = pd.DataFrame({
        "mkt_ret": mkt_ret,
        "rf": rf,
        "realized_vol_21": realized_vol,
        "realized_vol_63": (
            pd.Series(mkt_ret, index=dates)
            .rolling(63, min_periods=30)
            .std()
            .mul(math.sqrt(TRADING_DAYS))
        ),
        "drawdown": 0.0,
        "vix": vix,
    }, index=dates)
    market_daily.index.name = "date"

    return factor_daily, market_daily


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------
def run_backtest(cfg: BacktestConfig) -> dict:
    """Execute the full backtest pipeline.

    Returns a dict with keys:
        is_metrics, oos_metrics, sweep_results, lw_result,
        full_run_results, elapsed_s
    """
    from backtest.costs import CostConfig, cost_sweep as _cost_sweep
    from backtest.engine import BacktestEngine, EngineConfig, _compute_regime_series
    from backtest.metrics import (
        MetricsResult, compute_metrics, ledoit_wolf_sharpe_test, metrics_table
    )
    from backtest.strategy import build_strategy
    from backtest.walkforward import (
        WalkForwardConfig, WalkForwardSplitter,
        concatenate_oos_returns, run_walk_forward
    )

    t0_total = time.time()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    factor_daily, market_daily = load_data(cfg)
    benchmark_ret = market_daily["mkt_ret"]
    rf_series = market_daily["rf"]

    # ── 2. Engine + strategy setup ───────────────────────────────────────────
    engine_cfg = EngineConfig.from_dict({
        "costs": cfg.costs_cfg,
        "sizing": cfg.sizing_cfg,
        "strategies": cfg.strategies_cfg,
    })

    strat_cfgs = {
        k: v for k, v in cfg.strategies_cfg.items()
        if isinstance(v, dict) and v.get("enabled", True)
    }

    is_metrics_all: List[MetricsResult] = []
    oos_metrics_all: List[MetricsResult] = []
    full_run_results: dict = {}
    sweep_results: dict = {}

    wf_cfg = WalkForwardConfig.from_dict(cfg.walkforward_cfg)

    for strat_name, strat_dict in strat_cfgs.items():
        log.info("[backtest] Running strategy: %s", strat_name)
        t_strat = time.time()

        strategy = build_strategy(strat_name, strat_dict)

        # Full in-sample run
        engine = BacktestEngine(strategy, factor_daily, market_daily, engine_cfg)
        run_result = engine.run()
        full_run_results[strat_name] = run_result

        # IS metrics
        is_m = compute_metrics(
            run_result.net_returns, benchmark_ret, rf_series,
            strategy_name=strat_name, period="in_sample",
            bootstrap_n=cfg.bootstrap_n, bootstrap_p=cfg.bootstrap_p,
            bootstrap_seed=cfg.bootstrap_seed,
        )
        is_metrics_all.append(is_m)

        # Walk-forward OOS
        def _engine_factory(fd, md):
            return BacktestEngine(build_strategy(strat_name, strat_dict), fd, md, engine_cfg)

        fold_results = run_walk_forward(
            _engine_factory, factor_daily, market_daily, wf_cfg, strat_name
        )
        oos_rets = concatenate_oos_returns(fold_results)

        if len(oos_rets) >= 5:
            oos_m = compute_metrics(
                oos_rets, benchmark_ret.reindex(oos_rets.index),
                rf_series.reindex(oos_rets.index).fillna(0.0),
                strategy_name=strat_name, period="out_of_sample",
                bootstrap_n=cfg.bootstrap_n, bootstrap_p=cfg.bootstrap_p,
                bootstrap_seed=cfg.bootstrap_seed,
            )
        else:
            log.warning("[backtest] %s: insufficient OOS data (%d days)", strat_name, len(oos_rets))
            oos_m = MetricsResult(
                strategy_name=strat_name, period="out_of_sample", n_days=0,
                ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                sharpe_ci_lo=np.nan, sharpe_ci_hi=np.nan, sortino=np.nan,
                max_drawdown=np.nan, calmar=np.nan, win_rate=np.nan,
                profit_factor=np.nan, alpha=np.nan, beta=np.nan,
                alpha_pval=np.nan, skew=np.nan, kurtosis=np.nan,
            )
        oos_metrics_all.append(oos_m)

        log.info(
            "[backtest] %s done in %.1fs | IS Sharpe=%.3f | OOS Sharpe=%.3f",
            strat_name, time.time() - t_strat,
            is_m.sharpe, oos_m.sharpe,
        )

    # ── 3. Cost sweep for regime strategy ────────────────────────────────────
    regime_strat_name = "regime_momentum"
    if (
        cfg.cost_sweep_enabled
        and regime_strat_name in strat_cfgs
        and regime_strat_name in full_run_results
    ):
        log.info("[backtest] Running cost sweep for %s", regime_strat_name)
        strat_dict = strat_cfgs[regime_strat_name]
        regime_strategy = build_strategy(regime_strat_name, strat_dict)

        def _run_with_tc(tc_bps: float) -> pd.Series:
            cc = CostConfig(
                tc_bps=tc_bps,
                slippage_bps=cfg.costs_cfg.get("slippage_bps", 5.0),
                state_dependent_slippage=bool(
                    cfg.costs_cfg.get("state_dependent_slippage", True)
                ),
                turbulent_slippage_multiplier=float(
                    cfg.costs_cfg.get("turbulent_slippage_multiplier", 2.0)
                ),
            )
            eng = BacktestEngine(regime_strategy, factor_daily, market_daily, engine_cfg)
            result = eng.run(cost_cfg_override=cc)
            return result.net_returns

        sweep_df = _cost_sweep(
            run_fn=_run_with_tc,
            bps_grid=cfg.bps_grid,
            strategy_name=regime_strat_name,
            rf_series=rf_series,
        )
        sweep_results[regime_strat_name] = sweep_df
        log.info("[backtest] Cost sweep complete:\n%s", sweep_df.to_string(index=False))

    # Also sweep the unconditional momentum for comparison
    ls_strat_name = "long_short_momentum"
    if cfg.cost_sweep_enabled and ls_strat_name in strat_cfgs:
        ls_dict = strat_cfgs[ls_strat_name]
        ls_strategy = build_strategy(ls_strat_name, ls_dict)

        def _run_ls_with_tc(tc_bps: float) -> pd.Series:
            cc = CostConfig(tc_bps=tc_bps, slippage_bps=cfg.costs_cfg.get("slippage_bps", 5.0))
            eng = BacktestEngine(ls_strategy, factor_daily, market_daily, engine_cfg)
            return eng.run(cost_cfg_override=cc).net_returns

        ls_sweep = _cost_sweep(
            run_fn=_run_ls_with_tc,
            bps_grid=cfg.bps_grid,
            strategy_name=ls_strat_name,
            rf_series=rf_series,
        )
        sweep_results[ls_strat_name] = ls_sweep

    # ── 4. Ledoit-Wolf test (regime vs unconditional) ─────────────────────────
    lw_result = None
    if (
        regime_strat_name in full_run_results
        and ls_strat_name in full_run_results
    ):
        regime_rets = full_run_results[regime_strat_name].net_returns
        ls_rets = full_run_results[ls_strat_name].net_returns
        lw_result = ledoit_wolf_sharpe_test(
            regime_rets, ls_rets,
            name_a=regime_strat_name, name_b=ls_strat_name,
        )
        log.info(
            "[backtest] Ledoit-Wolf test: SR_regime=%.3f vs SR_unc=%.3f | "
            "t=%.3f | p=%.4f | sig@5%%=%s",
            lw_result.sr_a, lw_result.sr_b,
            lw_result.t_stat, lw_result.p_value,
            lw_result.significant_5pct,
        )
        print(
            f"\n{'='*60}\n"
            f"LEDOIT-WOLF SHARPE TEST\n"
            f"  Regime Momentum Sharpe:       {lw_result.sr_a:+.3f}\n"
            f"  Unconditional L/S Sharpe:     {lw_result.sr_b:+.3f}\n"
            f"  Sharpe difference:            {lw_result.sr_diff:+.3f}\n"
            f"  t-statistic (HAC):            {lw_result.t_stat:.3f}\n"
            f"  p-value:                      {lw_result.p_value:.4f}\n"
            f"  Significant at 5%:            {lw_result.significant_5pct}\n"
            f"{'='*60}\n"
        )

    # ── 5. Subperiod & crash-window breakdown ────────────────────────────────
    subperiod_rows = []
    crash_rows = []

    key_strats = {n: full_run_results[n] for n in [regime_strat_name, ls_strat_name]
                  if n in full_run_results}

    # Define subperiods
    all_dates = factor_daily.index
    breaks = [all_dates.min()] + [pd.Timestamp(b) for b in cfg.subperiod_breaks] + [all_dates.max()]
    subperiod_labels = []
    for i in range(len(breaks) - 1):
        a = breaks[i].year
        b = breaks[i + 1].year
        subperiod_labels.append(f"{a}-{b}")

    for strat_name, run_res in key_strats.items():
        rets = run_res.net_returns
        bench_aligned = benchmark_ret.reindex(rets.index).fillna(0.0)
        rf_aligned = rf_series.reindex(rets.index).fillna(0.0)

        # Subperiods
        for i in range(len(breaks) - 1):
            t0, t1 = breaks[i], breaks[i + 1]
            sub = rets[(rets.index >= t0) & (rets.index <= t1)]
            if len(sub) < 60:
                continue
            sub_bench = bench_aligned.reindex(sub.index).fillna(0.0)
            sub_rf = rf_aligned.reindex(sub.index).fillna(0.0)
            m = compute_metrics(
                sub, sub_bench, sub_rf,
                strategy_name=strat_name, period=subperiod_labels[i],
                bootstrap_n=min(cfg.bootstrap_n, 200),
                bootstrap_p=cfg.bootstrap_p, bootstrap_seed=cfg.bootstrap_seed,
            )
            subperiod_rows.append(m.to_dict())

        # Crash windows
        for crash_name, crash_cfg in cfg.crash_windows.items():
            cs = pd.Timestamp(crash_cfg["start"])
            ce = pd.Timestamp(crash_cfg["end"])
            sub = rets[(rets.index >= cs) & (rets.index <= ce)]
            if len(sub) < 5:
                continue
            sub_bench = bench_aligned.reindex(sub.index).fillna(0.0)
            sub_rf = rf_aligned.reindex(sub.index).fillna(0.0)
            m = compute_metrics(
                sub, sub_bench, sub_rf,
                strategy_name=strat_name, period=crash_name,
                bootstrap_n=50, bootstrap_p=0.5, bootstrap_seed=cfg.bootstrap_seed,
            )
            crash_rows.append(m.to_dict())

    subperiod_df = pd.DataFrame(subperiod_rows) if subperiod_rows else pd.DataFrame()
    crash_df = pd.DataFrame(crash_rows) if crash_rows else pd.DataFrame()

    if len(subperiod_df):
        log.info("\n[backtest] Subperiod breakdown:\n%s",
                 subperiod_df[["strategy","period","sharpe","ann_return","max_drawdown"]].to_string(index=False))
        print("\n=== SUBPERIOD BREAKDOWN (regime vs unconditional) ===")
        print(subperiod_df[["strategy","period","ann_return","sharpe","max_drawdown"]].to_string(index=False))

    if len(crash_df):
        log.info("\n[backtest] Crash-window breakdown:\n%s",
                 crash_df[["strategy","period","sharpe","ann_return","max_drawdown"]].to_string(index=False))
        print("\n=== CRASH-WINDOW BREAKDOWN ===")
        print(crash_df[["strategy","period","ann_return","sharpe","max_drawdown"]].to_string(index=False))

    # ── 6. Save tables ────────────────────────────────────────────────────────
    tables_dir = Path(cfg.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    is_df = metrics_table(is_metrics_all)
    oos_df = metrics_table(oos_metrics_all)
    is_df.to_csv(tables_dir / "metrics_in_sample.csv")
    oos_df.to_csv(tables_dir / "metrics_oos.csv")
    if len(subperiod_df):
        subperiod_df.to_csv(tables_dir / "metrics_subperiod.csv", index=False)
    if len(crash_df):
        crash_df.to_csv(tables_dir / "metrics_crash_windows.csv", index=False)

    for name, sweep in sweep_results.items():
        sweep.to_csv(tables_dir / f"cost_sweep_{name}.csv", index=False)

    # Print summary
    log.info("[backtest] In-sample metrics:\n%s", is_df.to_string())
    log.info("[backtest] OOS metrics:\n%s", oos_df.to_string())

    elapsed = time.time() - t0_total
    log.info("[backtest] Total elapsed: %.1fs", elapsed)

    return {
        "is_metrics": is_metrics_all,
        "oos_metrics": oos_metrics_all,
        "sweep_results": sweep_results,
        "lw_result": lw_result,
        "full_run_results": full_run_results,
        "subperiod_df": subperiod_df,
        "crash_df": crash_df,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------
_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{{ title }}</title>
<style>
  body { font-family:'Segoe UI',Arial,sans-serif; max-width:1200px; margin:40px auto;
         padding:0 24px; color:#1a1a2e; background:#f8f9fa; line-height:1.6; }
  h1   { font-size:1.8em; color:#16213e; border-bottom:3px solid #0f3460; padding-bottom:10px; }
  h2   { font-size:1.2em; color:#0f3460; margin-top:2em; border-bottom:1px solid #dee2e6; }
  h3   { font-size:1em; color:#555; margin-top:1.5em; }
  .meta { color:#555; font-size:0.9em; margin-bottom:1.5em; }
  .lw-box { background:#fff3cd; border:2px solid #ffc107; border-radius:8px;
             padding:18px 24px; margin:1.5em 0; }
  .lw-sig  { background:#d4edda; border-color:#28a745; }
  .lw-ns   { background:#f8d7da; border-color:#dc3545; }
  .lw-stat { font-size:1.1em; font-weight:bold; }
  table { border-collapse:collapse; width:100%; font-size:0.82em; margin-top:10px; }
  th { background:#0f3460; color:white; padding:6px 10px; text-align:left; }
  td { padding:5px 10px; border-bottom:1px solid #e5e5e5; }
  tr:nth-child(even) { background:#f9f9f9; }
  .positive { color:#155724; font-weight:bold; }
  .negative { color:#721c24; font-weight:bold; }
  .chart { text-align:center; margin:1.5em 0; }
  .chart img { max-width:100%; border:1px solid #dee2e6; border-radius:4px; }
  .note { font-size:0.82em; color:#666; font-style:italic; margin-top:0.5em; }
  footer { text-align:center; color:#aaa; font-size:0.8em; margin-top:3em; }
  .badge { display:inline-block; padding:3px 10px; border-radius:4px; font-size:0.8em;
           font-weight:bold; }
  .badge-pass { background:#d4edda; color:#155724; }
  .badge-fail { background:#f8d7da; color:#721c24; }
  .breakeven-highlight { font-size:1.3em; font-weight:bold; color:#0f3460;
                          background:#e3f2fd; padding:8px 16px; border-radius:4px;
                          display:inline-block; margin:6px 0; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="meta">
  <strong>Generated:</strong> {{ generated }} &nbsp;|&nbsp;
  <strong>Data:</strong> {{ data_window }} &nbsp;|&nbsp;
  <strong>Strategies:</strong> {{ n_strategies }}
</div>

{{ lw_section }}

<h2>Cumulative Return Chart</h2>
<div class="chart">
  <img src="data:image/png;base64,{{ cum_ret_chart }}" alt="Cumulative Returns"/>
</div>

<h2>In-Sample Metrics</h2>
{{ is_table }}

<h2>Out-of-Sample Metrics (Walk-Forward)</h2>
{{ oos_table }}
<p class="note">OOS metrics are computed on concatenated out-of-sample windows
from the expanding walk-forward procedure (embargo=21 trading days).</p>

<h2>Cost Sweep Analysis</h2>
{{ cost_section }}

<footer>equity-regime-research backtesting framework &mdash; {{ generated }}</footer>
</body>
</html>
"""


def render_backtest_report(results: dict, cfg: BacktestConfig) -> Path:
    """Render the HTML backtest report."""
    from jinja2 import Environment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = Environment()

    # ── Cumulative return chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, run) in enumerate(results.get("full_run_results", {}).items()):
        cum = (1 + run.net_returns).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=colors[i % len(colors)], linewidth=1.2)
    ax.set_title("Cumulative Net-of-Cost Returns (Full Period)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    cum_chart_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── Metrics tables ────────────────────────────────────────────────────────
    def _metrics_html(metrics_list) -> str:
        if not metrics_list:
            return "<p>No metrics computed.</p>"
        rows_html = []
        cols = [
            "strategy", "n_days", "ann_return", "ann_vol", "sharpe",
            "sharpe_ci_lo", "sharpe_ci_hi", "sortino", "max_drawdown",
            "calmar", "win_rate", "profit_factor", "alpha", "beta",
            "alpha_pval", "skew", "kurtosis",
        ]
        display_cols = [
            "Strategy", "Days", "Ann Ret", "Ann Vol", "Sharpe",
            "SR CI Lo", "SR CI Hi", "Sortino", "Max DD",
            "Calmar", "Win%", "Profit F", "Alpha", "Beta",
            "Alpha p", "Skew", "Kurt",
        ]
        header = "<tr>" + "".join(f"<th>{c}</th>" for c in display_cols) + "</tr>"
        for m in metrics_list:
            d = m.to_dict()
            cells = []
            for col in cols:
                val = d.get(col, "")
                if col == "ann_return":
                    pct = val * 100 if isinstance(val, float) and not np.isnan(val) else val
                    cls = "positive" if isinstance(pct, float) and pct > 0 else "negative"
                    cells.append(f'<td class="{cls}">{pct:.2f}%</td>' if isinstance(pct, float) else f"<td>{pct}</td>")
                elif col in ("ann_vol",):
                    pct = val * 100 if isinstance(val, float) and not np.isnan(val) else val
                    cells.append(f"<td>{pct:.2f}%</td>" if isinstance(pct, float) else f"<td>{pct}</td>")
                elif col == "max_drawdown":
                    pct = val * 100 if isinstance(val, float) and not np.isnan(val) else val
                    cls = "negative" if isinstance(pct, float) and pct < 0 else ""
                    cells.append(f'<td class="{cls}">{pct:.2f}%</td>' if isinstance(pct, float) else f"<td>{pct}</td>")
                elif col in ("win_rate",):
                    pct = val * 100 if isinstance(val, float) and not np.isnan(val) else val
                    cells.append(f"<td>{pct:.1f}%</td>" if isinstance(pct, float) else f"<td>{pct}</td>")
                elif col == "alpha":
                    pct = val * 100 if isinstance(val, float) and not np.isnan(val) else val
                    cls = "positive" if isinstance(pct, float) and pct > 0 else "negative"
                    cells.append(f'<td class="{cls}">{pct:.3f}%</td>' if isinstance(pct, float) else f"<td>{pct}</td>")
                elif isinstance(val, float):
                    cells.append(f"<td>{val:.4f}</td>" if not np.isnan(val) else "<td>—</td>")
                else:
                    cells.append(f"<td>{val}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><thead>" + header + "</thead><tbody>" + "".join(rows_html) + "</tbody></table>"

    is_table = _metrics_html(results.get("is_metrics", []))
    oos_table = _metrics_html(results.get("oos_metrics", []))

    # ── Ledoit-Wolf section ───────────────────────────────────────────────────
    lw = results.get("lw_result")
    if lw is not None:
        sig_class = "lw-sig" if lw.significant_5pct else "lw-ns"
        sig_label = "SIGNIFICANT at 5%" if lw.significant_5pct else "NOT significant at 5%"
        lw_section = f"""
<h2>Headline Test: Ledoit-Wolf (2008) Sharpe-Difference Test</h2>
<div class="lw-box {sig_class}">
  <p class="lw-stat">H<sub>0</sub>: SR(<em>regime_momentum</em>) = SR(<em>long_short_momentum</em>)</p>
  <p>
    <strong>Regime momentum Sharpe:</strong> {lw.sr_a:.4f} &nbsp;|&nbsp;
    <strong>Unconditional Sharpe:</strong> {lw.sr_b:.4f} &nbsp;|&nbsp;
    <strong>Difference:</strong> {lw.sr_diff:+.4f}
  </p>
  <p>
    <strong>t-statistic (HAC):</strong> {lw.t_stat:.4f} &nbsp;|&nbsp;
    <strong>p-value:</strong> {lw.p_value:.6f}
  </p>
  <p><span class="badge {'badge-pass' if lw.significant_5pct else 'badge-fail'}">{sig_label}</span></p>
  <p class="note">Test uses Newey-West HAC standard errors for the per-period Sharpe
  influence function difference (Ledoit &amp; Wolf 2008).  The null hypothesis is
  that both strategies have equal population Sharpe ratios.</p>
</div>"""
    else:
        lw_section = "<p>Ledoit-Wolf test not computed (one or both strategies disabled).</p>"

    # ── Cost sweep section ────────────────────────────────────────────────────
    cost_parts = []
    for strat_name, sweep_df in results.get("sweep_results", {}).items():
        if sweep_df is None or len(sweep_df) == 0:
            continue

        # Find breakeven (first tc_bps where ann_return <= 0)
        neg_mask = sweep_df["ann_return"] <= 0
        if neg_mask.any():
            breakeven_val = sweep_df.loc[neg_mask, "tc_bps"].iloc[0]
            breakeven_str = f"<span class='breakeven-highlight'>~{breakeven_val:.0f} bps</span>"
        else:
            breakeven_str = f"<span class='breakeven-highlight'>&gt;{max(cfg.bps_grid):.0f} bps (out of sweep range)</span>"

        # Sweep chart
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(sweep_df["tc_bps"], sweep_df["sharpe"], "o-", color="#0f3460", linewidth=1.5)
        ax2.axhline(0, color="#dc3545", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Transaction cost (bps one-way)")
        ax2.set_ylabel("Net Sharpe ratio")
        ax2.set_title(f"Cost sweep: {strat_name}")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=100)
        plt.close(fig2)
        chart_b64 = base64.b64encode(buf2.getvalue()).decode()

        # Table
        tbl_rows = ""
        for _, row in sweep_df.iterrows():
            sr_cls = "positive" if row["sharpe"] > 0 else "negative"
            ret_pct = row["ann_return"] * 100
            ret_cls = "positive" if ret_pct > 0 else "negative"
            tbl_rows += (
                f"<tr><td>{row['tc_bps']:.0f}</td>"
                f"<td class='{ret_cls}'>{ret_pct:.2f}%</td>"
                f"<td>{row['ann_vol']*100:.2f}%</td>"
                f"<td class='{sr_cls}'>{row['sharpe']:.4f}</td></tr>"
            )
        cost_parts.append(f"""
<h3>{strat_name} — Breakeven cost: {breakeven_str}</h3>
<div class="chart"><img src="data:image/png;base64,{chart_b64}" alt="Cost sweep"/></div>
<table>
  <thead><tr><th>TC (bps)</th><th>Ann Return</th><th>Ann Vol</th><th>Sharpe</th></tr></thead>
  <tbody>{tbl_rows}</tbody>
</table>
<p class="note">Breakeven = first TC level where annualised net return turns negative.
For a regime strategy this is the number that decides whether the conditional edge survives
real-world implementation costs.</p>
""")

    cost_section = "".join(cost_parts) if cost_parts else "<p>Cost sweep disabled.</p>"

    # ── Render ────────────────────────────────────────────────────────────────
    tmpl = env.from_string(_REPORT_TEMPLATE)
    run_results = results.get("full_run_results", {})
    if run_results:
        all_ret = pd.concat(
            [r.net_returns for r in run_results.values()], axis=1
        )
        start_date = all_ret.index.min().date()
        end_date = all_ret.index.max().date()
        data_window = f"{start_date} to {end_date}"
    else:
        data_window = "N/A"

    html = tmpl.render(
        title="Equity Regime Backtest Report",
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_window=data_window,
        n_strategies=len(results.get("full_run_results", {})),
        lw_section=lw_section,
        cum_ret_chart=cum_chart_b64,
        is_table=is_table,
        oos_table=oos_table,
        cost_section=cost_section,
    )

    out_path = Path(cfg.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("[report] Backtest report written -> %s", out_path)
    return out_path
