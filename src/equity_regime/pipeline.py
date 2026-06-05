"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress known benign numerical warnings from statsmodels EM algorithm
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message="The test statistic is outside")

import numpy as np
import pandas as pd

from equity_regime.config import Config


def run_pipeline(
    cfg: Config,
    use_synthetic: bool = True,
    stock_panel_path: Optional[str] = None,
    market_panel_path: Optional[str] = None,
) -> dict:
    """
    Execute the full research pipeline.

    Returns a dict of result artifacts (DataFrames, paths, etc.).
    """
    t0 = time.time()
    print(f"[pipeline] Starting run: {cfg.run.name}")
    print(f"[pipeline] Synthetic: {use_synthetic}")

    # --- Output directories ---
    fig_dir = cfg.output_path("figures")
    tbl_dir = cfg.output_path("tables")
    rep_dir = cfg.output_path("reports")
    for d in [fig_dir, tbl_dir, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1. INGEST
    # =========================================================
    print("[pipeline] Stage 1/7: Ingest")
    from equity_regime.data.ingest import ingest

    stock_day_raw, market_day_raw, true_regime = ingest(
        cfg,
        use_synthetic=use_synthetic,
        stock_panel_path=stock_panel_path,
        market_panel_path=market_panel_path,
    )
    print(f"  stock_day shape: {stock_day_raw.shape}")
    print(f"  market_day shape: {market_day_raw.shape}")

    # =========================================================
    # 2. CLEAN
    # =========================================================
    print("[pipeline] Stage 2/7: Clean")
    from equity_regime.data.clean import clean

    stock_day, market_day = clean(stock_day_raw, market_day_raw, cfg.data)
    print(f"  After cleaning — stock_day: {stock_day.shape}, unique stocks: {stock_day['permno'].nunique()}")

    # Stationarity check on market returns
    from equity_regime.stats.tests import stationarity_tests

    stat_res = stationarity_tests(market_day["mkt_ret"])
    if not stat_res["is_stationary_adf"]:
        print("  WARNING: Market returns may not be stationary (ADF p={:.4f})".format(stat_res["adf_pvalue"]))
    stat_df = pd.DataFrame([stat_res])
    stat_df.to_csv(tbl_dir / "stationarity_tests.csv", index=False)

    # =========================================================
    # 3. FACTORS
    # =========================================================
    print("[pipeline] Stage 3/7: Factors")
    from equity_regime.factors.momentum import compute_momentum_signal, verify_no_lookahead as mom_nla
    from equity_regime.factors.mean_reversion import compute_reversal_signal, verify_no_lookahead as rev_nla
    from equity_regime.factors.portfolios import build_portfolios, decile_monotonicity_check

    mom_df = compute_momentum_signal(stock_day, cfg.factors)
    print(f"  Momentum signal: {mom_df.shape} rows, {mom_df['date'].nunique()} dates")

    rev_df = compute_reversal_signal(stock_day, cfg.factors)
    print(f"  Reversal signal: {rev_df.shape} rows, {rev_df['date'].nunique()} dates")

    # No look-ahead validation
    print("  Verifying no look-ahead in momentum...")
    mom_nla(stock_day, mom_df, cfg.factors)
    print("  Verifying no look-ahead in reversal...")
    rev_nla(stock_day, rev_df, cfg.factors)
    print("  Look-ahead checks PASSED")

    # Portfolios
    mom_port, mom_ls = build_portfolios(stock_day, mom_df, "mom_signal", cfg.factors, "momentum")
    rev_port, rev_ls = build_portfolios(stock_day, rev_df, "rev_signal", cfg.factors, "reversal")

    print(f"  Mom L/S rows: {len(mom_ls)}, Rev L/S rows: {len(rev_ls)}")

    # Monotonicity check
    if not mom_port.empty:
        mono = decile_monotonicity_check(mom_port)
        print(f"  Momentum decile Spearman rho={mono['spearman_corr']:.3f} "
              f"(p={mono['spearman_pval']:.4f}), monotone={mono['is_monotone_increasing']}")
        mono_df = mono["mean_by_decile"].rename("mean_ret").reset_index()
        mono_df.to_csv(tbl_dir / "momentum_decile_means.csv", index=False)

    # Save factor data
    mom_ls.to_csv(tbl_dir / "momentum_ls_returns.csv", index=False)
    rev_ls.to_csv(tbl_dir / "reversal_ls_returns.csv", index=False)

    # =========================================================
    # 4. REGIME DETECTION
    # =========================================================
    print("[pipeline] Stage 4/7: Regime Detection")
    from equity_regime.regime.rule_based import compute_rule_based_regimes
    from equity_regime.regime.markov import fit_markov_model

    # Rule-based
    market_day_rb = compute_rule_based_regimes(market_day, cfg.regime)

    # Markov
    markov_result = fit_markov_model(market_day_rb, cfg.regime, seed=cfg.run.seed)
    market_day_final = markov_result.market_day

    # Validate: filtered != smoothed (they should differ)
    fp = market_day_final.get("filtered_prob_state1", pd.Series(dtype=float))
    sp = market_day_final.get("smoothed_prob_state1", pd.Series(dtype=float))
    if len(fp) > 0 and len(sp) > 0:
        corr = fp.corr(sp)
        print(f"  Filtered vs smoothed correlation: {corr:.4f} (should be high but <1)")
        assert corr < 1.0, "Filtered and smoothed probs are identical — check model"
    print(f"  Markov state means: {markov_result.state_means}")
    print(f"  Markov state vols:  {markov_result.state_vols}")

    # Validate regime recovery on synthetic data
    if true_regime is not None:
        _validate_regime_recovery(
            true_regime, market_day_final.set_index("date")["filtered_regime"]
        )

    # Save transition matrix
    pd.DataFrame(
        markov_result.transition_matrix,
        index=[f"from_state_{i}" for i in range(len(markov_result.state_means))],
        columns=[f"to_state_{i}" for i in range(len(markov_result.state_means))],
    ).to_csv(tbl_dir / "markov_transition_matrix.csv")

    # =========================================================
    # 5. STATISTICAL TESTS
    # =========================================================
    print("[pipeline] Stage 5/7: Statistical Tests")
    from equity_regime.stats.tests import (
        compute_acf, variance_ratio_test, predictive_regression, regime_interaction_regression
    )
    from equity_regime.stats.performance import (
        full_performance_table, train_test_split_with_embargo, oos_r_squared
    )

    mkt_ret = market_day_final.set_index("date")["mkt_ret"]

    # ACF
    acf_df = compute_acf(mkt_ret, cfg.stats.acf_lags)
    acf_df.to_csv(tbl_dir / "acf_market_returns.csv", index=False)

    # Variance ratio
    vr_df = variance_ratio_test(mkt_ret, cfg.stats.variance_ratio_q)
    vr_df.to_csv(tbl_dir / "variance_ratio.csv", index=False)
    print(f"  VR results:\n{vr_df[['q','vr','z_stat','pvalue']].to_string(index=False)}")

    # Predictive regression: momentum L/S ~ momentum signal (cross-section mean)
    if not mom_ls.empty and not mom_df.empty:
        mom_signal_mean = mom_df.groupby("date")["mom_zscore"].mean().rename("mom_signal")
        pred_reg = predictive_regression(
            mom_ls.set_index("date")["ls_ret"],
            mom_signal_mean.to_frame(),
            nw_lags=cfg.stats.newey_west_lags,
        )
        pred_reg.to_csv(tbl_dir / "predictive_regression.csv", index=False)

    # Regime-interaction regression
    regime_col = "filtered_regime"
    if not mom_ls.empty and regime_col in market_day_final.columns:
        regime_s = market_day_final.set_index("date")[regime_col]
        mom_signal_mean2 = mom_df.groupby("date")["mom_zscore"].mean()
        reg_result = regime_interaction_regression(
            mom_ls.set_index("date")["ls_ret"],
            mom_signal_mean2,
            regime_s,
            nw_lags=cfg.stats.newey_west_lags,
        )
        reg_result["coef_table"].to_csv(tbl_dir / "regime_interaction_regression.csv", index=False)
        print(f"  Regime invariance rejected: {reg_result['regime_invariance_rejected']} "
              f"(Wald p={reg_result['wald_pvalue']:.4f})")
    else:
        reg_result = {"coef_table": pd.DataFrame(), "regime_invariance_rejected": False,
                      "wald_pvalue": np.nan}

    # Performance tables
    strategies = {}
    if not mom_ls.empty:
        strategies["Momentum L/S"] = mom_ls.set_index("date")["ls_ret"]
    if not rev_ls.empty:
        strategies["Reversal L/S"] = rev_ls.set_index("date")["ls_ret"]

    rf_s = market_day_final.set_index("date")["rf"] if "rf" in market_day_final else None
    perf_tbl = full_performance_table(strategies, rf_series=rf_s)
    perf_tbl.to_csv(tbl_dir / "performance_summary.csv")
    print(f"  Performance summary:\n{perf_tbl[['ann_return','ann_vol','sharpe','max_drawdown']].to_string()}")

    # OOS split — fall back to 75/25 if configured split date is outside data range
    all_dates = pd.DatetimeIndex(market_day_final["date"].unique())
    split_ts = pd.Timestamp(cfg.stats.oos_split_date)
    if split_ts >= all_dates.max() or split_ts <= all_dates.min():
        fallback_idx = int(len(all_dates) * 0.75)
        split_ts = all_dates[fallback_idx]
        print(f"  [info] oos_split_date outside data range; using 75/25 split at {split_ts.date()}")
    train_dates, test_dates = train_test_split_with_embargo(
        all_dates, str(split_ts.date()), cfg.stats.embargo_days
    )
    print(f"  OOS split: train={len(train_dates)} days, test={len(test_dates)} days, embargo={cfg.stats.embargo_days} days")

    # =========================================================
    # 6. FIGURES
    # =========================================================
    print("[pipeline] Stage 6/7: Figures")
    from equity_regime.viz.figures import (
        plot_cumulative_returns, plot_rolling_autocorrelation, plot_variance_ratio,
        plot_regime_overlay, plot_state_stats, plot_momentum_by_regime,
        plot_equity_curves, plot_drawdowns,
    )

    saved_figs = []
    saved_figs.append(plot_cumulative_returns(mom_ls, rev_ls, fig_dir))
    saved_figs.append(plot_rolling_autocorrelation(market_day_final, [1, 5, 21], fig_dir))
    saved_figs.append(plot_variance_ratio(vr_df, fig_dir))
    saved_figs.append(plot_regime_overlay(market_day_final, out_dir=fig_dir))
    saved_figs.append(plot_state_stats(markov_result.state_means, markov_result.state_vols, fig_dir))

    coef_tbl = reg_result.get("coef_table", pd.DataFrame())
    saved_figs.append(plot_momentum_by_regime(coef_tbl, fig_dir))
    saved_figs.append(plot_equity_curves(strategies, fig_dir))
    saved_figs.append(plot_drawdowns(strategies, fig_dir))

    print(f"  Saved {len(saved_figs)} figures to {fig_dir}")

    # =========================================================
    # 7. REPORT
    # =========================================================
    print("[pipeline] Stage 7/7: Report")
    from equity_regime.report.builder import build_report

    report_path = build_report(
        cfg=cfg,
        figure_dir=fig_dir,
        table_dir=tbl_dir,
        out_path=rep_dir / "research_report.html",
        perf_table=perf_tbl.reset_index() if not perf_tbl.empty else None,
        vr_table=vr_df if not vr_df.empty else None,
        acf_table=acf_df if not acf_df.empty else None,
        stationarity_table=stat_df,
        regime_coef_table=coef_tbl if not coef_tbl.empty else None,
        markov_summary=markov_result.model_summary,
    )
    print(f"  Report: {report_path}")

    elapsed = time.time() - t0
    print(f"[pipeline] Done in {elapsed:.1f}s")

    return {
        "stock_day": stock_day,
        "market_day": market_day_final,
        "mom_ls": mom_ls,
        "rev_ls": rev_ls,
        "perf_table": perf_tbl,
        "vr_table": vr_df,
        "markov_result": markov_result,
        "report_path": report_path,
        "figures": saved_figs,
    }


def _validate_regime_recovery(
    true_regime: pd.Series,
    filtered_regime: pd.Series,
) -> None:
    """Check that the Markov model recovers the injected regimes better than chance."""
    aligned = pd.concat([true_regime.rename("true"), filtered_regime.rename("pred")], axis=1).dropna()
    if aligned.empty:
        print("  [warning] Could not align true vs filtered regimes for validation")
        return

    # Try both labelings (states may be flipped)
    acc0 = (aligned["true"] == aligned["pred"]).mean()
    acc1 = (aligned["true"] == (1 - aligned["pred"])).mean()
    best_acc = max(acc0, acc1)
    print(f"  Regime recovery accuracy: {best_acc:.3f} (random baseline=0.500)")
    if best_acc < 0.55:
        print("  [warning] Regime recovery near random — model may need tuning")


def main_cli() -> None:
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Equity Regime Research Pipeline"
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--stock-panel", default=None, help="Path to stock parquet panel")
    parser.add_argument("--market-panel", default=None, help="Path to market parquet panel")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.validate()

    use_syn = args.synthetic or (args.stock_panel is None)

    run_pipeline(
        cfg,
        use_synthetic=use_syn,
        stock_panel_path=args.stock_panel,
        market_panel_path=args.market_panel,
    )


if __name__ == "__main__":
    main_cli()
