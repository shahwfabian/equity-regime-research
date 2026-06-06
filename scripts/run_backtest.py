#!/usr/bin/env python
"""CLI entry point for the equity-regime backtesting framework.

Usage examples
--------------
# Full run with OOS and cost sweep (uses French parquet data):
python scripts/run_backtest.py --config configs/backtest.yaml --oos --cost-sweep

# Specific strategies only:
python scripts/run_backtest.py --config configs/backtest.yaml --strategies regime_momentum long_short_momentum

# Use synthetic data (no ETL run needed):
python scripts/run_backtest.py --config configs/backtest.yaml --synthetic

# Override TC cost:
python scripts/run_backtest.py --config configs/backtest.yaml --tc-bps 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _setup_logging(cfg) -> None:
    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equity Regime Research — Backtest Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/backtest.yaml",
        help="Path to backtest config YAML",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Restrict to specific strategy names (space-separated)",
    )
    parser.add_argument(
        "--oos", action="store_true", default=True,
        help="Run walk-forward OOS validation (default: on)",
    )
    parser.add_argument(
        "--no-oos", action="store_true",
        help="Disable walk-forward OOS validation",
    )
    parser.add_argument(
        "--cost-sweep", action="store_true", default=True,
        help="Run cost sweep analysis (default: on)",
    )
    parser.add_argument(
        "--no-cost-sweep", action="store_true",
        help="Disable cost sweep",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of parquet ETL tables",
    )
    parser.add_argument(
        "--tc-bps", type=float, default=None,
        help="Override transaction cost in basis points",
    )
    parser.add_argument(
        "--bootstrap-n", type=int, default=None,
        help="Override number of bootstrap resamples",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    from backtest.run import BacktestConfig, render_backtest_report, run_backtest

    cfg = BacktestConfig.load(args.config)
    _setup_logging(cfg)

    log = logging.getLogger("run_backtest")
    log.info("=" * 60)
    log.info("Equity Regime Backtest: %s v%s", cfg.name, cfg.version)
    log.info("Config: %s", args.config)
    log.info("=" * 60)

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    if args.synthetic:
        cfg.data_source = "synthetic"
        log.info("CLI override: data_source=synthetic")

    if args.tc_bps is not None:
        cfg.costs_cfg["tc_bps"] = args.tc_bps
        log.info("CLI override: tc_bps=%.1f", args.tc_bps)

    if args.bootstrap_n is not None:
        cfg.bootstrap_n = args.bootstrap_n

    if args.no_oos:
        cfg.walkforward_cfg["initial_train_days"] = 10_000   # effectively disables OOS
        log.info("CLI: OOS disabled")

    if args.no_cost_sweep:
        cfg.cost_sweep_enabled = False
        log.info("CLI: cost sweep disabled")

    if args.strategies:
        # Disable all strategies not in the list
        for name in list(cfg.strategies_cfg.keys()):
            if name not in args.strategies:
                cfg.strategies_cfg[name]["enabled"] = False
        log.info("CLI: restricted to strategies: %s", args.strategies)

    # ── Run backtest ──────────────────────────────────────────────────────────
    results = run_backtest(cfg)

    # ── Render report ─────────────────────────────────────────────────────────
    report_path = render_backtest_report(results, cfg)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("BACKTEST COMPLETE in %.1fs", results["elapsed_s"])
    log.info("Report: %s", report_path)

    is_metrics = results.get("is_metrics", [])
    if is_metrics:
        log.info("\nIn-sample Sharpe ratios:")
        for m in is_metrics:
            log.info("  %-30s %.4f", m.strategy_name, m.sharpe)

    oos_metrics = results.get("oos_metrics", [])
    if oos_metrics:
        log.info("\nOOS Sharpe ratios:")
        for m in oos_metrics:
            log.info("  %-30s %.4f", m.strategy_name, m.sharpe)

    lw = results.get("lw_result")
    if lw is not None:
        log.info("\nLedoit-Wolf test result: p=%.4f | sig@5%%=%s", lw.p_value, lw.significant_5pct)

    # Breakeven cost summary
    for name, sweep_df in results.get("sweep_results", {}).items():
        if sweep_df is None or len(sweep_df) == 0:
            continue
        neg_mask = sweep_df["ann_return"] <= 0
        if neg_mask.any():
            be = sweep_df.loc[neg_mask, "tc_bps"].iloc[0]
            print(f"\n*** BREAKEVEN COST [{name}]: {be:.0f} bps ***")
        else:
            print(f"\n*** BREAKEVEN COST [{name}]: > {max(cfg.bps_grid):.0f} bps (sweep limit) ***")

    log.info("=" * 60)
    print(f"\nReport saved: {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
