"""ETL orchestrator: extract -> clean -> transform -> split_adjust -> load -> validate.

Each stage logs row counts in/out and timing.
All failures are caught, logged, and surfaced in the validation report.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _setup_logging(cfg) -> None:
    """Configure structured logging: INFO to console + rotating file."""
    log_path = Path(cfg.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    log.info("Logging initialised -> %s (level=%s)", log_path, cfg.logging.level)


def run_etl(
    cfg,
    force_download: bool = False,
    skip_yfinance: bool = False,
) -> dict:
    """
    Execute the full ETL pipeline.

    Parameters
    ----------
    cfg            : ETLConfig
    force_download : re-download even if cached
    skip_yfinance  : disable yfinance regardless of config

    Returns
    -------
    dict with keys: store, validation_results, report_path, elapsed_s
    """
    t_pipeline = time.time()

    if skip_yfinance:
        cfg.sources.yfinance.enabled = False
        log.info("[etl] yfinance disabled via --skip-yfinance")

    log.info("=" * 60)
    log.info("[etl] Pipeline: %s v%s", cfg.name, cfg.version)
    log.info("[etl] Window: %s -> %s", cfg.window.start_date, cfg.window.end_date)
    log.info("[etl] Backend: %s", cfg.store.backend)
    log.info("=" * 60)

    # ── 1. EXTRACT ────────────────────────────────────────────────
    from pipeline.stages.extract import extract
    t0 = time.time()
    log.info("[etl] Stage 1/5: Extract")
    extracted = extract(cfg, force=force_download)
    log.info("[etl] Extract done in %.1fs", time.time() - t0)

    # ── 2. CLEAN ──────────────────────────────────────────────────
    from pipeline.stages.clean import clean
    t0 = time.time()
    log.info("[etl] Stage 2/5: Clean")
    cleaned = clean(extracted, cfg)
    log.info("[etl] Clean done in %.1fs", time.time() - t0)

    # ── 3. TRANSFORM ──────────────────────────────────────────────
    from pipeline.stages.transform import transform
    t0 = time.time()
    log.info("[etl] Stage 3/5: Transform")
    transformed = transform(cleaned)
    log.info("[etl] Transform done in %.1fs", time.time() - t0)

    # ── 4. SPLIT ADJUST ───────────────────────────────────────────
    from pipeline.stages.split_adjust import split_adjust
    t0 = time.time()
    log.info("[etl] Stage 4/5: Split Adjust")
    transformed = split_adjust(transformed)
    log.info("[etl] Split adjust done in %.1fs", time.time() - t0)

    # ── 5. LOAD ───────────────────────────────────────────────────
    from pipeline.stages.load import load
    t0 = time.time()
    log.info("[etl] Stage 5/5: Load")
    store = load(transformed, cfg)
    log.info("[etl] Load done in %.1fs", time.time() - t0)

    # ── VALIDATE ──────────────────────────────────────────────────
    from pipeline.validate.checks import run_all_checks
    from pipeline.validate.report import render_report
    log.info("[etl] Running validation checks…")
    t0 = time.time()
    results = run_all_checks(
        factor_daily=transformed.factor_daily,
        factor_monthly=transformed.factor_monthly,
        market_daily=transformed.market_daily,
        store=store,
        cfg=cfg,
    )
    log.info("[etl] Validation done in %.1fs", time.time() - t0)

    report_path = render_report(results, cfg)

    total_elapsed = time.time() - t_pipeline
    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass

    log.info("=" * 60)
    log.info("[etl] PIPELINE COMPLETE in %.1fs", total_elapsed)
    log.info("[etl] Validation: %d PASS / %d FAIL", n_pass, n_fail)
    log.info("[etl] Report: %s", report_path)
    log.info("=" * 60)

    return {
        "store": store,
        "validation_results": results,
        "report_path": report_path,
        "elapsed_s": total_elapsed,
        "n_pass": n_pass,
        "n_fail": n_fail,
    }


def main_cli() -> None:
    """Console script entry point: equity-etl"""
    parser = argparse.ArgumentParser(
        description="Equity Regime Research — ETL Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/etl.yaml",
        help="Path to ETL config YAML",
    )
    parser.add_argument(
        "--backend", choices=["parquet", "postgres"], default=None,
        help="Override store backend (parquet or postgres)",
    )
    parser.add_argument(
        "--skip-yfinance", action="store_true",
        help="Disable yfinance download regardless of config",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download, ignore cached files",
    )
    args = parser.parse_args()

    from pipeline.config import ETLConfig
    cfg = ETLConfig.load(args.config)
    cfg.validate()

    if args.backend:
        cfg.store.backend = args.backend
        log.info("Backend overridden via CLI: %s", args.backend)

    _setup_logging(cfg)

    result = run_etl(
        cfg,
        force_download=args.force,
        skip_yfinance=args.skip_yfinance,
    )

    sys.exit(0 if result["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main_cli()
