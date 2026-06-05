"""Smoke test: full pipeline end-to-end on synthetic data (fast config)."""

import pytest
from pathlib import Path


@pytest.mark.slow
def test_full_pipeline_synthetic(tmp_path):
    """Run the entire pipeline on synthetic data and verify key outputs exist."""
    from equity_regime.config import Config
    from equity_regime.pipeline import run_pipeline

    cfg = Config.load(Path("configs/default.yaml"))
    # Override for speed
    cfg.synthetic.n_stocks = 30
    cfg.synthetic.n_days = 600
    cfg.run.output_dir = str(tmp_path)
    cfg.factors.momentum_lookback = 120
    cfg.factors.momentum_skip = 21
    cfg.data.min_history_days = 60

    results = run_pipeline(cfg, use_synthetic=True)

    # Verify key outputs
    assert "stock_day" in results
    assert "mom_ls" in results
    assert "perf_table" in results
    assert "report_path" in results

    report = Path(results["report_path"])
    assert report.exists(), f"Report not found at {report}"
    assert report.stat().st_size > 1000, "Report file is too small"

    # Check figures were created
    assert len(results["figures"]) >= 6

    # Perf table should have strategies
    perf = results["perf_table"]
    assert len(perf) >= 1
