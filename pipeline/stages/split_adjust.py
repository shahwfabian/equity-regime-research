"""Split adjustment stage.

French data: NO split adjustment. Factor returns are computed from
the full CRSP universe using already-adjusted prices. We assert this
with a docstring-level declaration and a no-op function.

yfinance data: 'Adj Close' from Yahoo Finance is already adjusted for
splits AND dividends. We use it directly. This module also provides
a synthetic unit test to verify that our split-ratio logic would
round-trip correctly if raw Close were used instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.stages.transform import TransformResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# French assertion (no-op)
# ---------------------------------------------------------------------------

def assert_french_no_split_needed(factor_daily: pd.DataFrame) -> None:
    """
    Assert that French factor data requires no split adjustment.

    French factor returns are computed from the full CRSP universe using
    prices that are already split- and dividend-adjusted by CRSP prior to
    factor construction. The returns we receive are already in percent (÷100
    in our parser). There is no individual-stock price to adjust here.

    This function is a formal documentation point — it logs the assertion
    and checks that returns are in a sane range (not 100x off from a
    forgotten ÷100 step).
    """
    log.info(
        "[split_adjust] French factors: NO split adjustment needed. "
        "CRSP prices are pre-adjusted before factor construction."
    )
    # Sanity: all factor return columns should have |value| < 1.0 (i.e., < 100%)
    ret_cols = [c for c in factor_daily.columns if c != "rf"]
    for col in ret_cols:
        max_abs = factor_daily[col].abs().max()
        assert max_abs < 1.0, (
            f"French factor '{col}' has |value|={max_abs:.4f} ≥ 1.0. "
            "Did you forget to divide by 100?"
        )
    log.info("[split_adjust] French return-scale assertion passed (all |returns| < 1.0).")


# ---------------------------------------------------------------------------
# yfinance: already adjusted — verify via synthetic round-trip test
# ---------------------------------------------------------------------------

def verify_split_adjustment_roundtrip() -> dict:
    """
    Unit-testable synthetic 2:1 split round-trip test.

    Generates a synthetic price series, introduces a 2:1 split at t=50,
    applies backward-adjustment, and verifies that adjusted returns match
    the pre-split returns.

    Returns a dict with 'passed', 'max_abs_error', 'n_points'.
    This is test V8 in the validation report.
    """
    rng = np.random.default_rng(42)
    n = 100
    # Pre-split price starting at 100
    raw_prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))

    # Introduce a 2:1 split at t=50: prices halve, shares double
    split_t = 50
    split_ratio = 2.0
    raw_prices_with_split = raw_prices.copy()
    raw_prices_with_split[split_t:] /= split_ratio

    # Compute raw returns (biased at split point)
    raw_returns = pd.Series(raw_prices_with_split).pct_change().dropna()

    # Backward-adjust: multiply all prices before the split by 1/split_ratio
    adj_prices = raw_prices_with_split.copy()
    adj_prices[:split_t] /= split_ratio

    # Adjusted returns should match the true continuous returns
    adj_returns = pd.Series(adj_prices).pct_change().dropna()
    true_returns = pd.Series(raw_prices).pct_change().dropna()

    # Compare (skip the split row itself — index split_t-1 in returns)
    compare_idx = [i for i in range(len(true_returns)) if i != split_t - 1]
    max_err = float(
        (adj_returns.iloc[compare_idx] - true_returns.iloc[compare_idx]).abs().max()
    )
    passed = max_err < 1e-10

    log.info(
        "[split_adjust] Synthetic 2:1 split round-trip: passed=%s, max_abs_error=%.2e",
        passed, max_err,
    )
    return {"passed": passed, "max_abs_error": max_err, "n_points": len(compare_idx)}


# ---------------------------------------------------------------------------
# Main stage entry point
# ---------------------------------------------------------------------------

def split_adjust(transformed: TransformResult) -> TransformResult:
    """
    Apply split adjustment stage.

    French: no-op + assertion.
    yfinance (stock_daily): adj_close already adjusted — compute returns if not done.
    """
    # French assertion
    assert_french_no_split_needed(transformed.factor_daily)

    # yfinance: adj_close is already split-adjusted by Yahoo Finance
    if transformed.stock_daily is not None:
        log.info(
            "[split_adjust] stock_daily: adj_close sourced from Yahoo Finance "
            "'Adj Close' field — already split + dividend adjusted. "
            "No further adjustment applied."
        )

    # Run synthetic test (always, for V8 validation)
    verify_split_adjustment_roundtrip()

    return transformed   # pass-through; data unchanged
