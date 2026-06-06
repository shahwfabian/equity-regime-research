"""Transaction cost and slippage models.

Cost model
----------
On each rebalance date, the engine computes one-way turnover:

    turnover = sum(|w_new[f] - w_old[f]| for f in all_factors)

The total cost drag (as a fraction of NAV, applied that day) is:

    effective_slip = slippage_bps * turbulent_multiplier  if state_dependent AND regime==1
                   = slippage_bps                          otherwise
    cost_drag      = turnover * (tc_bps + effective_slip) / 10_000

State-dependent slippage motivation
------------------------------------
Regime-conditioned strategies trade most actively *at regime transitions*,
when bid-ask spreads widen and market-impact is highest.  Penalising
turbulent-state trades with higher slippage captures this asymmetry.  The
multiplier is configurable; set ``state_dependent_slippage: false`` in the
config to disable it.

Cost sweep
----------
:func:`cost_sweep` re-runs a strategy across a grid of ``tc_bps`` values
and returns the Sharpe ratio and net annualised return at each cost level.
The *breakeven cost* is the TC where net alpha → 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class CostConfig:
    """Transaction-cost and slippage configuration."""

    tc_bps: float = 10.0                    # one-way TC in basis points
    slippage_bps: float = 5.0               # base slippage in basis points
    state_dependent_slippage: bool = True   # scale slippage with regime
    turbulent_slippage_multiplier: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "CostConfig":
        return cls(
            tc_bps=float(d.get("tc_bps", 10.0)),
            slippage_bps=float(d.get("slippage_bps", 5.0)),
            state_dependent_slippage=bool(d.get("state_dependent_slippage", True)),
            turbulent_slippage_multiplier=float(
                d.get("turbulent_slippage_multiplier", 2.0)
            ),
        )

    @classmethod
    def zero(cls) -> "CostConfig":
        """No-cost model (gross returns)."""
        return cls(tc_bps=0.0, slippage_bps=0.0,
                   state_dependent_slippage=False, turbulent_slippage_multiplier=1.0)


def compute_cost_drag(
    turnover: float,
    regime: int,
    cost_cfg: CostConfig,
) -> float:
    """Compute the cost drag on a rebalance date as a fraction of NAV.

    Parameters
    ----------
    turnover:
        One-way turnover = sum(|w_new - w_old|).
    regime:
        Current regime label (0 = calm, 1 = turbulent).
    cost_cfg:
        Cost configuration.

    Returns
    -------
    float
        Cost drag as a decimal fraction (e.g. 0.0015 = 15 bps).
    """
    if turnover <= 0.0:
        return 0.0

    effective_slip = cost_cfg.slippage_bps
    if cost_cfg.state_dependent_slippage and regime == 1:
        effective_slip *= cost_cfg.turbulent_slippage_multiplier

    total_bps = cost_cfg.tc_bps + effective_slip
    return turnover * total_bps / 10_000.0


# ---------------------------------------------------------------------------
# Cost sweep
# ---------------------------------------------------------------------------
def cost_sweep(
    run_fn,          # callable(tc_bps) -> pd.Series of daily net returns
    bps_grid: List[float],
    strategy_name: str = "strategy",
    rf_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Re-run a strategy across a grid of TC levels and report key metrics.

    Parameters
    ----------
    run_fn:
        Callable that accepts ``tc_bps: float`` and returns a pd.Series of
        daily total net returns indexed by date.
    bps_grid:
        List of one-way TC values in basis points to sweep.
    strategy_name:
        Label used in the output DataFrame index.
    rf_series:
        Optional daily risk-free rate aligned to the return series.  Used
        for Sharpe calculation; defaults to 0 if not provided.

    Returns
    -------
    pd.DataFrame with columns [tc_bps, ann_return, ann_vol, sharpe,
                                  net_alpha_pct]
        ``net_alpha_pct`` is annualised net return relative to the 0-cost run.
    """
    rows = []
    gross_ann = None

    for bps in bps_grid:
        rets = run_fn(float(bps))
        if rets is None or len(rets) == 0:
            continue

        rf_daily = rf_series if rf_series is not None else pd.Series(0.0, index=rets.index)
        rf_daily = rf_daily.reindex(rets.index).fillna(0.0)

        ann_ret = float(rets.mean() * TRADING_DAYS)
        ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS))
        excess = rets - rf_daily
        ex_vol = excess.std()
        sharpe = float(excess.mean() / ex_vol * np.sqrt(TRADING_DAYS)) if ex_vol > 0 else np.nan

        if bps == 0:
            gross_ann = ann_ret

        rows.append({
            "strategy": strategy_name,
            "tc_bps": float(bps),
            "ann_return": round(ann_ret, 6),
            "ann_vol": round(ann_vol, 6),
            "sharpe": round(sharpe, 4),
        })

    result = pd.DataFrame(rows)
    if gross_ann is not None and len(result) > 0:
        result["net_alpha_pct"] = ((result["ann_return"] - gross_ann) * 100).round(4)

    # Identify breakeven cost (first cost level where ann_return turns negative
    # or falls below the RF level)
    if len(result) > 0:
        neg_mask = result["ann_return"] <= 0
        if neg_mask.any():
            breakeven = result.loc[neg_mask, "tc_bps"].iloc[0]
            log.info(
                "[cost_sweep] %s breakeven cost: ~%.0f bps (first non-positive ann return)",
                strategy_name, breakeven,
            )
        else:
            log.info(
                "[cost_sweep] %s: positive alpha at all tested costs (max %.0f bps)",
                strategy_name, max(bps_grid),
            )

    return result
