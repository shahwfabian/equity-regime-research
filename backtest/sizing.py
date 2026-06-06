"""Position sizing: weight normalisation and ex-ante volatility targeting.

Volatility targeting
---------------------
When enabled, the gross factor exposure is scaled so that the portfolio
targets a constant *annualised* volatility.  The scale factor is:

    scale = target_ann_vol / trailing_ann_vol
    scale = clip(scale, 0, max_leverage)

where ``trailing_ann_vol`` is computed from the *past* ``lookback_days``
returns (strictly historical -- no full-sample variance).

Look-ahead trap
---------------
A common mistake (Barroso & Santa-Clara 2015, Liu et al.) is to compute vol
using the full sample (or test-window) standard deviation.  This constitutes
look-ahead: any parameter estimated on data through T and applied at t < T
uses future information.

Here the vol estimate at date t uses ONLY returns from day (t-lookback) to
day (t-1), accessed from the ``history`` argument supplied by the engine.
A unit test in :mod:`tests.test_backtest` asserts this formally.

Weight normalisation
--------------------
:func:`normalize_weights` ensures the signed-weight vector respects the
gross-exposure target:
  * ``mode="long_only"``  → sum of absolute weights = 1 (fully invested)
  * ``mode="long_short"`` → weights are already zero-net; no rescaling needed
                            (convention: gross = 2 for standard L/S factor)
  * ``mode="none"``       → pass through unchanged
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252
Weights = Dict[str, float]


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------
def normalize_weights(weights: Weights, mode: str = "none") -> Weights:
    """Normalise factor weights to enforce a gross-exposure target.

    Parameters
    ----------
    weights:
        Raw dict of ``{factor: exposure}``.
    mode:
        ``"long_only"``  – rescale so sum of exposures = 1.
        ``"long_short"`` – pass through (WML etc. are already zero-net).
        ``"none"``       – no normalisation (default).

    Returns
    -------
    Normalised weight dict.
    """
    if not weights:
        return weights
    if mode == "none":
        return weights
    if mode == "long_only":
        total = sum(abs(v) for v in weights.values())
        if total > 1e-12:
            return {k: v / total for k, v in weights.items()}
        return weights
    if mode == "long_short":
        return weights   # convention: factor is already zero-cost
    raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Ex-ante volatility targeting
# ---------------------------------------------------------------------------
@dataclass
class VolTargetConfig:
    """Configuration for ex-ante volatility targeting."""

    enabled: bool = False
    target_ann_vol: float = 0.10     # 10 % target
    lookback_days: int = 63          # trailing window
    max_leverage: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> "VolTargetConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            target_ann_vol=float(d.get("target_ann_vol", 0.10)),
            lookback_days=int(d.get("lookback_days", 63)),
            max_leverage=float(d.get("max_leverage", 2.0)),
        )


def vol_target_scalar(
    history: pd.DataFrame,
    current_weights: Weights,
    cfg: VolTargetConfig,
    factors: list[str] | None = None,
) -> float:
    """Compute the ex-ante volatility-targeting scale factor.

    Uses only the trailing ``cfg.lookback_days`` rows from *history* (all of
    which are strictly before the current date).

    Parameters
    ----------
    history:
        DataFrame of factor excess returns, index = date, strictly before
        current date.
    current_weights:
        Weight dict (used to compute the portfolio return series).
    cfg:
        Volatility targeting configuration.
    factors:
        Factor columns to use (defaults to intersection of weight keys and
        history columns).

    Returns
    -------
    float
        Scale multiplier ∈ (0, max_leverage].  Returns 1.0 if not enough
        history or vol target is disabled.
    """
    if not cfg.enabled or len(history) < cfg.lookback_days:
        return 1.0

    if factors is None:
        factors = [f for f in current_weights if f in history.columns]

    if not factors:
        return 1.0

    # Trailing portfolio returns (gross, excess) – strictly historical
    tail = history.iloc[-cfg.lookback_days:]
    port_ret = sum(
        current_weights.get(f, 0.0) * tail[f] for f in factors
    )
    if isinstance(port_ret, (int, float)):
        return 1.0

    trailing_vol = float(port_ret.std()) * np.sqrt(TRADING_DAYS)
    if trailing_vol < 1e-9:
        return 1.0

    scale = cfg.target_ann_vol / trailing_vol
    scale = float(np.clip(scale, 0.0, cfg.max_leverage))
    return scale


def apply_vol_targeting(
    weights: Weights,
    history: pd.DataFrame,
    cfg: VolTargetConfig,
) -> Weights:
    """Apply vol-targeting scalar to all factor weights.

    The scaling is purely multiplicative; uninvested cash is implicitly
    reduced/increased to maintain unit notional.

    Parameters
    ----------
    weights:
        Target weights before scaling.
    history:
        Historical factor returns strictly before current date.
    cfg:
        Vol-targeting configuration.

    Returns
    -------
    Scaled weight dict.
    """
    if not cfg.enabled:
        return weights

    scale = vol_target_scalar(history, weights, cfg)
    return {k: v * scale for k, v in weights.items()}
