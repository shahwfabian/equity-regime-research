"""Strategy definitions for the portfolio-level backtester.

All strategies share a common interface enforced by the abstract base class
:class:`Strategy`.  The no-look-ahead contract is enforced by design:

  * ``generate_weights(date, history, regime_history)`` receives *only* data
    strictly before *date* (the engine passes ``data.loc[:date_prev]``).
  * A perturbation test in :mod:`tests.test_backtest` verifies that adding a
    future observation never changes today's weights.

Strategies
----------
LongOnlyMomentum         – long UMD when prior return > 0, else cash
LongShortMomentum        – always hold UMD (unconditional WML)
MeanReversion            – always hold ST_Rev (short-term reversal factor)
RegimeConditionedMomentum– scale UMD exposure by filtered regime state
MarketBuyAndHold         – always hold the market (Mkt-RF + RF)
SixtyForty               – 60 % market + 40 % cash

Weight conventions
------------------
Weights are factor *exposures* (not portfolio shares):

  r_excess_port = sum(w_f * r_f_excess[t])
  r_total_port  = r_excess_port + rf[t]

Uninvested capital implicitly earns the daily risk-free rate.  A weight of
1.0 on ``umd`` means a 100 % notional exposure to the WML momentum factor;
a weight of 0.0 means no exposure (all cash, earning ``rf``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Weights = Dict[str, float]   # factor_name -> exposure


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Strategy(ABC):
    """Abstract base strategy enforcing the no-look-ahead contract.

    Subclasses must implement :meth:`generate_weights`.  The
    :meth:`is_rebalance_day` method controls when the engine calls
    ``generate_weights``; between rebalance dates weights are held constant
    (no drift approximation -- valid for small daily factor returns).
    """

    name: str = "base"
    rebalance: str = "monthly"   # "daily" | "weekly" | "monthly"

    @abstractmethod
    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        """Return target factor exposures using ONLY data through t-1.

        Parameters
        ----------
        date:
            Current trading date (NOT included in *history*).
        history:
            ``factor_daily`` DataFrame with all rows strictly before *date*.
            Columns include at minimum ``umd``, ``st_rev``, ``mkt_rf``, ``rf``.
        regime_history:
            Daily regime labels (0 = calm, 1 = turbulent) strictly before
            *date*, aligned to the same index as *history*.

        Returns
        -------
        dict mapping factor names (``"umd"``, ``"st_rev"``, ``"mkt_rf"``) to
        float exposure values.  Missing factors default to 0.0.
        """

    def is_rebalance_day(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
    ) -> bool:
        """Return True if the engine should rebalance on *date*."""
        if len(history) == 0:
            return True   # always initialise on first day
        if self.rebalance == "daily":
            return True
        if self.rebalance == "weekly":
            # Rebalance on the first trading day of each ISO week (Monday or
            # the first available trading day if Monday is a holiday).
            last_date = history.index[-1]
            return date.isocalendar()[1] != last_date.isocalendar()[1]
        if self.rebalance == "monthly":
            last_date = history.index[-1]
            return (date.year, date.month) != (last_date.year, last_date.month)
        raise ValueError(f"Unknown rebalance schedule: {self.rebalance!r}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rebalance={self.rebalance!r})"


# ---------------------------------------------------------------------------
# 1. Long-only momentum
# ---------------------------------------------------------------------------
class LongOnlyMomentum(Strategy):
    """Hold the UMD factor when the prior-period UMD return was positive.

    Signal: sign of the most-recent UMD return in *history*.
      * signal > 0 → full exposure (w_umd = 1.0, long momentum)
      * signal <= 0 → no exposure (w_umd = 0.0, all cash)

    Uses only 1 lag → zero risk of look-ahead.
    """

    name = "long_only_momentum"

    def __init__(self, rebalance: str = "monthly"):
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        if len(history) == 0:
            return {"umd": 0.0}
        prior_umd = float(history["umd"].iloc[-1])
        exposure = 1.0 if prior_umd > 0 else 0.0
        return {"umd": exposure}


# ---------------------------------------------------------------------------
# 2. Long-short momentum (unconditional WML)
# ---------------------------------------------------------------------------
class LongShortMomentum(Strategy):
    """Always hold the WML momentum factor at 100 % notional exposure.

    This is the *unconditional* long-short momentum benchmark used as the
    baseline against which the regime-conditioned strategy is tested.
    """

    name = "long_short_momentum"

    def __init__(self, rebalance: str = "monthly"):
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        return {"umd": 1.0}


# ---------------------------------------------------------------------------
# 3. Mean reversion (short-term reversal)
# ---------------------------------------------------------------------------
class MeanReversion(Strategy):
    """Always hold the ST_Rev factor at 100 % notional exposure.

    ST_Rev is the Kenneth French short-term reversal factor, which captures
    the 1-month mean-reversion anomaly.  Weekly rebalancing is appropriate
    because the signal decays within 4–5 weeks.
    """

    name = "mean_reversion"

    def __init__(self, rebalance: str = "weekly"):
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        return {"st_rev": 1.0}


# ---------------------------------------------------------------------------
# 4. Regime-conditioned momentum  [HEADLINE STRATEGY]
# ---------------------------------------------------------------------------
class RegimeConditionedMomentum(Strategy):
    """Scale UMD exposure by the filtered regime state.

    This is the paper's headline strategy: momentum works best in calm
    regimes (state 0) and exhibits crash risk in turbulent regimes (state 1).
    Gating exposure by the regime should improve the Sharpe ratio relative to
    the unconditional long-short momentum strategy.

    Parameters
    ----------
    calm_exposure:
        UMD weight when regime == 0 (calm).  Default 1.0.
    turbulent_exposure:
        UMD weight when regime == 1 (turbulent).  Default 0.0 (all cash).
    rebalance:
        Rebalancing schedule.  Default "monthly".

    No-look-ahead guarantee
    ----------------------
    The regime used at date *t* is the regime label for date *t-1*, read
    from ``regime_history.iloc[-1]``.  The regime classifier itself also
    uses only expanding-window statistics computed from data through *t-1*.
    """

    name = "regime_momentum"

    def __init__(
        self,
        calm_exposure: float = 1.0,
        turbulent_exposure: float = 0.0,
        rebalance: str = "monthly",
    ):
        self.calm_exposure = calm_exposure
        self.turbulent_exposure = turbulent_exposure
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        if len(regime_history) == 0:
            regime = 0   # default to calm before any regime data
        else:
            regime = int(regime_history.iloc[-1])   # last KNOWN regime (t-1)

        exposure = (
            self.calm_exposure if regime == 0 else self.turbulent_exposure
        )
        return {"umd": exposure}


# ---------------------------------------------------------------------------
# 5. Market buy-and-hold
# ---------------------------------------------------------------------------
class MarketBuyAndHold(Strategy):
    """Always hold the market: 100 % exposure to Mkt-RF (plus RF earned on cash).

    This is the simplest benchmark: buy the market and hold forever.
    Net return = Mkt-RF + RF = total market return.
    """

    name = "market_bah"

    def __init__(self, rebalance: str = "monthly"):
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        return {"mkt_rf": 1.0}


# ---------------------------------------------------------------------------
# 6. 60/40 constant mix
# ---------------------------------------------------------------------------
class SixtyForty(Strategy):
    """60 % market exposure + 40 % cash (earns RF).

    A common balanced-fund benchmark.  Net return = 0.60 * Mkt-RF + RF.
    Monthly rebalancing to maintain the target allocation.
    """

    name = "sixty_forty"

    def __init__(self, mkt_weight: float = 0.60, rebalance: str = "monthly"):
        self.mkt_weight = mkt_weight
        self.rebalance = rebalance

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        regime_history: pd.Series,
    ) -> Weights:
        return {"mkt_rf": self.mkt_weight}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type] = {
    "long_only_momentum":  LongOnlyMomentum,
    "long_short_momentum": LongShortMomentum,
    "mean_reversion":      MeanReversion,
    "regime_momentum":     RegimeConditionedMomentum,
    "market_bah":          MarketBuyAndHold,
    "sixty_forty":         SixtyForty,
}


def build_strategy(name: str, cfg_dict: dict) -> Strategy:
    """Instantiate a Strategy from a config-dict entry."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}.  Available: {list(_REGISTRY)}")

    kwargs: dict = {}
    if "rebalance" in cfg_dict:
        kwargs["rebalance"] = cfg_dict["rebalance"]
    if name == "regime_momentum":
        kwargs["calm_exposure"] = cfg_dict.get("calm_exposure", 1.0)
        kwargs["turbulent_exposure"] = cfg_dict.get("turbulent_exposure", 0.0)
    if name == "sixty_forty":
        kwargs["mkt_weight"] = cfg_dict.get("mkt_weight", 0.60)

    return cls(**kwargs)
