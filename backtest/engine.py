"""Backtest engine: event loop with rebalancing, drift, and cost application.

Event loop (per trading day t)
-------------------------------
1.  *Regime classification*: compute regime state for t-1 from history using
    the expanding-window vol-percentile rule (inherently causal).
2.  *Rebalance check*: call ``strategy.is_rebalance_day(date, history)``.
3.  *If rebalance day*:
    a. Generate new target weights from ``strategy.generate_weights(date,
       history[:-0], regime_history[:-0])``.  History ends at t-1 so no
       future data reaches the strategy.
    b. Scale weights via ex-ante vol targeting (trailing vol only).
    c. Compute one-way turnover = sum(|w_new[f] - w_old[f]|).
    d. Compute cost drag = f(turnover, regime[t-1], cost_cfg).
    e. Update current_weights = new_weights.
4.  *Gross return*: r_excess = sum(w[f] * factor[f][t] for each factor).
    r_gross = r_excess + rf[t].
5.  *Net return*: r_net = r_gross - cost_drag   (cost only on rebalance days).
6.  Record daily result.

Regime classification (vol_rule)
---------------------------------
regime_t = 1  if realized_vol_21[t] > expanding_quantile(vol_pct, 1..t-1)
         = 0  otherwise

Using the realised vol from t itself would be look-ahead.  We shift by 1: the
regime used at time t is derived from data through t-1.  ``realized_vol_21``
in market_daily is already a trailing 21-day measure (no future obs); the
regime threshold is the *expanding* quantile up to t-1 (no future obs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.costs import CostConfig, compute_cost_drag
from backtest.sizing import VolTargetConfig, apply_vol_targeting
from backtest.strategy import Strategy, Weights

log = logging.getLogger(__name__)

TRADING_DAYS = 252
FACTOR_COLS = ["umd", "st_rev", "mkt_rf", "smb", "hml", "rmw", "cma", "lt_rev"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class EngineConfig:
    """Engine configuration (extracted from backtest.yaml)."""

    cost: CostConfig = field(default_factory=CostConfig)
    sizing: VolTargetConfig = field(default_factory=VolTargetConfig)
    regime_vol_pct: float = 0.75    # expanding percentile for vol rule

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        cost = CostConfig.from_dict(d.get("costs", {}))
        sizing = VolTargetConfig.from_dict(
            d.get("sizing", {}).get("vol_targeting", {})
        )
        regime_vol_pct = float(
            d.get("strategies", {})
            .get("regime_momentum", {})
            .get("regime_vol_pct", 0.75)
        )
        return cls(cost=cost, sizing=sizing, regime_vol_pct=regime_vol_pct)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DayResult:
    date: pd.Timestamp
    r_gross: float
    r_net: float
    r_excess: float
    turnover: float
    cost_drag: float
    regime: int
    is_rebalance: bool


@dataclass
class RunResult:
    """Output of one strategy run over a date range."""

    strategy_name: str
    returns: pd.DataFrame       # columns: r_gross, r_net, r_excess, turnover, cost_drag, regime
    regime_series: pd.Series    # 0/1 regime per day (causal)
    n_rebalances: int
    total_cost_drag: float

    @property
    def net_returns(self) -> pd.Series:
        return self.returns["r_net"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.returns["r_gross"]


# ---------------------------------------------------------------------------
# Regime classification (expanding vol-percentile rule)
# ---------------------------------------------------------------------------
def _compute_regime_series(
    market_daily: pd.DataFrame,
    vol_pct: float = 0.75,
) -> pd.Series:
    """Classify each day as calm (0) or turbulent (1) via expanding vol rule.

    Uses ``realized_vol_21`` from *market_daily*, shifted by 1 day so the
    regime assigned to date t is based entirely on data through t-1.

    Parameters
    ----------
    market_daily:
        DataFrame with at least ``realized_vol_21`` and (optionally)
        ``mkt_ret`` columns.  Index = DatetimeIndex.
    vol_pct:
        Expanding quantile threshold.  If realised vol > this quantile of
        all past vol, classify as turbulent.

    Returns
    -------
    pd.Series
        Daily regime labels (0 = calm, 1 = turbulent), same index as input.
    """
    vol_col = "realized_vol_21"
    if vol_col in market_daily.columns:
        vol = market_daily[vol_col].copy()
    else:
        # Fall back to computing from mkt_ret
        vol = (
            market_daily["mkt_ret"]
            .rolling(21, min_periods=11)
            .std()
            .mul(np.sqrt(TRADING_DAYS))
        )

    # Expanding quantile threshold (no look-ahead): each value uses all past
    vol_ffilled = vol.ffill()
    threshold = vol_ffilled.expanding().quantile(vol_pct).shift(1)

    regime = (vol_ffilled > threshold).astype(int)
    regime = regime.fillna(0).astype(int)
    return regime


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    """Simulate a strategy on daily factor return data.

    Parameters
    ----------
    strategy:
        A :class:`~backtest.strategy.Strategy` instance.
    factor_daily:
        DataFrame of daily factor excess returns (index = DatetimeIndex).
        Must include at least the factors referenced by the strategy.
    market_daily:
        DataFrame with ``rf``, ``realized_vol_21``, etc.
    engine_cfg:
        Engine configuration (costs, sizing, regime threshold).
    """

    def __init__(
        self,
        strategy: Strategy,
        factor_daily: pd.DataFrame,
        market_daily: pd.DataFrame,
        engine_cfg: EngineConfig | None = None,
    ):
        self.strategy = strategy
        # Align both tables to a common date index
        common_idx = factor_daily.index.intersection(market_daily.index)
        self.factor = factor_daily.loc[common_idx].sort_index()
        self.market = market_daily.loc[common_idx].sort_index()
        self.cfg = engine_cfg if engine_cfg is not None else EngineConfig()
        self._regime_series: Optional[pd.Series] = None

    @property
    def regime_series(self) -> pd.Series:
        """Lazy-compute regime series (cached)."""
        if self._regime_series is None:
            self._regime_series = _compute_regime_series(
                self.market, vol_pct=self.cfg.regime_vol_pct
            )
        return self._regime_series

    def run(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        cost_cfg_override: Optional[CostConfig] = None,
    ) -> RunResult:
        """Execute the backtest and return daily results.

        Uses a vectorised approach:
        1. Determine rebalance dates by scanning the schedule (O(N) but fast).
        2. For each rebalance date call ``generate_weights`` once (O(R) calls
           where R is the number of rebalance events, typically << N).
        3. Assign constant weights to each holding period.
        4. Compute gross returns via a vectorised dot product (O(N) NumPy).
        5. Apply cost drag only on rebalance rows.

        Parameters
        ----------
        start / end:
            Optional date range filter.  Defaults to full data window.
        cost_cfg_override:
            If provided, overrides ``self.cfg.cost`` (used by cost sweep).

        Returns
        -------
        :class:`RunResult`
        """
        factor = self.factor
        market = self.market
        regime_all = self.regime_series

        if start is not None:
            factor = factor.loc[start:]
            market = market.loc[start:]
            regime_all = regime_all.loc[start:]
        if end is not None:
            factor = factor.loc[:end]
            market = market.loc[:end]
            regime_all = regime_all.loc[:end]

        cost_cfg = cost_cfg_override if cost_cfg_override is not None else self.cfg.cost
        dates = factor.index
        n = len(dates)
        if n == 0:
            empty = pd.DataFrame(columns=["r_gross","r_net","r_excess","turnover","cost_drag","regime","is_rebalance"])
            empty.index.name = "date"
            return RunResult(self.strategy.name, empty, regime_all, 0, 0.0)

        avail_factors = [c for c in FACTOR_COLS if c in factor.columns]
        factor_arr = factor[avail_factors].values           # (N, F) float64
        rf_arr = market["rf"].values if "rf" in market.columns else np.zeros(n)
        regime_arr = regime_all.values.astype(int)

        # ── Phase 1: identify rebalance indices and compute weights ─────────
        rebal_indices: List[int] = []
        weights_per_rebal: List[np.ndarray] = []   # each (F,)
        current_weights: Weights = {}
        w_vec = np.zeros(len(avail_factors))

        for i in range(n):
            history = factor.iloc[:i]
            reg_history = regime_all.iloc[:i]
            is_rebal = self.strategy.is_rebalance_day(dates[i], history)
            if is_rebal:
                new_weights = self.strategy.generate_weights(
                    dates[i], history, reg_history
                )
                new_weights = apply_vol_targeting(
                    new_weights, history, self.cfg.sizing
                )
                new_vec = np.array([new_weights.get(f, 0.0) for f in avail_factors])
                rebal_indices.append(i)
                weights_per_rebal.append(new_vec)
                current_weights = dict(new_weights)
                w_vec = new_vec.copy()

        # ── Phase 2: build weight matrix W (N, F) ──────────────────────────
        W = np.zeros((n, len(avail_factors)))
        if rebal_indices:
            for j, idx in enumerate(rebal_indices):
                next_idx = rebal_indices[j + 1] if j + 1 < len(rebal_indices) else n
                W[idx:next_idx] = weights_per_rebal[j]

        # ── Phase 3: vectorised return computation ──────────────────────────
        r_excess_arr = (W * factor_arr).sum(axis=1)         # (N,) dot product
        r_gross_arr = r_excess_arr + rf_arr
        r_net_arr = r_gross_arr.copy()

        # Cost drag: only on rebalance days
        turnover_arr = np.zeros(n)
        cost_drag_arr = np.zeros(n)
        rebal_mask = np.zeros(n, dtype=bool)
        prev_w = np.zeros(len(avail_factors))
        for j, idx in enumerate(rebal_indices):
            rebal_mask[idx] = True
            to = float(np.abs(weights_per_rebal[j] - prev_w).sum())
            reg = int(regime_arr[idx - 1]) if idx > 0 else 0
            cd = compute_cost_drag(to, reg, cost_cfg)
            turnover_arr[idx] = to
            cost_drag_arr[idx] = cd
            r_net_arr[idx] -= cd
            prev_w = weights_per_rebal[j].copy()

        n_rebalances = len(rebal_indices)
        total_cost_drag = float(cost_drag_arr.sum())

        df = pd.DataFrame({
            "r_gross": r_gross_arr,
            "r_net": r_net_arr,
            "r_excess": r_excess_arr,
            "turnover": turnover_arr,
            "cost_drag": cost_drag_arr,
            "regime": regime_arr,
            "is_rebalance": rebal_mask,
        }, index=dates)
        df.index.name = "date"

        log.info(
            "[engine] %s: %d days, %d rebalances, total cost drag %.4f%%",
            self.strategy.name, n, n_rebalances, total_cost_drag * 100,
        )

        return RunResult(
            strategy_name=self.strategy.name,
            returns=df,
            regime_series=regime_all,
            n_rebalances=n_rebalances,
            total_cost_drag=total_cost_drag,
        )
