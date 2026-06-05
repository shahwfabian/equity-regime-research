"""Generate synthetic stock×day panels via a 2-state Markov regime process.

Regime 0 (calm):   low vol, positive intermediate-horizon autocorrelation (momentum)
Regime 1 (turbulent): high vol, negative short-horizon autocorrelation (reversal)

Returns
-------
stock_day  : pd.DataFrame  columns [date, permno, ret, prc, mktcap, vol]
market_day : pd.DataFrame  columns [date, mkt_ret, rf, realized_vol, vix_like]
regime_path: pd.Series     index=date, values in {0,1}  (TRUE hidden states)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import SyntheticConfig


def _simulate_regimes(
    n_days: int,
    trans: list[list[float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate hidden Markov regime path via forward recursion."""
    P = np.array(trans)
    n_states = P.shape[0]
    states = np.empty(n_days, dtype=int)
    # Stationary distribution as initial state probabilities
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.abs(eigvecs[:, idx])
    pi /= pi.sum()
    states[0] = rng.choice(n_states, p=pi)
    for t in range(1, n_days):
        states[t] = rng.choice(n_states, p=P[states[t - 1]])
    return states


def generate(cfg: SyntheticConfig, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate synthetic equity panel data.

    Parameters
    ----------
    cfg  : SyntheticConfig
    seed : int

    Returns
    -------
    stock_day, market_day, regime_path
    """
    rng = np.random.default_rng(seed)

    n_stocks = cfg.n_stocks
    n_days = cfg.n_days
    mu = np.array(cfg.regime_mu)
    sigma = np.array(cfg.regime_sigma)
    mom_str = cfg.momentum_strength
    rev_str = cfg.reversal_strength

    # 1. Regime path
    regime_path = _simulate_regimes(n_days, cfg.transition_matrix, rng)

    # 2. Market factor return (drives systematic component)
    mkt_noise = rng.standard_normal(n_days)
    mkt_ret = np.where(
        regime_path == 0,
        mu[0] + sigma[0] * mkt_noise,
        mu[1] + sigma[1] * mkt_noise,
    )

    # 3. Rolling lookback windows for momentum / reversal signals
    MOM_LB = 252   # match factor config defaults
    REV_LB = 21

    # Pre-allocate stock return matrix (n_days x n_stocks)
    stock_rets = np.zeros((n_days, n_stocks))

    # Independent idiosyncratic shocks
    idio = rng.standard_normal((n_days, n_stocks))

    # Beta drawn once per stock in [0.5, 1.5]
    betas = rng.uniform(0.5, 1.5, size=n_stocks)

    for t in range(n_days):
        reg = regime_path[t]
        s = sigma[reg]

        # Systematic + idiosyncratic
        base = betas * mkt_ret[t] + 0.5 * s * idio[t]

        # Momentum: positive autocorrelation at intermediate horizon (calm regime)
        if t >= MOM_LB and reg == 0:
            past_cum = stock_rets[t - MOM_LB : t - 21].mean(axis=0)
            base += mom_str * past_cum

        # Reversal: negative short-horizon autocorrelation (stronger in turbulent)
        rev_mult = rev_str * (1.0 if reg == 0 else 2.0)
        if t >= REV_LB:
            past_short = stock_rets[t - REV_LB : t].mean(axis=0)
            base -= rev_mult * past_short

        stock_rets[t] = base

    # 4. Simulate prices starting at $20–$100 range
    init_prices = rng.uniform(20.0, 100.0, size=n_stocks)
    prices = np.empty((n_days, n_stocks))
    prices[0] = init_prices
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * np.exp(stock_rets[t])
    prices = np.maximum(prices, 0.01)  # floor at penny

    # 5. Market cap (shares * price) — simulate shares once
    shares = rng.uniform(1e6, 1e9, size=n_stocks)
    mktcaps = prices * shares[np.newaxis, :]

    # 6. Volume (simulated, not used analytically)
    base_vol = rng.uniform(1e5, 5e6, size=n_stocks)
    volume = (
        base_vol[np.newaxis, :]
        * rng.lognormal(0, 0.5, size=(n_days, n_stocks))
    ).astype(int)

    # 7. Build business dates starting 2009-01-05
    dates = pd.bdate_range("2009-01-05", periods=n_days, freq="B")

    # 8. Stock permno IDs
    permnos = [f"SYN{i:05d}" for i in range(n_stocks)]

    # --- Assemble stock_day panel ---
    rows = []
    for i, perm in enumerate(permnos):
        df_i = pd.DataFrame(
            {
                "date": dates,
                "permno": perm,
                "ret": stock_rets[:, i],
                "prc": prices[:, i],
                "mktcap": mktcaps[:, i],
                "vol": volume[:, i],
            }
        )
        rows.append(df_i)
    stock_day = pd.concat(rows, ignore_index=True)

    # --- Market day panel ---
    realized_vol = pd.Series(np.abs(mkt_ret)).rolling(21).std().bfill().values
    vix_like = realized_vol * np.sqrt(252) * 100  # annualized % approximation
    rf = np.full(n_days, 0.00015)  # ~4% annual / 252

    market_day = pd.DataFrame(
        {
            "date": dates,
            "mkt_ret": mkt_ret,
            "rf": rf,
            "realized_vol": realized_vol,
            "vix_like": vix_like,
        }
    )

    # --- True regime path ---
    regime_series = pd.Series(regime_path, index=dates, name="true_regime")

    return stock_day, market_day, regime_series
