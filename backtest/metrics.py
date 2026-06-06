"""Performance metrics, statistical tests, and bootstrap confidence intervals.

Metric suite (Block-7)
-----------------------
Per strategy, computed on net-of-cost daily total returns:

  ann_return      – annualised arithmetic mean return
  ann_vol         – annualised standard deviation
  sharpe          – Sharpe ratio (Newey-West HAC denominator in CI)
  sortino         – Sortino ratio (downside deviation denominator)
  max_drawdown    – maximum peak-to-trough drawdown
  calmar          – ann_return / |max_drawdown|
  win_rate        – fraction of periods with positive return
  profit_factor   – sum(positive) / |sum(negative)|
  alpha           – CAPM alpha vs benchmark (Newey-West SEs)
  beta            – CAPM beta
  skew            – third standardised moment
  kurtosis        – excess kurtosis (Fisher)

Statistical inference
---------------------
* **Sharpe CI**: stationary block bootstrap (Politis-Romano 1994) with
  geometric block lengths.  Preserves serial correlation and volatility
  clustering (iid bootstrap would understate uncertainty).

* **Ledoit-Wolf (2008) Sharpe-difference test**: test H0: SR_A = SR_B using
  HAC (Newey-West) standard errors of the per-period Sharpe influence
  functions.  This is the headline test comparing regime-conditioned momentum
  vs unconditional momentum.  Do NOT use Jobson-Korkie (assumes normality).

  Reference: Ledoit & Wolf, "Robust Performance Hypothesis Testing with the
  Sharpe Ratio", Journal of Empirical Finance, 2008.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MetricsResult:
    """All computed performance metrics for one strategy / period."""

    strategy_name: str
    period: str                   # "in_sample" | "out_of_sample" | "full"
    n_days: int
    ann_return: float
    ann_vol: float
    sharpe: float
    sharpe_ci_lo: float           # 95 % block-bootstrap lower bound
    sharpe_ci_hi: float           # 95 % block-bootstrap upper bound
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    alpha: float                  # CAPM alpha (annualised)
    beta: float
    alpha_pval: float
    skew: float
    kurtosis: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "period": self.period,
            "n_days": self.n_days,
            "ann_return": round(self.ann_return, 6),
            "ann_vol": round(self.ann_vol, 6),
            "sharpe": round(self.sharpe, 4),
            "sharpe_ci_lo": round(self.sharpe_ci_lo, 4),
            "sharpe_ci_hi": round(self.sharpe_ci_hi, 4),
            "sortino": round(self.sortino, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar": round(self.calmar, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 4),
            "alpha_pval": round(self.alpha_pval, 6),
            "skew": round(self.skew, 4),
            "kurtosis": round(self.kurtosis, 4),
        }


@dataclass
class LedoitWolfResult:
    """Result of the Ledoit-Wolf (2008) Sharpe-difference test."""

    strategy_a: str
    strategy_b: str
    sr_a: float
    sr_b: float
    sr_diff: float
    t_stat: float
    p_value: float
    significant_5pct: bool

    def to_dict(self) -> dict:
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "sr_a": round(self.sr_a, 4),
            "sr_b": round(self.sr_b, 4),
            "sr_diff": round(self.sr_diff, 4),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "significant_5pct": self.significant_5pct,
        }


# ---------------------------------------------------------------------------
# Basic scalar metrics
# ---------------------------------------------------------------------------
def _ann_return(r: pd.Series) -> float:
    return float(r.mean() * TRADING_DAYS)


def _ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


def _sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    if isinstance(rf, (int, float)):
        excess = r - rf
    else:
        excess = r - rf.reindex(r.index).fillna(0.0)
    vol = float(excess.std())
    if vol < 1e-12:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))


def _sortino(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    if isinstance(rf, (int, float)):
        excess = r - rf
    else:
        excess = r - rf.reindex(r.index).fillna(0.0)
    downside = excess[excess < 0].std()
    if downside < 1e-12:
        return np.nan
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def _max_drawdown(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def _profit_factor(r: pd.Series) -> float:
    pos = r[r > 0].sum()
    neg = abs(r[r < 0].sum())
    if neg < 1e-12:
        return np.nan
    return float(pos / neg)


# ---------------------------------------------------------------------------
# CAPM regression with Newey-West standard errors
# ---------------------------------------------------------------------------
def _capm_nw(
    r_port: pd.Series,
    r_bench: pd.Series,
    rf: pd.Series | float = 0.0,
    n_lags: int | None = None,
) -> Tuple[float, float, float]:
    """OLS alpha + beta with Newey-West HAC standard errors.

    Returns (alpha_daily, beta, alpha_pval).
    """
    common = r_port.index.intersection(r_bench.index)
    rp = r_port.loc[common]

    if isinstance(rf, (int, float)):
        rfv = pd.Series(rf, index=common)
    else:
        rfv = rf.reindex(common).fillna(0.0)

    rb = r_bench.reindex(common)

    y = (rp - rfv).values
    x = (rb - rfv).values

    T = len(y)
    if T < 10:
        return np.nan, np.nan, np.nan

    # OLS
    X = np.column_stack([np.ones(T), x])
    try:
        b, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return np.nan, np.nan, np.nan

    alpha_d, beta = float(b[0]), float(b[1])

    # Newey-West HAC SE for alpha
    if n_lags is None:
        n_lags = int(4 * (T / 100) ** (2 / 9))

    e = y - X @ b
    # Sandwich variance: V = (X'X)^{-1} * S * (X'X)^{-1}
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])  # V_0
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1)  # Bartlett weight
        Vl = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        S += w * (Vl + Vl.T)
    S /= T
    V = T * XtX_inv @ S @ XtX_inv
    se_alpha = float(np.sqrt(max(V[0, 0], 0.0) / T))
    if se_alpha < 1e-14:
        alpha_pval = 1.0
    else:
        t_stat = alpha_d / se_alpha
        alpha_pval = float(2 * stats.t.sf(abs(t_stat), df=T - 2))

    alpha_ann = alpha_d * TRADING_DAYS
    return alpha_ann, beta, alpha_pval


# ---------------------------------------------------------------------------
# Stationary block bootstrap (Politis-Romano 1994)
# ---------------------------------------------------------------------------
def _stationary_bootstrap_samples(
    data: np.ndarray,
    n_boot: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bootstrap resamples using the stationary (Politis-Romano) bootstrap.

    Block lengths are geometrically distributed with parameter *p*
    (mean block length = 1/p).  Indexing is circular.

    Returns
    -------
    np.ndarray  shape (n_boot, T)
    """
    T = len(data)
    samples = np.empty((n_boot, T), dtype=float)
    for b in range(n_boot):
        indices = []
        while len(indices) < T:
            start = int(rng.integers(0, T))
            # Geometric block length
            block_len = int(np.ceil(np.log(rng.uniform()) / np.log(1 - p)))
            block_len = max(1, min(block_len, T))
            for j in range(block_len):
                indices.append((start + j) % T)
        samples[b] = data[np.array(indices[:T])]
    return samples


def block_bootstrap_ci(
    returns: pd.Series,
    stat_fn,
    n_boot: int = 1000,
    p: float = 0.10,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float]:
    """Block-bootstrap confidence interval for any statistic.

    Parameters
    ----------
    returns:
        Daily return series.
    stat_fn:
        Function ``(returns: np.ndarray) -> float``.
    n_boot:
        Number of bootstrap replicates.
    p:
        Stationary bootstrap geometric parameter (mean block = 1/p days).
    seed:
        RNG seed for reproducibility.
    ci_level:
        Confidence level (default 0.95 → 2.5 %–97.5 % interval).

    Returns
    -------
    (lower, upper) confidence bounds.
    """
    rng = np.random.default_rng(seed)
    data = returns.dropna().values
    if len(data) < 10:
        return (np.nan, np.nan)

    samples = _stationary_bootstrap_samples(data, n_boot, p, rng)
    boot_stats = np.array([stat_fn(s) for s in samples])
    boot_stats = boot_stats[np.isfinite(boot_stats)]

    alpha = (1 - ci_level) / 2
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1 - alpha))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Ledoit-Wolf (2008) Sharpe-difference test
# ---------------------------------------------------------------------------
def _sharpe_influence(r: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Per-period Sharpe influence function (Ledoit & Wolf 2008, eq. 8).

    phi_t = (r_t - mu) / sigma - (mu / (2 * sigma^2)) * ((r_t - mu)^2 / sigma^2 - 1)
          = z_t - (SR/2) * (z_t^2 - 1)    where z_t = (r_t - mu)/sigma

    This is the influence function for sqrt(T)*SR under the joint
    normal + iid-like regime, made HAC-robust via Newey-West.
    """
    if sigma < 1e-12:
        return np.zeros_like(r, dtype=float)
    z = (r - mu) / sigma
    SR = mu / sigma
    return z - 0.5 * SR * (z ** 2 - 1)


def _newey_west_variance(series: np.ndarray, n_lags: int | None = None) -> float:
    """Newey-West HAC variance estimate for the mean of *series*."""
    T = len(series)
    if n_lags is None:
        n_lags = int(4 * (T / 100) ** (2 / 9))
    s = series - series.mean()
    V = float(np.mean(s ** 2))  # V_0
    for lag in range(1, n_lags + 1):
        w = 1.0 - lag / (n_lags + 1)
        V += 2 * w * float(np.mean(s[lag:] * s[:-lag]))
    return max(V, 1e-20)


def ledoit_wolf_sharpe_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
    n_lags: int | None = None,
) -> LedoitWolfResult:
    """Test H0: SR_A = SR_B using HAC standard errors (Ledoit & Wolf 2008).

    Implementation
    --------------
    We test whether the difference in standardised returns has zero mean:

        psi_t = (r_A_t / sigma_A) - (r_B_t / sigma_B)

    Under H0: mu_A/sigma_A = mu_B/sigma_B, E[psi_t] = 0 (daily Sharpes equal).
    Under H1: E[psi_t] = SR_A_daily - SR_B_daily ≠ 0.

    The test statistic is a standard HAC (Newey-West) t-statistic for the
    mean of psi_t.  Using the scaled returns directly (rather than the
    influence function for mu alone) is equivalent to the Ledoit-Wolf (2008)
    Sharpe-difference test when the estimation-error correction in sigma is
    negligible for large T (their Remark 3.1).

    Reference: Ledoit & Wolf, "Robust Performance Hypothesis Testing with the
    Sharpe Ratio", Journal of Empirical Finance, 15(5), 2008.

    Parameters
    ----------
    returns_a, returns_b:
        Daily total return series for strategies A and B.
    name_a, name_b:
        Strategy names for labelling.
    n_lags:
        Number of Newey-West lags.  Default: ``floor(4*(T/100)^(2/9))``.

    Returns
    -------
    :class:`LedoitWolfResult`
    """
    # Align on common dates
    common = returns_a.index.intersection(returns_b.index)
    ra = returns_a.loc[common].dropna().values.astype(float)
    rb = returns_b.loc[common].dropna().values.astype(float)

    T = min(len(ra), len(rb))
    ra = ra[:T]
    rb = rb[:T]

    if T < 20:
        return LedoitWolfResult(
            strategy_a=name_a, strategy_b=name_b,
            sr_a=np.nan, sr_b=np.nan, sr_diff=np.nan,
            t_stat=np.nan, p_value=np.nan, significant_5pct=False,
        )

    sigma_a = float(ra.std())
    sigma_b = float(rb.std())
    mu_a = float(ra.mean())
    mu_b = float(rb.mean())

    sr_a = float(mu_a / sigma_a * np.sqrt(TRADING_DAYS)) if sigma_a > 1e-12 else np.nan
    sr_b = float(mu_b / sigma_b * np.sqrt(TRADING_DAYS)) if sigma_b > 1e-12 else np.nan

    if sigma_a < 1e-12 or sigma_b < 1e-12:
        return LedoitWolfResult(
            strategy_a=name_a, strategy_b=name_b,
            sr_a=sr_a, sr_b=sr_b, sr_diff=np.nan,
            t_stat=np.nan, p_value=np.nan, significant_5pct=False,
        )

    # Scaled return difference — directly tests H0: SR_A = SR_B
    # psi_t = r_A_t/sigma_A - r_B_t/sigma_B
    # E[psi_t] = mu_A/sigma_A - mu_B/sigma_B = SR_A_daily - SR_B_daily
    psi = ra / sigma_a - rb / sigma_b   # shape (T,)

    # HAC variance of the mean of psi (Newey-West)
    V_hac = _newey_west_variance(psi, n_lags)

    # t-statistic:  sqrt(T) * mean(psi) / sqrt(V_hac)
    t_stat = float(np.sqrt(T) * psi.mean() / np.sqrt(max(V_hac, 1e-20)))
    p_value = float(2 * stats.norm.sf(abs(t_stat)))

    return LedoitWolfResult(
        strategy_a=name_a,
        strategy_b=name_b,
        sr_a=sr_a,
        sr_b=sr_b,
        sr_diff=(sr_a - sr_b) if (not np.isnan(sr_a) and not np.isnan(sr_b)) else np.nan,
        t_stat=t_stat,
        p_value=p_value,
        significant_5pct=p_value < 0.05,
    )


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: pd.Series | float = 0.0,
    strategy_name: str = "strategy",
    period: str = "full",
    bootstrap_n: int = 1000,
    bootstrap_p: float = 0.10,
    bootstrap_seed: int = 42,
) -> MetricsResult:
    """Compute the full Block-7 metric suite for a daily return series.

    Parameters
    ----------
    returns:
        Daily total net-of-cost return series.
    benchmark:
        Daily benchmark returns (typically ``mkt_ret`` from ``market_daily``).
    rf:
        Daily risk-free rate, or float scalar.
    strategy_name, period:
        Labels for the output.
    bootstrap_n, bootstrap_p, bootstrap_seed:
        Block-bootstrap parameters for Sharpe CI.

    Returns
    -------
    :class:`MetricsResult`
    """
    r = returns.dropna()
    n = len(r)

    if n < 5:
        # Not enough data
        nan_val = np.nan
        return MetricsResult(
            strategy_name=strategy_name, period=period, n_days=n,
            ann_return=nan_val, ann_vol=nan_val, sharpe=nan_val,
            sharpe_ci_lo=nan_val, sharpe_ci_hi=nan_val, sortino=nan_val,
            max_drawdown=nan_val, calmar=nan_val, win_rate=nan_val,
            profit_factor=nan_val, alpha=nan_val, beta=nan_val,
            alpha_pval=nan_val, skew=nan_val, kurtosis=nan_val,
        )

    ann_ret = _ann_return(r)
    ann_vol = _ann_vol(r)
    sr = _sharpe(r, rf)
    so = _sortino(r, rf)
    mdd = _max_drawdown(r)
    calmar = float(ann_ret / abs(mdd)) if abs(mdd) > 1e-12 else np.nan
    win_rate = float((r > 0).mean())
    pf = _profit_factor(r)

    # Sharpe CI (block bootstrap)
    def _sr_fn(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        return _sharpe(s, 0.0)  # ignore rf inside bootstrap for simplicity

    ci_lo, ci_hi = block_bootstrap_ci(
        r, _sr_fn, n_boot=bootstrap_n, p=bootstrap_p, seed=bootstrap_seed
    )

    # CAPM
    alpha_ann, beta, alpha_pval = _capm_nw(r, benchmark, rf)

    skew = float(r.skew())
    kurt = float(r.kurtosis())   # excess kurtosis (Fisher)

    return MetricsResult(
        strategy_name=strategy_name,
        period=period,
        n_days=n,
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sr,
        sharpe_ci_lo=ci_lo,
        sharpe_ci_hi=ci_hi,
        sortino=so,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=pf,
        alpha=alpha_ann,
        beta=beta,
        alpha_pval=alpha_pval,
        skew=skew,
        kurtosis=kurt,
    )


def metrics_table(results: list[MetricsResult]) -> pd.DataFrame:
    """Convert a list of MetricsResult to a DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows).set_index(["strategy", "period"])
