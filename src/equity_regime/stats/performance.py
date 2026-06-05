"""Performance analytics: risk metrics, Diebold-Mariano, out-of-sample R²."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


def annualized_mean(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf
    vol = excess.std()
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf
    downside = excess[excess < 0].std()
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return float(annualized_mean(returns) / mdd) if mdd > 0 else np.nan


def skewness(returns: pd.Series) -> float:
    return float(returns.skew())


def excess_kurtosis(returns: pd.Series) -> float:
    return float(returns.kurtosis())


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    return float(returns.quantile(level))


def conditional_var(returns: pd.Series, level: float = 0.05) -> float:
    var = value_at_risk(returns, level)
    return float(returns[returns <= var].mean())


def full_performance_table(
    returns_dict: Dict[str, pd.Series],
    rf_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute all performance metrics for a dict of {name: return_series}.

    Parameters
    ----------
    returns_dict : dict of name -> daily return Series
    rf_series    : daily risk-free rate Series (aligned to same index), or None

    Returns
    -------
    pd.DataFrame with metrics as columns, strategy names as rows
    """
    rows = []
    for name, ret in returns_dict.items():
        ret = ret.dropna()
        rf_scalar = rf_series.reindex(ret.index).fillna(0.0).mean() if rf_series is not None else 0.0

        rows.append(
            {
                "strategy": name,
                "ann_return": annualized_mean(ret),
                "ann_vol": annualized_vol(ret),
                "sharpe": sharpe_ratio(ret, rf_scalar),
                "sortino": sortino_ratio(ret, rf_scalar),
                "max_drawdown": max_drawdown(ret),
                "calmar": calmar_ratio(ret),
                "skewness": skewness(ret),
                "excess_kurtosis": excess_kurtosis(ret),
                "var_5pct": value_at_risk(ret),
                "cvar_5pct": conditional_var(ret),
                "n_obs": len(ret),
            }
        )

    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Walk-forward OOS split with embargo
# ---------------------------------------------------------------------------

def train_test_split_with_embargo(
    dates: pd.DatetimeIndex,
    split_date: str,
    embargo_days: int = 21,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Split dates into train/test with an embargo gap between them.

    Returns (train_dates, test_dates) where test starts >= split_date + embargo_days.
    """
    split = pd.Timestamp(split_date)
    embargo_end = split + pd.Timedelta(days=embargo_days)

    train = dates[dates < split]
    test = dates[dates >= embargo_end]

    assert len(train) > 0, "No training dates before split"
    assert len(test) > 0, "No test dates after embargo"
    assert train.max() < test.min(), "Train/test overlap detected"

    return train, test


def oos_r_squared(
    y_actual: pd.Series,
    y_pred_model: pd.Series,
    y_pred_bench: Optional[pd.Series] = None,
) -> float:
    """
    Out-of-sample R² (Campbell-Thompson) relative to a benchmark forecast.

    OOS-R² = 1 - MSE_model / MSE_benchmark
    Benchmark default: historical mean (prevailing mean forecast).
    """
    y = y_actual.dropna()
    if y_pred_bench is None:
        bench = pd.Series(y.expanding().mean().shift(1), index=y.index)
    else:
        bench = y_pred_bench

    aligned = pd.concat(
        [y.rename("_y"), y_pred_model.rename("_model"), bench.rename("_bench")], axis=1
    ).dropna()

    mse_model = ((aligned["_y"] - aligned["_model"]) ** 2).mean()
    mse_bench = ((aligned["_y"] - aligned["_bench"]) ** 2).mean()

    if mse_bench == 0:
        return np.nan

    return float(1 - mse_model / mse_bench)


def diebold_mariano_test(
    y_actual: pd.Series,
    y_pred1: pd.Series,
    y_pred2: pd.Series,
    h: int = 1,
) -> Dict:
    """
    Diebold-Mariano test: H0: equal predictive accuracy between model 1 and model 2.

    Uses Harvey-Newbold-Leybourne small-sample correction.
    """
    aligned = pd.concat(
        [y_actual.rename("y"), y_pred1.rename("p1"), y_pred2.rename("p2")], axis=1
    ).dropna()

    e1 = (aligned["y"] - aligned["p1"]) ** 2
    e2 = (aligned["y"] - aligned["p2"]) ** 2
    d = e1 - e2

    n = len(d)
    d_bar = d.mean()

    # Newey-West variance of d
    gamma0 = d.var()
    gamma_k = sum(
        (1 - k / (h + 1)) * ((d - d_bar).iloc[k:] * (d - d_bar).iloc[:-k]).mean()
        for k in range(1, h + 1)
        if k < n
    )
    lrv = gamma0 + 2 * gamma_k
    se = np.sqrt(max(lrv, 1e-20) / n)

    dm_stat = float(d_bar / se) if se > 0 else np.nan
    pvalue = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))) if not np.isnan(dm_stat) else np.nan

    return {
        "dm_stat": dm_stat,
        "pvalue": pvalue,
        "n_obs": n,
        "mean_loss_diff": float(d_bar),
        "model1_wins": bool(d_bar < 0),
    }
