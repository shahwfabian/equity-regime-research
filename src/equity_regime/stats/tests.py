"""Statistical tests: ACF/Ljung-Box, variance ratio, ADF/KPSS, predictive regressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Autocorrelation and Ljung-Box
# ---------------------------------------------------------------------------

def compute_acf(series: pd.Series, nlags: int) -> pd.DataFrame:
    """Compute sample autocorrelations and Ljung-Box p-values."""
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.tsa.stattools import acf, q_stat

    clean = series.dropna().values
    ac, confint, qstat, pvalues = acf(clean, nlags=nlags, qstat=True, alpha=0.05)

    return pd.DataFrame(
        {
            "lag": np.arange(0, nlags + 1),
            "acf": ac,
            "ci_lower": confint[:, 0] - ac,
            "ci_upper": confint[:, 1] - ac,
            "q_stat": np.concatenate([[np.nan], qstat]),
            "lb_pvalue": np.concatenate([[np.nan], pvalues]),
        }
    )


# ---------------------------------------------------------------------------
# Lo-MacKinlay Variance Ratio (heteroskedasticity-robust)
# ---------------------------------------------------------------------------

def variance_ratio_test(
    series: pd.Series,
    q_values: List[int],
) -> pd.DataFrame:
    """
    Lo-MacKinlay (1988) heteroskedasticity-robust variance ratio test.

    VR(q) = Var(q-period ret) / (q * Var(1-period ret))
    Under RW: VR(q) = 1 for all q.
    VR > 1 implies positive autocorrelation (momentum).
    VR < 1 implies negative autocorrelation (mean reversion).
    """
    x = series.dropna().values
    n = len(x)
    mu = x.mean()

    # Variance of 1-period returns (unbiased)
    sigma2_1 = np.sum((x - mu) ** 2) / (n - 1)

    rows = []
    for q in q_values:
        if n < 2 * q:
            continue

        # q-period overlapping returns
        x_q = np.array([x[t:t + q].sum() for t in range(n - q + 1)])
        mu_q = mu * q
        sigma2_q = np.sum((x_q - mu_q) ** 2) / (len(x_q) - 1)

        vr = sigma2_q / (q * sigma2_1) if sigma2_1 > 0 else np.nan

        # Heteroskedasticity-robust delta (Lo-MacKinlay eq. 17)
        # delta_j = sum_{t=j+1}^{n} (x_t - mu)^2 (x_{t-j} - mu)^2
        # theta = 4/n * sum_{j=1}^{q-1} (1 - j/q)^2 * delta_j / sigma2_1^2
        def _delta(j: int) -> float:
            return (
                np.sum(
                    ((x[j:] - mu) ** 2) * ((x[: n - j] - mu) ** 2)
                )
                / (n ** 2)
            )

        theta_hat = (
            4.0
            * sum(
                (1 - j / q) ** 2 * _delta(j)
                for j in range(1, q)
            )
            / (sigma2_1 ** 2)
        )

        # z-stat (heteroskedasticity-robust)
        z_stat = (vr - 1) / np.sqrt(theta_hat) if theta_hat > 0 else np.nan
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

        rows.append(
            {
                "q": q,
                "vr": vr,
                "z_stat": z_stat,
                "pvalue": pvalue,
                "ci_lower": vr - 1.96 * np.sqrt(theta_hat),
                "ci_upper": vr + 1.96 * np.sqrt(theta_hat),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stationarity Tests
# ---------------------------------------------------------------------------

def stationarity_tests(series: pd.Series) -> Dict[str, float]:
    """Run ADF and KPSS stationarity tests."""
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = series.dropna()

    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(clean, autolag="AIC")
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(clean, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_lags = np.nan, np.nan, 0
        kpss_crit = {}

    return {
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "adf_lags": int(adf_lags),
        "adf_crit_1pct": float(adf_crit.get("1%", np.nan)),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "kpss_lags": int(kpss_lags),
        "is_stationary_adf": bool(adf_p < 0.05),
        "is_stationary_kpss": bool(kpss_p > 0.05),
    }


# ---------------------------------------------------------------------------
# Predictive Regression with Newey-West HAC
# ---------------------------------------------------------------------------

def predictive_regression(
    y: pd.Series,
    X: pd.DataFrame,
    nw_lags: int = 5,
) -> pd.DataFrame:
    """
    OLS regression of y on X with Newey-West HAC standard errors.

    Returns coefficient table with columns:
    [variable, coef, se_nw, t_stat, pvalue, ci_lower, ci_upper]
    """
    import statsmodels.api as sm

    aligned = pd.concat([y.rename("y"), X], axis=1).dropna()
    Y = aligned["y"]
    Xmat = sm.add_constant(aligned.drop(columns="y"))

    model = sm.OLS(Y, Xmat)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    coef_df = pd.DataFrame(
        {
            "variable": result.params.index,
            "coef": result.params.values,
            "se_nw": result.bse.values,
            "t_stat": result.tvalues.values,
            "pvalue": result.pvalues.values,
            "ci_lower": result.conf_int().iloc[:, 0].values,
            "ci_upper": result.conf_int().iloc[:, 1].values,
        }
    )
    return coef_df


# ---------------------------------------------------------------------------
# Regime-Interaction Regression + Wald Test
# ---------------------------------------------------------------------------

def regime_interaction_regression(
    y: pd.Series,
    signal: pd.Series,
    regime: pd.Series,
    nw_lags: int = 5,
) -> Dict:
    """
    Regime-interaction regression:
        y_t = alpha + beta*signal_t + gamma*regime_t + delta*(signal_t * regime_t) + eps

    Performs Wald test for delta=0 (H0: regime invariance).

    Returns dict with coef_table, wald_stat, wald_pvalue.
    """
    import statsmodels.api as sm

    df = pd.DataFrame(
        {"y": y, "signal": signal, "regime": regime}
    ).dropna()
    df["signal_x_regime"] = df["signal"] * df["regime"]

    Xmat = sm.add_constant(df[["signal", "regime", "signal_x_regime"]])
    model = sm.OLS(df["y"], Xmat)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    coef_table = pd.DataFrame(
        {
            "variable": result.params.index,
            "coef": result.params.values,
            "se_nw": result.bse.values,
            "t_stat": result.tvalues.values,
            "pvalue": result.pvalues.values,
        }
    )

    # Wald test: H0: delta (signal_x_regime) = 0
    wald_stat = float(result.tvalues["signal_x_regime"] ** 2)
    wald_pvalue = float(result.pvalues["signal_x_regime"])

    return {
        "coef_table": coef_table,
        "wald_stat": wald_stat,
        "wald_pvalue": wald_pvalue,
        "regime_invariance_rejected": bool(wald_pvalue < 0.05),
        "r_squared": float(result.rsquared),
    }
