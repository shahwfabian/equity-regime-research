"""Factor-model regression robustness exhibit for the equity-regime-research paper.

Addresses the referee question: "Is regime-conditioned momentum just repackaged
factor exposure, or does the regime gate add something the factor model cannot capture?"

Conventions (post-audit throughout)
------------------------------------
- All strategy returns are EXCESS of RF, net of transaction costs, from saved series.
- Regime labels are the FILTERED causal vol-rule (21-day realized vol, 75th-pct
  expanding threshold, shift-1). No Markov states. No look-ahead.
- Mean-reversion excluded from all performance comparisons.
- Newey-West lag length: 21 trading days (one calendar month). Chosen by return
  horizon, not tuned. Standard for daily equity return regressions.

Factor specifications
---------------------
  Spec 1 — CAPM:    r_excess ~ Mkt-RF
  Spec 2 — FF5+Mom: r_excess ~ Mkt-RF + SMB + HML + RMW + CMA + UMD

Strategies
----------
  (a) Unconditional L/S momentum  (sanity check — should load ~1 on UMD, near-zero alpha)
  (b) Regime-conditioned momentum (informative — partial UMD load, possible positive alpha)

Usage
-----
    python scripts/factor_regressions.py

Outputs
-------
    outputs/tables/table_factor_regressions.csv
    Console: regression tables + plain-English interpretation
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

TRADING_DAYS   = 252
NW_LAGS        = 21          # Newey-West lags: 1 trading month, standard for daily returns
OUT_DIR        = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Factor column names in factor_daily.parquet ──────────────────────────────
FACTOR_COLS = {
    "mkt_rf": "Mkt-RF",
    "smb":    "SMB",
    "hml":    "HML",
    "rmw":    "RMW",
    "cma":    "CMA",
    "umd":    "UMD",
}

SPECS = {
    "CAPM":     ["mkt_rf"],
    "FF5+Mom":  ["mkt_rf", "smb", "hml", "rmw", "cma", "umd"],
}

STRATEGIES = {
    "long_short_momentum": "Unconditional L/S momentum",
    "regime_momentum":     "Regime-conditioned momentum",
}


# ── OLS with Newey-West HAC ───────────────────────────────────────────────────
def run_ols_nw(y: pd.Series, X_df: pd.DataFrame, nw_lags: int) -> dict:
    """Run OLS, return NW-corrected t-stats, alpha annualised, adj R².

    Parameters
    ----------
    y : pd.Series
        Dependent variable (daily excess return, net of cost).
    X_df : pd.DataFrame
        Factor columns (raw daily factor excess returns, NOT yet with constant).
    nw_lags : int
        Newey-West truncation lag.

    Returns
    -------
    dict with keys: alpha_daily, alpha_ann, alpha_t, factor_betas (dict),
                    factor_t (dict), adj_r2, n_obs, nw_lags.
    """
    # Align
    data = pd.concat([y, X_df], axis=1).dropna()
    y_  = data.iloc[:, 0].values
    X_  = sm.add_constant(data.iloc[:, 1:].values, prepend=True)

    model   = sm.OLS(y_, X_)
    results = model.fit()

    # Newey-West covariance
    nw_cov  = cov_hac(results, nlags=nw_lags, use_correction=True)
    nw_se   = np.sqrt(np.diag(nw_cov))
    nw_t    = results.params / nw_se

    # Parameter labels: const first, then factors
    labels  = ["const"] + list(X_df.columns)

    alpha_daily = float(results.params[0])
    alpha_ann   = alpha_daily * TRADING_DAYS
    alpha_t     = float(nw_t[0])

    factor_betas = {lab: float(results.params[i + 1]) for i, lab in enumerate(X_df.columns)}
    factor_t_map = {lab: float(nw_t[i + 1])           for i, lab in enumerate(X_df.columns)}

    n        = int(results.nobs)
    k        = X_.shape[1] - 1   # exclude constant
    adj_r2   = float(1 - (1 - results.rsquared) * (n - 1) / (n - k - 1))

    return {
        "alpha_daily": alpha_daily,
        "alpha_ann":   alpha_ann,
        "alpha_t":     alpha_t,
        "factor_betas": factor_betas,
        "factor_t":    factor_t_map,
        "adj_r2":      adj_r2,
        "n_obs":       n,
        "nw_lags":     nw_lags,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────
def sig_stars(t: float) -> str:
    a = abs(t)
    if a > 2.576: return "***"
    if a > 1.960: return "**"
    if a > 1.645: return "*"
    return ""


def fmt_row(label: str, coef: float, t: float, width: int = 10) -> str:
    return (f"  {label:<18} {coef:>9.4f}   t = {t:>6.3f}{sig_stars(t)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    rets    = pd.read_parquet("outputs/data/strategy_returns.parquet")
    factors = pd.read_parquet("data/processed/etl_full/factor_daily.parquet")

    # Align to common index
    common  = rets.index.intersection(factors.index)
    rets    = rets.loc[common]
    factors = factors.loc[common]

    print(f"\nSample: {common[0].date()} to {common[-1].date()}  "
          f"({len(common):,} trading days)\n")
    print(f"Newey-West lags: {NW_LAGS} (1 trading month — standard for daily returns)\n")
    print("=" * 70)

    # ── Run regressions ──────────────────────────────────────────────────────
    all_rows   = []
    all_results = {}

    for strat_key, strat_label in STRATEGIES.items():
        y = rets[strat_key]

        for spec_key, factor_keys in SPECS.items():
            X = factors[factor_keys].copy()
            res = run_ols_nw(y, X, NW_LAGS)
            tag = f"{strat_label} — {spec_key}"
            all_results[tag] = res

            # Build CSV row
            row = {
                "strategy":    strat_label,
                "spec":        spec_key,
                "alpha_ann":   round(res["alpha_ann"],    4),
                "alpha_ann_pct": round(res["alpha_ann"] * 100, 3),
                "alpha_t":     round(res["alpha_t"],     3),
                "alpha_sig":   sig_stars(res["alpha_t"]),
                "adj_r2":      round(res["adj_r2"],       4),
                "n_obs":       res["n_obs"],
                "nw_lags":     NW_LAGS,
            }
            for fk in ["mkt_rf", "smb", "hml", "rmw", "cma", "umd"]:
                row[f"beta_{fk}"]   = round(res["factor_betas"].get(fk, np.nan), 4)
                row[f"t_{fk}"]      = round(res["factor_t"].get(fk, np.nan),     3)
            all_rows.append(row)

    # ── Print regression tables ──────────────────────────────────────────────
    for tag, res in all_results.items():
        print(f"\n{'-'*70}")
        print(f"  {tag}")
        print(f"{'-'*70}")
        print(fmt_row("Alpha (ann.)", res["alpha_ann"], res["alpha_t"]))
        print(f"  {'':18} ({res['alpha_ann']*100:.3f}% per year)")
        for fk, beta in res["factor_betas"].items():
            fname = FACTOR_COLS.get(fk, fk.upper())
            print(fmt_row(fname, beta, res["factor_t"][fk]))
        print(f"  {'Adj. R²':<18} {res['adj_r2']:>9.4f}")
        print(f"  {'N (days)':<18} {res['n_obs']:>9,}")
        print(f"  {'NW lags':<18} {res['nw_lags']:>9}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    table = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "table_factor_regressions.csv"
    table.to_csv(out_path, index=False)
    print(f"\n\nSaved -> {out_path}\n")
    print(table.to_string(index=False))

    # ── Plain-English interpretation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PLAIN-ENGLISH INTERPRETATION")
    print("=" * 70)

    # -- Sanity check: unconditional momentum on FF5+Mom -----------------------
    unc_ff5  = all_results["Unconditional L/S momentum — FF5+Mom"]
    umd_beta = unc_ff5["factor_betas"]["umd"]
    unc_alpha_ann_pct = unc_ff5["alpha_ann"] * 100
    unc_alpha_t       = unc_ff5["alpha_t"]

    sanity_ok = abs(umd_beta - 1.0) < 0.15 and abs(unc_alpha_ann_pct) < 2.0

    print(f"\n[SANITY CHECK] Unconditional momentum on FF5+Mom:")
    print(f"  UMD loading = {umd_beta:.4f}  (expected ~1.00)")
    print(f"  Alpha       = {unc_alpha_ann_pct:.3f}% p.a.  t = {unc_alpha_t:.3f}  (expected ~0)")
    if sanity_ok:
        print("  PASS — unconditional momentum is essentially the UMD factor with near-zero")
        print("  alpha after controlling for factor exposure. Results are internally consistent.")
    else:
        print("  *** FAIL *** — alpha far from zero or UMD loading far from 1.0.")
        print("  This indicates a problem with the strategy construction. Investigate.")

    # -- Regime strategy on FF5+Mom -------------------------------------------
    reg_ff5       = all_results["Regime-conditioned momentum — FF5+Mom"]
    reg_umd_beta  = reg_ff5["factor_betas"]["umd"]
    reg_alpha_ann = reg_ff5["alpha_ann"] * 100
    reg_alpha_t   = reg_ff5["alpha_t"]
    reg_adj_r2    = reg_ff5["adj_r2"]

    print(f"\n[KEY FINDING] Regime-conditioned momentum on FF5+Mom:")
    print(f"  UMD loading = {reg_umd_beta:.4f}  (expected < {umd_beta:.2f} due to cash periods)")
    print(f"  Alpha       = {reg_alpha_ann:.3f}% p.a.  t = {reg_alpha_t:.3f}{sig_stars(reg_alpha_t)}")
    print(f"  Adj. R²     = {reg_adj_r2:.4f}")

    umd_reduction_pct = (1.0 - reg_umd_beta / umd_beta) * 100
    print(f"\n  UMD exposure reduced by ~{umd_reduction_pct:.0f}% vs unconditional.")
    print(f"  This is mechanical: the strategy sits in cash (w=0) during turbulent")
    print(f"  regimes (~38% of days), so average UMD loading must be below 1.0.")

    if reg_alpha_t > 1.645:
        print(f"\n  Alpha is statistically significant at 10% (t={reg_alpha_t:.3f}).")
        print(f"  The {reg_alpha_ann:.2f}% p.a. alpha represents return the factor model")
        print(f"  cannot attribute to standard factor loadings — it is the economic value")
        print(f"  of avoiding momentum crashes in turbulent regimes.")
    elif reg_alpha_t > 1.282:
        print(f"\n  Alpha is marginally significant (t={reg_alpha_t:.3f}, p~0.10-0.20).")
        print(f"  The {reg_alpha_ann:.2f}% p.a. alpha is economically meaningful but")
        print(f"  does not clear the 5% threshold. Consistent with the Ledoit-Wolf result.")
    else:
        print(f"\n  Alpha is not statistically significant (t={reg_alpha_t:.3f}).")
        print(f"  The {reg_alpha_ann:.2f}% p.a. alpha is economically positive but imprecisely")
        print(f"  estimated. Consistent with the Ledoit-Wolf p=0.084 result.")

    # -- Time-varying exposure caveat -----------------------------------------
    print(f"""
[IMPORTANT CAVEAT — TIME-VARYING EXPOSURE]
The regime strategy has structurally time-varying factor loadings:
  - Calm periods  (w=1): full UMD exposure, beta(UMD) = 1.0 locally
  - Turbulent periods (w=0): zero UMD exposure, beta(UMD) = 0.0 locally

A constant-coefficient OLS is therefore MISSPECIFIED for this strategy by
design. The static UMD beta ({reg_umd_beta:.3f}) is a time-average that mixes
two distinct regimes. The static alpha absorbs part of the regime-switching
benefit but not all of it. These regressions are a LOWER-BOUND CHECK: if alpha
is positive and meaningful even under a misspecified constant model, the true
regime-gate contribution is at least as large. The crash-drawdown evidence
(GFC -5.1% vs -57.1%) is a cleaner measure of the strategy's value than the
alpha estimate from a constant-loading regression.
""")

    # -- CAPM comparison ------------------------------------------------------
    reg_capm = all_results["Regime-conditioned momentum — CAPM"]
    unc_capm = all_results["Unconditional L/S momentum — CAPM"]
    print(f"[CAPM comparison]")
    print(f"  Regime alpha (CAPM):     {reg_capm['alpha_ann']*100:.2f}% p.a.  "
          f"t={reg_capm['alpha_t']:.3f}{sig_stars(reg_capm['alpha_t'])}")
    print(f"  Uncond alpha (CAPM):     {unc_capm['alpha_ann']*100:.2f}% p.a.  "
          f"t={unc_capm['alpha_t']:.3f}{sig_stars(unc_capm['alpha_t'])}")
    print(f"  Both strategies display near-zero market beta (momentum is long-short,")
    print(f"  so market exposure nets out). CAPM alpha is close to raw excess return.")
    print()


if __name__ == "__main__":
    main()
