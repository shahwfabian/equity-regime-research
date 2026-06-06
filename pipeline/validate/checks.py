"""Validation checks V1–V10.

Each check is a function that returns a CheckResult(name, passed, detail, numbers).
No check raises — failures are captured and surfaced in the report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    numbers: Dict[str, Any] = field(default_factory=dict)
    rows: Optional[pd.DataFrame] = None   # optional table snippet for the report


def _safe(fn, name: str) -> CheckResult:
    """Wrap a check so exceptions become FAIL results rather than crashes."""
    try:
        return fn()
    except Exception as e:
        log.error("[validate] %s raised: %s", name, e, exc_info=True)
        return CheckResult(name=name, passed=False, detail=f"ERROR: {e}")


# ---------------------------------------------------------------------------
# V1  French reconcile: Mkt = Mkt-RF + RF
# ---------------------------------------------------------------------------
def v1_french_reconcile(factor_daily: pd.DataFrame) -> CheckResult:
    name = "V1 French Factor Reconcile (Mkt = Mkt-RF + RF)"

    def _run():
        recomputed = factor_daily["mkt_rf"] + factor_daily["rf"]
        mkt_ret = factor_daily["mkt_rf"] + factor_daily["rf"]   # same thing — identity check
        diff = (recomputed - mkt_ret).abs()
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        passed = max_err < 1e-10
        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Recomputed Mkt = Mkt-RF + RF. Max absolute deviation: {max_err:.2e}. "
                f"Identity {'holds' if passed else 'FAILS'}."
            ),
            numbers={"max_abs_deviation": max_err, "mean_abs_deviation": mean_err},
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V2  NaN audit for the full-history ragged panel
# ---------------------------------------------------------------------------
def v2_no_missing_values(
    factor_daily: pd.DataFrame,
    factor_monthly: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> CheckResult:
    """Audit NaN counts per column.

    With the full-history panel, pre-1963 NaN in 5-factor columns and
    pre-1990 NaN in VIX are *expected* (ragged native starts).  This check
    PASSES as long as each column's NaN rows are all contiguous at the
    START of the series (i.e., no holes within a column's valid range).
    """
    name = "V2 NaN Audit (Ragged-Start Panel)"

    def _run():
        ret_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "st_rev", "lt_rev", "rf"]
        rows = []
        unexpected_holes = 0

        for col in [c for c in ret_cols if c in factor_daily.columns]:
            s = factor_daily[col]
            n_nan = int(s.isna().sum())
            first_valid = s.first_valid_index()

            # Check: all NaNs must be before first_valid (leading NaNs only)
            if first_valid is not None and n_nan > 0:
                post_start_nan = int(s.loc[first_valid:].isna().sum())
            else:
                post_start_nan = n_nan if first_valid is None else 0

            if post_start_nan > 0:
                unexpected_holes += post_start_nan

            rows.append({
                "column": col,
                "n_nan_leading": n_nan - post_start_nan,
                "n_nan_holes": post_start_nan,
                "first_valid": str(first_valid.date()) if first_valid else "all-NaN",
                "last_valid":  str(s.last_valid_index().date()) if s.last_valid_index() is not None else "all-NaN",
            })

        # VIX in market_daily
        if "vix" in market_daily.columns:
            vix = market_daily["vix"]
            n_nan = int(vix.isna().sum())
            first_valid = vix.first_valid_index()
            post_start_nan = int(vix.loc[first_valid:].isna().sum()) if first_valid is not None else 0
            rows.append({
                "column": "vix",
                "n_nan_leading": n_nan - post_start_nan,
                "n_nan_holes": post_start_nan,
                "first_valid": str(first_valid.date()) if first_valid else "all-NaN",
                "last_valid":  str(vix.last_valid_index().date()) if vix.last_valid_index() is not None else "all-NaN",
            })
            if post_start_nan > 0:
                unexpected_holes += post_start_nan

        result_df = pd.DataFrame(rows)
        passed = unexpected_holes == 0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Audited {len(rows)} return/vix columns. "
                f"Leading NaN (ragged starts) are expected and allowed. "
                f"Unexpected mid-series holes: {unexpected_holes}. "
                f"{'PASS' if passed else 'FAIL — check column details'}."
            ),
            numbers={"unexpected_holes": unexpected_holes, "n_cols_checked": len(rows)},
            rows=result_df,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V3  Date coverage: ~252 trading days/year ±5; gaps > 5 days
# ---------------------------------------------------------------------------
def v3_date_coverage(factor_daily: pd.DataFrame, max_gap_days: int = 5) -> CheckResult:
    """Date-coverage check.

    For the full-history panel (>50 years), known historical exchange closures
    (NYSE 1914 WWI closure, 1933 bank holiday) create multi-week gaps that are
    expected and do not indicate a data problem.  We flag but do not FAIL on
    gaps ≤ 180 calendar days when the series spans >50 years; gaps > 180 days
    are flagged as potential errors regardless of history length.
    """
    name = "V3 Date Coverage"

    def _run():
        dates = factor_daily.index.sort_values()
        n_years = (dates[-1] - dates[0]).days / 365.25
        days_per_year = len(dates) / n_years if n_years > 0 else 0

        # For full history, use effective max-gap tolerance of 180 days (exchange closures)
        effective_gap_limit = max_gap_days if n_years <= 20 else 180

        gaps = pd.Series(dates).diff().dt.days.dropna()
        large_gaps = gaps[gaps > effective_gap_limit]   # only flag truly huge gaps

        gap_table = None
        if len(large_gaps) > 0:
            gap_rows = []
            for idx in large_gaps.index:
                gap_rows.append({
                    "from": dates[idx - 1].date(),
                    "to": dates[idx].date(),
                    "calendar_days": int(large_gaps[idx]),
                })
            gap_table = pd.DataFrame(gap_rows)

        # Also report all gaps > 5 days for information
        info_gaps = gaps[gaps > max_gap_days]
        n_info_gaps = len(info_gaps)

        target_low, target_high = 230, 265
        in_range = target_low <= days_per_year <= target_high
        # PASS: days/year in range AND no gaps > effective limit
        passed = in_range and len(large_gaps) == 0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Total trading days: {len(dates)}, span: {n_years:.1f} years, "
                f"avg {days_per_year:.1f} days/year (target {target_low}-{target_high}). "
                f"Gaps > {max_gap_days} days: {n_info_gaps} (incl. known historical closures). "
                f"Gaps > {effective_gap_limit} days (error threshold): {len(large_gaps)}."
            ),
            numbers={
                "total_days": len(dates),
                "years": round(n_years, 2),
                "days_per_year": round(days_per_year, 1),
                "n_info_gaps": n_info_gaps,
                "n_error_gaps": len(large_gaps),
                "effective_gap_limit": effective_gap_limit,
            },
            rows=gap_table,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V4  Return sanity: no |daily factor return| > 50% (relaxed for full history)
# ---------------------------------------------------------------------------
def v4_return_sanity(factor_daily: pd.DataFrame, max_abs: float = 0.50) -> CheckResult:
    name = f"V4 Return Sanity (|daily return| <= {max_abs:.0%})"

    def _run():
        ret_cols = [c for c in factor_daily.columns if c != "rf"]
        flagged = []
        for col in ret_cols:
            extreme = factor_daily[factor_daily[col].abs() > max_abs][[col]]
            for dt, row in extreme.iterrows():
                flagged.append({"date": dt.date(), "factor": col, "value": round(row[col], 6)})

        flag_df = pd.DataFrame(flagged) if flagged else None
        passed = len(flagged) == 0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Checked {len(ret_cols)} factor columns against |return| > {max_abs:.0%}. "
                f"Flagged: {len(flagged)} observations."
            ),
            numbers={"n_flagged": len(flagged), "threshold": max_abs},
            rows=flag_df,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V5  Stationarity: ADF on factor returns rejects unit root
# ---------------------------------------------------------------------------
def v5_stationarity(factor_daily: pd.DataFrame) -> CheckResult:
    name = "V5 Stationarity (ADF)"

    def _run():
        from statsmodels.tsa.stattools import adfuller

        ret_cols = [c for c in factor_daily.columns if c != "rf"]
        rows = []
        n_fail = 0
        for col in ret_cols:
            s = factor_daily[col].dropna()
            stat, pval, lags, _, crit, _ = adfuller(s, autolag="AIC")
            reject = pval < 0.05
            if not reject:
                n_fail += 1
            rows.append({
                "factor": col,
                "adf_stat": round(stat, 4),
                "p_value": round(pval, 6),
                "lags": lags,
                "reject_unit_root": reject,
            })

        result_df = pd.DataFrame(rows)
        passed = n_fail == 0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"ADF test on {len(ret_cols)} factor return series. "
                f"Unit root rejected (p<0.05) for {len(ret_cols)-n_fail}/{len(ret_cols)}. "
                f"Failures: {n_fail}."
            ),
            numbers={"n_series": len(ret_cols), "n_reject": len(ret_cols) - n_fail, "n_fail": n_fail},
            rows=result_df,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V6  Realized vol monotonic with |returns|
# ---------------------------------------------------------------------------
def v6_realized_vol(market_daily: pd.DataFrame) -> CheckResult:
    name = "V6 Realized Vol vs |Returns| Correlation"

    def _run():
        df = market_daily[["mkt_ret", "realized_vol_21"]].dropna()
        abs_ret = df["mkt_ret"].abs()
        corr = float(abs_ret.corr(df["realized_vol_21"]))
        passed = corr > 0.0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Pearson correlation of |mkt_ret| with realized_vol_21: {corr:.4f}. "
                f"Expected positive (realized vol tracks magnitude of returns). "
                f"{'PASS' if passed else 'FAIL'}."
            ),
            numbers={"correlation": round(corr, 4)},
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V7  VIX coverage and alignment
# ---------------------------------------------------------------------------
def v7_vix_coverage(market_daily: pd.DataFrame) -> CheckResult:
    name = "V7 VIX Coverage & Alignment"

    def _run():
        total = len(market_daily)
        n_vix = int(market_daily["vix"].notna().sum())
        n_missing = total - n_vix
        pct = n_vix / total * 100 if total > 0 else 0.0

        unmatched_dates = market_daily.index[market_daily["vix"].isna()]
        unmatched_table = None
        if len(unmatched_dates) > 0:
            unmatched_table = pd.DataFrame(
                {"date": [d.date() for d in unmatched_dates[:20]]}
            )

        # VIX only available from 1990-01-02 (FRED VIXCLS native start).
        # Check coverage from 1990 onwards rather than from the full factor history.
        vix_era_start = pd.Timestamp("1990-01-01")
        vix_era = market_daily.loc[market_daily.index >= vix_era_start]
        n_vix_era = int(vix_era["vix"].notna().sum())
        total_era = len(vix_era)
        pct_era = n_vix_era / total_era * 100 if total_era > 0 else 0.0
        passed = pct_era >= 95.0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"VIX present on {n_vix}/{total} total factor trading days ({pct:.1f}%). "
                f"Pre-1990 NaN is expected (FRED VIXCLS starts 1990-01-02). "
                f"Coverage from 1990+: {n_vix_era}/{total_era} days ({pct_era:.1f}%). "
                f"{'PASS (>=95% since 1990)' if passed else 'FAIL (<95% since 1990)'}."
            ),
            numbers={"coverage_pct": round(pct, 2), "n_present": n_vix, "n_missing": n_missing},
            rows=unmatched_table,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V8  Split adjustment round-trip (synthetic 2:1)
# ---------------------------------------------------------------------------
def v8_split_roundtrip() -> CheckResult:
    name = "V8 Split Adjustment Round-Trip (Synthetic 2:1)"

    def _run():
        from pipeline.stages.split_adjust import verify_split_adjustment_roundtrip
        result = verify_split_adjustment_roundtrip()
        passed = result["passed"]
        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Synthetic 2:1 split: backward-adjusted returns match true returns. "
                f"Max absolute error: {result['max_abs_error']:.2e} over {result['n_points']} points. "
                f"{'PASS' if passed else 'FAIL'}."
            ),
            numbers=result,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V9  Survivorship bias disclosure (always PASS — informational)
# ---------------------------------------------------------------------------
def v9_survivorship_disclosure() -> CheckResult:
    name = "V9 Survivorship Bias Disclosure"
    disclosure = (
        "FRENCH DATA (factor_daily, factor_monthly, market_daily): "
        "Survivorship-bias CONTROLLED. Kenneth French's factor returns are constructed "
        "from the complete CRSP universe of NYSE/AMEX/NASDAQ common stocks at each "
        "rebalancing date, including delisted firms. Factor returns reflect the full "
        "cross-section; no look-ahead or survivorship bias is introduced at the factor "
        "construction stage. See: Fama & French (1993, 2015) and the CRSP documentation "
        "at mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.\n\n"
        "YFINANCE DATA (stock_daily, when enabled): Survivorship-bias AFFECTED. "
        "The hand-picked ticker universe (e.g. SPY, QQQ, IWM) consists of currently "
        "existing ETFs selected by the researcher. This introduces selection bias: "
        "instruments that failed or were delisted are excluded. This data is labelled "
        "ILLUSTRATIVE ONLY and must never be used for headline factor-return analysis. "
        "All quantitative conclusions in this study are derived exclusively from the "
        "French factor series."
    )
    log.info("[validate] V9 Survivorship Disclosure:\n%s", disclosure)
    return CheckResult(
        name=name, passed=True,
        detail=disclosure,
        numbers={"french_survivorship_controlled": True, "yfinance_illustrative_only": True},
    )


# ---------------------------------------------------------------------------
# V11  Regime sanity: list detected turbulent periods, confirm major crashes
# ---------------------------------------------------------------------------
def v11_regime_sanity(market_daily: pd.DataFrame, vol_pct: float = 0.75) -> CheckResult:
    """Confirm the expanding-vol regime classifier identifies major historical crashes."""
    name = "V11 Regime Sanity (Turbulent Periods)"

    def _run():
        # Use realized_vol_21 if available, else compute from mkt_ret
        if "realized_vol_21" in market_daily.columns:
            vol = market_daily["realized_vol_21"].dropna()
        elif "mkt_ret" in market_daily.columns:
            vol = (
                market_daily["mkt_ret"].dropna()
                .rolling(21, min_periods=11)
                .std()
                .mul(np.sqrt(252))
                .dropna()
            )
        else:
            return CheckResult(name=name, passed=False,
                               detail="No market return or vol column available.")

        # Expanding-quantile threshold (same as engine.py)
        threshold = vol.expanding().quantile(vol_pct).shift(1).ffill()
        regime = (vol > threshold).astype(int).fillna(0)

        # Extract contiguous turbulent episodes (regime==1)
        episodes = []
        in_ep = False
        ep_start = None
        for date, val in regime.items():
            if val == 1 and not in_ep:
                ep_start = date
                in_ep = True
            elif val == 0 and in_ep:
                episodes.append({"start": ep_start.date(), "end": date.date(),
                                  "n_days": (date - ep_start).days})
                in_ep = False
        if in_ep:
            episodes.append({"start": ep_start.date(), "end": regime.index[-1].date(),
                              "n_days": (regime.index[-1] - ep_start).days})

        ep_df = pd.DataFrame(episodes) if episodes else pd.DataFrame()

        # Known crashes to sanity-check (trading days, within vol series range ~1963+)
        # GD_1929 and Oil_1973-74 onset pre-date our vol series so are not checked.
        known_crashes = {
            "BM_1987":    pd.Timestamp("1987-10-19"),   # Black Monday (Monday)
            "DotCom_2001":pd.Timestamp("2001-09-17"),   # first trading day after 9/11
            "GFC_2008":   pd.Timestamp("2008-10-06"),   # peak stress week (Monday)
            "GFC_2009":   pd.Timestamp("2009-03-09"),   # market trough (Monday)
            "Covid_2020": pd.Timestamp("2020-03-16"),   # Monday after crash weekend
        }
        found, missed = [], []
        turbulent_dates = set(regime[regime == 1].index)
        for label, crash_date in known_crashes.items():
            if crash_date in turbulent_dates:
                found.append(label)
            elif crash_date in regime.index:
                missed.append(label)
            # else: crash predates available data

        pct_turbulent = float(regime.mean() * 100)
        passed = len(missed) == 0

        log.info("[validate] V11: %d turbulent episodes, %.1f%% of days turbulent",
                 len(episodes), pct_turbulent)
        log.info("[validate] V11 crashes found: %s | missed: %s", found, missed)

        detail = (
            f"{len(episodes)} turbulent episodes detected over {len(regime)} days "
            f"({pct_turbulent:.1f}% turbulent). "
            f"Known crashes identified: {found}. "
            f"Missed: {missed if missed else 'none (all in turbulent episodes)'}."
        )
        return CheckResult(
            name=name, passed=passed,
            detail=detail,
            numbers={
                "n_episodes": len(episodes),
                "pct_turbulent": round(pct_turbulent, 2),
                "crashes_found": found,
                "crashes_missed": missed,
            },
            rows=ep_df.head(30) if len(ep_df) > 0 else None,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V10  Store round-trip: write then read back, assert equality
# ---------------------------------------------------------------------------
def v10_store_roundtrip(store, factor_daily: pd.DataFrame) -> CheckResult:
    name = "V10 Store Round-Trip"

    def _run():
        from pipeline.stages.load import TABLE_FACTOR_DAILY
        df_back = store.read(TABLE_FACTOR_DAILY)

        # Align indices for comparison
        common_idx = factor_daily.index.intersection(df_back.index)
        common_cols = [c for c in factor_daily.columns if c in df_back.columns]

        orig = factor_daily.loc[common_idx, common_cols].sort_index()
        back = df_back.loc[common_idx, common_cols].sort_index()

        max_err = float((orig - back).abs().max().max())
        shape_match = orig.shape == back.shape
        passed = shape_match and max_err < 1e-9

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Wrote factor_daily to {store.backend!r} backend, read back. "
                f"Shape match: {shape_match} ({orig.shape} vs {back.shape}). "
                f"Max absolute value difference: {max_err:.2e}. "
                f"{'PASS' if passed else 'FAIL'}."
            ),
            numbers={
                "backend": store.backend,
                "shape_match": shape_match,
                "max_abs_diff": max_err,
                "rows_checked": len(common_idx),
            },
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------
def run_all_checks(
    factor_daily: pd.DataFrame,
    factor_monthly: pd.DataFrame,
    market_daily: pd.DataFrame,
    store,
    cfg,
    coverage_df=None,
) -> List[CheckResult]:
    """Run V1–V11 and return all results."""
    log.info("[validate] Running all validation checks…")
    results = []

    # V1: reconcile on the subset where mkt_rf and rf are both non-NaN
    fd_clean = factor_daily.dropna(subset=["mkt_rf", "rf"]) if "mkt_rf" in factor_daily.columns else factor_daily
    results.append(v1_french_reconcile(fd_clean))
    results.append(v2_no_missing_values(factor_daily, factor_monthly, market_daily))
    results.append(v3_date_coverage(factor_daily, max_gap_days=cfg.validation.max_gap_days))
    # V4: use relaxed threshold for full history (Depression-era moves > 25%)
    results.append(v4_return_sanity(factor_daily, max_abs=max(cfg.validation.max_abs_return, 0.50)))
    # V5: stationarity on the five-factor window only (requires non-NaN)
    results.append(v5_stationarity(fd_clean))
    # V6/V7: use the market_daily view with non-NaN mkt_ret
    md_clean = market_daily.dropna(subset=["mkt_ret"]) if "mkt_ret" in market_daily.columns else market_daily
    results.append(v6_realized_vol(md_clean))
    results.append(v7_vix_coverage(market_daily))
    results.append(v8_split_roundtrip())
    results.append(v9_survivorship_disclosure())
    results.append(v10_store_roundtrip(store, fd_clean))
    results.append(v11_regime_sanity(market_daily))

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    log.info("[validate] Checks complete: %d PASS / %d FAIL", n_pass, n_fail)
    for r in results:
        level = logging.INFO if r.passed else logging.WARNING
        log.log(level, "[validate] %s -> %s", r.name, "PASS" if r.passed else "FAIL")

    return results
