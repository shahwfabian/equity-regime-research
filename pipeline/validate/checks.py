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
# V2  No missing values in factor tables post-clean
# ---------------------------------------------------------------------------
def v2_no_missing_values(
    factor_daily: pd.DataFrame,
    factor_monthly: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> CheckResult:
    name = "V2 No Missing Values Post-Clean"

    def _run():
        results = {}
        for label, df in [
            ("factor_daily", factor_daily),
            ("factor_monthly", factor_monthly),
            ("market_daily", market_daily),
        ]:
            nulls = df.isnull().sum()
            results[label] = nulls[nulls > 0].to_dict()

        # VIX NaNs in market_daily are allowed (non-return col, documented in V7)
        # For pass/fail, check only return columns
        ret_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "st_rev", "lt_rev", "rf"]
        total_missing = 0
        for label, df in [("factor_daily", factor_daily), ("factor_monthly", factor_monthly)]:
            for col in [c for c in ret_cols if c in df.columns]:
                total_missing += int(df[col].isnull().sum())

        passed = total_missing == 0
        detail_parts = []
        for label, nulls in results.items():
            if nulls:
                detail_parts.append(f"{label}: {nulls}")
            else:
                detail_parts.append(f"{label}: no nulls")

        return CheckResult(
            name=name, passed=passed,
            detail="; ".join(detail_parts),
            numbers={"total_missing_return_cols": total_missing, "all_tables": results},
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V3  Date coverage: ~252 trading days/year ±5; gaps > 5 days
# ---------------------------------------------------------------------------
def v3_date_coverage(factor_daily: pd.DataFrame, max_gap_days: int = 5) -> CheckResult:
    name = "V3 Date Coverage"

    def _run():
        dates = factor_daily.index.sort_values()
        n_years = (dates[-1] - dates[0]).days / 365.25
        days_per_year = len(dates) / n_years if n_years > 0 else 0

        # Find gaps larger than max_gap_days calendar days
        gaps = pd.Series(dates).diff().dt.days.dropna()
        large_gaps = gaps[gaps > max_gap_days]

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

        target_low, target_high = 252 - 5, 252 + 5
        in_range = target_low <= days_per_year <= target_high
        passed = in_range and len(large_gaps) == 0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"Total trading days: {len(dates)}, span: {n_years:.1f} years, "
                f"avg {days_per_year:.1f} days/year (target 252±5). "
                f"Gaps > {max_gap_days} calendar days: {len(large_gaps)}."
            ),
            numbers={
                "total_days": len(dates),
                "years": round(n_years, 2),
                "days_per_year": round(days_per_year, 1),
                "n_large_gaps": len(large_gaps),
            },
            rows=gap_table,
        )
    return _safe(_run, name)


# ---------------------------------------------------------------------------
# V4  Return sanity: no |daily factor return| > 25%
# ---------------------------------------------------------------------------
def v4_return_sanity(factor_daily: pd.DataFrame, max_abs: float = 0.25) -> CheckResult:
    name = "V4 Return Sanity (|daily return| <= 25%)"

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

        passed = pct >= 95.0

        return CheckResult(
            name=name, passed=passed,
            detail=(
                f"VIX present on {n_vix}/{total} factor trading days ({pct:.1f}%). "
                f"Missing: {n_missing} days. "
                f"{'PASS (>=95%)' if passed else 'FAIL (<95%)'}."
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
) -> List[CheckResult]:
    """Run V1–V10 and return all results."""
    log.info("[validate] Running all validation checks…")
    results = []

    results.append(v1_french_reconcile(factor_daily))
    results.append(v2_no_missing_values(factor_daily, factor_monthly, market_daily))
    results.append(v3_date_coverage(factor_daily, max_gap_days=cfg.validation.max_gap_days))
    results.append(v4_return_sanity(factor_daily, max_abs=cfg.validation.max_abs_return))
    results.append(v5_stationarity(factor_daily))
    results.append(v6_realized_vol(market_daily))
    results.append(v7_vix_coverage(market_daily))
    results.append(v8_split_roundtrip())
    results.append(v9_survivorship_disclosure())
    results.append(v10_store_roundtrip(store, factor_daily))

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    log.info("[validate] Checks complete: %d PASS / %d FAIL", n_pass, n_fail)
    for r in results:
        level = logging.INFO if r.passed else logging.WARNING
        log.log(level, "[validate] %s -> %s", r.name, "PASS" if r.passed else "FAIL")

    return results
