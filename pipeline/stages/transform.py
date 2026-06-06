"""Transform stage: realized vol, market features, factor assembly.

Outputs the final schema tables ready for loading:
  factor_daily   : [date PK, mkt_rf, smb, hml, rmw, cma, umd, st_rev, lt_rev, rf]
  factor_monthly : [date PK, mkt_rf, smb, hml, rmw, cma, rf]
  market_daily   : [date PK, mkt_ret, rf, realized_vol_21, realized_vol_63, drawdown, vix]
  stock_daily    : [date, ticker, adj_close, ret, log_ret, volume]  (optional)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.stages.clean import CleanResult

log = logging.getLogger(__name__)


@dataclass
class TransformResult:
    factor_daily: pd.DataFrame
    factor_monthly: pd.DataFrame
    market_daily: pd.DataFrame
    stock_daily: Optional[pd.DataFrame] = None


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility using a rolling std (252 trading days/year)."""
    return returns.rolling(window, min_periods=window // 2).std() * math.sqrt(252)


def _drawdown(returns: pd.Series) -> pd.Series:
    """Running drawdown from peak of cumulative return index."""
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    return (cum - roll_max) / roll_max


def _compute_stock_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ret (simple) and log_ret columns computed from adj_close."""
    df = df.copy().sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_ret"] = np.log(
        df.groupby("ticker")["adj_close"].transform(lambda x: x / x.shift(1))
    )
    n_before = len(df)
    df = df.dropna(subset=["ret"])
    log.info(
        "[transform] stock_daily returns: dropped %d NaN first-rows", n_before - len(df)
    )
    return df


def transform(cleaned: CleanResult) -> TransformResult:
    """
    Apply all transformations and produce final schema-aligned tables.
    """
    # ---- factor_daily: pass-through (already in correct schema) ----
    factor_daily = cleaned.factor_daily.copy()
    factor_daily.index.name = "date"
    log.info("[transform] factor_daily: %d rows", len(factor_daily))

    # ---- factor_monthly ----
    factor_monthly = cleaned.factor_monthly.copy()
    factor_monthly.index.name = "date"
    log.info("[transform] factor_monthly: %d rows", len(factor_monthly))

    # ---- market_daily ----
    md = cleaned.market_daily.copy()
    # mkt_rf and rf may be NaN for pre-1963 rows; mkt_ret is also NaN there
    mkt_rf_col = md["mkt_rf"] if "mkt_rf" in md.columns else pd.Series(dtype="float64", name="mkt_rf")
    rf_col     = md["rf"]     if "rf"     in md.columns else pd.Series(dtype="float64", name="rf")
    mkt_ret = (mkt_rf_col + rf_col).astype("float64")   # NaN where either is NaN (pre-1963)

    # Realized vol and drawdown are only meaningful where mkt_ret exists
    md_out = pd.DataFrame(index=md.index)
    md_out.index.name = "date"
    md_out["mkt_ret"]          = mkt_ret
    md_out["rf"]               = rf_col.astype("float64")
    md_out["realized_vol_21"]  = _realized_vol(mkt_ret, 21).astype("float64")
    md_out["realized_vol_63"]  = _realized_vol(mkt_ret, 63).astype("float64")
    md_out["drawdown"]         = _drawdown(mkt_ret.fillna(0)).astype("float64")  # 0-filled for pre-1963
    md_out["vix"]              = md["vix"].astype("float64") if "vix" in md.columns else float("nan")

    n_mkt_nan = md_out["mkt_ret"].isna().sum()
    n_vix_nan = md_out["vix"].isna().sum()
    log.info(
        "[transform] market_daily: %d rows. mkt_ret NaN=%d (pre-1963 expected), "
        "realized_vol_21 NaN=%d, realized_vol_63 NaN=%d, vix NaN=%d (pre-1990 expected)",
        len(md_out), n_mkt_nan,
        md_out["realized_vol_21"].isna().sum(),
        md_out["realized_vol_63"].isna().sum(),
        n_vix_nan,
    )

    # ---- stock_daily (optional) ----
    stock_daily = None
    if cleaned.stocks is not None and not cleaned.stocks.empty:
        stock_daily = _compute_stock_returns(cleaned.stocks)
        stock_daily = stock_daily[["date", "ticker", "adj_close", "ret", "log_ret", "volume"]]
        log.info("[transform] stock_daily: %d rows, %d tickers",
                 len(stock_daily), stock_daily["ticker"].nunique())

    return TransformResult(
        factor_daily=factor_daily,
        factor_monthly=factor_monthly,
        market_daily=md_out,
        stock_daily=stock_daily,
    )
