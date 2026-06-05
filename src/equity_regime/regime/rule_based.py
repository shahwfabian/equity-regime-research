"""Rule-based regime detection: realized vol + 200-day moving average trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_regime.config import RegimeConfig


def label_vol_regime(
    market_day: pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.DataFrame:
    """
    Assign high/low volatility regimes based on rolling realized vol percentile.

    Regime 1 (turbulent): rolling vol > vol_high_pct percentile of all history
    Regime 0 (calm):      otherwise

    Returns market_day with added column 'vol_regime'.
    """
    df = market_day.copy().sort_values("date").reset_index(drop=True)

    vol_col = "realized_vol"
    if vol_col not in df.columns:
        df[vol_col] = df["mkt_ret"].rolling(cfg.vol_window).std().bfill()

    # Expanding percentile threshold (no look-ahead)
    expanding_pct = (
        df[vol_col]
        .expanding()
        .quantile(cfg.vol_high_pct)
        .shift(1)  # use yesterday's threshold
    )
    df["vol_regime"] = (df[vol_col] > expanding_pct).astype(int)
    df["vol_regime"] = df["vol_regime"].fillna(0).astype(int)

    return df


def label_trend_regime(
    market_day: pd.DataFrame,
    ma_window: int = 200,
) -> pd.DataFrame:
    """
    Assign up/down trend regimes based on 200-day moving average.

    Regime 1 (uptrend):   price > 200d MA
    Regime 0 (downtrend): price <= 200d MA

    Uses a synthetic index level constructed from cumulative market returns.
    """
    df = market_day.copy().sort_values("date").reset_index(drop=True)

    # Build index level from cumulative market returns
    df["index_level"] = np.exp(df["mkt_ret"].fillna(0).cumsum()) * 100.0
    df["ma200"] = df["index_level"].rolling(ma_window, min_periods=ma_window // 2).mean().shift(1)
    df["trend_regime"] = (df["index_level"] > df["ma200"]).astype(int)
    df["trend_regime"] = df["trend_regime"].fillna(0).astype(int)

    return df


def compute_rule_based_regimes(
    market_day: pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.DataFrame:
    """
    Combine vol and trend regime labels.

    Returns market_day with columns: vol_regime, trend_regime, combined_regime.
    combined_regime = 1 if vol_regime == 1 OR trend_regime == 0 (bear).
    """
    df = label_vol_regime(market_day, cfg)
    df = label_trend_regime(df)
    df["combined_regime"] = ((df["vol_regime"] == 1) | (df["trend_regime"] == 0)).astype(int)
    return df
