"""Equity Regime Backtesting Framework.

Portfolio-level backtester operating on daily factor/portfolio return series.

IMPORTANT SCOPE NOTE
--------------------
This is a *portfolio-level* (factor-space) backtester, NOT a stock-level
order-book event-driven engine.  All inputs are daily factor / portfolio
returns from the Kenneth French Data Library (UMD, ST_Rev, Mkt-RF, RF) and
the market_daily ETL table.  Because we have NO individual-stock OHLCV data,
we cannot simulate per-share fills, market-impact curves, or intraday
dynamics.

The "event layer" handles only:
  * Rebalancing schedule (daily / weekly / monthly triggers)
  * Turnover-based transaction-cost and slippage models
  * Portfolio-weight drift between rebalances

For a true stock-level engine you would need tick-level or at minimum OHLCV
data for the full CRSP / Compustat universe -- a commercial data subscription.
This limitation is intentional, transparent, and consistent with the study's
free-data mandate.

Modules
-------
strategy    : Strategy base class + 6 concrete strategy classes
engine      : BacktestEngine -- event loop, drift, cost application
costs       : CostConfig, per-rebalance cost model, cost-sweep utility
sizing      : Weight normalisation + ex-ante volatility targeting
walkforward : WalkForwardSplitter with embargo gap
metrics     : Full metric suite, Ledoit-Wolf Sharpe test, block bootstrap
run         : Orchestrator + HTML report renderer
"""
__version__ = "1.0.0"
