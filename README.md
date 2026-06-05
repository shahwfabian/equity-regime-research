# Equity Regime Research

**Statistical Evaluation of Momentum, Mean Reversion, and Regime Dynamics in U.S. Equity Markets**

A fully reproducible quantitative finance research pipeline that runs on synthetic data with zero external data dependencies, and also accepts real CRSP/French Fama parquet panels.

## Architecture

```
equity_regime_research/
├── configs/default.yaml          # All tunable parameters
├── src/equity_regime/
│   ├── config.py                 # YAML → typed dataclasses (Config.load)
│   ├── pipeline.py               # End-to-end orchestrator + CLI
│   ├── data/
│   │   ├── synthetic.py          # 2-state Markov regime data generator
│   │   ├── ingest.py             # Parquet loader + schema validation
│   │   └── clean.py              # Universe filters, log returns, winsorization
│   ├── factors/
│   │   ├── momentum.py           # Intermediate-horizon cumulative return signal
│   │   ├── mean_reversion.py     # Short-horizon reversal signal
│   │   └── portfolios.py         # Decile sorts, L/S portfolios, monotonicity check
│   ├── regime/
│   │   ├── rule_based.py         # Rolling-vol + 200d MA regime labels
│   │   └── markov.py             # 2-state Markov-switching (statsmodels)
│   ├── stats/
│   │   ├── tests.py              # ACF/LB, variance ratio, ADF/KPSS, NW regressions
│   │   └── performance.py        # Sharpe, MDD, OOS R², Diebold-Mariano
│   ├── viz/figures.py            # 8 matplotlib figures → outputs/figures/
│   └── report/builder.py         # Self-contained HTML report
├── scripts/run_pipeline.py       # Main CLI entry point
└── tests/                        # pytest unit + integration tests
```

## Quickstart

```bash
pip install -e ".[dev]"

# Synthetic data run (no external data required)
python scripts/run_pipeline.py --config configs/default.yaml --synthetic

# Real data run
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --stock-panel data/raw/stock_day.parquet \
  --market-panel data/raw/market_day.parquet

# Tests
pytest                        # all tests
pytest -m "not slow"          # skip full pipeline test
pytest tests/test_factors.py  # specific module
```

## Pipeline Stages

1. **Ingest** — load or generate stock×day + market×day panels
2. **Clean** — price/history filters, log returns, cross-sectional winsorization, excess returns
3. **Factors** — momentum (252-21d skip) and short-horizon reversal signals, decile portfolios
4. **Regime Detection** — Markov-switching (statsmodels) + rule-based (vol/trend) labels
5. **Statistical Tests** — ACF, Lo-MacKinlay VR, ADF/KPSS, NW predictive regressions, regime-interaction Wald test, OOS R², Diebold-Mariano
6. **Figures** — 8 PNGs saved to `outputs/figures/`
7. **Report** — `outputs/reports/research_report.html` (self-contained, all figures embedded as base64)

## Key Design Decisions

- **No look-ahead**: momentum signal at t uses returns from [t-lookback, t-skip], shift(skip) on rolling window. A perturbation test verifies this formally.
- **Filtered vs smoothed regime probs**: clearly named separate columns; only `filtered_*` is used for factor conditioning.
- **Walk-forward OOS with embargo**: 21-day gap between train/test to prevent information leakage.
- **Heteroskedasticity-robust inference**: Lo-MacKinlay VR statistic and Newey-West HAC throughout.
- **Pure functions + DataFrames**: each stage is independently testable; no global state.

## Config (`configs/default.yaml`)

All parameters are centralized — lookbacks, portfolio construction, regime detection, test lags, OOS split date, report metadata. Override any section without touching code.

## Data Schema

**stock_day**: `[date, permno, ret, prc, mktcap, vol]`  
**market_day**: `[date, mkt_ret, rf, realized_vol, vix_like]`
