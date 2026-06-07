"""Export daily excess-return series and regime labels for figure generation.

Runs the backtest engine on the full momentum_core window and saves:
  outputs/data/strategy_returns.parquet  -- daily r_net and r_excess per strategy
  outputs/data/regime_series.parquet     -- causal vol-rule regime labels (0/1)
  outputs/data/oos_fold_sharpes.csv      -- per-fold OOS Sharpe for rolling figure

Excludes mean_reversion from the main file (it is unimplementable).
Mean-reversion series saved separately as:
  outputs/data/mean_reversion_returns.parquet

Call from repo root:
    python scripts/export_series.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRADING_DAYS = 252
OUT = Path("outputs/data")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---- Load config & data ------------------------------------------------
    from backtest.run import BacktestConfig, load_data
    from backtest.engine import BacktestEngine, EngineConfig, _compute_regime_series
    from backtest.strategy import (
        LongOnlyMomentum,
        LongShortMomentum,
        MeanReversion,
        RegimeConditionedMomentum,
        MarketBuyAndHold,
        SixtyForty,
    )
    from backtest.costs import CostConfig

    cfg = BacktestConfig.load("configs/backtest.yaml")
    fd, md = load_data(cfg)
    log.info("Data window: %s -> %s  (%d days)", fd.index.min().date(), fd.index.max().date(), len(fd))

    # ---- Regime series (causal, vol-rule) ----------------------------------
    regime = _compute_regime_series(md, vol_pct=cfg.strategies_cfg.get("regime_momentum", {}).get("regime_vol_pct", 0.75))
    regime.name = "regime"
    regime.to_frame().to_parquet(OUT / "regime_series.parquet")
    log.info("Saved regime_series.parquet  turbulent_pct=%.1f%%", 100 * regime.mean())

    # ---- Engine config ------------------------------------------------------
    eng_cfg = EngineConfig.from_dict({"costs": cfg.costs_cfg,
                                      "sizing": cfg.sizing_cfg,
                                      "strategies": cfg.strategies_cfg})
    cost_cfg = CostConfig.from_dict(cfg.costs_cfg)

    # ---- Strategy map -------------------------------------------------------
    strat_map = {
        "regime_momentum":    RegimeConditionedMomentum(
            calm_exposure=float(cfg.strategies_cfg.get("regime_momentum", {}).get("calm_exposure", 1.0)),
            turbulent_exposure=float(cfg.strategies_cfg.get("regime_momentum", {}).get("turbulent_exposure", 0.0)),
            rebalance=cfg.strategies_cfg.get("regime_momentum", {}).get("rebalance", "monthly"),
        ),
        "long_short_momentum": LongShortMomentum(
            rebalance=cfg.strategies_cfg.get("long_short_momentum", {}).get("rebalance", "monthly"),
        ),
        "long_only_momentum":  LongOnlyMomentum(
            rebalance=cfg.strategies_cfg.get("long_only_momentum", {}).get("rebalance", "monthly"),
        ),
        "market_bah":          MarketBuyAndHold(
            rebalance=cfg.strategies_cfg.get("market_bah", {}).get("rebalance", "monthly"),
        ),
        "sixty_forty":         SixtyForty(
            mkt_weight=float(cfg.strategies_cfg.get("sixty_forty", {}).get("mkt_weight", 0.6)),
            rebalance=cfg.strategies_cfg.get("sixty_forty", {}).get("rebalance", "monthly"),
        ),
        "mean_reversion":      MeanReversion(
            rebalance=cfg.strategies_cfg.get("mean_reversion", {}).get("rebalance", "weekly"),
        ),
    }

    # ---- Run each strategy and collect daily series ------------------------
    all_excess: dict[str, pd.Series] = {}
    all_net:    dict[str, pd.Series] = {}

    for name, strat in strat_map.items():
        log.info("Running %s ...", name)
        engine = BacktestEngine(strat, fd, md, eng_cfg)
        result = engine.run()
        r = result.returns
        # r_excess = excess return only (net of RF); r_net = r_excess after costs
        # The engine stores: r_gross = r_excess + rf; r_excess = factor exposure only
        # We want: excess_net = r_gross - rf - cost_drag = r_excess_factor - cost_drag
        # But r_net already = r_gross - cost_drag; so excess_net = r_net - rf
        rf_series = md.reindex(r.index)["rf"].fillna(0.0)
        # excess_return (net of cost and RF)
        excess_net = r["r_net"] - rf_series
        # excess_return gross (before cost, net of RF)
        excess_gross = r["r_gross"] - rf_series
        all_excess[name] = excess_net
        all_net[name] = excess_gross
        log.info("  %s: Sharpe=%.4f", name, excess_net.mean() / excess_net.std() * np.sqrt(TRADING_DAYS))

    # ---- Save main series (all strategies) ---------------------------------
    excess_df = pd.DataFrame(all_excess)
    excess_df.index.name = "date"
    excess_df.to_parquet(OUT / "strategy_returns_excess_net.parquet")
    log.info("Saved strategy_returns_excess_net.parquet  shape=%s", excess_df.shape)

    # Separate mean_reversion for appendix only
    mr_df = excess_df[["mean_reversion"]].copy()
    mr_df.to_parquet(OUT / "mean_reversion_returns.parquet")

    # Main returns = exclude mean_reversion
    main_cols = [c for c in excess_df.columns if c != "mean_reversion"]
    excess_df[main_cols].to_parquet(OUT / "strategy_returns.parquet")
    log.info("Saved strategy_returns.parquet  cols=%s", main_cols)

    # ---- OOS fold Sharpes (walk-forward) -----------------------------------
    log.info("Running walk-forward for OOS fold Sharpes ...")
    from backtest.walkforward import WalkForwardSplitter, WalkForwardConfig

    wf_cfg_dict = cfg.walkforward_cfg
    wf_cfg = WalkForwardConfig(
        mode=wf_cfg_dict.get("mode", "expanding"),
        initial_train_days=int(wf_cfg_dict.get("initial_train_days", 1260)),
        test_days=int(wf_cfg_dict.get("test_days", 252)),
        step_days=int(wf_cfg_dict.get("step_days", 252)),
        embargo_days=int(wf_cfg_dict.get("embargo_days", 21)),
    )
    splitter = WalkForwardSplitter(fd.index, wf_cfg)
    folds = splitter.folds()
    log.info("  %d folds", len(folds))

    fold_rows = []
    for fold in folds:
        test_mask = (fd.index >= fold.test_start) & (fd.index <= fold.test_end)
        test_dates = fd.index[test_mask]
        if len(test_dates) == 0:
            continue

        for name in ["regime_momentum", "long_short_momentum"]:
            oos_ret = all_excess[name].reindex(test_dates).dropna()
            if len(oos_ret) < 10:
                continue
            sr = oos_ret.mean() / oos_ret.std() * np.sqrt(TRADING_DAYS)
            fold_rows.append({
                "fold":       fold.fold_id,
                "strategy":   name,
                "test_start": fold.test_start,
                "test_end":   fold.test_end,
                "sharpe":     sr,
                "n_days":     len(oos_ret),
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "oos_fold_sharpes.csv", index=False)
    log.info("Saved oos_fold_sharpes.csv  rows=%d", len(fold_df))

    # ---- Mean-reversion cost-sweep series for appendix ---------------------
    log.info("Computing mean-reversion cost sweep for appendix ...")
    bps_grid = [0, 50, 100, 200]
    mr_sweep: dict[str, pd.Series] = {}
    for bps in bps_grid:
        from backtest.costs import CostConfig
        cc = CostConfig(tc_bps=bps, slippage_bps=0,
                        state_dependent_slippage=False, turbulent_slippage_multiplier=1.0)
        strat = strat_map["mean_reversion"]
        engine = BacktestEngine(strat, fd, md, eng_cfg)
        result = engine.run(cost_cfg_override=cc)
        rf_s = md.reindex(result.returns.index)["rf"].fillna(0.0)
        mr_sweep[f"mr_{bps}bps"] = result.returns["r_net"] - rf_s

    mr_sweep_df = pd.DataFrame(mr_sweep)
    mr_sweep_df.index.name = "date"
    mr_sweep_df.to_parquet(OUT / "mean_reversion_cost_sweep.parquet")
    log.info("Saved mean_reversion_cost_sweep.parquet")

    log.info("Export complete. Files in %s/", OUT)
    for f in sorted(OUT.iterdir()):
        log.info("  %s  (%d KB)", f.name, f.stat().st_size // 1024)


if __name__ == "__main__":
    main()
