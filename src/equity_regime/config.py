"""Configuration loading: YAML -> typed nested dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class RunConfig:
    name: str = "equity_regime_study"
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    start_date: str = "2009-01-01"
    end_date: str = "2023-12-31"
    min_history_days: int = 252
    min_price: float = 1.0
    winsor_low: float = 0.005
    winsor_high: float = 0.995


@dataclass
class SyntheticConfig:
    n_stocks: int = 300
    n_days: int = 3780
    n_regimes: int = 2
    regime_mu: List[float] = field(default_factory=lambda: [0.0005, -0.0003])
    regime_sigma: List[float] = field(default_factory=lambda: [0.008, 0.020])
    transition_matrix: List[List[float]] = field(
        default_factory=lambda: [[0.98, 0.02], [0.10, 0.90]]
    )
    momentum_strength: float = 0.04
    reversal_strength: float = 0.06


@dataclass
class FactorsConfig:
    reversal_lookback: int = 21
    momentum_lookback: int = 252
    momentum_skip: int = 21
    n_portfolios: int = 10
    weighting: str = "value"


@dataclass
class RegimeConfig:
    method: str = "markov"
    n_states: int = 2
    vol_window: int = 63
    vol_high_pct: float = 0.75


@dataclass
class StatsConfig:
    acf_lags: int = 20
    variance_ratio_q: List[int] = field(default_factory=lambda: [2, 5, 10, 20])
    newey_west_lags: int = 5
    oos_split_date: str = "2018-01-01"
    embargo_days: int = 21


@dataclass
class ReportConfig:
    title: str = "Statistical Evaluation of Momentum, Mean Reversion, and Regime Dynamics"
    author: str = "Quantitative Research"
    format: str = "html"


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    factors: FactorsConfig = field(default_factory=FactorsConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load config from YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw or {})

    @classmethod
    def _from_dict(cls, d: dict) -> "Config":
        def _merge(dc_cls, section: dict):
            obj = dc_cls()
            for k, v in section.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        return cls(
            run=_merge(RunConfig, d.get("run", {})),
            data=_merge(DataConfig, d.get("data", {})),
            synthetic=_merge(SyntheticConfig, d.get("synthetic", {})),
            factors=_merge(FactorsConfig, d.get("factors", {})),
            regime=_merge(RegimeConfig, d.get("regime", {})),
            stats=_merge(StatsConfig, d.get("stats", {})),
            report=_merge(ReportConfig, d.get("report", {})),
        )

    def validate(self) -> None:
        """Raise ValueError for invalid config combinations."""
        if self.factors.momentum_skip >= self.factors.momentum_lookback:
            raise ValueError("momentum_skip must be < momentum_lookback")
        if not (0 < self.data.winsor_low < self.data.winsor_high < 1):
            raise ValueError("winsor bounds must satisfy 0 < low < high < 1")
        if self.regime.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if len(self.synthetic.regime_mu) != self.synthetic.n_regimes:
            raise ValueError("regime_mu length must equal n_regimes")
        if len(self.synthetic.regime_sigma) != self.synthetic.n_regimes:
            raise ValueError("regime_sigma length must equal n_regimes")

    def output_path(self, *parts: str) -> Path:
        """Resolve a path under the configured output_dir."""
        return Path(self.run.output_dir).joinpath(*parts)
