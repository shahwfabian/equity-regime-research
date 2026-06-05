"""ETL pipeline configuration: YAML → typed dataclasses."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2024-12-31"


@dataclass
class StoreConfig:
    backend: str = "parquet"           # "parquet" | "postgres"
    parquet_dir: str = "data/processed/etl"
    postgres_url: str = "postgresql+psycopg2://localhost:5432/equity_regime"


@dataclass
class FrenchConfig:
    enabled: bool = True
    cache_dir: str = "data/raw/french"
    daily_dataset: str = "F-F_Research_Data_5_Factors_2x3_daily"
    momentum_dataset: str = "F-F_Momentum_Factor_daily"
    st_rev_dataset: str = "F-F_ST_Reversal_Factor_daily"
    lt_rev_dataset: str = "F-F_LT_Reversal_Factor_daily"
    monthly_dataset: str = "F-F_Research_Data_5_Factors_2x3"
    base_url: str = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


@dataclass
class VixConfig:
    enabled: bool = True
    cache_dir: str = "data/raw/vix"
    fred_series: str = "VIXCLS"
    stooq_ticker: str = "^VIX.US"
    fred_api_key: str = ""


@dataclass
class YFinanceConfig:
    enabled: bool = False
    cache_dir: str = "data/raw/yfinance"
    tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    illustrative_only: bool = True


@dataclass
class SourcesConfig:
    french: FrenchConfig = field(default_factory=FrenchConfig)
    vix: VixConfig = field(default_factory=VixConfig)
    yfinance: YFinanceConfig = field(default_factory=YFinanceConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "outputs/logs/etl.log"


@dataclass
class ValidationConfig:
    report_path: str = "outputs/reports/data_validation.html"
    max_gap_days: int = 5
    max_abs_return: float = 0.25
    vix_ffill_limit: int = 1


@dataclass
class ETLConfig:
    name: str = "equity_etl"
    version: str = "0.2.0"
    window: WindowConfig = field(default_factory=WindowConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ETLConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "ETLConfig":
        pip = d.get("pipeline", {})
        win = d.get("window", {})
        sto = d.get("store", {})
        src = d.get("sources", {})
        log_ = d.get("logging", {})
        val = d.get("validation", {})

        french_d = src.get("french", {})
        vix_d = src.get("vix", {})
        yf_d = src.get("yfinance", {})

        def _fill(dc, raw):
            obj = dc()
            for k, v in raw.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        return cls(
            name=pip.get("name", "equity_etl"),
            version=pip.get("version", "0.2.0"),
            window=_fill(WindowConfig, win),
            store=_fill(StoreConfig, sto),
            sources=SourcesConfig(
                french=_fill(FrenchConfig, french_d),
                vix=_fill(VixConfig, vix_d),
                yfinance=_fill(YFinanceConfig, yf_d),
            ),
            logging=_fill(LoggingConfig, log_),
            validation=_fill(ValidationConfig, val),
        )

    def resolve_path(self, p: str) -> Path:
        """Resolve a config path relative to the project root (CWD)."""
        return Path(p)

    def validate(self) -> None:
        if self.store.backend not in ("parquet", "postgres"):
            raise ValueError(f"store.backend must be 'parquet' or 'postgres', got {self.store.backend!r}")
        if self.window.start_date >= self.window.end_date:
            raise ValueError("window.start_date must be before end_date")
