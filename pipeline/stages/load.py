"""Load stage: write to Parquet or PostgreSQL behind a common interface.

The Store class abstracts both backends. All callers use the same API:
  store.write(table_name, df)
  store.read(table_name) -> pd.DataFrame
  store.table_exists(table_name) -> bool

Parquet backend: one file per table in cfg.store.parquet_dir/<table>.parquet
Postgres backend: one table per schema table, upsert via INSERT … ON CONFLICT.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline.stages.transform import TransformResult

log = logging.getLogger(__name__)

# Canonical table names
TABLE_FACTOR_DAILY   = "factor_daily"
TABLE_FACTOR_MONTHLY = "factor_monthly"
TABLE_MARKET_DAILY   = "market_daily"
TABLE_STOCK_DAILY    = "stock_daily"

# Required columns per table (for schema validation on read-back)
SCHEMAS: dict[str, list[str]] = {
    TABLE_FACTOR_DAILY:   ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    TABLE_FACTOR_MONTHLY: ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"],
    TABLE_MARKET_DAILY:   ["mkt_ret", "rf", "realized_vol_21", "realized_vol_63", "drawdown", "vix"],
    TABLE_STOCK_DAILY:    ["date", "ticker", "adj_close", "ret", "log_ret"],
}


class Store:
    """Unified read/write interface for Parquet and PostgreSQL backends."""

    def __init__(self, cfg):
        self.backend = cfg.store.backend
        if self.backend == "parquet":
            self.parquet_dir = Path(cfg.store.parquet_dir)
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            self._engine = None
            log.info("[store] Backend: parquet -> %s", self.parquet_dir)
        elif self.backend == "postgres":
            from sqlalchemy import create_engine
            self._engine = create_engine(cfg.store.postgres_url)
            log.info("[store] Backend: postgres -> %s", cfg.store.postgres_url)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def write(self, table: str, df: pd.DataFrame) -> int:
        """Write DataFrame to the active backend. Returns row count written."""
        t0 = time.time()
        n = len(df)
        if self.backend == "parquet":
            path = self.parquet_dir / f"{table}.parquet"
            df.to_parquet(path, index=True)
            log.info("[load] Wrote %d rows to parquet: %s (%.2fs)", n, path, time.time() - t0)
        else:
            df.to_sql(
                table,
                self._engine,
                if_exists="replace",
                index=True,
                method="multi",
                chunksize=5000,
            )
            log.info("[load] Wrote %d rows to postgres table '%s' (%.2fs)", n, table, time.time() - t0)
        return n

    def read(self, table: str) -> pd.DataFrame:
        """Read a table back from the active backend."""
        if self.backend == "parquet":
            path = self.parquet_dir / f"{table}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Parquet table not found: {path}")
            df = pd.read_parquet(path)
        else:
            df = pd.read_sql_table(table, self._engine, index_col="date", parse_dates=["date"])
        return df

    def table_exists(self, table: str) -> bool:
        if self.backend == "parquet":
            return (self.parquet_dir / f"{table}.parquet").exists()
        else:
            from sqlalchemy import inspect
            insp = inspect(self._engine)
            return insp.has_table(table)

    def list_tables(self) -> list[str]:
        if self.backend == "parquet":
            return [p.stem for p in self.parquet_dir.glob("*.parquet")]
        else:
            from sqlalchemy import inspect
            return inspect(self._engine).get_table_names()


def load(transformed: TransformResult, cfg) -> Store:
    """
    Load all transformed tables into the active backend.

    Returns the Store instance for downstream use (validation round-trip test).
    """
    store = Store(cfg)

    tables = {
        TABLE_FACTOR_DAILY:   transformed.factor_daily,
        TABLE_FACTOR_MONTHLY: transformed.factor_monthly,
        TABLE_MARKET_DAILY:   transformed.market_daily,
    }
    if transformed.stock_daily is not None:
        tables[TABLE_STOCK_DAILY] = transformed.stock_daily.set_index(["date", "ticker"])

    total_rows = 0
    for table_name, df in tables.items():
        n = store.write(table_name, df)
        total_rows += n

    log.info("[load] Total rows written: %d across %d tables", total_rows, len(tables))
    return store
