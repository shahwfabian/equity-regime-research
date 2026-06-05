"""Shared pytest fixtures for equity_regime tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def default_config():
    from equity_regime.config import Config
    cfg = Config.load(Path(__file__).parent.parent / "configs" / "default.yaml")
    return cfg


@pytest.fixture(scope="session")
def small_config():
    """Tiny config for fast tests."""
    from equity_regime.config import Config, SyntheticConfig
    cfg = Config()
    cfg.synthetic.n_stocks = 30
    cfg.synthetic.n_days = 800
    cfg.run.seed = 123
    return cfg


@pytest.fixture(scope="session")
def synthetic_data(small_config):
    """Generate small synthetic dataset once per session."""
    from equity_regime.data.synthetic import generate
    stock_day, market_day, true_regime = generate(small_config.synthetic, seed=small_config.run.seed)
    return stock_day, market_day, true_regime


@pytest.fixture(scope="session")
def clean_data(synthetic_data, small_config):
    from equity_regime.data.clean import clean
    stock_day, market_day, _ = synthetic_data
    # Relax min_history for small data
    small_config.data.min_history_days = 100
    return clean(stock_day, market_day, small_config.data)
