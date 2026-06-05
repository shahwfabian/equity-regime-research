"""Tests for config loading and validation."""

import pytest
from pathlib import Path
from equity_regime.config import Config


def test_load_default_config():
    cfg = Config.load(Path("configs/default.yaml"))
    assert cfg.run.seed == 42
    assert cfg.synthetic.n_stocks == 300
    assert cfg.factors.momentum_lookback == 252
    assert cfg.factors.momentum_skip == 21
    assert cfg.regime.n_states == 2


def test_validate_passes():
    cfg = Config.load(Path("configs/default.yaml"))
    cfg.validate()  # should not raise


def test_validate_rejects_bad_skip():
    cfg = Config()
    cfg.factors.momentum_skip = 300
    cfg.factors.momentum_lookback = 252
    with pytest.raises(ValueError, match="momentum_skip"):
        cfg.validate()


def test_validate_rejects_bad_winsor():
    cfg = Config()
    cfg.data.winsor_low = 0.9
    cfg.data.winsor_high = 0.1
    with pytest.raises(ValueError, match="winsor"):
        cfg.validate()


def test_output_path():
    cfg = Config()
    cfg.run.output_dir = "outputs"
    p = cfg.output_path("figures", "test.png")
    assert str(p).endswith("test.png")
