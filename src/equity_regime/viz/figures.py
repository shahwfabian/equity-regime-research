"""Figure generation: all plots saved as PNGs to outputs/figures/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


_STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def _save(fig: plt.Figure, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cumulative_returns(
    mom_ls: pd.DataFrame,
    rev_ls: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Cumulative returns of momentum vs. reversal long-short portfolios."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        for ls_df, label, color in [
            (mom_ls, "Momentum L/S", "steelblue"),
            (rev_ls, "Reversal L/S", "darkorange"),
        ]:
            if ls_df.empty or "ls_ret" not in ls_df.columns:
                continue
            s = ls_df.set_index("date")["ls_ret"].dropna()
            cum = (1 + s).cumprod()
            ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.5)

        ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Cumulative Return: Momentum vs. Reversal Long-Short")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (indexed to 1)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    return _save(fig, out_dir, "01_cumulative_returns.png")


def plot_rolling_autocorrelation(
    market_day: pd.DataFrame,
    lags: List[int],
    out_dir: Path,
    window: int = 252,
) -> Path:
    """Rolling autocorrelation of market returns at multiple horizons."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        rets = market_day.set_index("date")["mkt_ret"].dropna()

        for lag in lags[:4]:  # cap at 4 for readability
            roll_ac = (
                rets.rolling(window)
                .apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False)
            )
            ax.plot(roll_ac.index, roll_ac.values, label=f"lag={lag}", linewidth=1.2, alpha=0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_title(f"Rolling {window}d Autocorrelation of Market Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Autocorrelation")
        ax.legend(ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    return _save(fig, out_dir, "02_rolling_autocorrelation.png")


def plot_variance_ratio(vr_df: pd.DataFrame, out_dir: Path) -> Path:
    """Variance ratio with 95% CI bands."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        if not vr_df.empty:
            ax.plot(vr_df["q"], vr_df["vr"], "o-", color="steelblue", label="VR(q)", linewidth=2)
            ax.fill_between(
                vr_df["q"],
                vr_df["ci_lower"],
                vr_df["ci_upper"],
                alpha=0.2,
                color="steelblue",
                label="95% CI",
            )
        ax.axhline(1, color="black", linewidth=1, linestyle="--", label="RW benchmark")
        ax.set_xlabel("Holding period q (days)")
        ax.set_ylabel("Variance Ratio VR(q)")
        ax.set_title("Lo-MacKinlay Variance Ratio Test")
        ax.legend()

    return _save(fig, out_dir, "03_variance_ratio.png")


def plot_regime_overlay(
    market_day: pd.DataFrame,
    regime_col: str = "filtered_regime",
    out_dir: Path = Path("outputs/figures"),
) -> Path:
    """Filtered regime probabilities overlaid on market level."""
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        df = market_day.set_index("date").sort_index()

        # Market level
        mkt_level = np.exp(df["mkt_ret"].fillna(0).cumsum()) * 100
        ax1.plot(df.index, mkt_level, color="black", linewidth=1.2)
        ax1.set_ylabel("Market Level (indexed)")
        ax1.set_title("Market Level and Regime Detection")

        # Shade turbulent regimes
        if regime_col in df.columns:
            turb = df[regime_col].fillna(0).astype(bool)
            in_regime = False
            start = None
            for date, val in turb.items():
                if val and not in_regime:
                    start = date
                    in_regime = True
                elif not val and in_regime:
                    ax1.axvspan(start, date, alpha=0.15, color="red")
                    in_regime = False
            if in_regime and start is not None:
                ax1.axvspan(start, df.index[-1], alpha=0.15, color="red")

        # Filtered probability
        prob_col = "filtered_prob_state1"
        if prob_col in df.columns:
            ax2.fill_between(df.index, df[prob_col], alpha=0.6, color="red", label="P(turbulent | filtered)")
            ax2.set_ylabel("Filtered P(turbulent)")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right")
        else:
            ax2.plot(df.index, df.get(regime_col, pd.Series(dtype=float)), color="red")
            ax2.set_ylabel("Regime Label")

        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    return _save(fig, out_dir, "04_regime_overlay.png")


def plot_state_stats(
    state_means: np.ndarray,
    state_vols: np.ndarray,
    out_dir: Path,
) -> Path:
    """State-dependent mean/vol bar chart."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        n_states = len(state_means)
        labels = [f"State {i}" for i in range(n_states)]
        x = np.arange(n_states)

        axes[0].bar(x, state_means * TRADING_DAYS, color=["steelblue", "darkorange"][:n_states])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_title("Annualized Mean Return by State")
        axes[0].set_ylabel("Ann. Mean (%)")
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

        axes[1].bar(x, state_vols * np.sqrt(TRADING_DAYS), color=["steelblue", "darkorange"][:n_states])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title("Annualized Volatility by State")
        axes[1].set_ylabel("Ann. Vol (%)")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

        fig.suptitle("Markov Regime State Statistics")
        fig.tight_layout()

    return _save(fig, out_dir, "05_state_stats.png")


TRADING_DAYS = 252


def plot_momentum_by_regime(
    coef_table: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Momentum regression coefficient by regime with confidence intervals."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))

        if coef_table.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        else:
            vars_to_plot = coef_table[coef_table["variable"] != "const"]
            y_pos = np.arange(len(vars_to_plot))
            coefs = vars_to_plot["coef"].values
            ci_low = vars_to_plot["ci_lower"].values if "ci_lower" in vars_to_plot else coefs - 0.01
            ci_high = vars_to_plot["ci_upper"].values if "ci_upper" in vars_to_plot else coefs + 0.01

            ax.barh(y_pos, coefs, xerr=[coefs - ci_low, ci_high - coefs],
                    color="steelblue", alpha=0.8, capsize=4)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(vars_to_plot["variable"].values)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Coefficient")
            ax.set_title("Momentum Signal Predictive Coefficients (Newey-West HAC)")

    return _save(fig, out_dir, "06_momentum_by_regime.png")


def plot_equity_curves(
    strategies: Dict[str, pd.Series],
    out_dir: Path,
) -> Path:
    """Strategy equity curves."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        colors = plt.cm.tab10.colors
        for i, (name, ret) in enumerate(strategies.items()):
            s = ret.dropna()
            if s.empty:
                continue
            cum = (1 + s).cumprod()
            ax.plot(cum.index, cum.values, label=name, color=colors[i % len(colors)], linewidth=1.5)

        ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Strategy Equity Curves")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    return _save(fig, out_dir, "07_equity_curves.png")


def plot_drawdowns(
    strategies: Dict[str, pd.Series],
    out_dir: Path,
) -> Path:
    """Drawdown plot for each strategy."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))

        colors = plt.cm.tab10.colors
        for i, (name, ret) in enumerate(strategies.items()):
            s = ret.dropna()
            if s.empty:
                continue
            cum = (1 + s).cumprod()
            roll_max = cum.cummax()
            dd = (cum - roll_max) / roll_max
            ax.plot(dd.index, dd.values, label=name, color=colors[i % len(colors)],
                    linewidth=1.2, alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.05, color="red")
        ax.set_title("Strategy Drawdowns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()

    return _save(fig, out_dir, "08_drawdowns.png")
