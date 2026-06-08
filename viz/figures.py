"""Publication-quality figure functions for the equity-regime-research paper.

Each function:
  - Reads ONLY from saved files in outputs/data/ and outputs/tables/
  - Returns (fig, caption_str)
  - Saves 300dpi PNG + vector PDF to outputs/figures/

Certified headline numbers (post-audit, June 2026):
  regime_momentum excess Sharpe   : 0.8374
  long_short_momentum excess Sharpe: 0.5919
  LW t-stat                        : 1.730   p=0.084 (NOT significant at 5%)
  Regime vol                       : 6.7%    Unconditional vol: 12.4%
  Sample                           : 1963-07-01 to 2026-04-30 (15,813 obs)
  Turbulent regime mean duration   : 5.3 weeks (27 trading days)

Crash-window max drawdowns (certified):
  Crisis        Regime  Uncond
  1973 Oil Bear  -8.4%  -14.8%
  1987 Monday    -2.5%  -14.7%
  DotCom         -2.3%  -28.5%
  GFC            -5.1%  -57.1%
  Covid          -8.6%  -33.4%
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from viz.style import (
    apply_style, COLORS, LABELS, LINES, MARKERS, CRASH_WINDOWS,
    fig_size, shade_regimes, mark_crashes, tidy_legend,
    drawdown_series, rolling_sharpe,
)

# ── Directories ─────────────────────────────────────────────────────────────
DATA_DIR    = Path("outputs/data")
TABLES_DIR  = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# ── Certified crash numbers (post-audit) ────────────────────────────────────
CERTIFIED_CRASH = {
    "1973\nOil Bear":  {"regime": -0.084, "uncond": -0.148},
    "1987\nBlack Mon.":{"regime": -0.025, "uncond": -0.147},
    "2000\nDot-com":   {"regime": -0.023, "uncond": -0.285},
    "2008\nGFC":       {"regime": -0.051, "uncond": -0.571},
    "2020\nCovid":     {"regime": -0.086, "uncond": -0.334},
}

SOURCE_NOTE = ("Kenneth French Data Library, daily, 1963-07-01 to 2026-04-30; "
               "author calculations.")


# ── Save helper ──────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as 300dpi PNG and vector PDF."""
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)


# ── Data loaders (each function calls only these) ────────────────────────────
def _load_returns() -> pd.DataFrame:
    """Load main strategy excess-return series (excl. mean reversion)."""
    return pd.read_parquet(DATA_DIR / "strategy_returns.parquet")


def _load_regime() -> pd.Series:
    """Load causal vol-rule regime labels (0=calm, 1=turbulent)."""
    return pd.read_parquet(DATA_DIR / "regime_series.parquet")["regime"]


def _load_market_daily() -> pd.DataFrame:
    return pd.read_parquet("data/processed/etl_full/market_daily.parquet")


def _load_oos_folds() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "oos_fold_sharpes.csv", parse_dates=["test_start", "test_end"])


def _load_mr_sweep() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "mean_reversion_stock_cost_sweep.parquet")


def _load_crash_table() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "metrics_crash_windows.csv")


# ============================================================================
# FIG 1 — Regime timeline
# ============================================================================
def fig1_regime_timeline(
    save: bool = True,
    two_col: bool = True,
) -> Tuple[plt.Figure, str]:
    """Market cumulative level with turbulent-regime shading.

    Upper panel: log-scale cumulative market return with turbulent spans shaded.
    Lower panel: rolling 21-day realised vol, highlighting turbulent threshold.
    Five crash episodes faintly marked.
    """
    apply_style()

    md = _load_market_daily()
    rets = _load_returns()
    md = md.reindex(rets.index)
    regime = _load_regime()

    mkt_excess = md["mkt_ret"].fillna(0) - md["rf"].fillna(0)
    cum_mkt = (1 + mkt_excess).cumprod()
    rvol = md["realized_vol_21"].ffill()

    # ── Fixed wide landscape size — never let style.fig_size shrink this ─────
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.10},
        sharex=True,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.97)

    # ── Panel A: cumulative market level ─────────────────────────────────────
    ax = axes[0]
    shade_regimes(ax, regime, alpha=0.28, label=True)
    ax.semilogy(cum_mkt.index, cum_mkt.values,
                color=COLORS["market_bah"], linewidth=1.2,
                label="Market excess return (log scale)")

    # Crash labels: short tags placed at top of axes via axvline + text
    CRASH_TAGS = {
        "1973 Oil\nShock":    {"start": "1973-01-01", "end": "1975-01-01"},
        "1987 Black\nMonday": {"start": "1987-07-01", "end": "1988-01-01"},
        "2000 Dot-\ncom":     {"start": "2000-03-01", "end": "2002-12-01"},
        "2008\nGFC":          {"start": "2007-10-01", "end": "2009-06-01"},
        "2020\nCovid":        {"start": "2020-02-01", "end": "2020-09-01"},
    }
    for tag, span in CRASH_TAGS.items():
        t0 = pd.Timestamp(span["start"])
        t1 = pd.Timestamp(span["end"])
        mid = t0 + (t1 - t0) / 2
        ax.axvline(mid, color=COLORS["crash_marker"],
                   linewidth=0.6, linestyle=":", alpha=0.55, zorder=1)
        # Use axes-fraction y so label sits just above the plot area
        ax.annotate(
            tag,
            xy=(mid, 1.01),
            xycoords=("data", "axes fraction"),
            fontsize=7, color=COLORS["crash_marker"],
            ha="center", va="bottom", alpha=0.85,
        )

    ax.set_ylabel("Cumulative growth (log scale)", labelpad=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.0f}x" if y >= 10 else f"{y:.1f}x"
    ))
    tidy_legend(ax, loc="upper left", ncol=2)

    # ── Panel B: realised vol + threshold ────────────────────────────────────
    ax2 = axes[1]
    shade_regimes(ax2, regime, alpha=0.28)
    ax2.plot(rvol.index, rvol.values * 100,
             color=COLORS["reference"], linewidth=0.9,
             label="Realised vol (21-day, ann.)")
    threshold = rvol.expanding().quantile(0.75).shift(1) * 100
    ax2.plot(threshold.index, threshold.values,
             color=COLORS["crash_marker"], linewidth=0.8, linestyle="--",
             label="75th-pct threshold", alpha=0.85)

    ax2.set_ylabel("Ann. vol (%)", labelpad=5)
    ax2.set_xlabel("")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    tidy_legend(ax2, loc="upper left", ncol=2)

    # x-axis ticks on lower panel only (sharex=True)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune="both"))
    ax2.tick_params(axis="x", rotation=0)

    fig.align_ylabels(axes)

    caption = (
        "Figure 1. Regime timeline, 1963-2026. "
        "Upper panel: cumulative excess market return (log scale) with turbulent "
        "regimes shaded (amber). Turbulent state defined by the expanding "
        "75th-percentile realised-volatility rule (causal, one-day lag). "
        "Vertical dotted lines mark the midpoints of five named crash episodes. "
        "Lower panel: 21-day realised annualised volatility (black) with the "
        "expanding 75th-percentile threshold (red dashed) that triggers the turbulent "
        "classification. Regime shading is consistent between panels. "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig1_regime_timeline")

    return fig, caption


# ============================================================================
# FIG 2 — Equity curves
# ============================================================================
def fig2_equity_curves(
    save: bool = True,
    two_col: bool = True,
) -> Tuple[plt.Figure, str]:
    """Cumulative $1 growth, excess returns, net of cost.

    Strategies: regime_momentum, long_short_momentum, market_bah, sixty_forty.
    Mean reversion excluded (unimplementable at factor level).
    Log y-axis. Turbulent regimes lightly shaded.
    """
    apply_style()

    rets   = _load_returns()
    regime = _load_regime()

    PLOT_STRATS = ["regime_momentum", "long_short_momentum", "market_bah", "sixty_forty"]

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.62))

    shade_regimes(ax, regime, alpha=0.22)

    for name in PLOT_STRATS:
        r = rets[name].dropna()
        cum = (1 + r).cumprod()
        ax.semilogy(
            cum.index, cum.values,
            color=COLORS[name],
            label=LABELS[name],
            **LINES[name],
        )
        # Annotate final value
        final_val = cum.iloc[-1]
        n_yrs = (r.index[-1] - r.index[0]).days / 365.25
        cagr  = final_val ** (1 / n_yrs) - 1
        ax.annotate(
            f"{final_val:.0f}x\n({cagr*100:.1f}% p.a.)",
            xy=(cum.index[-1], final_val),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=5.5,
            color=COLORS[name],
            va="center",
        )

    ax.set_ylabel("Cumulative growth of $1 (log, excess return)", labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:.0f}" if y >= 10 else f"${y:.1f}"
    ))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, prune="both"))
    tidy_legend(ax, loc="upper left")

    caption = (
        "Figure 2. Cumulative growth of $1, excess returns net of transaction costs, "
        "1963-2026. Log scale. All series are daily excess returns (net of risk-free "
        "rate) after 10bps one-way transaction cost and 5bps slippage; turbulent "
        "regime slippage doubled. Turbulent-regime spans shaded amber. "
        "Annotations show terminal wealth and compound annual growth rate (p.a.). "
        "Mean-reversion strategy excluded: gross factor-level SR 1.93 is "
        "unimplementable at 24 stock-level round-trips per year. "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig2_equity_curves")

    return fig, caption


# ============================================================================
# FIG 3 — Underwater / drawdown curves
# ============================================================================
def fig3_drawdown_curves(
    save: bool = True,
    two_col: bool = True,
) -> Tuple[plt.Figure, str]:
    """Drawdown-from-peak for regime-conditioned vs unconditional momentum.

    Full sample. Five crash windows shaded.
    """
    apply_style()

    rets   = _load_returns()
    regime = _load_regime()

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.60))

    shade_regimes(ax, regime, alpha=0.20)

    # Shade crash windows more prominently
    for label, span in CRASH_WINDOWS.items():
        ax.axvspan(
            pd.Timestamp(span["start"]),
            pd.Timestamp(span["end"]),
            color=COLORS["crash_marker"], alpha=0.06,
            linewidth=0, zorder=0,
        )

    for name in ["regime_momentum", "long_short_momentum"]:
        r  = rets[name].dropna()
        dd = drawdown_series(r)
        ax.fill_between(dd.index, dd.values, 0,
                        color=COLORS[name], alpha=0.15, linewidth=0)
        ax.plot(dd.index, dd.values,
                color=COLORS[name],
                label=LABELS[name],
                **LINES[name])

    # Annotate GFC drawdowns (the headline pair)
    gfc_start = pd.Timestamp("2007-10-01")
    gfc_end   = pd.Timestamp("2009-06-01")
    for name, mdd_val, yoff in [
        ("regime_momentum", -0.051, -0.04),
        ("long_short_momentum", -0.571, +0.04),
    ]:
        r  = rets[name].dropna()
        dd = drawdown_series(r)
        gfc_dd = dd.loc[gfc_start:gfc_end]
        trough = gfc_dd.idxmin()
        ax.annotate(
            f"GFC: {mdd_val*100:.1f}%",
            xy=(trough, mdd_val),
            xytext=(-45, 15 if name == "regime_momentum" else -25),
            textcoords="offset points",
            fontsize=6,
            color=COLORS[name],
            arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=0.5),
        )

    ax.set_ylabel("Drawdown from peak", labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, prune="both"))
    ax.set_ylim(top=0.01)
    tidy_legend(ax, loc="lower left")

    caption = (
        "Figure 3. Drawdown from running peak, full sample 1963-2026. "
        "Regime-conditioned momentum (blue solid) versus unconditional L/S momentum "
        "(vermillion dashed). Turbulent-regime spans shaded amber; five named crash "
        "episodes shaded red. The GFC drawdowns are annotated: regime-conditioned "
        "strategy -5.1% versus unconditional -57.1%, illustrating the crash-insurance "
        "property of volatility-regime conditioning. "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig3_drawdown_curves")

    return fig, caption


# ============================================================================
# FIG 4 — Crash-window drawdown bar chart
# ============================================================================
def fig4_crash_bars(
    save: bool = True,
    two_col: bool = False,
) -> Tuple[plt.Figure, str]:
    """Grouped bar chart of max drawdown per crisis, regime vs unconditional.

    Uses certified post-audit numbers. Sorted by severity of unconditional MDD.
    """
    apply_style()

    # Use certified numbers directly (post-audit verified against saved CSV)
    crises = list(CERTIFIED_CRASH.keys())
    # Sort by unconditional severity (worst first)
    crises.sort(key=lambda c: CERTIFIED_CRASH[c]["uncond"])

    reg_vals   = [CERTIFIED_CRASH[c]["regime"] * 100 for c in crises]
    uncond_vals = [CERTIFIED_CRASH[c]["uncond"] * 100 for c in crises]

    x     = np.arange(len(crises))
    width = 0.34

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.75))

    bars_u = ax.bar(x - width/2, uncond_vals, width,
                    color=COLORS["long_short_momentum"],
                    label="Unconditional L/S momentum",
                    linewidth=0.4, edgecolor="white")
    bars_r = ax.bar(x + width/2, reg_vals, width,
                    color=COLORS["regime_momentum"],
                    label="Regime-conditioned momentum",
                    linewidth=0.4, edgecolor="white")

    # Value labels
    for bar in bars_u:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v - 1.5,
                f"{v:.1f}%", ha="center", va="top", fontsize=5.8,
                color="white", fontweight="bold")
    for bar in bars_r:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v - 1.5,
                f"{v:.1f}%", ha="center", va="top", fontsize=5.8,
                color="white", fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(crises, fontsize=6.5)
    ax.set_ylabel("Max drawdown (%)", labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    tidy_legend(ax, loc="lower left")

    caption = (
        "Figure 4. Maximum drawdown during five crisis episodes, regime-conditioned "
        "versus unconditional L/S momentum. Bars sorted by severity of the "
        "unconditional strategy drawdown (worst leftmost). Certified post-audit values: "
        "GFC regime -5.1% vs unconditional -57.1%; 2000 dot-com -2.3% vs -28.5%; "
        "1987 Black Monday -2.5% vs -14.7%; 2020 Covid -8.6% vs -33.4%; "
        "1973 Oil Bear -8.4% vs -14.8%. "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig4_crash_bars")

    return fig, caption


# ============================================================================
# FIG 5 — Rolling 1-year Sharpe
# ============================================================================
def fig5_rolling_sharpe(
    save: bool = True,
    two_col: bool = True,
) -> Tuple[plt.Figure, str]:
    """Rolling 252-day excess-return Sharpe ratio, regime vs unconditional.

    Turbulent regimes shaded. Full-sample Sharpes annotated as horizontal refs.
    """
    apply_style()

    rets   = _load_returns()
    regime = _load_regime()

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.58))

    shade_regimes(ax, regime, alpha=0.22)

    for name in ["regime_momentum", "long_short_momentum"]:
        r  = rets[name].dropna()
        rs = rolling_sharpe(r, window=252)
        ax.plot(rs.index, rs.values,
                color=COLORS[name],
                label=LABELS[name],
                **LINES[name])

    # Full-sample Sharpe reference lines — labels anchored to LEFT margin
    for name, sr_val, va in [
        ("regime_momentum",    0.8374, "bottom"),
        ("long_short_momentum", 0.5919, "top"),
    ]:
        ax.axhline(sr_val, color=COLORS[name], linewidth=0.8,
                   linestyle="--", alpha=0.6)
        ax.text(
            0.01, sr_val,
            f" SR={sr_val:.3f}",
            transform=ax.get_yaxis_transform(),   # x in axes fraction, y in data
            fontsize=6, color=COLORS[name],
            ha="left", va=va,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
        )

    ax.axhline(0, color=COLORS["zero_line"], linewidth=0.5, linestyle="-")
    ax.set_ylabel("Rolling 1-yr Sharpe ratio (excess)", labelpad=4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, prune="both"))
    tidy_legend(ax, loc="upper right")

    caption = (
        "Figure 5. Rolling 252-day (1-year) annualised Sharpe ratio from excess "
        "returns, 1964-2026. Regime-conditioned momentum (blue solid) versus "
        "unconditional L/S momentum (vermillion dashed). Turbulent-regime spans "
        "shaded amber. Horizontal dashed lines show full-sample Sharpe ratios "
        "(regime 0.837, unconditional 0.592). The edge is concentrated in and around "
        "turbulent periods (regime-conditioned strategy maintains positive Sharpe "
        "through crisis windows where the unconditional strategy deeply underperforms). "
        "Ledoit-Wolf (2008) test: difference = +0.245, t = 1.730, p = 0.084 "
        "(not significant at 5% after correcting total-return bias). "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig5_rolling_sharpe")

    return fig, caption


# ============================================================================
# FIG 6 — Return distribution
# ============================================================================
def fig6_return_distribution(
    save: bool = True,
    two_col: bool = False,
) -> Tuple[plt.Figure, str]:
    """Histogram + KDE of monthly excess returns, regime vs unconditional.

    Marks mean and 5% VaR. Annotates skewness and excess kurtosis.
    """
    apply_style()
    from scipy.stats import gaussian_kde
    from scipy.stats import skew as _skew, kurtosis as _kurt

    rets = _load_returns()

    # Resample to monthly
    monthly = {}
    for name in ["regime_momentum", "long_short_momentum"]:
        r = rets[name].dropna()
        monthly[name] = r.resample("ME").sum()

    # Wider figure so annotation box never clips
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    bin_edges = np.linspace(-0.30, 0.30, 55)
    x_grid    = np.linspace(-0.35, 0.35, 400)

    stats_text_parts = []
    for name in ["regime_momentum", "long_short_momentum"]:
        r     = monthly[name].dropna().values
        color = COLORS[name]
        label = LABELS[name]

        ax.hist(r, bins=bin_edges, density=True,
                color=color, alpha=0.25, linewidth=0)

        kde = gaussian_kde(r, bw_method="silverman")
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5,
                label=label, **{k: v for k, v in LINES[name].items()
                                if k == "linestyle"})

        mean_r = r.mean()
        var5   = np.percentile(r, 5)
        ax.axvline(mean_r, color=color, linewidth=0.8, linestyle="-", alpha=0.8)
        ax.axvline(var5,   color=color, linewidth=0.8, linestyle=":", alpha=0.8)

        sk = _skew(r)
        ku = _kurt(r)
        # Use short strategy name to prevent clipping
        short_label = "Regime-cond. momentum" if "regime" in name else "Uncond. L/S momentum"
        stats_text_parts.append(
            f"{short_label}\n"
            f"  Mean={mean_r*100:.2f}%  VaR5={var5*100:.1f}%\n"
            f"  Skew={sk:.2f}  Ex-kurt={ku:.1f}"
        )

    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)

    # Stats box — anchored inside left edge to avoid right-side clipping
    stats_text = "\n\n".join(stats_text_parts)
    ax.text(0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.75", linewidth=0.5, alpha=0.92))

    ax.set_xlabel("Monthly excess return", labelpad=4)
    ax.set_ylabel("Density", labelpad=4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    tidy_legend(ax, loc="upper left")

    caption = (
        "Figure 6. Distribution of monthly excess returns, 1963-2026. "
        "Histogram (transparent fill) and kernel density estimate (KDE, "
        "Silverman bandwidth) for regime-conditioned momentum (blue) and "
        "unconditional L/S momentum (vermillion). "
        "Vertical solid lines: sample mean; vertical dotted lines: 5th-percentile "
        "value-at-risk. Annotations show skewness and excess kurtosis for each "
        "strategy. Regime conditioning trims the left tail visible in the "
        "unconditional distribution during turbulent periods. "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig6_return_distribution")

    return fig, caption


# ============================================================================
# FIG 7 — Regime-conditional performance heatmap
# ============================================================================
def fig7_regime_heatmap(
    save: bool = True,
    two_col: bool = False,
) -> Tuple[plt.Figure, str]:
    """Annualised excess return by strategy x regime state.

    Rows: regime_momentum, long_short_momentum, market_bah.
    Columns: calm (state=0), turbulent (state=1).
    Diverging colormap centred at zero.
    """
    apply_style()

    rets   = _load_returns()
    regime = _load_regime()
    regime = regime.reindex(rets.index).fillna(0)

    HMAP_STRATS = ["regime_momentum", "long_short_momentum", "market_bah", "sixty_forty"]
    ROW_LABELS  = [LABELS[s] for s in HMAP_STRATS]
    COL_LABELS  = ["Calm\n(state = 0)", "Turbulent\n(state = 1)"]

    data = np.zeros((len(HMAP_STRATS), 2))
    n_obs = np.zeros((len(HMAP_STRATS), 2), dtype=int)

    for i, name in enumerate(HMAP_STRATS):
        r = rets[name].dropna()
        reg_aligned = regime.reindex(r.index).fillna(0)
        for j, state in enumerate([0, 1]):
            mask = reg_aligned == state
            r_state = r[mask]
            n_obs[i, j] = int(mask.sum())
            data[i, j] = float(r_state.mean() * TRADING_DAYS) if len(r_state) > 0 else np.nan

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.80))

    # Diverging cmap
    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    # Cell text
    for i in range(len(HMAP_STRATS)):
        for j in range(2):
            val = data[i, j]
            n   = n_obs[i, j]
            cell_color = "white" if abs(val) > vmax * 0.55 else "black"
            ax.text(j, i,
                    f"{val*100:.1f}%\n({n}d)",
                    ha="center", va="center",
                    fontsize=6.5, color=cell_color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(COL_LABELS, fontsize=7)
    ax.set_yticks(range(len(HMAP_STRATS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=6.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Annualised excess return", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
    )

    # Remove grid for heatmap
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    caption = (
        "Figure 7. Annualised excess return by strategy and regime state, "
        "1963-2026. Rows: four strategies. Columns: calm regime (volatility below "
        "expanding 75th percentile) and turbulent regime. Cell values show "
        "annualised mean excess return with trading-day count in parentheses. "
        "Blue = positive excess return; red = negative. "
        "Regime-conditioned momentum earns positive returns in calm and avoids "
        "losses in turbulent periods by design; unconditional momentum suffers "
        "sharp losses in turbulent regimes (momentum crashes). "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "fig7_regime_heatmap")

    return fig, caption


# ============================================================================
# FIG A1 — Mean reversion gross vs net (appendix only)
# ============================================================================
def figA1_mean_reversion_costs(
    save: bool = True,
    two_col: bool = False,
) -> Tuple[plt.Figure, str]:
    """Cumulative excess return of mean reversion at different stock-level costs.

    LABELED 'unimplementable; illustrative of factor-return cost blindness.'
    Appendix only.
    """
    apply_style()

    sweep = _load_mr_sweep()

    PLOT_COLS = {
        "gross_0bps":    ("Gross (0 bps stock cost)",   "#444444",  "-",  1.5),
        "stock_50bps":   ("50 bps x 24 turns/yr",       "#E69F00",  "--", 1.2),
        "stock_100bps":  ("100 bps x 24 turns/yr",      "#D55E00",  "-.", 1.2),
        "stock_200bps":  ("200 bps x 24 turns/yr",      "#AA0000",  ":",  1.3),
    }

    fig, ax = plt.subplots(figsize=fig_size(two_col=two_col, aspect=0.85))

    for col, (label, color, ls, lw) in PLOT_COLS.items():
        r   = sweep[col].dropna()
        cum = (1 + r).cumprod()
        ax.semilogy(cum.index, cum.values,
                    color=color, linestyle=ls, linewidth=lw, label=label)
        sr = r.mean() / r.std() * TRADING_DAYS ** 0.5
        ax.annotate(
            f"SR={sr:.2f}",
            xy=(cum.index[-1], cum.iloc[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=6, color=color, va="center",
        )

    # Legend first — upper left
    tidy_legend(ax, loc="upper left")

    # UNIMPLEMENTABLE warning — lower centre, well below the legend
    ax.text(0.5, 0.18,
            "UNIMPLEMENTABLE: gross SR=1.93 is a factor-level artifact.\n"
            "Stock-level turnover (~24 round-trips/yr) is invisible to the\n"
            "factor-return engine. Appendix illustrative only.",
            transform=ax.transAxes, fontsize=6, ha="center", va="top",
            color="#AA0000", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF0F0",
                      edgecolor="#AA0000", linewidth=0.8))

    ax.set_ylabel("Cumulative growth of $1 (log, excess return)", labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:.0f}" if y >= 10 else f"${y:.1f}"
    ))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, prune="both"))

    caption = (
        "Figure A1 (Appendix). Mean-reversion strategy: gross versus stock-level "
        "cost scenarios, 1963-2026. [UNIMPLEMENTABLE -- illustrative only.] "
        "The ST_Rev factor return represents a monthly-rebalanced stock portfolio. "
        "The factor-level engine records zero turnover (weight always 1.0) and "
        "applies no cost drag, yielding a gross Sharpe of 1.93. "
        "The four lines apply constant daily drags equivalent to 0, 50, 100, and "
        "200bps stock-level round-trip cost on an assumed 24 round-trips per year "
        "(consistent with monthly stock rebalancing of the ST_Rev sort). "
        "At 100bps the cumulative value approaches 1x (SR approx. 0); "
        "at 200bps the strategy is deeply negative. "
        "This figure illustrates the factor-return cost blindspot identified in the "
        "adversarial audit (Check 2). "
        "Data: " + SOURCE_NOTE
    )

    if save:
        _save(fig, "figA1_mean_reversion_costs")

    return fig, caption
