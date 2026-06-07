"""Shared journal-style aesthetics for the equity-regime-research figure suite.

Journal target: Journal of Finance / JFE
  - Single column: 3.35 in wide
  - Two column:    6.85 in wide
  - Base font:     8 pt (axis labels), 7 pt (tick labels), 9 pt (panel labels)

Colorblind-safe palette (Okabe-Ito 2008):
  Blue      #0072B2  -- regime-conditioned momentum  (primary)
  Vermillion #D55E00 -- unconditional L/S momentum
  Sky blue  #56B4E9  -- market buy-and-hold
  Teal      #009E73  -- 60/40
  Pink      #CC79A7  -- long-only momentum
  Amber     #E69F00  -- highlight / VIX
  Black     #000000  -- reference lines

Regime shading:
  Calm:       no fill (white)
  Turbulent:  very light amber '#FEF0D9' at alpha=0.40

Linestyles ensure grayscale survival:
  solid      -- regime_momentum
  dashed     -- long_short_momentum
  dash-dot   -- market_bah
  dotted     -- sixty_forty
  solid thin -- long_only_momentum

Usage
-----
    from viz.style import apply_style, COLORS, LINES, shade_regimes, fig_size

    apply_style()   # call once at module import or top of script
    fig, ax = plt.subplots(figsize=fig_size())
    shade_regimes(ax, regime_series)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Semantic color map  (reused consistently across ALL figures)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "regime_momentum":     "#0072B2",   # blue
    "long_short_momentum": "#D55E00",   # vermillion
    "long_only_momentum":  "#CC79A7",   # pink/mauve
    "market_bah":          "#56B4E9",   # sky blue
    "sixty_forty":         "#009E73",   # teal
    "mean_reversion":      "#E69F00",   # amber
    # Regime shading
    "calm_fill":           "#FFFFFF",   # white (no fill)
    "turbulent_fill":      "#FEF0D9",   # very light amber
    "turbulent_edge":      "#E69F00",   # amber
    # Reference / annotation
    "reference":           "#444444",
    "zero_line":           "#888888",
    "crash_marker":        "#CC0000",
}

# Display labels for legend
LABELS: dict[str, str] = {
    "regime_momentum":     "Regime-conditioned momentum",
    "long_short_momentum": "Unconditional L/S momentum",
    "long_only_momentum":  "Long-only momentum",
    "market_bah":          "Market (buy-and-hold)",
    "sixty_forty":         "60/40 portfolio",
    "mean_reversion":      "Mean reversion (gross)",
}

# Linestyle assignments (grayscale-survivable)
LINES: dict[str, dict] = {
    "regime_momentum":     {"linestyle": "-",  "linewidth": 1.6},
    "long_short_momentum": {"linestyle": "--", "linewidth": 1.4},
    "long_only_momentum":  {"linestyle": "-",  "linewidth": 1.0},
    "market_bah":          {"linestyle": "-.", "linewidth": 1.1},
    "sixty_forty":         {"linestyle": ":",  "linewidth": 1.3},
    "mean_reversion":      {"linestyle": "-",  "linewidth": 1.4},
}

# Marker assignments for scatter / fold plots
MARKERS: dict[str, str] = {
    "regime_momentum":     "o",
    "long_short_momentum": "s",
    "market_bah":          "^",
    "sixty_forty":         "D",
}

# Named crash episodes (for vertical annotation)
CRASH_WINDOWS: dict[str, dict] = {
    "1973 Oil\nBear":    {"start": "1973-01-01", "end": "1975-01-01"},
    "1987 Black\nMonday":{"start": "1987-07-01", "end": "1988-01-01"},
    "2000 Dot-com\nbust": {"start": "2000-03-01", "end": "2002-12-01"},
    "2008\nGFC":          {"start": "2007-10-01", "end": "2009-06-01"},
    "2020\nCovid":        {"start": "2020-02-01", "end": "2020-09-01"},
}


# ---------------------------------------------------------------------------
# Figure sizing
# ---------------------------------------------------------------------------
SINGLE_COL_W = 3.35   # inches
TWO_COL_W    = 6.85   # inches

def fig_size(two_col: bool = False, aspect: float = 0.65) -> tuple[float, float]:
    """Return (width, height) in inches.

    Parameters
    ----------
    two_col:
        If True, use two-column width (6.85 in); else single-column (3.35 in).
    aspect:
        Height = width * aspect.  Default 0.65 gives a slightly landscape panel.
    """
    w = TWO_COL_W if two_col else SINGLE_COL_W
    return (w, w * aspect)


# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Apply journal-quality rcParams.  Call once before any figure creation."""
    mpl.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.title_fontsize": 7,
        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "axes.grid":          True,
        "axes.grid.which":    "major",
        "grid.alpha":         0.25,
        "grid.linewidth":     0.5,
        "grid.linestyle":     "-",
        # Ticks
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Lines
        "lines.linewidth":    1.2,
        "lines.markersize":   3.5,
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # Legend
        "legend.frameon":     True,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.75",
        "legend.borderpad":   0.4,
        "legend.handlelength": 1.8,
    })


# ---------------------------------------------------------------------------
# Helper: shade turbulent regime spans
# ---------------------------------------------------------------------------
def shade_regimes(
    ax: "plt.Axes",
    regime: pd.Series,
    alpha: float = 0.30,
    label: bool = False,
) -> None:
    """Shade spans where regime==1 (turbulent) on *ax*.

    Parameters
    ----------
    ax:
        The matplotlib Axes to shade.
    regime:
        Boolean or 0/1 Series with DatetimeIndex.
    alpha:
        Fill transparency.
    label:
        If True, add one legend proxy for the shaded regions.
    """
    if regime is None or len(regime) == 0:
        return

    turbulent = (regime == 1).astype(int)
    # Find contiguous blocks of turbulent=1
    diff = turbulent.diff().fillna(turbulent.iloc[0])
    starts = regime.index[diff == 1].tolist()
    ends   = regime.index[diff == -1].tolist()

    # Handle edge cases: series starts/ends in turbulent
    if turbulent.iloc[0] == 1:
        starts = [regime.index[0]] + starts
    if turbulent.iloc[-1] == 1:
        ends = ends + [regime.index[-1]]

    patch_added = False
    for s, e in zip(starts, ends):
        ax.axvspan(
            s, e,
            color=COLORS["turbulent_fill"],
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )
        if not patch_added and label:
            ax.axvspan(
                s, e,
                color=COLORS["turbulent_fill"],
                alpha=alpha,
                linewidth=0,
                label="Turbulent regime",
                zorder=0,
            )
            patch_added = True


# ---------------------------------------------------------------------------
# Helper: annotate crash episodes with vertical markers
# ---------------------------------------------------------------------------
def mark_crashes(
    ax: "plt.Axes",
    windows: Optional[dict] = None,
    ypos: float = 0.97,
    fontsize: float = 5.5,
    color: str = "#AA0000",
) -> None:
    """Add subtle vertical lines + labels for named crash episodes.

    Parameters
    ----------
    windows:
        Dict of {label: {start: str, end: str}}.  Defaults to CRASH_WINDOWS.
    ypos:
        Axes-fraction y position for the text label (0=bottom, 1=top).
    """
    if windows is None:
        windows = CRASH_WINDOWS

    ylim = ax.get_ylim()
    for label, span in windows.items():
        mid = pd.Timestamp(span["start"]) + (
            pd.Timestamp(span["end"]) - pd.Timestamp(span["start"])
        ) / 2
        ax.axvline(mid, color=color, linewidth=0.5, linestyle=":", alpha=0.6, zorder=1)
        ax.text(
            mid, ypos,
            label,
            transform=ax.get_xaxis_transform(),
            fontsize=fontsize,
            color=color,
            ha="center", va="top",
            rotation=0,
            alpha=0.8,
        )


# ---------------------------------------------------------------------------
# Helper: clean legend placement
# ---------------------------------------------------------------------------
def tidy_legend(ax: "plt.Axes", loc: str = "best", ncol: int = 1) -> None:
    """Apply consistent legend formatting."""
    leg = ax.legend(loc=loc, ncol=ncol, handlelength=1.6,
                    columnspacing=1.0, handletextpad=0.5)
    if leg:
        leg.get_frame().set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Utility: compute drawdown series
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown-from-peak series from daily returns."""
    wealth = (1 + returns).cumprod()
    peak   = wealth.cummax()
    dd     = wealth / peak - 1
    return dd


# ---------------------------------------------------------------------------
# Utility: compute rolling annualised Sharpe
# ---------------------------------------------------------------------------
def rolling_sharpe(excess_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualised Sharpe ratio (excess returns assumed)."""
    return (
        excess_returns.rolling(window, min_periods=int(window * 0.8))
        .apply(lambda x: x.mean() / x.std() * (252 ** 0.5) if x.std() > 1e-12 else np.nan,
               raw=True)
    )
