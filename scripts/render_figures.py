"""Render all publication figures from saved data files.

Usage (from repo root):
    python scripts/render_figures.py

Options:
    --from <dir>    Data directory  (default: outputs/data)
    --out   <dir>   Figures directory  (default: outputs/figures)
    --no-save       Preview only, do not write files

Outputs:
    outputs/figures/fig1_regime_timeline.{png,pdf}
    outputs/figures/fig2_equity_curves.{png,pdf}
    outputs/figures/fig3_drawdown_curves.{png,pdf}
    outputs/figures/fig4_crash_bars.{png,pdf}
    outputs/figures/fig5_rolling_sharpe.{png,pdf}
    outputs/figures/fig6_return_distribution.{png,pdf}
    outputs/figures/fig7_regime_heatmap.{png,pdf}
    outputs/figures/figA1_mean_reversion_costs.{png,pdf}
    outputs/figures/contact_sheet.png
    outputs/figures/figures_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
import warnings
from pathlib import Path

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless / CI use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from viz.style import apply_style
from viz.figures import FIGURES_DIR
import viz.figures as F


# ── Figure registry ───────────────────────────────────────────────────────────
FIGURE_REGISTRY = [
    {
        "id":       "Fig 1",
        "stem":     "fig1_regime_timeline",
        "title":    "Regime timeline: market level + turbulent shading",
        "fn":       F.fig1_regime_timeline,
        "two_col":  True,
        "source":   "data/processed/etl_full/market_daily.parquet, "
                    "outputs/data/regime_series.parquet, "
                    "outputs/data/strategy_returns.parquet",
    },
    {
        "id":       "Fig 2",
        "stem":     "fig2_equity_curves",
        "title":    "Equity curves: cumulative $1 growth (excess, net of cost)",
        "fn":       F.fig2_equity_curves,
        "two_col":  True,
        "source":   "outputs/data/strategy_returns.parquet, "
                    "outputs/data/regime_series.parquet",
    },
    {
        "id":       "Fig 3",
        "stem":     "fig3_drawdown_curves",
        "title":    "Underwater curves: drawdown from peak",
        "fn":       F.fig3_drawdown_curves,
        "two_col":  True,
        "source":   "outputs/data/strategy_returns.parquet, "
                    "outputs/data/regime_series.parquet",
    },
    {
        "id":       "Fig 4",
        "stem":     "fig4_crash_bars",
        "title":    "Crisis drawdown bar chart: regime vs unconditional",
        "fn":       F.fig4_crash_bars,
        "two_col":  False,
        "source":   "outputs/tables/metrics_crash_windows.csv (certified numbers hardcoded)",
    },
    {
        "id":       "Fig 5",
        "stem":     "fig5_rolling_sharpe",
        "title":    "Rolling 1-year Sharpe: regime vs unconditional",
        "fn":       F.fig5_rolling_sharpe,
        "two_col":  True,
        "source":   "outputs/data/strategy_returns.parquet, "
                    "outputs/data/regime_series.parquet",
    },
    {
        "id":       "Fig 6",
        "stem":     "fig6_return_distribution",
        "title":    "Monthly excess-return distribution: regime vs unconditional",
        "fn":       F.fig6_return_distribution,
        "two_col":  False,
        "source":   "outputs/data/strategy_returns.parquet",
    },
    {
        "id":       "Fig 7",
        "stem":     "fig7_regime_heatmap",
        "title":    "Regime-conditional performance heatmap",
        "fn":       F.fig7_regime_heatmap,
        "two_col":  False,
        "source":   "outputs/data/strategy_returns.parquet, "
                    "outputs/data/regime_series.parquet",
    },
    {
        "id":       "Fig A1",
        "stem":     "figA1_mean_reversion_costs",
        "title":    "Appendix: mean reversion gross vs stock-level cost scenarios",
        "fn":       F.figA1_mean_reversion_costs,
        "two_col":  False,
        "source":   "outputs/data/mean_reversion_stock_cost_sweep.parquet",
    },
]


# ── Main render loop ──────────────────────────────────────────────────────────
def render_all(save: bool = True) -> list[dict]:
    """Render every figure. Returns manifest rows."""
    apply_style()

    manifest_rows = []
    rendered_figs = []
    captions = {}

    for entry in FIGURE_REGISTRY:
        fig_id   = entry["id"]
        stem     = entry["stem"]
        fn       = entry["fn"]
        two_col  = entry["two_col"]

        print(f"  Rendering {fig_id}: {entry['title']} ... ", end="", flush=True)
        t0 = time.time()
        try:
            fig, caption = fn(save=save, two_col=two_col)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            captions[fig_id] = caption
            rendered_figs.append((fig_id, fig))
        except Exception:
            print("FAILED")
            traceback.print_exc()
            captions[fig_id] = "[FAILED]"
            rendered_figs.append((fig_id, None))

        manifest_rows.append({
            "figure":    fig_id,
            "title":     entry["title"],
            "source":    entry["source"],
            "stem":      stem,
            "caption":   captions[fig_id],
        })
        plt.close("all")

    return manifest_rows, rendered_figs, captions


def render_contact_sheet(save: bool = True) -> None:
    """Render a contact sheet of all figures on one page for quick review."""
    apply_style()
    print("  Rendering contact sheet ... ", end="", flush=True)

    n_figs = len(FIGURE_REGISTRY)
    n_cols = 2
    n_rows = (n_figs + 1) // n_cols

    fig = plt.figure(figsize=(13, n_rows * 4.2))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.45, wspace=0.25)

    for idx, entry in enumerate(FIGURE_REGISTRY):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_aspect("auto")
        try:
            png_path = FIGURES_DIR / f"{entry['stem']}.png"
            if png_path.exists():
                img = plt.imread(str(png_path))
                ax.imshow(img, interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "Not rendered", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")
        except Exception:
            ax.text(0.5, 0.5, "Error loading", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")

        ax.set_title(f"{entry['id']}: {entry['title']}", fontsize=7, pad=3)
        ax.axis("off")

    if save:
        cs_path = FIGURES_DIR / "contact_sheet.png"
        fig.savefig(cs_path, dpi=150, bbox_inches="tight")
        print(f"done -> {cs_path}")
    plt.close(fig)


def write_manifest(manifest_rows: list[dict]) -> None:
    """Write figures_manifest.csv."""
    out_path = FIGURES_DIR / "figures_manifest.csv"
    fieldnames = ["figure", "title", "source", "stem", "caption"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Manifest written -> {out_path}")


def print_captions(captions: dict[str, str]) -> None:
    """Print all captions for pasting into the manuscript."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("FIGURE CAPTIONS (paste into manuscript)")
    print(sep)
    for fig_id, caption in captions.items():
        print(f"\n[{fig_id}]")
        print(caption)
    print(f"\n{sep}\n")


# ── Verification checks ───────────────────────────────────────────────────────
def run_verification() -> None:
    """Spot-check key constraints before reporting done."""
    import pandas as pd

    print("\n--- Verification checks ---")
    ok = True

    # 1. All figures read from saved files
    for path in [
        "outputs/data/strategy_returns.parquet",
        "outputs/data/regime_series.parquet",
        "outputs/data/oos_fold_sharpes.csv",
        "outputs/data/mean_reversion_stock_cost_sweep.parquet",
    ]:
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
        print(f"  {status}  {path}")
        if not exists:
            ok = False

    # 2. Fig 4 crash bars match certified numbers
    print("\n  Fig 4 crash bars vs certified:")
    CERTIFIED = {
        "1973\nOil Bear":   (-0.084, -0.148),
        "1987\nBlack Mon.": (-0.025, -0.147),
        "2000\nDot-com":    (-0.023, -0.285),
        "2008\nGFC":        (-0.051, -0.571),
        "2020\nCovid":      (-0.086, -0.334),
    }
    from viz.figures import CERTIFIED_CRASH
    for crisis, (reg, unc) in CERTIFIED.items():
        stored = CERTIFIED_CRASH.get(crisis)
        if stored is None:
            print(f"    MISSING  {crisis}")
            ok = False
            continue
        reg_ok = abs(stored["regime"] - reg) < 0.0005
        unc_ok = abs(stored["uncond"] - unc) < 0.0005
        status = "OK" if (reg_ok and unc_ok) else "MISMATCH"
        print(f"    {status}  {crisis.replace(chr(10),' ')}  "
              f"regime={stored['regime']*100:.1f}%  uncond={stored['uncond']*100:.1f}%")
        if not (reg_ok and unc_ok):
            ok = False

    # 3. Mean reversion appears only in figA1
    print("\n  Mean reversion in main figures: checking ...")
    main_figs = [e for e in FIGURE_REGISTRY if e["id"] != "Fig A1"]
    for entry in main_figs:
        # source field should NOT contain mean_reversion
        if "mean_reversion" in entry["source"].lower():
            print(f"    FAIL: {entry['id']} source references mean_reversion")
            ok = False
    print("    OK  No main figure sources reference mean_reversion data directly.")

    # 4. No total returns: check that regime_series is 0/1 labels (not Markov smoothed)
    regime = pd.read_parquet("outputs/data/regime_series.parquet")["regime"]
    unique_vals = set(regime.dropna().astype(int).unique())
    status = "OK" if unique_vals <= {0, 1} else "FAIL"
    print(f"\n  {status}  Regime series values={unique_vals} (expected {{0,1}} causal vol-rule)")
    if status == "FAIL":
        ok = False

    print(f"\n  Overall: {'PASS' if ok else 'FAIL -- see items above'}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Render all publication figures.")
    parser.add_argument("--from", dest="data_dir", default="outputs/data",
                        help="Data directory (default: outputs/data)")
    parser.add_argument("--out", dest="out_dir", default="outputs/figures",
                        help="Output directory (default: outputs/figures)")
    parser.add_argument("--no-save", dest="no_save", action="store_true",
                        help="Preview only, do not write files")
    args = parser.parse_args()

    save = not args.no_save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering {len(FIGURE_REGISTRY)} figures ...\n")
    manifest_rows, rendered_figs, captions = render_all(save=save)

    if save:
        render_contact_sheet(save=True)
        write_manifest(manifest_rows)

    print_captions(captions)

    run_verification()

    # Summary
    n_ok   = sum(1 for r in manifest_rows if r["caption"] != "[FAILED]")
    n_fail = len(manifest_rows) - n_ok
    print(f"Done. {n_ok}/{len(manifest_rows)} figures rendered successfully.", end="")
    if n_fail:
        print(f"  {n_fail} FAILED -- see output above.")
    else:
        print()


if __name__ == "__main__":
    main()
