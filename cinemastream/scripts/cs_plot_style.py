"""
cs_plot_style.py — CinemaStream matplotlib style standard.

Import this at the top of EVERY script that generates a plot for the book.
Usage:
    from cinemastream.scripts.cs_plot_style import apply_cs_style, CS
    apply_cs_style()

All plots using this style will have:
- Consistent CinemaStream brand colours
- 150 DPI output (print-quality)
- Minimum 900px wide figures
- Clean, readable fonts
- No chart junk
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Brand Palette ──────────────────────────────────────────────────────────────
class CS:
    """CinemaStream colour constants. Use these in every plot."""
    CREAM    = "#F5EFE6"   # background
    CHARCOAL = "#2D2D2D"   # primary text, axes
    TEAL     = "#3D7370"   # primary data series, positive
    ORANGE   = "#E07A3B"   # highlight, second series, alerts
    TEAL2    = "#6BA8A5"   # third series (lighter teal)
    ORANGE2  = "#F0AA7A"   # fourth series (lighter orange)
    GREY     = "#9E9E9E"   # gridlines, secondary text

    # Ordered palette for multi-series charts
    PALETTE  = [TEAL, ORANGE, TEAL2, ORANGE2, CHARCOAL, GREY]


def apply_cs_style():
    """
    Apply CinemaStream house style to all subsequent matplotlib figures.
    Call once at the top of any script that generates plots for the book.
    """
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":     CS.CREAM,
        "figure.figsize":       (10, 5.6),   # 16:9 aspect, ≥900px at 150dpi
        "figure.dpi":           150,          # print quality — do NOT reduce
        "savefig.dpi":          150,
        "savefig.facecolor":    CS.CREAM,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.15,

        # Axes
        "axes.facecolor":       CS.CREAM,
        "axes.edgecolor":       CS.CHARCOAL,
        "axes.labelcolor":      CS.CHARCOAL,
        "axes.titlecolor":      CS.CHARCOAL,
        "axes.titlesize":       14,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "axes.linewidth":       1.2,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.prop_cycle":      mpl.cycler(color=CS.PALETTE),

        # Grid
        "grid.color":           CS.GREY,
        "grid.alpha":           0.35,
        "grid.linewidth":       0.8,

        # Ticks
        "xtick.color":          CS.CHARCOAL,
        "ytick.color":          CS.CHARCOAL,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "xtick.direction":      "out",
        "ytick.direction":      "out",

        # Legend
        "legend.frameon":       False,
        "legend.fontsize":      9,
        "legend.labelcolor":    CS.CHARCOAL,

        # Font — use a clean sans-serif available on all platforms
        "font.family":          "sans-serif",
        "font.sans-serif":      ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "text.color":           CS.CHARCOAL,

        # Lines
        "lines.linewidth":      2.0,
        "lines.markersize":     6,

        # Bars
        "patch.edgecolor":      "none",
    })


def save_figure(fig, path: str, title: str = None):
    """
    Save a figure with CinemaStream standards.
    Validates: minimum size 5KB, correct DPI, correct dimensions.

    Args:
        fig: matplotlib Figure object
        path: output path (e.g. 'cinemastream/images/020_dataviz/watch_bar.png')
        title: optional figure title (added to top if provided)
    """
    import os
    from pathlib import Path

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold",
                     color=CS.CHARCOAL, y=1.01)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=CS.CREAM, edgecolor="none")

    size_kb = out.stat().st_size / 1024
    width_px, height_px = fig.get_size_inches() * fig.dpi

    assert size_kb >= 5, f"Figure too small ({size_kb:.1f}KB) — likely blank"
    assert width_px >= 900, f"Figure too narrow ({width_px:.0f}px) — increase figsize"
    assert height_px >= 450, f"Figure too short ({height_px:.0f}px) — increase figsize"

    print(f"✅ Saved: {out} ({size_kb:.0f}KB, {width_px:.0f}×{height_px:.0f}px)")
    return str(out)


# ── Quick demo (run this file directly to verify style) ───────────────────────
if __name__ == "__main__":
    import numpy as np

    apply_cs_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax = axes[0]
    countries = ["SG", "MY", "ID", "PH", "TH"]
    revenue   = [42, 28, 19, 15, 8]
    bars = ax.bar(countries, revenue, color=CS.TEAL)
    bars[0].set_color(CS.ORANGE)  # highlight top performer
    ax.set_title("Monthly Revenue by Country (S$K)")
    ax.set_xlabel("Country")
    ax.set_ylabel("Revenue (S$K)")

    # Line chart
    ax2 = axes[1]
    days = np.arange(1, 31)
    churn = 3.2 + 0.05 * days + np.random.normal(0, 0.2, 30)
    ax2.plot(days, churn, color=CS.TEAL, label="Churn rate (%)")
    ax2.axhline(4.5, color=CS.ORANGE, linestyle="--", linewidth=1.5, label="Alert threshold")
    ax2.set_title("Daily Churn Rate — May 2024")
    ax2.set_xlabel("Day of month")
    ax2.set_ylabel("Churn rate (%)")
    ax2.legend()

    save_figure(fig, "/tmp/cs_style_demo.png", "CinemaStream Plot Style Demo")
    plt.show()
    print("Style demo complete. Check /tmp/cs_style_demo.png")
