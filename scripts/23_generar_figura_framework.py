"""Generate the conceptual Figure 1 for the manuscript.

The figure is intentionally metric-free: it shows the D2 target operator, the
biophysical gate, the decoupled M3/M1 inference and the evaluation harness.
No performance KPI, legacy percentage or project branding is used.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.constantes import DIR_FIGURAS


COLORS = {
    "ink": "#1f2d3d",
    "muted": "#5b6b7c",
    "data": "#2c7fb8",
    "target": "#2a9d8f",
    "gate": "#e9a23b",
    "model": "#6a4c93",
    "eval": "#c44536",
    "line": "#9aa7b4",
    "panel": "#f7f9fb",
    "white": "#ffffff",
}


def panel(ax, x, y, w, h, title, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            linewidth=1.0,
            edgecolor="#d6dee6",
            facecolor=COLORS["panel"],
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 5.0), w, 5.0,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            linewidth=0,
            facecolor=color,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h - 2.5, title,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=COLORS["white"],
        zorder=3,
    )


def bullet(ax, x, y, text, color, size=8.6, weight="normal"):
    ax.text(
        x, y, "•", ha="left", va="top",
        fontsize=size + 1.0, color=color,
        fontweight="bold", zorder=3,
    )
    ax.text(
        x + 1.4, y, text, ha="left", va="top",
        fontsize=size, color=COLORS["ink"],
        fontweight=weight, linespacing=1.22, zorder=3,
    )


def arrow(ax, x0, x1, y):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=COLORS["line"],
            shrinkA=0, shrinkB=0, zorder=4,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 8.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["white"])

    ax.text(
        2, 59.8,
        "A reproducible computational framework for decoupled wildfire ignition and propagation benchmarking",
        ha="left", va="top", fontsize=15.5, fontweight="bold", color=COLORS["ink"],
    )
    ax.text(
        2, 56.3,
        "D2 targets  |  gated M3/M1 inference  |  open benchmark against Canadian FWI and INPE",
        ha="left", va="top", fontsize=9.4, color=COLORS["muted"],
    )

    y, h = 8.0, 43.5
    columns = [
        (1.5, 16.5, "INPUTS", COLORS["data"]),
        (20.0, 19.0, "D2 TARGET OPERATOR", COLORS["target"]),
        (41.0, 14.0, "BIOPHYSICAL GATE", COLORS["gate"]),
        (57.0, 23.5, "DECOUPLED INFERENCE", COLORS["model"]),
        (82.5, 16.0, "EVALUATION + OUTPUTS", COLORS["eval"]),
    ]
    for x, w, title, color in columns:
        panel(ax, x, y, w, h, title, color)

    for x0, x1 in [(18.2, 19.7), (39.2, 40.7), (55.2, 56.7), (80.7, 82.2)]:
        arrow(ax, x0, x1, 29.5)

    # 1. Inputs
    x = 3.0
    bullet(ax, x, 44.2, "VIIRS active fires", COLORS["data"], weight="bold")
    bullet(ax, x, 39.2, "Suomi-NPP, NOAA-20/21\n375-m detections", COLORS["muted"], size=8.0)
    bullet(ax, x, 31.0, "ERA5-Land weather", COLORS["data"], weight="bold")
    bullet(ax, x, 26.0, "VPD, rain, wind, FFMC/ISI", COLORS["muted"], size=8.0)
    bullet(ax, x, 19.8, "MapBiomas C2 fuel", COLORS["data"], weight="bold")
    bullet(ax, x, 14.8, "Pasture, savanna, forest", COLORS["muted"], size=8.0)
    bullet(ax, x, 10.8, "Causal fire history", COLORS["data"], weight="bold")

    # 2. D2 targets
    x = 21.6
    ax.text(x, 43.9, "Primary ignition", fontsize=9.2, fontweight="bold", color=COLORS["target"])
    bullet(ax, x, 40.2, "fire at cell i, day t", COLORS["target"], size=8.3)
    bullet(ax, x, 36.1, "no t-1 fire within 25 km", COLORS["target"], size=8.3)
    bullet(ax, x, 32.0, "no same-cell fire in 10 d", COLORS["target"], size=8.3)
    ax.text(x, 26.6, "Tactical propagation", fontsize=9.2, fontweight="bold", color=COLORS["gate"])
    bullet(ax, x, 23.0, "fire at cell i, day t", COLORS["gate"], size=8.3)
    bullet(ax, x, 18.9, "within 25 km of a t-1 fire", COLORS["gate"], size=8.3)
    ax.text(
        x, 13.4, "The two target domains are disjoint.",
        fontsize=8.0, style="italic", color=COLORS["muted"],
    )
    ax.text(
        x, 10.3, "D2 is applied once and reused\nfor training, test and alerts.",
        fontsize=8.0, color=COLORS["muted"], linespacing=1.2,
    )

    # 3. Gate
    x = 42.5
    ax.text(x, 43.9, "Applied before inference", fontsize=8.2, style="italic", color=COLORS["muted"])
    bullet(ax, x, 39.6, "fuel > 0", COLORS["gate"], size=8.5, weight="bold")
    bullet(ax, x, 34.0, "VPD ≥ 0.8 kPa", COLORS["gate"], size=8.5, weight="bold")
    bullet(ax, x, 28.4, "48-h rain < 10 mm", COLORS["gate"], size=8.5, weight="bold")
    ax.text(
        x, 21.0,
        "Removes water, urban\nsurfaces and saturated\nfuels from inference.",
        fontsize=8.0, color=COLORS["muted"], linespacing=1.25,
    )
    ax.text(
        x, 12.8,
        "The gate is not a\nlabel filter.",
        fontsize=8.0, style="italic", color=COLORS["muted"], linespacing=1.2,
    )

    # 4. Models
    x = 58.6
    ax.text(x, 43.9, "M3  hybrid ignition", fontsize=9.4, fontweight="bold", color=COLORS["model"])
    ax.text(
        x, 40.2,
        "weather + fuel + ecotone\n+ anthropogenic priors\nno t-1 fire input",
        fontsize=8.1, color=COLORS["ink"], linespacing=1.2,
    )
    ax.text(x, 31.0, "M1  fallback", fontsize=9.4, fontweight="bold", color=COLORS["model"])
    ax.text(
        x, 27.3,
        "weather + fuel only\nused where fire history\nis absent",
        fontsize=8.1, color=COLORS["ink"], linespacing=1.2,
    )
    ax.text(x, 18.6, "Propagation", fontsize=9.4, fontweight="bold", color=COLORS["gate"])
    ax.text(
        x, 15.0,
        "causal t-1 kernel:\ndistance, counts and FRP",
        fontsize=8.1, color=COLORS["ink"], linespacing=1.2,
    )
    ax.text(
        x, 9.4,
        "Decoupled, not persistence.",
        fontsize=8.0, style="italic", color=COLORS["muted"],
    )

    # 5. Evaluation and outputs
    x = 84.0
    bullet(ax, x, 43.7, "Temporal test 2024–2025", COLORS["eval"], size=8.2)
    bullet(ax, x, 39.0, "50-km spatial blocks", COLORS["eval"], size=8.2)
    bullet(ax, x, 34.3, "Grouped bootstrap", COLORS["eval"], size=8.2)
    bullet(ax, x, 29.6, "Lead-time analysis", COLORS["eval"], size=8.2)
    bullet(ax, x, 24.9, "Cost-loss and conformal", COLORS["eval"], size=8.2)
    ax.text(
        x, 18.3,
        "Alert maps\nBenchmark tables\nOpen code and data",
        fontsize=8.4, fontweight="bold", color=COLORS["ink"], linespacing=1.35,
    )
    ax.text(
        x, 10.6,
        "Every result is tied\nto a frozen split.",
        fontsize=8.0, style="italic", color=COLORS["muted"], linespacing=1.2,
    )

    # Bottom banner
    ax.add_patch(
        FancyBboxPatch(
            (1.5, 1.3), 97.0, 4.4,
            boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=0, facecolor=COLORS["ink"], zorder=1,
        )
    )
    ax.text(
        50, 3.5,
        "Contribution: an auditable target operator, gated inference and evaluation harness — not a new boosting algorithm.",
        ha="center", va="center", fontsize=9.2, color=COLORS["white"],
        fontweight="bold", zorder=3,
    )

    plt.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    png = DIR_FIGURAS / "fig01_framework_conceptual.png"
    pdf = DIR_FIGURAS / "fig01_framework_conceptual.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
