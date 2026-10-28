"""Generate Figure 1 (conceptual framework) at final C&G print size."""

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


def panel(ax, x, y, w, h, title, color, header=4.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.7",
        linewidth=0.7, edgecolor="#d6dee6", facecolor=COLORS["panel"], zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (x, y + h - header), w, header, boxstyle="round,pad=0.22,rounding_size=0.7",
        linewidth=0, facecolor=color, zorder=2,
    ))
    ax.text(x + w / 2, y + h - header / 2, title, ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=COLORS["white"], zorder=3)


def bullet(ax, x, y, text, color, size=6.2, weight="normal"):
    ax.text(x, y, "•", ha="left", va="top", fontsize=size + 1.0, color=color,
            fontweight="bold", zorder=3)
    ax.text(x + 1.1, y, text, ha="left", va="top", fontsize=size,
            color=COLORS["ink"], fontweight=weight, linespacing=1.22, zorder=3)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6,
        linewidth=0.8, color=COLORS["line"], shrinkA=0, shrinkB=0, zorder=4,
    ))


def main():
    fig, ax = plt.subplots(figsize=(5.83, 3.21), dpi=600)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["white"])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    y, h = 18.5, 39.5
    columns = [
        (1.0, 18.0, "INPUTS", COLORS["data"]),
        (20.2, 22.0, "D2 TARGET", COLORS["target"]),
        (43.4, 18.0, "GATE", COLORS["gate"]),
        (62.6, 36.4, "INFERENCE", COLORS["model"]),
    ]
    for x, w, title, color in columns:
        panel(ax, x, y, w, h, title, color)

    for x0, x1 in [(19.1, 20.1), (42.3, 43.3), (61.5, 62.5)]:
        arrow(ax, x0, x1, 35.0)

    x = 2.3
    bullet(ax, x, 50.6, "VIIRS 375 m", COLORS["data"], weight="bold")
    bullet(ax, x, 43.8, "ERA5-Land", COLORS["data"], weight="bold")
    bullet(ax, x, 37.0, "MapBiomas C2", COLORS["data"], weight="bold")
    bullet(ax, x, 30.2, "Fire history", COLORS["data"], weight="bold")

    x = 21.5
    ax.text(x, 50.3, "Primary ignition", fontsize=6.4, fontweight="bold", color=COLORS["target"])
    bullet(ax, x, 46.5, "no fire ≤25 km at t−1", COLORS["target"])
    bullet(ax, x, 41.0, "no same-cell fire 10 d", COLORS["target"])
    ax.text(x, 34.0, "Propagation", fontsize=6.4, fontweight="bold", color=COLORS["gate"])
    bullet(ax, x, 30.2, "fire ≤25 km at t−1", COLORS["gate"])
    ax.text(x, 23.0, "domains disjoint", fontsize=6.0, style="italic", color=COLORS["muted"])

    x = 44.7
    bullet(ax, x, 50.6, "Fuel > 0", COLORS["gate"], weight="bold")
    bullet(ax, x, 43.8, "VPD ≥ 0.8 kPa", COLORS["gate"], weight="bold")
    bullet(ax, x, 37.0, "Rain < 10 mm", COLORS["gate"], weight="bold")
    ax.text(x, 28.5, "removes non-\ncombustible and\nwet cells", fontsize=6.0,
            color=COLORS["muted"], linespacing=1.22)

    x = 64.0
    ax.text(x, 50.3, "M3 hybrid", fontsize=6.4, fontweight="bold", color=COLORS["model"])
    ax.text(x, 46.8, "weather + fuel + ecotone\nno t−1 fire input", fontsize=6.0,
            color=COLORS["ink"], linespacing=1.22)
    ax.text(x, 39.0, "M2 fallback", fontsize=6.4, fontweight="bold", color=COLORS["model"])
    ax.text(x, 35.5, "weather + fuel + ecotone\n+ historical anchor", fontsize=6.0,
            color=COLORS["ink"], linespacing=1.22)
    ax.text(x, 27.5, "Propagation", fontsize=6.4, fontweight="bold", color=COLORS["gate"])
    ax.text(x, 24.0, "causal t−1 kernel", fontsize=6.0, color=COLORS["ink"])

    # Evaluation + outputs, full width
    x0, w0, y0, h0 = 1.0, 98.0, 2.0, 13.0
    panel(ax, x0, y0, w0, h0, "EVALUATION + OUTPUTS", COLORS["eval"], header=3.7)
    yb = 7.0
    for xx, txt in [
        (2.5, "Temporal 2024–2025"),
        (22.5, "50-km blocks"),
        (39.5, "Grouped bootstrap"),
        (58.5, "Cost-loss"),
        (71.5, "Open code/data"),
    ]:
        bullet(ax, xx, yb, txt, COLORS["eval"], size=6.2)
    ax.text(50.5, 3.4, "Auditable target operator  ·  gated inference  ·  open evaluation harness",
            ha="center", va="center", fontsize=5.9, style="italic", color=COLORS["muted"])

    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    png = DIR_FIGURAS / "fig01_framework_conceptual.png"
    pdf = DIR_FIGURAS / "fig01_framework_conceptual.pdf"
    fig.savefig(png, dpi=600, facecolor=fig.get_facecolor(), pad_inches=0)
    fig.savefig(pdf, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
