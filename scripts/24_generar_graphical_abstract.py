"""Generate the graphical abstract from a clean, metric-aware brief.

The layout is deliberately simple: separate the targets, show the gated
decoupled pipeline, and report only the current blind-test benchmark. It does
not reuse the legacy architecture graphic or its obsolete percentages.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from src.constantes import DIR_FIGURAS


INK = "#1f2d3d"
MUTED = "#5b6b7c"
TEAL = "#2a9d8f"
ORANGE = "#e9a23b"
BLUE = "#2c7fb8"
PURPLE = "#6a4c93"
RED = "#c44536"
PANEL = "#f7f9fb"
LINE = "#9aa7b4"
WHITE = "#ffffff"


def box(ax, x, y, w, h, fc, ec="#d6dee6", lw=1.0, r=0.7, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.25,rounding_size={r}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
        )
    )


def label(ax, x, y, text, size=8.0, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=5, **kw)


def arrow(ax, x0, y0, x1, y1, color=LINE):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
            linewidth=1.2, color=color, shrinkA=0, shrinkB=0, zorder=4,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    label(ax, 1.5, 38.2,
          "Decoupled ignition and propagation benchmarking for operational wildfire nowcasting",
          size=10.8, weight="bold")
    label(ax, 1.5, 35.9,
          "A reproducible computational framework — not a new boosting algorithm",
          size=7.2, color=MUTED)

    # Panels
    for x, w, color, title in [
        (1.0, 29.5, TEAL, "1 | SEPARATE THE TARGETS"),
        (32.5, 32.5, PURPLE, "2 | GATE AND DECOUPLE"),
        (67.0, 32.0, RED, "3 | BLIND BENCHMARK"),
    ]:
        box(ax, x, 4.5, w, 29.5, PANEL, "#dbe3ea", 0.9)
        box(ax, x, 29.7, w, 4.3, color, color, 0.0, 0.5)
        label(ax, x + 1.3, 31.9, title, size=7.4, color=WHITE, weight="bold")

    # Panel 1: problem and D2
    label(ax, 2.5, 27.6, "Primary ignition", size=7.6, color=TEAL, weight="bold")
    label(ax, 2.5, 25.1, "fire at (i,t) without t-1\nfire within 25 km", size=6.8, color=INK)
    label(ax, 2.5, 20.8, "Tactical propagation", size=7.6, color=ORANGE, weight="bold")
    label(ax, 2.5, 18.3, "fire at (i,t) within\n25 km of a t-1 fire", size=6.8, color=INK)
    box(ax, 2.5, 12.4, 26.0, 3.3, "#eef6f4", TEAL, 0.8, 0.4)
    label(ax, 3.2, 14.05, "No persistence leakage", size=6.6, color=TEAL, weight="bold")
    label(ax, 2.5, 8.7,
          "Same fire, two tasks.\nD2 keeps the labels disjoint.",
          size=6.4, color=MUTED, linespacing=1.2)

    # Panel 2: pipeline
    label(ax, 34.0, 26.9, "Inputs", size=6.4, color=MUTED)
    box(ax, 34.0, 22.8, 29.5, 3.3, "#eef3f8", BLUE, 0.8, 0.4)
    label(ax, 34.8, 24.45, "VIIRS  ·  ERA5-Land  ·  MapBiomas  ·  history",
          size=6.2, color=INK)

    arrow(ax, 48.7, 22.5, 48.7, 20.3, BLUE)
    box(ax, 34.0, 17.3, 29.5, 3.0, "#fdf3e3", ORANGE, 0.8, 0.4)
    label(ax, 34.8, 18.8, "Biophysical gate", size=6.3, color=ORANGE, weight="bold")

    arrow(ax, 48.7, 17.0, 48.7, 14.8, ORANGE)
    box(ax, 34.0, 11.7, 14.0, 3.1, "#f0ecf7", PURPLE, 0.8, 0.4)
    label(ax, 34.7, 13.25, "M3 hybrid\n(no t-1 fire)", size=5.9, color=INK)
    box(ax, 49.5, 11.7, 14.0, 3.1, "#f0ecf7", PURPLE, 0.8, 0.4)
    label(ax, 50.2, 13.25, "M1 fallback\n(no history)", size=5.9, color=INK)
    arrow(ax, 48.7, 11.4, 48.7, 9.2, PURPLE)
    label(ax, 34.0, 8.0,
          "Causal t-1 propagation\n+ frozen evaluation harness",
          size=6.1, color=MUTED, linespacing=1.2)

    # Panel 3: benchmark bars
    label(ax, 68.5, 27.2, "ROC-AUC | Chaco | retrospective 2024–2025",
          size=6.2, color=MUTED)
    bars = [
        ("M3 hybrid", 0.802, PURPLE),
        ("Full FWI", 0.517, BLUE),
        ("INPE", 0.539, ORANGE),
    ]
    y = 23.0
    x0 = 75.0
    scale = 24.0 / 0.85
    for name, value, color in bars:
        ax.add_patch(Rectangle((x0, y - 0.9), value * scale, 1.8,
                               facecolor=color, edgecolor="none", zorder=3))
        label(ax, 68.5, y, name, size=6.3, color=INK)
        label(ax, x0 + value * scale + 0.6, y, f"{value:.3f}",
              size=6.1, color=INK, weight="bold")
        y -= 5.1
    ax.plot([x0, x0], [13.0, 25.1], color="#d6dee6", lw=0.8, zorder=2)
    label(ax, 68.5, 11.3,
          "M3 is stronger on the blind test;\nM1 is the fallback for unseen cells.",
          size=6.1, color=MUTED, linespacing=1.2)

    # Footer
    box(ax, 1.0, 0.8, 98.0, 2.4, INK, INK, 0.0, 0.4)
    label(ax, 50, 2.0,
          "D2 targets  ·  gated M3/M1  ·  causal propagation  ·  open benchmark",
          size=6.6, color=WHITE, weight="bold", ha="center")

    plt.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    png = DIR_FIGURAS / "graphical_abstract.png"
    pdf = DIR_FIGURAS / "graphical_abstract.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
