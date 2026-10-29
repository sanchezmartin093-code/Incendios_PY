"""Generate the Elsevier/Computers & Geosciences graphical abstract."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

from src.constantes import DIR_FIGURAS

W, H = 1328, 531
INK = "#111827"
MUTED = "#4B5563"
FAINT = "#6B7280"
RULE = "#D1D5DB"
WHITE = "#FFFFFF"
BLUE = "#0072B2"
BLUE_T = "#EAF3F8"
ORANGE = "#D55E00"
ORANGE_T = "#FBEFE8"
GREY_T = "#F6F7F8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


def text(ax, x, y, s, size=10, weight="normal", color=INK, ha="left", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, style=style, zorder=5)


def rule(ax, x0, x1, y, color=RULE, lw=0.7):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=1)


def box(ax, x, y, w, h, edge, face, lw=0.8):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(ax, points, color=MUTED, lw=1.0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops=dict(arrowstyle="-|>", mutation_scale=9, lw=lw, color=color, shrinkA=0, shrinkB=0),
        zorder=4,
    )


def flame(ax, cx, cy, scale=1.0):
    pts = [
        (cx - 12 * scale, cy - 32 * scale),
        (cx - 25 * scale, cy - 2 * scale),
        (cx - 18 * scale, cy + 24 * scale),
        (cx - 4 * scale, cy + 40 * scale),
        (cx + 2 * scale, cy + 22 * scale),
        (cx + 8 * scale, cy + 32 * scale),
        (cx + 16 * scale, cy + 8 * scale),
        (cx + 12 * scale, cy - 12 * scale),
        (cx + 22 * scale, cy - 2 * scale),
        (cx + 13 * scale, cy - 30 * scale),
    ]
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=ORANGE_T, edgecolor=ORANGE, linewidth=1.1, zorder=3)
    )
    ax.add_patch(
        Polygon(
            [
                (cx - 4 * scale, cy - 28 * scale),
                (cx - 10 * scale, cy - 4 * scale),
                (cx - 4 * scale, cy + 14 * scale),
                (cx + 4 * scale, cy + 2 * scale),
                (cx + 10 * scale, cy + 10 * scale),
                (cx + 8 * scale, cy - 14 * scale),
                (cx + 2 * scale, cy - 28 * scale),
            ],
            closed=True,
            facecolor=ORANGE,
            edgecolor="none",
            zorder=4,
        )
    )


def main() -> None:
    fig = plt.figure(figsize=(W / 300, H / 300), dpi=300, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    text(ax, 40, 482, "Temporal skill is not spatial skill", size=10.2, weight="bold")
    rule(ax, 40, 1294, 460, lw=0.55)
    text(ax, 40, 430, "D2 split", size=6.8, weight="bold")
    text(ax, 40, 398, "I  ignition", size=6.5, weight="bold", color=BLUE)
    text(ax, 40, 372, "no fire ≤25 km t−1", size=6.3, color=MUTED)
    text(ax, 40, 342, "no same-cell 10 d", size=6.3, color=MUTED)
    text(ax, 40, 285, "P  propagation", size=6.5, weight="bold", color=ORANGE)
    text(ax, 40, 259, "fire ≤25 km t−1", size=6.3, color=MUTED)

    cx, cy, r = 250.0, 170.0, 52.0
    ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=ORANGE, lw=0.7, linestyle=(0, (2.0, 1.8)), zorder=3))
    ax.add_patch(Rectangle((cx - 4, cy - 4), 8, 8, facecolor=ORANGE, edgecolor="none", zorder=4))
    for dx, dy in [(-28, 20), (26, 20), (-20, -30), (28, -24)]:
        ax.add_patch(Rectangle((cx + dx - 3, cy + dy - 3), 6, 6, facecolor=ORANGE_T, edgecolor=ORANGE, lw=0.4, zorder=3))
    for dx, dy in [(-62, 38), (60, 38), (-62, -42), (60, -42)]:
        ax.add_patch(Rectangle((cx + dx - 3, cy + dy - 3), 6, 6, facecolor=BLUE_T, edgecolor=BLUE, lw=0.4, zorder=3))
    text(ax, cx, cy + 8, "t−1", size=5.6, color=ORANGE, ha="center", weight="bold")
    text(ax, cx, 92, "25 km · 10 d", size=6.3, color=MUTED, ha="center")

    x_axis0, x_axis1 = 720.0, 1260.0
    axis_y = 105.0
    ax.plot([x_axis0, x_axis1], [axis_y, axis_y], color=RULE, lw=0.6, zorder=2)
    for tick in (0.5, 0.7, 0.9):
        x = x_axis0 + (tick - 0.4) / 0.55 * (x_axis1 - x_axis0)
        ax.plot([x, x], [axis_y, axis_y + 5], color=MUTED, lw=0.5, zorder=2)
        text(ax, x, 88, f"{tick:.1f}", size=5.9, color=FAINT, ha="center")

    x_chance = x_axis0 + (0.5 - 0.4) / 0.55 * (x_axis1 - x_axis0)
    ax.plot([x_chance, x_chance], [axis_y, 420], color="#9CA3AF", lw=0.6, linestyle=(0, (2.2, 2.0)), zorder=1)
    text(ax, x_chance, 430, "chance", size=5.8, color=FAINT, ha="center")

    def xpos(value: float) -> float:
        return x_axis0 + (value - 0.4) / 0.55 * (x_axis1 - x_axis0)

    def pair(y: float, a: float, b: float, color: str) -> None:
        xa, xb = xpos(a), xpos(b)
        ax.plot([xa, xb], [y, y], color=color, lw=1.1, zorder=3)
        ax.plot(xa, y, "o", ms=3.2, color=color, zorder=4)
        ax.plot(xb, y, "o", ms=3.2, color=color, zorder=4)

    text(ax, 400, 420, "Temporal · 2024–25", size=6.8, weight="bold")
    text(ax, 460, 388, "M3", size=6.1, color=BLUE)
    text(ax, 460, 358, "FWI", size=6.1, color=FAINT)
    text(ax, 460, 328, "INPE", size=6.1, color=FAINT)
    pair(388, 0.7534, 0.8021, BLUE)
    pair(358, 0.5059, 0.5611, FAINT)
    pair(328, 0.4973, 0.5266, FAINT)

    text(ax, 400, 285, "Spatial · 2023", size=6.8, weight="bold")
    text(ax, 460, 253, "M2", size=6.1, color=FAINT)
    text(ax, 460, 223, "M3", size=6.1, color=BLUE)
    pair(253, 0.6773, 0.7470, FAINT)
    pair(223, 0.6496, 0.7277, BLUE)

    text(ax, 400, 180, "Propagation · 2024–25", size=6.8, weight="bold")
    pair(150, 0.7859, 0.9039, ORANGE)

    text(ax, 720, 55, "ROC-AUC · Chaco/Oriental", size=6.0, color=MUTED)
    text(ax, 40, 48, "Ignition: screening only · 0.08–0.20% · lift 5×", size=6.1, color=FAINT)
    text(ax, 1294, 430, "Propagation: V=0.63/0.40 · α=0.002", size=5.8, color=ORANGE, ha="right")
    text(ax, 40, 20, "Gran Chaco · 1 km² · 2018–2025 · open benchmark", size=6.2, color=FAINT)

    out = DIR_FIGURAS / "graphical_abstract_v2"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", facecolor=WHITE, edgecolor="none", bbox_inches=None)
    fig.savefig(f"{out}.svg", facecolor=WHITE, edgecolor="none", bbox_inches=None)
    fig.savefig(f"{out}.png", dpi=600, facecolor=WHITE, edgecolor="none", bbox_inches=None)
    plt.close(fig)
    with Image.open(f"{out}.png") as im:
        im.convert("RGB").save(f"{out}.tiff", dpi=(600, 600), compression="tiff_lzw")
    print(f"Wrote {out}.pdf/.svg/.png/.tiff")


if __name__ == "__main__":
    main()
