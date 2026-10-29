"""Generate the Figure 1 conceptual framework at final print size."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from scripts.fig01_contract import FACTS, validate_contract
from src.constantes import DIR_FIGURAS

MM = 1.0
W, H = 183.0, 135.0
M, G = 4.0, 5.0
PW = (W - 2 * M - G) / 2.0
PH = 61.0

INK = "#111827"
MUTED = "#4B5563"
FAINT = "#6B7280"
RULE = "#C7CDD4"
GRID = "#E5E7EB"
WHITE = "#FFFFFF"

BLUE = "#0072B2"
BLUE_T = "#EAF3F8"
ORANGE = "#D55E00"
ORANGE_T = "#FBEFE8"
AMBER = "#B7791F"
AMBER_T = "#FDF8EC"
GREEN = "#2F6F5E"
GREEN_T = "#EEF5F2"
GREY_T = "#F6F7F8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


def text(ax, x, y, s, size=7.0, weight="normal", color=INK, ha="left", va="center", style="normal"):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        style=style,
        zorder=6,
    )


def box(ax, x, y, w, h, edge=RULE, face=WHITE, lw=0.7):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
    )


def arrow(ax, points, color=MUTED, lw=0.75):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw, solid_capstyle="round", zorder=4)
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=6.0,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
    )


def contract(ax, x, y, s, size=6.1, color=FAINT):
    text(
        ax,
        x,
        y,
        s,
        size=size,
        color=color,
        ha="center",
        style="italic",
    )


def header(ax, x, y, letter, title):
    ax.plot([x + 3, x + 3 + PW - 6], [y + PH - 6.0, y + PH - 6.0], color=RULE, lw=0.55, zorder=1)
    text(ax, x + 3, y + PH - 3.2, letter, size=8.0, weight="bold", va="center")
    text(ax, x + 9.5, y + PH - 3.2, title, size=8.0, weight="bold", va="center")


def panel_a(ax, x, y):
    header(ax, x, y, "a", "Ingestion and biophysical gate")

    sources = [
        ("VIIRS 375 m", "Suomi-NPP · NOAA-20/21"),
        ("ERA5-Land", "3-hourly → daily · 0.25°"),
        ("MapBiomas C2", "annual · prior-year join"),
        ("Hansen GFC", "tree-cover loss"),
        ("Historical priors", "causal · 3-year window"),
    ]
    sx = x + 3.0
    sy = y + 46.5
    for name, detail in sources:
        text(ax, sx, sy, name, size=7.0, weight="bold")
        text(ax, sx, sy - 3.1, detail, size=6.2, color=MUTED)
        sy -= 7.4

    gx, gy, gw, gh = x + 50.5, y + 15.0, 31.0, 32.0
    box(ax, gx, gy, gw, gh, edge=AMBER, face=AMBER_T, lw=0.7)
    text(ax, gx + gw / 2, gy + gh - 4.5, "Biophysical gate", size=7.0, weight="bold", ha="center")
    text(ax, gx + gw / 2, gy + gh - 10.0, "Fuel = 1", size=6.8, ha="center")
    text(ax, gx + gw / 2, gy + gh - 15.0, "VPD ≥ 0.8 kPa", size=6.8, ha="center")
    text(ax, gx + gw / 2, gy + gh - 20.0, r"$R_{48h}<10\ \mathrm{mm}$", size=6.8, ha="center")
    text(
        ax,
        gx + gw / 2,
        gy + 2.5,
        "excludes non-combustible\nand wet cell-days",
        size=6.2,
        color=MUTED,
        ha="center",
        va="bottom",
    )

    arrow(ax, [(x + 43.5, y + 34.5), (gx - 0.4, y + 34.5)], color=MUTED)
    contract(ax, x + 44.5, y + 38.0, "daily 1-km²\nstack", size=5.8)

    arrow(ax, [(gx + gw / 2, gy - 0.4), (gx + gw / 2, y + 10.8)], color=MUTED)
    text(ax, gx + gw / 2, y + 7.2, "Gated candidate\ncell-days", size=6.5, weight="bold", ha="center")
    text(ax, gx + gw / 2, y + 2.5, f"{FACTS['gate_removed_cell_days_pct']}% cell-days removed", size=6.2, color=FAINT, ha="center")


def panel_b(ax, x, y):
    header(ax, x, y, "b", "D2 target operator")

    bx = x + 4.0
    text(ax, bx, y + 47.5, r"$F_t=\{i:\ \mathrm{fire}(i,t)=1\}$", size=7.0)
    text(ax, bx, y + 41.7, r"$I_{i,t}=1\ \Leftrightarrow\ i\in F_t,$", size=7.0)
    text(ax, bx, y + 37.1, r"$N_{25}(i,t{-}1)=\emptyset,\ H_{10}(i,t)=0$", size=7.0)
    text(ax, bx, y + 30.0, r"$P_{i,t}=1\ \Leftrightarrow\ i\in F_t,$", size=7.0)
    text(ax, bx, y + 25.4, r"$N_{25}(i,t{-}1)\neq\emptyset$", size=7.0)
    text(ax, x + 70.0, y + 41.7, r"$I\cap P=\emptyset$", size=7.0, ha="right")
    text(ax, x + 70.0, y + 28.0, r"$25\ \mathrm{km}\ \cdot\ 10\ \mathrm{d}$", size=7.0, ha="right")

    bw = 36.0
    box(ax, bx, y + 8.0, bw, 8.0, edge=BLUE, face=BLUE_T)
    text(ax, bx + 2.0, y + 12.0, "I  primary ignition", size=6.4, weight="bold")
    box(ax, bx + 38.0, y + 8.0, 36.0, 8.0, edge=ORANGE, face=ORANGE_T)
    text(ax, bx + 40.0, y + 12.0, "P  tactical propagation", size=6.4, weight="bold")
    text(ax, bx + 37.0, y + 5.0, "domains disjoint", size=6.1, color=FAINT, ha="center", style="italic")


def panel_c(ax, x, y):
    header(ax, x, y, "c", "Decoupled inference")

    cx, cy = x + 16.0, y + 30.0
    diamond = Polygon(
        [(x + 6.0, cy), (cx, cy + 7.5), (x + 26.0, cy), (cx, cy - 7.5)],
        closed=True,
        facecolor=WHITE,
        edgecolor=MUTED,
        linewidth=0.7,
        zorder=3,
    )
    ax.add_patch(diamond)
    text(ax, cx, cy + 1.6, "freq_hist_1km", size=5.4, ha="center")
    text(ax, cx, cy - 1.8, "> 0 ?", size=5.6, ha="center")

    mx, mw = x + 30.0, 32.0
    box(ax, mx, y + 41.0, mw, 11.0, edge=BLUE, face=BLUE_T)
    text(ax, mx + 1.8, y + 49.8, "M3 hybrid", size=7.0, weight="bold")
    text(ax, mx + 1.8, y + 46.5, "44/45-D · no t−1 fire", size=6.2)
    text(ax, mx + 1.8, y + 43.5, "causal history", size=6.2, color=MUTED)

    box(ax, mx, y + 17.0, mw, 11.0, edge=MUTED, face=GREY_T)
    text(ax, mx + 1.8, y + 25.8, "M2 fallback", size=7.0, weight="bold")
    text(ax, mx + 1.8, y + 22.5, "35-D · 25-km anchor", size=6.2)
    text(ax, mx + 1.8, y + 19.5, "freq_hist_1km = 0", size=6.2, color=MUTED)

    arrow(ax, [(x + 26.0, cy + 1.0), (x + 28.0, cy + 1.0), (x + 28.0, y + 46.5), (mx - 0.3, y + 46.5)], color=MUTED)
    text(ax, x + 27.2, y + 39.0, "Yes", size=6.0, color=MUTED, ha="right")
    arrow(ax, [(x + 26.0, cy - 1.0), (x + 28.0, cy - 1.0), (x + 28.0, y + 22.5), (mx - 0.3, y + 22.5)], color=MUTED)
    text(ax, x + 27.2, y + 25.0, "No", size=6.0, color=MUTED, ha="right")

    px, py, pw, ph = x + 30.0, y + 5.0, 32.0, 10.0
    box(ax, px, py, pw, ph, edge=ORANGE, face=ORANGE_T)
    text(ax, px + 1.8, py + 6.9, "P  t−1 propagation kernel", size=6.4, weight="bold")
    text(ax, px + 1.8, py + 3.2, "distance · n25 · FRP25", size=6.2, color=MUTED)

    cal_x, cal_y, cal_w, cal_h = x + 66.0, y + 8.0, 16.0, 32.0
    box(ax, cal_x, cal_y, cal_w, cal_h, edge=MUTED, face=GREY_T)
    text(ax, cal_x + cal_w / 2, y + 29.0, "Platt 2023", size=6.8, weight="bold", ha="center")
    text(ax, cal_x + cal_w / 2, y + 25.5, "per task", size=6.0, ha="center", color=MUTED)
    text(ax, cal_x + cal_w / 2, y + 22.3, "prior correction", size=6.0, ha="center", color=MUTED)
    text(ax, cal_x + cal_w / 2, y + 19.2, "F0.5 frozen", size=6.0, ha="center", color=MUTED)

    arrow(ax, [(mx + mw - 0.2, y + 46.5), (x + 64.0, y + 46.5), (x + 64.0, y + 40.0), (cal_x - 0.3, y + 40.0)], color=MUTED)
    arrow(ax, [(mx + mw - 0.2, y + 22.5), (x + 64.0, y + 22.5), (x + 64.0, y + 35.0), (cal_x - 0.3, y + 35.0)], color=MUTED)
    arrow(ax, [(px + pw + 0.2, y + 10.0), (cal_x - 0.3, y + 10.0)], color=ORANGE)

    p_out_x = cal_x + cal_w + 2.5
    text(ax, p_out_x, y + 39.0, r"$p_{\mathrm{ign}}$", size=7.0, weight="bold")
    text(ax, p_out_x, y + 10.0, r"$p_{\mathrm{prop}}$", size=7.0, weight="bold")
    arrow(ax, [(cal_x + cal_w + 0.2, y + 39.0), (p_out_x - 0.8, y + 39.0)], color=MUTED)
    arrow(ax, [(cal_x + cal_w + 0.2, y + 10.0), (p_out_x - 0.8, y + 10.0)], color=ORANGE)


def panel_d(ax, x, y):
    header(ax, x, y, "d", "Auditable evaluation")

    bx = x + 4.0
    text(ax, bx, y + 48.0, "Temporal split", size=7.0, weight="bold")
    text(ax, x + 80.0, y + 48.0, "1:10 case–control", size=6.0, color=FAINT, ha="right")

    total = 76.0
    widths = [47.0, 11.0, 18.0]
    fills = [BLUE_T, AMBER_T, ORANGE_T]
    edges = [BLUE, AMBER, ORANGE]
    labels = [FACTS["train_years"], FACTS["validation_year"], FACTS["test_years"]]
    xx = bx
    for w, f, e, lab in zip(widths, fills, edges, labels):
        box(ax, xx, y + 40.0, w, 5.5, edge=e, face=f, lw=0.55)
        text(ax, xx + w / 2, y + 42.75, lab, size=6.2, ha="center")
        xx += w
    text(ax, bx, y + 36.5, "Train", size=6.2, color=MUTED, ha="left")
    text(ax, bx + widths[0], y + 36.5, "Val", size=6.2, color=MUTED, ha="center")
    text(ax, bx + widths[0] + widths[1], y + 36.5, "Test frozen", size=5.8, color=MUTED, ha="center")

    text(ax, bx, y + 29.5, "Spatial holdout", size=6.8, weight="bold")
    gx, gy, cell = bx + 1.0, y + 12.0, 3.2
    for i in range(4):
        for j in range(4):
            ax.add_patch(
                Rectangle(
                    (gx + i * cell, gy + j * cell),
                    cell,
                    cell,
                    facecolor=ORANGE_T if (i, j) == (2, 2) else WHITE,
                    edgecolor=GRID,
                    linewidth=0.45,
                    zorder=2,
                )
            )
    ax.add_patch(
        Rectangle(
            (gx + 6.4 - 1.2, gy + 6.4 - 1.2),
            2 * cell + 2.4,
            2 * cell + 2.4,
            fill=False,
            edgecolor=ORANGE,
            linewidth=0.65,
            linestyle=(0, (2.0, 1.5)),
            zorder=3,
        )
    )
    text(ax, gx + 16.0, y + 25.0, "50-km blocks", size=6.2)
    text(ax, gx + 16.0, y + 21.2, "20-km buffer", size=6.2)
    text(ax, gx + 16.0, y + 17.0, "spatial only", size=6.2, color=FAINT, style="italic")

    ux = x + 48.0
    text(ax, ux, y + 29.5, "Uncertainty", size=6.8, weight="bold")
    text(ax, ux, y + 25.0, "date-grouped", size=6.2)
    text(ax, ux, y + 21.2, "bootstrap", size=6.2)
    text(ax, ux, y + 16.8, "500 CI · 1,000 paired", size=6.2, color=MUTED)

    text(ax, bx, y + 7.5, "Decision value", size=6.6, weight="bold")
    text(ax, bx, y + 3.8, "cost-loss", size=6.2)
    text(ax, x + 48.0, y + 7.5, "Open code + data", size=6.6, weight="bold")
    text(ax, x + 48.0, y + 3.8, "reproducible release", size=6.2, color=MUTED)


def build(out_dir: Path) -> None:
    validate_contract()
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=600, facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    panel_a(ax, M, 70.0)
    panel_b(ax, M + PW + G, 70.0)
    panel_c(ax, M, M)
    panel_d(ax, M + PW + G, M)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig01_framework_conceptual"
    fig.savefig(f"{stem}.pdf", facecolor=WHITE, edgecolor="none", bbox_inches=None)
    fig.savefig(f"{stem}.svg", facecolor=WHITE, edgecolor="none", bbox_inches=None)
    fig.savefig(f"{stem}.png", dpi=600, facecolor=WHITE, edgecolor="none", bbox_inches=None)
    fig.savefig(f"{stem}.eps", facecolor=WHITE, edgecolor="none", bbox_inches=None)
    plt.close(fig)
    print(f"Wrote {stem}.pdf/.svg/.png/.eps")


if __name__ == "__main__":
    build(DIR_FIGURAS)
