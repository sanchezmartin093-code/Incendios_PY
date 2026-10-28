"""Figure S5: rolling-origin ROC-AUC by fold."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.constantes import DIR_FIGURAS, DIR_TABLAS


def main():
    df = pd.read_csv(DIR_TABLAS / "tablaS26_rolling_origin.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    colores = {"M1": "#7f7f7f", "M2": "#1f77b4", "M3": "#d62728"}
    for ax, region in zip(axes, ("Chaco", "Oriental")):
        sub = df[df.region == region].sort_values("test_year")
        for model in ("M1", "M2", "M3"):
            m = sub[sub.model == model]
            ax.plot(m.test_year, m.roc_auc, marker="o", label=model, color=colores[model])
        for year in sorted(sub.test_year.unique()):
            d = sub[(sub.test_year == year) & (sub.model == "M3")]["delta_roc_m3_m2"].iloc[0]
            ax.annotate(f"{d:+.3f}", (year, sub[(sub.test_year == year) & (sub.model == "M3")]["roc_auc"].iloc[0]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7, color="#d62728")
        ax.set_title(region)
        ax.set_xlabel("Test year")
        ax.set_xticks(sorted(sub.test_year.unique()))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    ruta = DIR_FIGURAS / "supp_fig05_rolling_origin"
    fig.savefig(ruta.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(ruta)


if __name__ == "__main__":
    main()
