"""
Generates figure 05 of D2 benchmarking: ROC and PR curves per target.
"""

from pathlib import Path
import importlib.util
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_FIGURAS

plt.rcParams.update({"font.family": "Arial", "font.size": 7})

spec = importlib.util.spec_from_file_location(
    "bench", ROOT_DIR / "scripts" / "13_benchmarking_modelos.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main():
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    fig, axes = plt.subplots(2, 2, figsize=(5.83, 5.2), dpi=600)
    colores = {"Chaco": "#c0392b", "Oriental": "#27ae60"}

    for fila, task in enumerate(["ignicion", "propagacion"]):
        target_col = "target_ignicion" if task == "ignicion" else "target_propagacion"
        for region in ["Chaco", "Oriental"]:
            df_val = pl.read_parquet(dir_targets / f"val_{task}.parquet").filter(pl.col("region") == region)
            df_test = pl.read_parquet(dir_targets / f"test_{task}.parquet").filter(pl.col("region") == region)
            y = df_test[target_col].to_numpy().astype(int)
            if task == "ignicion":
                (p_fwi, _), (p_inpe, _) = bench._benchmarks(df_val, df_test, region, target_col)
                curva_modelo = bench._p_modelo(region, "hibrido", df_test)
                nombre_modelo = "M3"
                wo = pl.read_parquet(
                    DIR_PROCESSED / "evaluacion_test_weather_only.parquet"
                ).filter(pl.col("region") == region).select(
                    ["cell_id", "fecha", "probabilidad_weather_only"]
                )
                p_wo = (
                    df_test.select(["cell_id", "fecha"])
                    .join(wo, on=["cell_id", "fecha"], how="left")["probabilidad_weather_only"]
                    .to_numpy()
                )
                curvas = [
                    (f"{region} {nombre_modelo}", curva_modelo, "-"),
                    (f"{region} FWI", p_fwi, ":"),
                    (f"{region} INPE", p_inpe, "--"),
                    (f"{region} Weather-only", p_wo, "-."),
                ]
            else:
                curva_modelo = bench._p_modelo(region, "propagacion", df_test)
                nombre_modelo = "Propagation"
                curvas = [(f"{region} {nombre_modelo}", curva_modelo, "-")]

            for nombre, p, ls in curvas:
                fpr, tpr, _ = roc_curve(y, p)
                auc = roc_auc_score(y, p)
                axes[fila, 0].plot(fpr, tpr, color=colores[region], linestyle=ls, linewidth=1.2, label=f"{nombre} ({auc:.3f})")
                prec, rec, _ = precision_recall_curve(y, p)
                ap = average_precision_score(y, p)
                axes[fila, 1].plot(rec, prec, color=colores[region], linestyle=ls, linewidth=1.2, label=f"{nombre} ({ap:.3f})")

        axes[fila, 0].plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        task_label = "Ignition" if task == "ignicion" else "Propagation"
        axes[fila, 0].set_title(f"ROC — {task_label}", fontsize=7.5, fontweight="bold")
        axes[fila, 0].set_xlabel("False positive rate", fontsize=7)
        axes[fila, 0].set_ylabel("True positive rate", fontsize=7)
        axes[fila, 1].set_title(f"Precision–recall — {task_label}", fontsize=7.5, fontweight="bold")
        axes[fila, 1].set_xlabel("Recall", fontsize=7)
        axes[fila, 1].set_ylabel("Precision", fontsize=7)

    legend_handles = [
        Line2D([0], [0], color=colores["Chaco"], ls="-", lw=1.2,
               label="Chaco M3 / Propagation"),
        Line2D([0], [0], color=colores["Chaco"], ls=":", lw=1.2, label="Chaco FWI"),
        Line2D([0], [0], color=colores["Chaco"], ls="--", lw=1.2, label="Chaco INPE"),
        Line2D([0], [0], color=colores["Oriental"], ls="-", lw=1.2,
               label="Oriental M3 / Propagation"),
        Line2D([0], [0], color=colores["Oriental"], ls=":", lw=1.2, label="Oriental FWI"),
        Line2D([0], [0], color=colores["Oriental"], ls="--", lw=1.2, label="Oriental INPE"),
        Line2D([0], [0], color=colores["Chaco"], ls="-.", lw=1.2, label="Chaco weather-only"),
        Line2D([0], [0], color=colores["Oriental"], ls="-.", lw=1.2, label="Oriental weather-only"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=6.2,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.18,
                        hspace=0.45, wspace=0.28)
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.png", dpi=600)
    fig.savefig(DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.pdf")
    plt.close(fig)
    print("✓ fig05 actualizada")


if __name__ == "__main__":
    main()
