"""
Genera la figura 05 de benchmarking D2: curvas ROC y PR por target.
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
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_FIGURAS

spec = importlib.util.spec_from_file_location(
    "bench", ROOT_DIR / "scripts" / "13_benchmarking_modelos.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main():
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    colores = {"Chaco": "#c0392b", "Oriental": "#27ae60"}

    for fila, task in enumerate(["ignicion", "propagacion"]):
        target_col = "target_ignicion" if task == "ignicion" else "target_propagacion"
        for region in ["Chaco", "Oriental"]:
            df_val = pl.read_parquet(dir_targets / f"val_{task}.parquet").filter(pl.col("region") == region)
            df_test = pl.read_parquet(dir_targets / f"test_{task}.parquet").filter(pl.col("region") == region)
            y = df_test[target_col].to_numpy().astype(int)
            (p_fwi, _), (p_inpe, _) = bench._benchmarks(df_val, df_test, region, target_col)

            if task == "ignicion":
                curva_modelo = bench._p_modelo(region, "hibrido", df_test)
                nombre_modelo = "M3 ignición"
            else:
                curva_modelo = bench._p_modelo(region, "propagacion", df_test)
                nombre_modelo = "Propagación"

            for nombre, p, ls in [
                (f"{region} {nombre_modelo}", curva_modelo, "-"),
                (f"{region} FWI-ISI", p_fwi, ":"),
                (f"{region} INPE proxy", p_inpe, "--"),
            ]:
                fpr, tpr, _ = roc_curve(y, p)
                auc = roc_auc_score(y, p)
                axes[fila, 0].plot(fpr, tpr, color=colores[region], linestyle=ls, linewidth=1.8, label=f"{nombre} ({auc:.3f})")
                prec, rec, _ = precision_recall_curve(y, p)
                ap = average_precision_score(y, p)
                axes[fila, 1].plot(rec, prec, color=colores[region], linestyle=ls, linewidth=1.8, label=f"{nombre} ({ap:.3f})")

        axes[fila, 0].plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        axes[fila, 0].set_title(f"ROC — {task.upper()}", fontweight="bold")
        axes[fila, 0].set_xlabel("1 - Especificidad")
        axes[fila, 0].set_ylabel("Sensibilidad")
        axes[fila, 0].legend(fontsize=7, loc="lower right")
        axes[fila, 1].set_title(f"Precision-Recall — {task.upper()}", fontweight="bold")
        axes[fila, 1].set_xlabel("Recall")
        axes[fila, 1].set_ylabel("Precisión")
        axes[fila, 1].legend(fontsize=7, loc="upper right")

    plt.suptitle("Benchmarking D2: Incendios_PY vs Canadian FWI (ISI) vs Proxy INPE", fontweight="bold", y=0.99)
    plt.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.png", dpi=300)
    fig.savefig(DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.pdf")
    plt.close(fig)
    print("✓ fig05 actualizada")


if __name__ == "__main__":
    main()
