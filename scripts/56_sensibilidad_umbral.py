"""Script 56 (R.5): sensibilidad al umbral y a la metrica.

Compara cortes F0.5, F1, F2 y Youden seleccionados en validacion 2023 y
evaluados en el test 2024-2025 para M3, M2 y propagacion. Genera la Figura S6
y una tabla de continuacion de la Tabla S14.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import confusion_matrix, fbeta_score

from src.constantes import DIR_FIGURAS, DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    calcular_umbral_optimo_f_beta,
    calcular_umbral_optimo_youden,
)
from src.modelo import (
    cargar_modelo_y_calibrador,
    predecir_probabilidad,
    preparar_matrices_xy,
)

MODELOS = [
    ("ignicion", "M3", "hibrido"),
    ("ignicion", "M2", "ignicion"),
    ("propagacion", "Propagation", "propagacion"),
]


def _pred(region, task, modo):
    val = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / f"val_{task}.parquet").filter(pl.col("region") == region)
    test = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / f"test_{task}.parquet").filter(pl.col("region") == region)
    b, c = cargar_modelo_y_calibrador(
        DIR_MODELOS / f"lightgbm_{modo}_{region.lower()}.booster",
        DIR_MODELOS / "calibradores" / f"calibrador_{modo}_{region.lower()}.joblib",
    )
    Xv, yv, _ = preparar_matrices_xy(val, region, modo=modo)
    Xt, yt, _ = preparar_matrices_xy(test, region, modo=modo)
    return val, test, yv, predecir_probabilidad(b, c, Xv), yt, predecir_probabilidad(b, c, Xt)


def _metricas(y, p, cut):
    yb = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": float(prec), "recall": float(rec),
        "f05": float(fbeta_score(y, yb, beta=0.5, zero_division=0)),
        "f1": float(fbeta_score(y, yb, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y, yb, beta=2.0, zero_division=0)),
    }


def _curva(y, p, rejilla):
    out = []
    for t in rejilla:
        m = _metricas(y, p, float(t))
        out.append((float(t), m["precision"], m["recall"], m["f05"], m["f1"], m["f2"]))
    return np.array(out)


def main():
    rejilla = np.linspace(0.0, 1.0, 201)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    filas = []
    paneles = [("Chaco", "ignicion"), ("Oriental", "ignicion"), ("Chaco", "propagacion"), ("Oriental", "propagacion")]
    for ax, (region, task) in zip(axes.ravel(), paneles):
        modelo = "M3" if task == "ignicion" else "Propagation"
        modo = "hibrido" if task == "ignicion" else "propagacion"
        _, _, yv, pv, yt, pt = _pred(region, task, modo)
        curva = _curva(yv, pv, rejilla)
        ax.plot(curva[:, 0], curva[:, 2], label="Recall", color="#1f77b4")
        ax.plot(curva[:, 0], curva[:, 1], label="Precision", color="#d62728")
        ax.plot(curva[:, 0], curva[:, 3], label="F0.5", color="#2ca02c", ls="--")
        ax.plot(curva[:, 0], curva[:, 4], label="F1", color="#9467bd", ls=":")
        ax.plot(curva[:, 0], curva[:, 5], label="F2", color="#ff7f0e", ls="-.")
        for beta, color in ((0.5, "#2ca02c"), (1.0, "#9467bd"), (2.0, "#ff7f0e")):
            cut, _ = calcular_umbral_optimo_f_beta(yv, pv, beta=beta)
            ax.axvline(cut, color=color, lw=0.8, alpha=0.7)
            m = _metricas(yt, pt, cut)
            filas.append({
                "region": region, "task": task, "model": modelo,
                "criterion": f"F{beta:g}", "cutoff": round(float(cut), 4),
                "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                "f05": round(m["f05"], 4), "f1": round(m["f1"], 4), "f2": round(m["f2"], 4),
            })
        cut, _ = calcular_umbral_optimo_youden(yv, pv)
        ax.axvline(cut, color="black", lw=0.8, ls="--", alpha=0.7)
        m = _metricas(yt, pt, cut)
        filas.append({
            "region": region, "task": task, "model": modelo,
            "criterion": "Youden", "cutoff": round(float(cut), 4),
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "f05": round(m["f05"], 4), "f1": round(m["f1"], 4), "f2": round(m["f2"], 4),
        })
        ax.set_title(f"{region} — {task}")
        ax.set_xlabel("Cutoff")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Metric")
    axes[1, 0].set_ylabel("Metric")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    ruta = DIR_FIGURAS / "supp_fig06_sensibilidad_umbral"
    fig.savefig(ruta.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS14b_sensibilidad_umbral.csv", index=False)
    _md(df)
    print(df)


def _md(df):
    lineas = [
        "| Region | Task | Model | Criterion | Cutoff | Precision | Recall | F0.5 | F1 | F2 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.task} | {r.model} | {r.criterion} | {r.cutoff:.4f} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.f05:.4f} | {r.f1:.4f} | {r.f2:.4f} |"
        )
    (DIR_TABLAS / "tablaS14b_sensibilidad_umbral.md").write_text(
        "# Table S14 (continued). Threshold and metric sensitivity\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
