"""
Script 22: análisis cost-loss/utilidad operativa D2.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_FIGURAS, DIR_MODELOS
from src.cost_loss import valor_cost_loss
from src.metricas import corregir_prior_probabilidades

MODELOS = [
    ("M3_ignicion_hibrida", "ignicion", "probabilidad_hibrido"),
    ("M2_ignicion_base", "ignicion", "probabilidad_calibrada"),
    ("Propagacion_tactica", "propagacion", "probabilidad_propagacion"),
]
ALPHAS = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])


def _conteos():
    return pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet")


def _prevalencia(c, region, task, ini, fin):
    s = c.filter(
        (pl.col("region") == region) & (pl.col("task") == task)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(ini, fin)
    )
    p = int(s["n_pos"].sum()); n = int(s["n_neg"].sum())
    return p / (p + n), p + n, p


def _pesos(preds, c):
    muestra = preds.group_by(["fecha", "region", "target"]).len().rename({"len": "n_muestra"})
    base = preds.select(["fecha", "region", "target"]).join(muestra, on=["fecha", "region", "target"], how="left").join(c, on=["fecha", "region"], how="left")
    y = base["target"].to_numpy()
    w = np.empty(len(base), dtype=float)
    pos = base.filter(pl.col("target") == 1)
    neg = base.filter(pl.col("target") == 0)
    w[y == 1] = pos["n_pos"].to_numpy() / np.maximum(pos["n_muestra"].to_numpy(), 1)
    w[y == 0] = neg["n_neg"].to_numpy() / np.maximum(neg["n_muestra"].to_numpy(), 1)
    return w


def main():
    c = _conteos()
    filas = []
    for region in ["Chaco", "Oriental"]:
        for nombre, task, col in MODELOS:
            target = "target_ignicion" if task == "ignicion" else "target_propagacion"
            preds = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(pl.col("region") == region)
            y = preds[target].to_numpy().astype(int)
            p = preds[col].to_numpy()
            pi_s = float(y.mean())
            prev, _, _ = _prevalencia(c, region, task, 2024, 2025)
            p_pob = corregir_prior_probabilidades(p, pi_s, prev)
            ctask = c.filter((pl.col("task") == task) & (pl.col("region") == region))
            w = _pesos(preds, ctask)
            for a in ALPHAS:
                v = valor_cost_loss(y, p_pob, w, a)
                filas.append({"region": region, "task": task, "modelo": nombre, **v})
    df = pd.DataFrame(filas)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_TABLAS / "tabla11_cost_loss.csv", index=False)
    (DIR_TABLAS / "tabla11_cost_loss.md").write_text(
        "# Tabla 11. Análisis cost-loss poblacional\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, task in zip(axes, ["ignicion", "propagacion"]):
        for reg, ls in [("Chaco", "-"), ("Oriental", "--")]:
            for nombre, _, _ in [m for m in MODELOS if m[1] == task]:
                sub = df[(df["region"] == reg) & (df["task"] == task) & (df["modelo"] == nombre)]
                label = {"M3_ignicion_hibrida": "M3 hybrid ignition",
                         "M2_ignicion_base": "M2 baseline ignition",
                         "Propagacion_tactica": "Tactical propagation"}.get(nombre, nombre)
                ax.plot(sub["alpha"], sub["valor"], ls, label=f"{reg} {label}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"Cost-loss — {'ignition' if task == 'ignicion' else 'propagation'}")
        ax.set_xlabel("α = C/L")
        ax.set_ylabel("V(α)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(DIR_FIGURAS / "fig08_cost_loss.png")
    fig.savefig(DIR_FIGURAS / "fig08_cost_loss.pdf")
    plt.close(fig)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
