"""Script 63 (R.4): sensibilidad a la razon de muestreo 1:5 y 1:20.

1:5 se obtiene submuestreando negativos del train 1:10. 1:20 se aproxima
duplicando negativos (bootstrap de negativos), porque el plan congelado no
guarda negativos adicionales. Validacion y test no cambian. Se reporta
precision poblacional corregida por prevalencia.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.metricas import calcular_umbral_optimo_f_beta
from src.modelo import (
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

RATIOS = ("1:10 (baseline)", "1:5", "1:20 (negativos duplicados)")


def _train(region, tr):
    out = {}
    for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
        Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
        out[nombre] = (Xtr, ytr, feats)
    return out


def _pesos_poblacionales(region, te):
    c = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet").filter(
        (pl.col("task") == "ignicion") & (pl.col("region") == region)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(2024, 2025)
    ).select(["fecha", "n_pos", "n_neg"])
    m = te.group_by(["fecha", "target_ignicion"]).len().rename({"len": "n_muestra"})
    b = te.with_row_index("_i").join(m, on=["fecha", "target_ignicion"]).join(c, on="fecha").sort("_i")
    y = b["target_ignicion"].to_numpy().astype(int)
    w = np.where(y == 1, b["n_pos"].to_numpy() / b["n_muestra"].to_numpy(),
                 b["n_neg"].to_numpy() / b["n_muestra"].to_numpy())
    return y, w


def _precision_pob(y, w, p, cut):
    h = p >= cut
    tp = float((w * ((y == 1) & h)).sum()); fp = float((w * ((y == 0) & h)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


def main():
    tr0 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    va0 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    te0 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
    filas = []
    for region in ("Chaco", "Oriental"):
        va = va0.filter(pl.col("region") == region)
        te = te0.filter(pl.col("region") == region)
        y_te, w_te = _pesos_poblacionales(region, te)
        for ratio in RATIOS:
            tr = tr0.filter(pl.col("region") == region)
            if ratio == "1:5":
                pos = tr.filter(pl.col("target_ignicion") == 1)
                neg = tr.filter(pl.col("target_ignicion") == 0).sample(fraction=0.5, seed=42)
                tr = pl.concat([pos, neg])
            elif ratio.startswith("1:20"):
                pos = tr.filter(pl.col("target_ignicion") == 1)
                neg = tr.filter(pl.col("target_ignicion") == 0)
                tr = pl.concat([pos, neg, neg])
            datos = _train(region, tr)
            for nombre in ("M2", "M3"):
                Xtr, ytr, feats = datos[nombre]
                Xva, yva, _ = preparar_matrices_xy(va, region, modo="ignicion" if nombre == "M2" else "hibrido")
                Xte, yte, _ = preparar_matrices_xy(te, region, modo="ignicion" if nombre == "M2" else "hibrido")
                b = entrenar_modelo_lightgbm(
                    Xtr, ytr, Xva, yva, feats,
                    scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                    num_leaves=23 if region == "Chaco" else 31,
                    min_child_samples=20 if region == "Chaco" else 10,
                    num_boost_round=45 if region == "Chaco" else 60,
                    early_stopping_rounds=None, semilla=42,
                )
                c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
                pva = predecir_probabilidad(b, c, Xva)
                pte = predecir_probabilidad(b, c, Xte)
                cut, _ = calcular_umbral_optimo_f_beta(yva, pva, beta=0.5)
                filas.append({
                    "region": region, "ratio": ratio, "model": nombre,
                    "n_train": int(Xtr.shape[0]),
                    "pos_train": int(ytr.sum()),
                    "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
                    "pr_auc": round(float(average_precision_score(yte, pte)), 4),
                    "cutoff_f05": round(float(cut), 4),
                    "precision_pob": round(float(_precision_pob(y_te, w_te, pte, cut)), 6),
                })
                print(region, ratio, nombre, filas[-1], flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS28_sensibilidad_muestreo.csv", index=False)
    _md(df)
    print(df)


def _md(df):
    lineas = [
        "| Region | Ratio | Model | n train | Pos train | ROC-AUC | PR-AUC | Cutoff F0.5 | Population precision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.ratio} | {r.model} | {r.n_train:,} | {r.pos_train:,} | "
            f"{r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.cutoff_f05:.4f} | {r.precision_pob:.6f} |"
        )
    (DIR_TABLAS / "tablaS28_sensibilidad_muestreo.md").write_text(
        "# Table S28. Sampling-ratio sensitivity\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
