"""Script 67 (R.4): 1:20 con negativos independientes.

Usa `targets_extra/gate_1_20` para entrenar con 20 negativos por positivo.
Validacion y test se mantienen en el muestreo 1:10 congelado para que las
metricas sean comparables.
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
from src.modelo import calibrar_probabilidades_isotonica, entrenar_modelo_lightgbm, predecir_probabilidad, preparar_matrices_xy


def _pesos(region, te):
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


def _precision(y, w, p, cut):
    h = p >= cut
    tp = float((w * ((y == 1) & h)).sum()); fp = float((w * ((y == 0) & h)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


def main():
    tr10 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    va10 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    te10 = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
    tr20 = pl.read_parquet(DIR_PROCESSED / "targets_extra" / "gate_1_20" / "train_ignicion.parquet")
    rows = []
    for region in ("Chaco", "Oriental"):
        va = va10.filter(pl.col("region") == region)
        te = te10.filter(pl.col("region") == region)
        y_te, w_te = _pesos(region, te)
        variants = {"1:10 (baseline)": tr10.filter(pl.col("region") == region),
                    "1:20 (independientes)": tr20.filter(pl.col("region") == region)}
        for ratio, tr in variants.items():
            for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
                Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
                Xva, yva, _ = preparar_matrices_xy(va, region, modo=modo)
                Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
                b = entrenar_modelo_lightgbm(
                    Xtr, ytr, Xva, yva, feats,
                    scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                    num_leaves=23 if region == "Chaco" else 31,
                    min_child_samples=20 if region == "Chaco" else 10,
                    num_boost_round=45 if region == "Chaco" else 60,
                    early_stopping_rounds=None, semilla=42,
                )
                c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
                pva = predecir_probabilidad(b, c, Xva); pte = predecir_probabilidad(b, c, Xte)
                cut, _ = calcular_umbral_optimo_f_beta(yva, pva, beta=0.5)
                rows.append({
                    "region": region, "ratio": ratio, "model": nombre,
                    "n_train": int(Xtr.shape[0]), "pos_train": int(ytr.sum()),
                    "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
                    "pr_auc": round(float(average_precision_score(yte, pte)), 4),
                    "cutoff_f05": round(float(cut), 4),
                    "precision_pob": round(float(_precision(y_te, w_te, pte, cut)), 6),
                })
                print(region, ratio, nombre, rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DIR_TABLAS / "tablaS28_muestreo_1_20_independiente.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
