"""Script 61 (R.9): sensibilidad del gate biofisico.

El muestreo 1:10 congelado ya esta dentro del gate (VPD >= 0.8, lluvia 48 h
< 10 mm). Este script evalua umbrales mas estrictos y documenta la retencion.
No reconstruye el archivo candidato completo; por tanto no es un "sin gate"
puro, sino una ablacion del umbral dentro del dominio muestreado.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.modelo import entrenar_modelo_lightgbm, preparar_matrices_xy

VARIANTES = {
    "gate 0.8/10 (baseline)": (0.8, 10.0),
    "gate 1.0/5": (1.0, 5.0),
    "gate 1.2/2": (1.2, 2.0),
}


def _fwi(df):
    grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"])
    fwi = pl.read_parquet(DIR_PROCESSED / "fwi_completo_diario.parquet", columns=["nodo_clima_id", "fecha", "fwi"])
    return df.join(grilla, on="cell_id", how="left").join(fwi, on=["nodo_clima_id", "fecha"], how="left")


def _train(region, tr, va, te):
    out = {}
    for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
        Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
        Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xtr, ytr, feats,
            scale_pos_weight=1.0 if region == "Chaco" else 2.0,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            num_boost_round=45 if region == "Chaco" else 60,
            early_stopping_rounds=None, semilla=42,
        )
        out[nombre] = float(roc_auc_score(yte, b.predict(Xte)))
    return out


def main():
    tr = _fwi(pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet"))
    va = _fwi(pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet"))
    te = _fwi(pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet"))
    filas = []
    for region in ("Chaco", "Oriental"):
        tr_r = tr.filter(pl.col("region") == region)
        va_r = va.filter(pl.col("region") == region)
        te_r = te.filter(pl.col("region") == region)
        n0, p0 = te_r.height, int(te_r["target_ignicion"].sum())
        for nombre, (vpd, lluvia) in VARIANTES.items():
            cond = (pl.col("vpd_max_kpa") >= vpd) & (pl.col("precipitacion_48h_mm") < lluvia)
            tr_v = tr_r.filter(cond)
            va_v = va_r.filter(cond)
            te_v = te_r.filter(cond)
            if int((tr_v["target_ignicion"] == 1).sum()) < 100 or int(te_v["target_ignicion"].sum()) < 20:
                continue
            m = _train(region, tr_v, va_v, te_v)
            y = te_v["target_ignicion"].to_numpy().astype(int)
            fwi = float(roc_auc_score(y, te_v["fwi"].to_numpy())) if te_v["fwi"].null_count() == 0 else np.nan
            filas.append({
                "region": region, "variant": nombre,
                "vpd_min": vpd, "rain_max": lluvia,
                "retention_rows": round(te_v.height / n0, 3),
                "retention_pos": round(float(te_v["target_ignicion"].sum()) / p0, 3),
                "n_train": tr_v.height, "n_test": te_v.height,
                "m3_roc": round(m["M3"], 4), "m2_roc": round(m["M2"], 4),
                "fwi_roc": round(fwi, 4) if np.isfinite(fwi) else np.nan,
                "delta_m3_m2": round(m["M3"] - m["M2"], 4),
            })
            print(region, nombre, filas[-1], flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS31_ablacion_gate.csv", index=False)
    _md(df)
    print(df)


def _md(df):
    lineas = [
        "| Region | Gate | Rows retained | Positives retained | M3 ROC | M2 ROC | FWI ROC | ΔROC M3−M2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.variant} | {r.retention_rows:.3f} | {r.retention_pos:.3f} | "
            f"{r.m3_roc:.4f} | {r.m2_roc:.4f} | {r.fwi_roc:.4f} | {r.delta_m3_m2:+.4f} |"
        )
    (DIR_TABLAS / "tablaS31_ablacion_gate.md").write_text(
        "# Table S31. Gate threshold ablation\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
