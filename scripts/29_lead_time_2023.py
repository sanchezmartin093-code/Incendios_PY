"""
Script 29: t-1 persistence lead-time evaluation on 2023 validation only.

No test data or confirmation holdout is read.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.espaciotemporal import agregar_features_antropicas
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    FEATURES_HIBRIDO_CHACO,
    FEATURES_HIBRIDO_ORIENTAL,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
    FEATURES_M1_CHACO,
    FEATURES_M1_ORIENTAL,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)

WEATHER_COLS = [
    "temp_max_c", "temp_min_c", "precipitacion_mm", "precipitacion_48h_mm",
    "viento_max_kmh", "vpd_max_kpa", "dias_secos_consecutivos", "ffmc", "isi",
    "interaccion_viento_vpd", "swvl1_m3m3", "fdd_heladas_60d", "deficit_hidro_60d",
]


def _con_nodo(df, grilla):
    return df.join(grilla.select(["cell_id", "nodo_clima_id"]), on="cell_id", how="left")


def _t1(df, clima):
    df = df.with_columns(
        (pl.col("fecha").str.to_date("%Y-%m-%d") - pl.duration(days=1))
        .dt.strftime("%Y-%m-%d").alias("_fecha_prev")
    )
    clim = clima.select(["nodo_clima_id", "fecha", *WEATHER_COLS]).rename(
        {c: f"{c}_t1" for c in WEATHER_COLS}
    ).rename({"fecha": "_fecha_prev"})
    df = df.join(clim, on=["nodo_clima_id", "_fecha_prev"], how="left")
    df = df.with_columns([pl.col(f"{c}_t1").alias(c) for c in WEATHER_COLS])
    df = df.drop([f"{c}_t1" for c in WEATHER_COLS] + ["_fecha_prev"])
    return agregar_features_antropicas(df.with_columns([
        (pl.col("temp_max_c") - pl.col("temp_min_c")).alias("rango_termico_diario"),
        (pl.col("vpd_max_kpa") * (pl.col("dias_secos_consecutivos") + 1.0)).alias("deficit_hidrico_acumulado"),
        ((pl.col("viento_max_kmh") / 10.0).pow(2)).alias("potencial_viento_cuadratico"),
    ]))


def _m(y, p, cutoff, region, modelo):
    y_bin = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (y_bin == 1)).sum())
    fp = int(((y == 0) & (y_bin == 1)).sum())
    fn = int(((y == 1) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1 + 0.25) * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) else 0.0
    return {
        "region": region, "modelo": modelo, "n": int(len(y)), "n_pos": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "cutoff_f05": round(float(cutoff), 6),
        "precision": round(prec, 4), "recall": round(rec, 4), "f05": round(f05, 4),
    }


def main():
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    grid = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    clima = pl.read_parquet(DIR_PROCESSED / "clima_era5_diario.parquet")
    train = _t1(_con_nodo(pl.read_parquet(dir_t / "train_ignicion.parquet"), grid), clima)
    val = _t1(_con_nodo(pl.read_parquet(dir_t / "val_ignicion.parquet"), grid), clima)
    train = train.drop_nulls(WEATHER_COLS)
    val = val.drop_nulls(WEATHER_COLS)
    filas = []
    for region in ["Chaco", "Oriental"]:
        tr = train.filter(pl.col("region") == region)
        va = val.filter(pl.col("region") == region)
        modelos = [
            ("M1_biofisico_t1", FEATURES_M1_CHACO if region == "Chaco" else FEATURES_M1_ORIENTAL),
            ("M2_base_t1", FEATURES_IGNICION_CHACO if region == "Chaco" else FEATURES_IGNICION_ORIENTAL),
            ("M3_hibrido_t1", FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL),
        ]
        ytr = tr["target_ignicion"].to_numpy().astype(np.float32)
        yva = va["target_ignicion"].to_numpy().astype(np.float32)
        for nombre, feats in modelos:
            Xtr, Xva = tr.select(feats).to_numpy(), va.select(feats).to_numpy()
            b = entrenar_modelo_lightgbm(
                Xtr, ytr, Xva, yva, feats,
                scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                num_boost_round=45 if region == "Chaco" else 60,
                num_leaves=23 if region == "Chaco" else 31,
                min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None, semilla=42,
            )
            c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
            p = predecir_probabilidad(b, c, Xva)
            cutoff, _ = calcular_umbral_optimo_f_beta(yva, p, beta=0.5)
            filas.append(_m(yva.astype(int), p, cutoff, region, nombre))
            print(region, nombre, filas[-1]["roc_auc"], filas[-1]["pr_auc"])
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tabla10_validacion_2023_lead_time.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
