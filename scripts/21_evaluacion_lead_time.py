"""
Script 21: Evaluation with t-1 weather (persistence) to estimate the cost
of moving from a nowcast with same-day weather to a forecast with 24 h of
lead time.

It is not a real numerical forecast: it uses yesterday's observed weather as a
persistence proxy. It serves as a lower bound/order of magnitude of performance
if same-day weather is removed.

Exports `tablas/tabla09_evaluacion_lead_time.csv/.md`.
"""

from pathlib import Path
import sys
from datetime import timedelta

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
    FEATURES_M1_BIOFISICO_BASE,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)

WEATHER_COLS = [
    "temp_max_c", "temp_min_c", "precipitacion_mm", "precipitacion_48h_mm",
    "viento_max_kmh", "vpd_max_kpa", "dias_secos_consecutivos",
    "ffmc", "isi", "interaccion_viento_vpd", "swvl1_m3m3",
]


def _con_nodo(df: pl.DataFrame, grilla: pl.DataFrame) -> pl.DataFrame:
    return df.join(
        grilla.select(["cell_id", "nodo_clima_id"]),
        on="cell_id",
        how="left",
    )


def _reemplazar_por_t1(df: pl.DataFrame, clima: pl.DataFrame) -> pl.DataFrame:
    """Replaces the weather variables with those from the previous day."""
    df = df.with_columns(
        (
            pl.col("fecha").str.to_date("%Y-%m-%d")
            - pl.duration(days=1)
        ).dt.strftime("%Y-%m-%d").alias("_fecha_prev")
    )
    clim = clima.select(["nodo_clima_id", "fecha", *WEATHER_COLS]).rename(
        {c: f"{c}_t1" for c in WEATHER_COLS}
    ).rename({"fecha": "_fecha_prev"})
    df = df.join(clim, on=["nodo_clima_id", "_fecha_prev"], how="left")
    df = df.with_columns([
        pl.col(f"{c}_t1").alias(c) for c in WEATHER_COLS
    ]).drop([f"{c}_t1" for c in WEATHER_COLS] + ["_fecha_prev"])
    df = df.with_columns([
        (pl.col("temp_max_c") - pl.col("temp_min_c")).alias("rango_termico_diario"),
        (pl.col("vpd_max_kpa") * (pl.col("dias_secos_consecutivos") + 1.0)).alias("deficit_hidrico_acumulado"),
        ((pl.col("viento_max_kmh") / 10.0).pow(2)).alias("potencial_viento_cuadratico"),
    ])
    return agregar_features_antropicas(df)


def _metricas(y, p, cutoff, region, modelo):
    y_bin = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (y_bin == 1)).sum())
    fp = int(((y == 0) & (y_bin == 1)).sum())
    fn = int(((y == 1) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1 + 0.25) * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) else 0.0
    return {
        "region": region,
        "modelo": modelo,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float(np.mean(y)),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "cutoff_f05": round(float(cutoff), 5),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "fp": fp,
        "fn": fn,
    }


def main():
    print("=" * 72)
    print("EVALUACIÓN CON CLIMA t-1 (PERSISTENCIA, 24 H DE ANTICIPACIÓN)")
    print("=" * 72)

    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    p_train = dir_targets / "train_ignicion.parquet"
    p_val = dir_targets / "val_ignicion.parquet"
    p_test = dir_targets / "test_ignicion.parquet"
    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    for p in [p_train, p_val, p_test, p_grilla, p_clima]:
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}")

    grilla = pl.read_parquet(p_grilla)
    clima = pl.read_parquet(p_clima)

    train = _reemplazar_por_t1(_con_nodo(pl.read_parquet(p_train), grilla), clima)
    val = _reemplazar_por_t1(_con_nodo(pl.read_parquet(p_val), grilla), clima)
    test = _reemplazar_por_t1(_con_nodo(pl.read_parquet(p_test), grilla), clima)

    nulos = int(train["ffmc"].is_null().sum() + val["ffmc"].is_null().sum() + test["ffmc"].is_null().sum())
    if nulos:
        print(f"[ADVERTENCIA] {nulos:,} filas sin clima t-1; se eliminan.")
        train = train.drop_nulls(WEATHER_COLS)
        val = val.drop_nulls(WEATHER_COLS)
        test = test.drop_nulls(WEATHER_COLS)

    filas = []
    for region in ["Chaco", "Oriental"]:
        print(f"\n--- {region.upper()} ---")
        tr = train.filter(pl.col("region") == region)
        va = val.filter(pl.col("region") == region)
        te = test.filter(pl.col("region") == region)
        b_col = "pct_bosque_seco" if region == "Chaco" else "pct_bosque_humedo"
        feats_m3 = FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL
        modelos = [
            ("M1_biofisico_t1", FEATURES_M1_BIOFISICO_BASE + [b_col]),
            ("M3_hibrido_t1", feats_m3),
        ]
        y_tr = tr["target_ignicion"].to_numpy().astype(np.float32)
        y_va = va["target_ignicion"].to_numpy().astype(np.float32)
        y_te = te["target_ignicion"].to_numpy().astype(int)
        spw = 1.0 if region == "Chaco" else 2.0
        leaves = 23 if region == "Chaco" else 31
        rounds = 45 if region == "Chaco" else 60

        for nombre, feats in modelos:
            X_tr = tr.select(feats).to_numpy()
            X_va = va.select(feats).to_numpy()
            X_te = te.select(feats).to_numpy()
            booster = entrenar_modelo_lightgbm(
                X_tr, y_tr, X_va, y_va, feats,
                scale_pos_weight=spw, num_boost_round=rounds,
                num_leaves=leaves, min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None,
            )
            cal = calibrar_probabilidades_isotonica(booster, X_va, y_va, metodo="platt_monotono")
            p_va = predecir_probabilidad(booster, cal, X_va)
            cutoff, _ = calcular_umbral_optimo_f_beta(y_va, p_va, beta=0.5)
            p_te = predecir_probabilidad(booster, cal, X_te)
            m = _metricas(y_te, p_te, cutoff, region, nombre)
            filas.append(m)
            print(f"  {nombre}: ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} Prec={100*m['precision']:.2f}%")

    df = pd.DataFrame(filas)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    p_csv = DIR_TABLAS / "tabla09_evaluacion_lead_time.csv"
    p_md = DIR_TABLAS / "tabla09_evaluacion_lead_time.md"
    df.to_csv(p_csv, index=False)
    p_md.write_text(
        "# Tabla 9. Evaluación con clima de t-1 (persistencia, 24 h de anticipación)\n\n"
        "Las variables meteorológicas se sustituyen por las del día anterior. "
        "No es un pronóstico numérico real; es una cota de persistencia. "
        "Comparar con la Tabla 5 (clima del mismo día).\n\n"
        + df.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n✓ Guardado en {p_csv.name} y {p_md.name}")
    print(df.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
