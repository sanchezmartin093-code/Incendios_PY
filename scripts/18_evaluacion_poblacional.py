"""
Script 18: D2 population evaluation with sampling reweighting.

Uses population counts by date/region/target (D2 plan) and the predictions
from the target-specific tests.
"""

from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_MODELOS
from src.metricas import (
    calcular_metricas_poblacionales,
    calcular_umbral_optimo_f_beta,
    corregir_prior_probabilidades,
)
from src.modelo import (
    cargar_modelo_y_calibrador,
    predecir_probabilidad,
    preparar_matrices_xy,
)

MODELOS = [
    ("M3_ignicion_hibrida", "hibrido", "ignicion", "probabilidad_hibrido"),
    ("M2_ignicion_base", "ignicion", "ignicion", "probabilidad_calibrada"),
    ("M2_propagacion_tactica", "propagacion", "propagacion", "probabilidad_propagacion"),
]


def _prevalencia(conteos, region, task, anio_ini, anio_fin):
    sub = conteos.filter(
        (pl.col("region") == region)
        & (pl.col("task") == task)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(anio_ini, anio_fin)
    )
    n_pos = int(sub["n_pos"].sum())
    n_neg = int(sub["n_neg"].sum())
    return n_pos / (n_pos + n_neg), n_pos + n_neg, n_pos


def _pesos(df, conteos):
    muestra = (
        df.group_by(["fecha", "region", "target"])
        .len()
        .rename({"len": "n_muestra"})
    )
    base = df.select(["fecha", "region", "target"]).join(
        muestra, on=["fecha", "region", "target"], how="left"
    ).join(conteos, on=["fecha", "region"], how="left")
    y = base["target"].to_numpy()
    n_pos_s = base.filter(pl.col("target") == 1)["n_muestra"].to_numpy()
    n_neg_s = base.filter(pl.col("target") == 0)["n_muestra"].to_numpy()
    w_pos = base.filter(pl.col("target") == 1)["n_pos"].to_numpy() / np.maximum(n_pos_s, 1)
    w_neg = base.filter(pl.col("target") == 0)["n_neg"].to_numpy() / np.maximum(n_neg_s, 1)
    pesos = np.empty(len(base), dtype=float)
    pesos[y == 1] = w_pos
    pesos[y == 0] = w_neg
    return base, pesos


def main():
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    conteos = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet")
    filas = []
    filas_anio = []

    for region in ["Chaco", "Oriental"]:
        for nombre, modo, task, col_prob in MODELOS:
            print(f"\n{region} / {task} / {nombre}")
            target_col = "target_ignicion" if task == "ignicion" else "target_propagacion"
            df_val = pl.read_parquet(dir_targets / f"val_{task}.parquet").filter(pl.col("region") == region)
            df_test = pl.read_parquet(dir_targets / f"test_{task}.parquet").filter(pl.col("region") == region)
            preds = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(pl.col("region") == region)

            b, c = cargar_modelo_y_calibrador(
                DIR_MODELOS / f"lightgbm_{modo}_{region.lower()}.booster",
                DIR_MODELOS / "calibradores" / f"calibrador_{modo}_{region.lower()}.joblib",
            )
            X_val, y_val, _ = preparar_matrices_xy(df_val, region, modo=modo)
            p_val = predecir_probabilidad(b, c, X_val)

            prev_val, n_pop_val, n_pos_val = _prevalencia(conteos, region, task, 2023, 2023)
            prev_test, n_pop_test, n_pos_test = _prevalencia(conteos, region, task, 2024, 2025)
            pi_s_val = float(df_val[target_col].mean())
            pi_s_test = float(df_test[target_col].mean())

            p_val_pob = corregir_prior_probabilidades(p_val, pi_s_val, prev_val)
            cutoff, _ = calcular_umbral_optimo_f_beta(y_val, p_val_pob, beta=0.5)

            p_test = preds[col_prob].to_numpy()
            y_test = preds[target_col].to_numpy().astype(int)
            p_test_pob = corregir_prior_probabilidades(p_test, pi_s_test, prev_test)

            _, pesos = _pesos(preds, conteos.filter(pl.col("task") == task))
            metricas = calcular_metricas_poblacionales(y_test, p_test_pob, pesos, cutoff=cutoff)

            filas.append({
                "region": region,
                "task": task,
                "modelo": nombre,
                "prevalencia_poblacion_test": prev_test,
                "prevalencia_muestra_test": pi_s_test,
                "cutoff_f05_poblacional": round(float(cutoff), 6),
                "roc_auc_poblacional": round(metricas["roc_auc_poblacional"], 4),
                "pr_auc_poblacional": round(metricas["pr_auc_poblacional"], 4),
                "precision_poblacional_pct": round(100 * metricas["precision_poblacional"], 3),
                "recall_poblacional_pct": round(100 * metricas["recall_poblacional"], 3),
                "f05_poblacional": round(metricas["f05_poblacional"], 4),
                "brier_poblacional": round(metricas["brier_poblacional"], 6),
                "bss_poblacional": round(metricas["bss_poblacional"], 4),
                "ece_poblacional": round(metricas["ece_poblacional"], 4),
                "n_poblacion_test": n_pop_test,
                "n_pos_poblacion_test": n_pos_test,
                "n_muestra_test": len(y_test),
                "n_pos_muestra_test": int(y_test.sum()),
            })

            anios_eval = preds["fecha"].str.slice(0, 4).cast(pl.Int32).to_numpy()
            for anio_eval in sorted(set(int(a) for a in anios_eval)):
                mask = anios_eval == anio_eval
                prev_yr, n_pop_yr, n_pos_yr = _prevalencia(conteos, region, task, anio_eval, anio_eval)
                p_yr = corregir_prior_probabilidades(p_test[mask], pi_s_test, prev_yr)
                met_yr = calcular_metricas_poblacionales(
                    y_test[mask], p_yr, pesos[mask], cutoff=cutoff
                )
                filas_anio.append({
                    "region": region,
                    "task": task,
                    "modelo": nombre,
                    "anio_eval": anio_eval,
                    "fuel_lag": 1 if anio_eval == 2024 else 2,
                    "prevalencia_poblacion": prev_yr,
                    "cutoff_f05_poblacional": round(float(cutoff), 6),
                    "roc_auc_poblacional": round(met_yr["roc_auc_poblacional"], 4),
                    "pr_auc_poblacional": round(met_yr["pr_auc_poblacional"], 4),
                    "precision_poblacional_pct": round(100 * met_yr["precision_poblacional"], 3),
                    "recall_poblacional_pct": round(100 * met_yr["recall_poblacional"], 3),
                    "n_poblacion": n_pop_yr,
                    "n_pos_poblacion": n_pos_yr,
                    "n_muestra": int(mask.sum()),
                })

    df_out = pd.DataFrame(filas)
    df_anio = pd.DataFrame(filas_anio)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    p_csv = DIR_TABLAS / "tabla07_evaluacion_poblacional.csv"
    df_out.to_csv(p_csv, index=False)
    df_anio.to_csv(DIR_TABLAS / "tabla07_evaluacion_poblacional_por_anio.csv", index=False)
    (DIR_TABLAS / "tabla07_evaluacion_poblacional_por_anio.md").write_text(
        "# Tabla 7b. Evaluación poblacional por año\n\n" + df_anio.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    (DIR_TABLAS / "tabla07_evaluacion_poblacional.md").write_text(
        "# Tabla 7. Evaluación poblacional D2\n\n" + df_out.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
