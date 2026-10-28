#!/usr/bin/env python3
"""Script 43: weather-only GBT baseline for the FWI/INPE benchmark (B.2).

Trains a gradient-boosted tree using only the meteorological information that
feeds the Canadian FWI (temperature, atmospheric demand, wind, rainfall and the
two FWI-derived codes). It deliberately excludes fuel fractions, ecotone,
anthropogenic priors, accessibility, calendar and fire history, so the
comparison isolates "how far can weather alone go".

Same capacity, calibration and cutoff rule as M1/M2/M3:
  * learning rate 0.03, 45/60 rounds, 23/31 leaves, scale_pos_weight 1/2,
    min_child_samples 20/10, monotonic physical constraints.
  * monotonic Platt calibration fitted on 2023.
  * F0.5 cutoff selected on 2023 validation and frozen for 2024-2025.

Outputs:
  * modelos/lightgbm_weatheronly_{region}.booster
  * modelos/calibradores/calibrador_weatheronly_{region}.joblib
  * modelos/calibradores/umbrales_weatheronly_{region}.json
  * datos/processed/evaluacion_test_weather_only.parquet
  * entregables/tablas/tablaS21_benchmark_weather_only.csv / .md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, roc_auc_score

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    bootstrap_diferencia_roc,
    calcular_bootstrap_ci_roc,
    calcular_brier_score,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    guardar_modelo_y_calibrador,
    predecir_probabilidad,
)

# Information set available to the Canadian FWI: no fuel, no anthropic priors,
# no accessibility, no calendar and no fire history.
FEATURES_WEATHER_ONLY = [
    "temp_max_c",
    "temp_min_c",
    "vpd_max_kpa",
    "viento_max_kmh",
    "precipitacion_mm",
    "precipitacion_48h_mm",
    "dias_secos_consecutivos",
    "ffmc",
    "isi",
]


def _metricas(y: np.ndarray, p: np.ndarray, cutoff: float) -> dict:
    pred = (p >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "cutoff_f05": float(cutoff),
        "precision": float(precision),
        "recall": float(recall),
        "f05_score": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
        "falsos_positivos": int(fp),
    }


def main() -> None:
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    tr = pl.read_parquet(dir_t / "train_ignicion.parquet")
    va = pl.read_parquet(dir_t / "val_ignicion.parquet")
    te = pl.read_parquet(dir_t / "test_ignicion.parquet")
    eval_m3 = pl.read_parquet(
        DIR_PROCESSED / "evaluacion_test_ignicion.parquet",
        columns=["cell_id", "fecha", "region", "probabilidad_hibrido"],
    )

    dir_cal = DIR_MODELOS / "calibradores"
    dir_cal.mkdir(parents=True, exist_ok=True)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)

    filas = []
    predicciones = []

    for region in ["Chaco", "Oriental"]:
        trr = tr.filter(pl.col("region") == region)
        vaa = va.filter(pl.col("region") == region)
        tee = te.filter(pl.col("region") == region)

        Xtr = trr.select(FEATURES_WEATHER_ONLY).to_numpy()
        ytr = trr["target_ignicion"].to_numpy().astype(float)
        Xva = vaa.select(FEATURES_WEATHER_ONLY).to_numpy()
        yva = vaa["target_ignicion"].to_numpy().astype(float)
        Xte = tee.select(FEATURES_WEATHER_ONLY).to_numpy()
        yte = tee["target_ignicion"].to_numpy().astype(int)

        es_chaco = region == "Chaco"
        booster = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, FEATURES_WEATHER_ONLY,
            scale_pos_weight=1.0 if es_chaco else 2.0,
            num_boost_round=45 if es_chaco else 60,
            num_leaves=23 if es_chaco else 31,
            min_child_samples=20 if es_chaco else 10,
            early_stopping_rounds=None,
            semilla=42,
        )
        calibrador = calibrar_probabilidades_isotonica(
            booster, Xva, yva, metodo="platt_monotono"
        )
        p_va = predecir_probabilidad(booster, calibrador, Xva)
        cutoff, _ = calcular_umbral_optimo_f_beta(yva, p_va, beta=0.5)
        p_te = predecir_probabilidad(booster, calibrador, Xte)

        nombre = region.lower()
        guardar_modelo_y_calibrador(
            booster, calibrador,
            DIR_MODELOS / f"lightgbm_weatheronly_{nombre}.booster",
            dir_cal / f"calibrador_weatheronly_{nombre}.joblib",
        )

        met = _metricas(yte, p_te, cutoff)
        prev_train = float(ytr.mean())
        met["brier_score"] = float(calcular_brier_score(yte, p_te))
        bs_clim = float(np.mean((yte - prev_train) ** 2))
        met["brier_skill_score"] = 1.0 - met["brier_score"] / (bs_clim + 1e-9)
        met["ece"] = float(calcular_expected_calibration_error(yte, p_te))
        fechas = tee["fecha"].to_numpy()
        ci_low, ci_high = calcular_bootstrap_ci_roc(
            yte, p_te, n_bootstrap=500, block_ids=fechas
        )

        # Paired comparison against M3 on exactly the same rows.
        clave = tee.select(["cell_id", "fecha"])
        p_m3 = (
            clave.join(eval_m3.filter(pl.col("region") == region),
                       on=["cell_id", "fecha"], how="left")["probabilidad_hibrido"]
            .to_numpy()
        )
        delta, p_val = bootstrap_diferencia_roc(
            yte, p_m3, p_te, n_bootstrap=1000, block_ids=fechas
        )

        filas.append({
            "region": region,
            "modelo": "Weather-only GBT",
            "roc_auc": round(met["roc_auc"], 4),
            "roc_ci_95": f"[{ci_low:.3f} - {ci_high:.3f}]",
            "pr_auc": round(met["pr_auc"], 4),
            "brier_score": round(met["brier_score"], 4),
            "brier_skill_score": round(met["brier_skill_score"], 4),
            "ece": round(met["ece"], 4),
            "cutoff_f05": round(met["cutoff_f05"], 4),
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f05_score": round(met["f05_score"], 4),
            "falsos_positivos": met["falsos_positivos"],
            "delta_roc_vs_m3": round(delta, 4),
            "p_vs_m3": round(float(p_val), 4),
            "n": int(len(yte)),
            "n_pos": int(yte.sum()),
        })
        predicciones.append(
            tee.select(["cell_id", "region", "fecha", "target_ignicion"])
            .with_columns(
                pl.Series("probabilidad_weather_only", p_te),
                pl.Series("probabilidad_m3", p_m3),
            )
        )

        meta = {
            "region": region,
            "modo": "weather_only",
            "features": FEATURES_WEATHER_ONLY,
            "c_f05": round(float(cutoff), 4),
            "roc_auc_val": round(float(roc_auc_score(yva, p_va)), 4),
            "pr_auc_val": round(float(average_precision_score(yva, p_va)), 4),
        }
        (dir_cal / f"umbrales_weatheronly_{nombre}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"{region}: ROC={met['roc_auc']:.4f} PR={met['pr_auc']:.4f} "
              f"cutoff={cutoff:.4f} prec={met['precision']:.4f} rec={met['recall']:.4f} "
              f"dROC(M3-WO)={delta:+.4f} p={p_val:.4f}")

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS21_benchmark_weather_only.csv")
    cols = df.columns
    lineas = [
        "# Table S21. Weather-only GBT baseline (FWI information set)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in df.rows():
        lineas.append("| " + " | ".join(str(v) for v in row) + " |")
    (DIR_TABLAS / "tablaS21_benchmark_weather_only.md").write_text(
        "\n".join(lineas) + "\n", encoding="utf-8"
    )
    pl.concat(predicciones).write_parquet(
        DIR_PROCESSED / "evaluacion_test_weather_only.parquet", compression="zstd"
    )

    # Merge the two rows into the main benchmark table, replacing any previous
    # weather-only rows so the script is idempotent. Run scripts/13 first.
    ruta_bench = DIR_TABLAS / "tabla04_benchmarking_internacional.csv"
    if ruta_bench.exists():
        base = pl.read_csv(ruta_bench)
        extra = df.select([
            pl.col("region"),
            pl.lit("ignicion").alias("task"),
            pl.lit("Weather-only GBT (FWI information set)").alias("modelo"),
            pl.col("roc_auc"),
            pl.col("roc_ci_95"),
            pl.col("pr_auc"),
            pl.col("ece"),
            pl.col("brier_score"),
            pl.col("brier_skill_score"),
            pl.col("cutoff_f05"),
            pl.col("recall").alias("sensibilidad"),
            pl.col("precision"),
            pl.col("f05_score"),
            pl.col("falsos_positivos"),
        ]).select(base.columns)
        base = base.filter(~pl.col("modelo").str.starts_with("Weather-only"))
        pl.concat([base, extra], how="vertical_relaxed").write_csv(ruta_bench)
        print(f"Filas weather-only integradas en {ruta_bench}")
    print(df)


if __name__ == "__main__":
    main()
