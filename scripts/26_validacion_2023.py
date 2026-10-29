"""
Script 26: Fase 2 scientific selection evidence on 2023 validation only.

This script never reads test_*.parquet or the confirmation holdout.

Outputs:
- auditoria/fase2_seleccion_modelo/FASE2_VALIDACION_2023/predicciones_val_2023_<region>.parquet
- tablas/tabla10_validacion_2023.csv
- tablas/tabla10_validacion_2023_fallback.csv
- tablas/tabla10_validacion_2023_poblacional.csv
- tablas/tabla10_comparaciones_pareadas_2023.csv
"""

from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_MODELOS
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
    calcular_metricas_poblacionales,
    calcular_umbral_optimo_f_beta,
    corregir_prior_probabilidades,
)
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy

MODELOS = [
    ("M1_biofisico", "m1"),
    ("M2_base", "ignicion"),
    ("M3_hibrido", "hibrido"),
]
SALIDA = ROOT_DIR / "auditoria" / "fase2_seleccion_modelo" / "FASE2_VALIDACION_2023"


def _metricas(y, p, cutoff, region, modelo, subset):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
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
        "subset": subset,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalencia": round(float(y.mean()), 6),
        "cutoff_f05": round(float(cutoff), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float(y.mean()),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "fp": fp,
        "fn": fn,
    }


def _pesos(df_val, conteos, task):
    muestra = (
        df_val.group_by(["fecha", "region", "target"])
        .len()
        .rename({"len": "n_muestra"})
    )
    base = (
        df_val.select(["fecha", "region", "target"])
        .join(muestra, on=["fecha", "region", "target"], how="left")
        .join(
            conteos.filter(pl.col("task") == task),
            on=["fecha", "region"],
            how="left",
        )
    )
    y = base["target"].to_numpy()
    w = np.empty(len(base), dtype=float)
    pos = base.filter(pl.col("target") == 1)
    neg = base.filter(pl.col("target") == 0)
    w[y == 1] = pos["n_pos"].to_numpy() / np.maximum(pos["n_muestra"].to_numpy(), 1)
    w[y == 0] = neg["n_neg"].to_numpy() / np.maximum(neg["n_muestra"].to_numpy(), 1)
    if np.any(np.isnan(w)):
        raise ValueError("Pesos poblacionales con NaN; revisar conteos 2023")
    return w


def _prevalencia(conteos, region, task):
    s = conteos.filter(
        (pl.col("task") == task)
        & (pl.col("region") == region)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).eq(2023)
    )
    p = int(s["n_pos"].sum())
    n = int(s["n_neg"].sum())
    return p / (p + n), p + n, p


def _delta_bootstrap(y, p1, p0, blocks, n_boot=1000, seed=42):
    """Paired grouped bootstrap CI for ΔROC-AUC and ΔPR-AUC."""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1).astype(float)
    p0 = np.asarray(p0).astype(float)
    blocks = np.asarray(blocks).astype(str)
    unicos, inv = np.unique(blocks, return_inverse=True)
    indices = [np.where(inv == i)[0] for i in range(len(unicos))]
    rng = np.random.default_rng(seed)
    d_auc = []
    d_pr = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unicos), size=len(unicos))
        idx = np.concatenate([indices[j] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
        d_pr.append(
            average_precision_score(y[idx], p1[idx])
            - average_precision_score(y[idx], p0[idx])
        )
    obs_auc = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    obs_pr = float(average_precision_score(y, p1) - average_precision_score(y, p0))
    if not d_auc:
        return obs_auc, np.nan, np.nan, obs_pr, np.nan, np.nan
    return (
        obs_auc,
        float(np.percentile(d_auc, 2.5)),
        float(np.percentile(d_auc, 97.5)),
        obs_pr,
        float(np.percentile(d_pr, 2.5)),
        float(np.percentile(d_pr, 97.5)),
    )


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    conteos = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet")
    val = pl.read_parquet(dir_t / "val_ignicion.parquet")
    if val["fecha"].min() >= "2024" or val["fecha"].max() >= "2024":
        raise ValueError("val_ignicion no es 2023; se aborta para evitar fuga temporal")

    filas = []
    filas_fallback = []
    filas_pob = []
    filas_delta = []
    filas_delta_fallback = []
    for region in ["Chaco", "Oriental"]:
        df_region = val.filter(pl.col("region") == region).filter(pl.col("target_ignicion").is_not_null())
        y = df_region["target_ignicion"].to_numpy().astype(int)
        fechas = df_region["fecha"].to_numpy()
        pesos = _pesos(df_region, conteos, "ignicion")
        prev_pob, n_pob, n_pos_pob = _prevalencia(conteos, region, "ignicion")
        pi_s = float(y.mean())
        preds = {"y": y, "fecha": fechas, "region": region}
        for nombre, modo in MODELOS:
            b, c = cargar_modelo_y_calibrador(
                DIR_MODELOS / f"lightgbm_{modo}_{region.lower()}.booster",
                DIR_MODELOS / "calibradores" / f"calibrador_{modo}_{region.lower()}.joblib",
            )
            X, y_ok, _ = preparar_matrices_xy(df_region, region, modo=modo)
            if not np.array_equal(y_ok.astype(int), y):
                raise ValueError(f"Desalineación de labels {region}/{nombre}")
            p = predecir_probabilidad(b, c, X)
            meta = json.loads(
                (DIR_MODELOS / "calibradores" / f"umbrales_{modo}_{region.lower()}.json").read_text()
            )
            cutoff = float(meta["c_f05"])
            filas.append(_metricas(y, p, cutoff, region, nombre, "todos"))
            mask0 = df_region["freq_hist_1km"].to_numpy() == 0
            filas_fallback.append(
                _metricas(y[mask0], p[mask0], cutoff, region, nombre, "sin_historia")
            )
            p_pob = corregir_prior_probabilidades(p, pi_s, prev_pob)
            cutoff_pob, _ = calcular_umbral_optimo_f_beta(y, p_pob, beta=0.5)
            met = calcular_metricas_poblacionales(y, p_pob, pesos, cutoff=cutoff_pob)
            filas_pob.append({
                "region": region,
                "modelo": nombre,
                "prevalencia_poblacion_2023": round(prev_pob, 8),
                "prevalencia_muestra_2023": round(pi_s, 6),
                "cutoff_poblacional_2023": round(float(cutoff_pob), 6),
                "roc_auc_poblacional": round(met["roc_auc_poblacional"], 4),
                "pr_auc_poblacional": round(met["pr_auc_poblacional"], 4),
                "precision_poblacional_pct": round(100 * met["precision_poblacional"], 3),
                "recall_poblacional_pct": round(100 * met["recall_poblacional"], 3),
                "f05_poblacional": round(met["f05_poblacional"], 4),
                "brier_poblacional": round(met["brier_poblacional"], 6),
                "bss_poblacional": round(met["bss_poblacional"], 4),
                "ece_poblacional": round(met["ece_poblacional"], 4),
                "n_poblacion_2023": n_pob,
                "n_pos_poblacion_2023": n_pos_pob,
                "n_muestra": len(y),
            })
            preds[nombre] = p
            print(f"{region}/{nombre}: ROC={filas[-1]['roc_auc']:.4f} PR={filas[-1]['pr_auc']:.4f}")

        for nombre_a, nombre_b in [("M3_hibrido", "M2_base"), ("M3_hibrido", "M1_biofisico"), ("M2_base", "M1_biofisico")]:
            d_auc, lo_auc, hi_auc, d_pr, lo_pr, hi_pr = _delta_bootstrap(
                y, preds[nombre_a], preds[nombre_b], fechas
            )
            filas_delta.append({
                "region": region,
                "comparacion": f"{nombre_a}_minus_{nombre_b}",
                "delta_roc_auc": round(d_auc, 4),
                "delta_roc_ci95_low": round(lo_auc, 4),
                "delta_roc_ci95_high": round(hi_auc, 4),
                "delta_pr_auc": round(d_pr, 4),
                "delta_pr_ci95_low": round(lo_pr, 4),
                "delta_pr_ci95_high": round(hi_pr, 4),
                "n_bootstrap": 1000,
                "agrupacion": "fecha",
            })

        for nombre_a, nombre_b in [("M2_base", "M3_hibrido"), ("M3_hibrido", "M1_biofisico"), ("M2_base", "M1_biofisico")]:
            d_auc, lo_auc, hi_auc, d_pr, lo_pr, hi_pr = _delta_bootstrap(
                y[mask0], preds[nombre_a][mask0], preds[nombre_b][mask0], fechas[mask0]
            )
            filas_delta_fallback.append({
                "region": region,
                "subset": "sin_historia",
                "comparacion": f"{nombre_a}_minus_{nombre_b}",
                "delta_roc_auc": round(d_auc, 4),
                "delta_roc_ci95_low": round(lo_auc, 4),
                "delta_roc_ci95_high": round(hi_auc, 4),
                "delta_pr_auc": round(d_pr, 4),
                "delta_pr_ci95_low": round(lo_pr, 4),
                "delta_pr_ci95_high": round(hi_pr, 4),
                "n_bootstrap": 1000,
                "agrupacion": "fecha",
            })

        df_pred = pl.DataFrame({
            "fecha": fechas,
            "region": region,
            "y": y,
            **{f"p_{k}": v for k, v in preds.items() if k not in {"y", "fecha", "region"}},
        })
        df_pred.write_parquet(SALIDA / f"predicciones_val_2023_{region.lower()}.parquet")

    pd.DataFrame(filas).to_csv(DIR_TABLAS / "tabla10_validacion_2023.csv", index=False)
    pd.DataFrame(filas_fallback).to_csv(DIR_TABLAS / "tabla10_validacion_2023_fallback.csv", index=False)
    pd.DataFrame(filas_pob).to_csv(DIR_TABLAS / "tabla10_validacion_2023_poblacional.csv", index=False)
    pd.DataFrame(filas_delta).to_csv(DIR_TABLAS / "tabla10_comparaciones_pareadas_2023.csv", index=False)
    pd.DataFrame(filas_delta_fallback).to_csv(
        DIR_TABLAS / "tabla10_comparaciones_fallback_2023.csv", index=False
    )
    print("\nComparaciones pareadas:")
    print(pd.DataFrame(filas_delta).to_string(index=False))
    print("\nComparaciones fallback (sin historia):")
    print(pd.DataFrame(filas_delta_fallback).to_string(index=False))


if __name__ == "__main__":
    main()
