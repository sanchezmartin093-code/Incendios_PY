"""Evaluate the operational gated M3/M2 ignition policy.

M3 is used where ``freq_hist_1km > 0`` and M2 is used where it is zero. The
gated F0.5 cutoff is selected on 2023 validation and frozen for the
retrospective 2024-2025 test. Outputs are separate from the r4 candidate tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
    calcular_metricas_poblacionales,
    calcular_umbral_optimo_f_beta,
    calcular_bootstrap_ci_roc,
    corregir_prior_probabilidades,
)
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy


def _prevalencia(conteos: pl.DataFrame, region: str, ini: int, fin: int) -> tuple[float, int, int]:
    sub = conteos.filter(
        (pl.col("region") == region)
        & (pl.col("task") == "ignicion")
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(ini, fin)
    )
    n_pos = int(sub["n_pos"].sum())
    n_neg = int(sub["n_neg"].sum())
    return n_pos / (n_pos + n_neg), n_pos + n_neg, n_pos


def _pesos(df: pl.DataFrame, conteos: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    muestra = df.group_by(["fecha", "region", "target"]).len().rename({"len": "n_muestra"})
    base = df.select(["fecha", "region", "target"]).join(muestra, on=["fecha", "region", "target"], how="left").join(
        conteos.filter(pl.col("task") == "ignicion"), on=["fecha", "region"], how="left"
    )
    y = base["target"].to_numpy()
    n_pos_s = base.filter(pl.col("target") == 1)["n_muestra"].to_numpy()
    n_neg_s = base.filter(pl.col("target") == 0)["n_muestra"].to_numpy()
    w_pos = base.filter(pl.col("target") == 1)["n_pos"].to_numpy() / np.maximum(n_pos_s, 1)
    w_neg = base.filter(pl.col("target") == 0)["n_neg"].to_numpy() / np.maximum(n_neg_s, 1)
    pesos = np.empty(len(base), dtype=float)
    pesos[y == 1] = w_pos
    pesos[y == 0] = w_neg
    return base, pesos


def _sample_metrics(y: np.ndarray, p: np.ndarray, cutoff: float, fechas: np.ndarray) -> dict:
    pred = (p >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ci_low, ci_high = calcular_bootstrap_ci_roc(y, p, n_bootstrap=500, block_ids=fechas)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_ci_95": f"[{ci_low:.3f} - {ci_high:.3f}]",
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(calcular_brier_score(y, p)),
        "bss": float(calcular_brier_skill_score(y, p)),
        "ece": float(calcular_expected_calibration_error(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f05": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main() -> None:
    val = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    test = pl.read_parquet(DIR_PROCESSED / "evaluacion_test_ignicion.parquet")
    conteos = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet")

    sample_rows = []
    pop_rows = []
    for region in ["Chaco", "Oriental"]:
        print(f"Evaluando política gated M3/M2: {region}")
        val_r = val.filter(pl.col("region") == region).sort("fecha")
        test_r = test.filter(pl.col("region") == region).sort("fecha")
        y_val = val_r["target_ignicion"].to_numpy().astype(int)
        y_test = test_r["target_ignicion"].to_numpy().astype(int)

        p_val = {}
        for mode, name in [("hibrido", "M3"), ("ignicion", "M2")]:
            b, c = cargar_modelo_y_calibrador(
                DIR_MODELOS / f"lightgbm_{mode}_{region.lower()}.booster",
                DIR_MODELOS / "calibradores" / f"calibrador_{mode}_{region.lower()}.joblib",
            )
            X_val, y_ok, _ = preparar_matrices_xy(val_r, region, modo=mode)
            if not np.array_equal(y_ok.astype(int), y_val):
                raise ValueError(f"Desalineación de labels en validación: {region}/{mode}")
            p_val[name] = predecir_probabilidad(b, c, X_val)

        fallback_val = val_r["freq_hist_1km"].to_numpy() <= 0
        p_gated_val = np.where(fallback_val, p_val["M2"], p_val["M3"])
        cutoff, _ = calcular_umbral_optimo_f_beta(y_val, p_gated_val, beta=0.5)

        p_test_m2 = test_r["probabilidad_ignicion"].to_numpy()
        p_test_m3 = test_r["probabilidad_hibrido"].to_numpy()
        fallback_test = test_r["freq_hist_1km"].to_numpy() <= 0
        p_gated_test = np.where(fallback_test, p_test_m2, p_test_m3)

        sample = _sample_metrics(y_test, p_gated_test, cutoff, test_r["fecha"].to_numpy())
        sample.update({
            "region": region,
            "policy": "M3/M2 gated",
            "cutoff_f05": round(float(cutoff), 6),
            "n": int(len(y_test)),
            "n_pos": int(y_test.sum()),
            "n_fallback": int(fallback_test.sum()),
        })
        sample_rows.append(sample)

        prev_test, n_pop, n_pos_pop = _prevalencia(conteos, region, 2024, 2025)
        pi_s = float(y_test.mean())
        p_pob = corregir_prior_probabilidades(p_gated_test, pi_s, prev_test)
        cutoff_pob = float(corregir_prior_probabilidades(np.array([cutoff]), pi_s, prev_test)[0])
        _, pesos = _pesos(test_r, conteos)
        met = calcular_metricas_poblacionales(y_test, p_pob, pesos, cutoff=cutoff_pob)
        pop_rows.append({
            "region": region,
            "modelo": "M3/M2 gated",
            "prevalencia_poblacion": prev_test,
            "prevalencia_muestra": pi_s,
            "cutoff_f05_poblacional": round(cutoff_pob, 6),
            "roc_auc_poblacional": round(met["roc_auc_poblacional"], 4),
            "pr_auc_poblacional": round(met["pr_auc_poblacional"], 4),
            "precision_poblacional_pct": round(100 * met["precision_poblacional"], 3),
            "recall_poblacional_pct": round(100 * met["recall_poblacional"], 3),
            "f05_poblacional": round(met["f05_poblacional"], 4),
            "brier_poblacional": round(met["brier_poblacional"], 6),
            "bss_poblacional": round(met["bss_poblacional"], 4),
            "ece_poblacional": round(met["ece_poblacional"], 4),
            "n_poblacion": n_pop,
            "n_pos_poblacion": n_pos_pop,
            "n_muestra": len(y_test),
        })

    df_sample = pd.DataFrame(sample_rows)
    df_pop = pd.DataFrame(pop_rows)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(DIR_TABLAS / "tabla13_politica_gated.csv", index=False)
    df_pop.to_csv(DIR_TABLAS / "tabla13_politica_gated_poblacional.csv", index=False)
    (DIR_TABLAS / "tabla13_politica_gated.md").write_text(
        "# Tabla 13. Política operativa gated M3/M2 en el test retrospectivo 2024–2025\n\n"
        "M3 se usa en celdas con historia (`freq_hist_1km > 0`); M2 en celdas sin historia. "
        "El cutoff F0.5 se seleccionó en validación 2023 y se congeló para test.\n\n"
        + df_sample.to_markdown(index=False) + "\n\n## Poblacional\n\n"
        + df_pop.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(df_sample.to_string(index=False))
    print(df_pop.to_string(index=False))


if __name__ == "__main__":
    main()
