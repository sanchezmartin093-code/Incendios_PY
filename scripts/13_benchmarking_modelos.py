"""
Script 13: benchmarking D2 por target (ignición y propagación).

Compara M3/M2/propagación con ISI e INPE proxy en el dominio D2 correspondiente.
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)

from src.constantes import DIR_PROCESSED, DIR_MODELOS, DIR_TABLAS
from src.metricas import (
    bootstrap_diferencia_roc,
    calcular_bootstrap_ci_roc,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy


def _p_modelo(region, modo, df):
    ruta_b = DIR_MODELOS / f"lightgbm_{modo}_{region.lower()}.booster"
    ruta_c = DIR_MODELOS / "calibradores" / f"calibrador_{modo}_{region.lower()}.joblib"
    booster, calibrador = cargar_modelo_y_calibrador(ruta_b, ruta_c)
    X, _, _ = preparar_matrices_xy(df, region, modo=modo)
    return predecir_probabilidad(booster, calibrador, X)


def _umbral_modelo(region, modo):
    ruta = DIR_MODELOS / "calibradores" / f"umbrales_{modo}_{region.lower()}.json"
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    return float(meta["c_f05"])


def _benchmarks(df_val, df_test, region, target_col):
    isi_val = df_val["isi"].to_numpy()
    isi_min, isi_max = float(isi_val.min()), float(isi_val.max())
    p_fwi_val = np.clip((isi_val - isi_min) / (isi_max - isi_min + 1e-6), 0, 1)
    p_fwi = np.clip((df_test["isi"].to_numpy() - isi_min) / (isi_max - isi_min + 1e-6), 0, 1)

    if region == "Oriental":
        veg_val = df_val["pct_pastura"].to_numpy()
        veg_test = df_test["pct_pastura"].to_numpy()
    else:
        veg_val = df_val["pct_bosque_seco"].to_numpy()
        veg_test = df_test["pct_bosque_seco"].to_numpy()

    cdd_vpd_val = (
        df_val["dias_secos_consecutivos"].to_numpy()
        * df_val["vpd_max_kpa"].to_numpy()
        * (0.5 + veg_val / 100.0)
    )
    cdd_vpd_test = (
        df_test["dias_secos_consecutivos"].to_numpy()
        * df_test["vpd_max_kpa"].to_numpy()
        * (0.5 + veg_test / 100.0)
    )
    p_inpe_val = np.clip(1.0 - np.exp(-0.05 * cdd_vpd_val), 0, 1)
    p_inpe = np.clip(1.0 - np.exp(-0.05 * cdd_vpd_test), 0, 1)

    y_val = df_val[target_col].to_numpy().astype(int)
    c_fwi, _ = calcular_umbral_optimo_f_beta(y_val, p_fwi_val, beta=0.5)
    c_inpe, _ = calcular_umbral_optimo_f_beta(y_val, p_inpe_val, beta=0.5)
    return (p_fwi, c_fwi), (p_inpe, c_inpe)


def main():
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    regiones = ["Oriental", "Chaco"]
    resultados = []
    significancia = []

    for region in regiones:
        for task, target_col in [("ignicion", "target_ignicion"), ("propagacion", "target_propagacion")]:
            print(f"\n--- {region} / {task} ---")
            df_train = pl.read_parquet(dir_targets / f"train_{task}.parquet").filter(pl.col("region") == region)
            df_val = pl.read_parquet(dir_targets / f"val_{task}.parquet").filter(pl.col("region") == region)
            df_test = pl.read_parquet(dir_targets / f"test_{task}.parquet").filter(pl.col("region") == region)
            y = df_test[target_col].to_numpy().astype(int)
            n_pos = int(y.sum())
            prevalencia = n_pos / len(y)

            modelos = []
            if task == "ignicion":
                modelos.append(("Incendios_PY (Ignición Híbrida M3)", _p_modelo(region, "hibrido", df_test), _umbral_modelo(region, "hibrido")))
                modelos.append(("Incendios_PY (Ignición Base M2)", _p_modelo(region, "ignicion", df_test), _umbral_modelo(region, "ignicion")))
            else:
                modelos.append(("Incendios_PY (Propagación Táctica)", _p_modelo(region, "propagacion", df_test), _umbral_modelo(region, "propagacion")))

            (p_fwi, c_fwi), (p_inpe, c_inpe) = _benchmarks(df_val, df_test, region, target_col)
            modelos.extend([
                ("Canadian FWI (ISI)", p_fwi, c_fwi),
                ("Proxy INPE Risco de Fogo", p_inpe, c_inpe),
            ])
            p_clim = np.full(len(y), float(df_train[target_col].mean()))
            modelos.append(("Línea Base Climatológica", p_clim, float(df_train[target_col].mean())))

            for nombre, p_mod, cutoff in modelos:
                if len(np.unique(p_mod)) > 1:
                    auc = float(roc_auc_score(y, p_mod))
                    pr_auc = float(average_precision_score(y, p_mod))
                else:
                    auc = 0.5
                    pr_auc = prevalencia
                es_prob = nombre.startswith("Incendios_PY") or "Climatológica" in nombre
                if es_prob:
                    bs = float(brier_score_loss(y, p_mod))
                    prev_train = float(df_train[target_col].mean())
                    bs_clim = float(brier_score_loss(y, np.full(len(y), prev_train)))
                    bss = 1.0 - bs / (bs_clim + 1e-9)
                    ece = float(calcular_expected_calibration_error(y, p_mod))
                else:
                    bs = bss = ece = np.nan
                ci_low, ci_high = calcular_bootstrap_ci_roc(y, p_mod, n_bootstrap=500) if len(np.unique(p_mod)) > 1 else (0.5, 0.5)

                if len(np.unique(p_mod)) > 1:
                    y_bin = (p_mod >= cutoff).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, y_bin).ravel()
                else:
                    fp, tp, fn = 0, 0, int(y.sum())
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05 = fbeta_score(y, (p_mod >= cutoff).astype(int), beta=0.5, zero_division=0) if len(np.unique(p_mod)) > 1 else 0.0

                resultados.append({
                    "region": region,
                    "task": task,
                    "modelo": nombre,
                    "roc_auc": round(auc, 4),
                    "roc_ci_95": f"[{ci_low:.3f} - {ci_high:.3f}]",
                    "pr_auc": round(pr_auc, 4),
                    "ece": round(float(ece), 4) if np.isfinite(ece) else np.nan,
                    "brier_score": round(float(bs), 4) if np.isfinite(bs) else np.nan,
                    "brier_skill_score": round(float(bss), 4) if np.isfinite(bss) else np.nan,
                    "cutoff_f05": round(float(cutoff), 4),
                    "sensibilidad": round(float(recall), 4),
                    "precision": round(float(precision), 4),
                    "f05_score": round(float(f05), 4),
                    "falsos_positivos": int(fp),
                })

            if task == "ignicion":
                pares = [
                    ("Incendios_PY (Ignición Híbrida M3)", "Canadian FWI (ISI)"),
                    ("Incendios_PY (Ignición Híbrida M3)", "Proxy INPE Risco de Fogo"),
                ]
            else:
                pares = [
                    ("Incendios_PY (Propagación Táctica)", "Canadian FWI (ISI)"),
                    ("Incendios_PY (Propagación Táctica)", "Proxy INPE Risco de Fogo"),
                ]
            prob_por_nombre = {n: p for n, p, _ in modelos}
            for a, b in pares:
                delta, p_val = bootstrap_diferencia_roc(y, prob_por_nombre[a], prob_por_nombre[b], n_bootstrap=1000)
                significancia.append({
                    "region": region,
                    "task": task,
                    "comparacion": f"{a} vs {b}",
                    "delta_roc": round(delta, 4),
                    "p_value_bootstrap": round(p_val, 5),
                    "n_bootstrap": 1000,
                })

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(DIR_TABLAS / "tabla04_benchmarking_internacional.csv", index=False)
    (DIR_TABLAS / "tabla04_benchmarking_internacional.md").write_text(
        "# Tabla 4. Benchmarking D2 por target\n\n" + df_res.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    df_sig = pd.DataFrame(significancia)
    df_sig.to_csv(DIR_TABLAS / "tabla10_significancia_pareada.csv", index=False)
    (DIR_TABLAS / "tabla10_significancia_pareada.md").write_text(
        "# Tabla 10. Significancia pareada D2\n\n" + df_sig.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(df_res.to_string(index=False))
    print(df_sig.to_string(index=False))


if __name__ == "__main__":
    main()
