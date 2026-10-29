"""
Script 13: D2 benchmarking per target (ignition and propagation).

Compares M3/M2/propagation with ISI and the INPE proxy in the corresponding D2 domain.
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
from src.inpe import constante_a_desde_fracciones, riesgo_basico, riesgo_basico_vectorizado
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
    grilla = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet",
        columns=["cell_id", "nodo_clima_id"],
    )
    fwi = pl.read_parquet(DIR_PROCESSED / "fwi_completo_diario.parquet")
    inpe = pl.read_parquet(DIR_PROCESSED / "inpe_diario.parquet")

    def _join(df):
        x = df.join(grilla, on="cell_id", how="left").join(
            fwi, on=["nodo_clima_id", "fecha"], how="left"
        ).join(inpe, on=["nodo_clima_id", "fecha"], how="left")
        a = np.array([
            constante_a_desde_fracciones(p, s, bs, bh, palm)
            for p, s, bs, bh, palm in zip(
                x["pct_pastura"].to_numpy(),
                x["pct_sabana_pastizal"].to_numpy(),
                x["pct_bosque_seco"].to_numpy(),
                x["pct_bosque_humedo"].to_numpy(),
                x["pct_humedal_palmar"].to_numpy(),
            )
        ])
        rb = riesgo_basico_vectorizado(x["pse"].to_numpy(), a)
        fur = -0.006 * x["rh_min_pct"].to_numpy() + 1.3
        ft = 0.02 * x["temp_max_c"].to_numpy() + 0.4
        flat = 1 + np.abs(x["lat"].to_numpy()) * 0.003
        p_inpe = np.clip(rb * fur * ft * flat, 0, 1)
        return x["fwi"].to_numpy(), p_inpe

    p_fwi_val, p_inpe_val = _join(df_val)
    p_fwi, p_inpe = _join(df_test)
    y_val = df_val[target_col].to_numpy().astype(int)
    c_fwi, _ = calcular_umbral_optimo_f_beta(y_val, p_fwi_val, beta=0.5)
    c_inpe, _ = calcular_umbral_optimo_f_beta(y_val, p_inpe_val, beta=0.5)
    return (p_fwi, c_fwi), (p_inpe, c_inpe)


def main():
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    regiones = ["Oriental", "Chaco"]
    resultados = []
    resultados_anio = []
    significancia = []

    for region in regiones:
        for task, target_col in [("ignicion", "target_ignicion"), ("propagacion", "target_propagacion")]:
            print(f"\n--- {region} / {task} ---")
            df_train = pl.read_parquet(dir_targets / f"train_{task}.parquet").filter(pl.col("region") == region)
            df_val = pl.read_parquet(dir_targets / f"val_{task}.parquet").filter(pl.col("region") == region)
            df_test = pl.read_parquet(dir_targets / f"test_{task}.parquet").filter(pl.col("region") == region)
            y = df_test[target_col].to_numpy().astype(int)
            bloques = df_test["fecha"].to_numpy()
            anios_eval = df_test["fecha"].str.slice(0, 4).cast(pl.Int32).to_numpy()
            n_pos = int(y.sum())
            prevalencia = n_pos / len(y)

            modelos = []
            if task == "ignicion":
                modelos.append(("Incendios_PY (Ignición Híbrida M3)", _p_modelo(region, "hibrido", df_test), _umbral_modelo(region, "hibrido")))
                modelos.append(("Incendios_PY (Ignición Base M2)", _p_modelo(region, "ignicion", df_test), _umbral_modelo(region, "ignicion")))
            else:
                modelos.append(("Incendios_PY (Propagación Táctica)", _p_modelo(region, "propagacion", df_test), _umbral_modelo(region, "propagacion")))

            if task == "ignicion":
                (p_fwi, c_fwi), (p_inpe, c_inpe) = _benchmarks(df_val, df_test, region, target_col)
                modelos.extend([
                    ("Canadian FWI completo", p_fwi, c_fwi),
                    ("INPE Risco de Fogo", p_inpe, c_inpe),
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
                ci_low, ci_high = calcular_bootstrap_ci_roc(y, p_mod, n_bootstrap=500, block_ids=bloques) if len(np.unique(p_mod)) > 1 else (0.5, 0.5)

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

                for anio_eval in sorted(set(int(a) for a in anios_eval)):
                    mask = anios_eval == anio_eval
                    y_yr = y[mask]
                    p_yr = p_mod[mask]
                    if len(y_yr) == 0:
                        continue
                    auc_yr = float(roc_auc_score(y_yr, p_yr)) if len(np.unique(y_yr)) > 1 and len(np.unique(p_yr)) > 1 else 0.5
                    pr_yr = float(average_precision_score(y_yr, p_yr)) if len(np.unique(y_yr)) > 1 and len(np.unique(p_yr)) > 1 else float(y_yr.mean())
                    y_bin = (p_yr >= cutoff).astype(int)
                    tn, fp_yr, fn_yr, tp_yr = confusion_matrix(y_yr, y_bin, labels=[0, 1]).ravel()
                    prec_yr = float(tp_yr / (tp_yr + fp_yr)) if (tp_yr + fp_yr) else 0.0
                    rec_yr = float(tp_yr / (tp_yr + fn_yr)) if (tp_yr + fn_yr) else 0.0
                    f05_yr = float(fbeta_score(y_yr, y_bin, beta=0.5, zero_division=0))
                    resultados_anio.append({
                        "region": region,
                        "task": task,
                        "modelo": nombre,
                        "anio_eval": anio_eval,
                        "fuel_lag": 1 if anio_eval == 2024 else 2,
                        "n": int(len(y_yr)),
                        "n_pos": int(y_yr.sum()),
                        "roc_auc": round(auc_yr, 4),
                        "pr_auc": round(pr_yr, 4),
                        "precision": round(prec_yr, 4),
                        "recall": round(rec_yr, 4),
                        "f05_score": round(f05_yr, 4),
                        "falsos_positivos": int(fp_yr),
                    })

            if task == "ignicion":
                pares = [
                    ("Incendios_PY (Ignición Híbrida M3)", "Canadian FWI completo"),
                    ("Incendios_PY (Ignición Híbrida M3)", "INPE Risco de Fogo"),
                    ("Incendios_PY (Ignición Híbrida M3)", "Incendios_PY (Ignición Base M2)"),
                ]
            else:
                pares = []
            prob_por_nombre = {n: p for n, p, _ in modelos}
            for a, b in pares:
                delta, p_val = bootstrap_diferencia_roc(y, prob_por_nombre[a], prob_por_nombre[b], n_bootstrap=1000, block_ids=bloques)
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
    df_res_anio = pd.DataFrame(resultados_anio)
    df_res.to_csv(DIR_TABLAS / "tabla04_benchmarking_internacional.csv", index=False)
    df_res_anio.to_csv(DIR_TABLAS / "tabla04_benchmarking_por_anio.csv", index=False)
    (DIR_TABLAS / "tabla04_benchmarking_por_anio.md").write_text(
        "# Tabla 4b. Benchmarking por año (2024 lag 1; 2025 lag 2)\n\n"
        + df_res_anio.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
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
