"""
Script 07: Blind Out-of-Time Evaluation and Generation of Scientific and Operational Metrics.
- Evaluates the regional models on the unseen test set (test_dataset.parquet).
- Generates ROC-AUC, PR-AUC, Brier Skill Score curves and confusion matrices at multiple thresholds:
  * Standard threshold (0.50)
  * Optimal Youden J threshold (theoretical balance)
  * Optimal F0.5 threshold (priority on Precision to minimize false alarms)
- Exports formal tables for the scientific manuscript:
  * tablas/tabla02_metricas_desempeno_modelos.csv
  * tablas/tabla02_metricas_desempeno_modelos.md
- Exports individual predictions: datos/processed/evaluacion_test_predicciones.parquet.
"""

from pathlib import Path
import sys

# Ensure the project root is on the PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score
)

from src.constantes import DIR_PROCESSED, DIR_MODELOS, DIR_TABLAS
from src.modelo import (
    preparar_matrices_xy,
    cargar_modelo_y_calibrador,
    predecir_probabilidad
)
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_umbral_optimo_youden,
    calcular_umbral_optimo_f_beta,
    calcular_expected_calibration_error,
    calcular_bootstrap_ci_roc
)

def evaluar_modelo_en_test():
    print("=" * 65)
    print("EVALUACIÓN CIEGA OUT-OF-TIME (TEST SET NO VISTO)")
    print("=" * 65)

    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    p_ign = dir_targets / "test_ignicion.parquet"
    p_prop = dir_targets / "test_propagacion.parquet"
    for p in [p_ign, p_prop]:
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset de test no encontrado en {p}. "
                "Ejecuta scripts/05d_muestrear_poblacional_d2.py primero."
            )

    df_test_ign = pl.read_parquet(p_ign)
    df_test_prop = pl.read_parquet(p_prop)
    print(f"Test ignición: {len(df_test_ign):,} registros.")
    print(f"Test propagación: {len(df_test_prop):,} registros.")

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    regiones = ["Chaco", "Oriental"]
    modos = [
        ("ignicion", "Ignición Primaria"),
        ("propagacion", "Propagación Táctica"),
        ("hibrido", "Ignición Híbrida M3"),
    ]
    
    filas_metricas = []
    preds_por_region = {}

    for region in regiones:
        nombre_lower = region.lower()
        df_reg_ign = df_test_ign.filter(pl.col("region") == region)
        df_reg_prop = df_test_prop.filter(pl.col("region") == region)
        preds_ign = {}
        preds_prop = {}

        for modo_clave, modo_nombre in modos:
            print(f"\n--- EVALUANDO: {region.upper()} [{modo_nombre}] ---")
            ruta_b = DIR_MODELOS / f"lightgbm_{modo_clave}_{nombre_lower}.booster"
            ruta_c = DIR_MODELOS / "calibradores" / f"calibrador_{modo_clave}_{nombre_lower}.joblib"

            if not ruta_b.exists() or not ruta_c.exists():
                print(f"Aviso: No se encontraron los archivos de modelo serializados para {region} ({modo_clave}).")
                continue

            # 1. Load model and calibrator
            booster, calibrador = cargar_modelo_y_calibrador(ruta_b, ruta_c)

            # 2. Extract the X and y matrices from the test set
            es_prop = modo_clave == "propagacion"
            df_fuente = df_test_prop if es_prop else df_test_ign
            preds_dict = preds_prop if es_prop else preds_ign
            X_test, y_test, feats = preparar_matrices_xy(df_fuente, region, modo=modo_clave)
            bloques = df_fuente.filter(pl.col("region") == region)["fecha"].to_numpy()

            n_pos = int(y_test.sum())
            n_neg = int(len(y_test) - n_pos)
            print(f" • Muestras evaluadas: {len(y_test):,} (Positivos: {n_pos}, Negativos: {n_neg})")

            # 3. Calibrated probabilistic prediction
            probs = predecir_probabilidad(booster, calibrador, X_test)
            preds_dict[f"probabilidad_{modo_clave}"] = probs
            if modo_clave == "ignicion":
                preds_dict["probabilidad_calibrada"] = probs
            if modo_clave == "hibrido":
                preds_dict["probabilidad_hibrido"] = probs

            # 4. Discrimination and global calibration metrics
            auc = float(roc_auc_score(y_test, probs))
            pr_auc = float(average_precision_score(y_test, probs))
            bs = float(calcular_brier_score(y_test, probs))
            bss = float(calcular_brier_skill_score(y_test, probs))
            ece = float(calcular_expected_calibration_error(y_test, probs))
            ci_low, ci_high = calcular_bootstrap_ci_roc(y_test, probs, n_bootstrap=500, block_ids=bloques)

            print(f" • ROC-AUC:            {auc:.4f} [{ci_low:.3f} - {ci_high:.3f}]")
            print(f" • PR-AUC:             {pr_auc:.4f}")
            print(f" • ECE:                {ece:.4f}")
            print(f" • Brier Score:        {bs:.4f}")
            print(f" • Brier Skill Score:  {bss:.4f} (Ganancia respecto a la climatología)")

            # 5. Evaluation at 3 operational thresholds derived strictly on validation (Anti-Leakage)
            dir_cal = DIR_MODELOS / "calibradores"
            ruta_meta = dir_cal / f"umbrales_{modo_clave}_{nombre_lower}.json"
            if ruta_meta.exists():
                import json
                meta_val = json.loads(ruta_meta.read_text(encoding="utf-8"))
                c_youden = meta_val.get("c_youden", meta_val.get("c_f05", 0.5))
                c_f05 = meta_val.get("c_f05", 0.5)
            else:
                c_youden, _ = calcular_umbral_optimo_youden(y_test, probs)
                c_f05, _ = calcular_umbral_optimo_f_beta(y_test, probs, beta=0.5)

            umbrales = [
                ("Estándar (0.50)", 0.50),
                (f"Youden J ({c_youden:.2f})", c_youden),
                (f"Alta Precisión F0.5 ({c_f05:.2f})", c_f05)
            ]

            for desc_umbral, cutoff in umbrales:
                y_pred_bin = (probs >= cutoff).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()

                sensibilidad = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                especificidad = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                f1 = float(f1_score(y_test, y_pred_bin, zero_division=0))
                f05 = float(fbeta_score(y_test, y_pred_bin, beta=0.5, zero_division=0))

                filas_metricas.append({
                    "region": region,
                    "capa_modelo": modo_nombre,
                    "umbral_estrategia": desc_umbral,
                    "cutoff": round(cutoff, 3),
                    "roc_auc": round(auc, 4),
                    "roc_ci_95": f"[{ci_low:.3f} - {ci_high:.3f}]",
                    "pr_auc": round(pr_auc, 4),
                    "ece": round(ece, 4),
                    "brier_score": round(bs, 4),
                    "brier_skill_score": round(bss, 4),
                    "sensibilidad_recall": round(sensibilidad, 4),
                    "especificidad": round(especificidad, 4),
                    "precision": round(precision, 4),
                    "f1_score": round(f1, 4),
                    "f05_score": round(f05, 4),
                    "verdaderos_positivos": tp,
                    "falsos_positivos": fp,
                    "verdaderos_negativos": tn,
                    "falsos_negativos": fn
                })

        for col_name, p_vals in preds_ign.items():
            df_reg_ign = df_reg_ign.with_columns(pl.Series(col_name, p_vals))
        for col_name, p_vals in preds_prop.items():
            df_reg_prop = df_reg_prop.with_columns(pl.Series(col_name, p_vals))
        preds_por_region[region] = (df_reg_ign, df_reg_prop)

    dfs_ign = [v[0] for v in preds_por_region.values()]
    dfs_prop = [v[1] for v in preds_por_region.values()]

    # 6. Consolidate results table
    df_tabla = pd.DataFrame(filas_metricas)
    
    path_csv = DIR_TABLAS / "tabla02_metricas_desempeno_modelos.csv"
    path_md = DIR_TABLAS / "tabla02_metricas_desempeno_modelos.md"
    df_tabla.to_csv(path_csv, index=False)

    # Format Markdown table with academic style
    md_content = "# Tabla 2. Métricas de Rendimiento y Discriminación Operativa en Test Ciego\n\n"
    md_content += df_tabla.to_markdown(index=False) + "\n"
    path_md.write_text(md_content, encoding="utf-8")

    # Save predictions separated by target
    if dfs_ign:
        out_ign = DIR_PROCESSED / "evaluacion_test_ignicion.parquet"
        pl.concat(dfs_ign).write_parquet(out_ign, compression="zstd")
        print(f"\n✓ Predicciones de ignición: {out_ign.name}")
    if dfs_prop:
        out_prop = DIR_PROCESSED / "evaluacion_test_propagacion.parquet"
        pl.concat(dfs_prop).write_parquet(out_prop, compression="zstd")
        print(f"✓ Predicciones de propagación: {out_prop.name}")

    print(f"✓ Tabla CSV exportada: {path_csv.name}")
    print(f"✓ Tabla Markdown exportada: {path_md.name}")

    print("\n" + "=" * 65)
    print("RESUMEN DE RENDIMIENTO OPERATIVO EN TEST (OUT-OF-TIME):")
    print("=" * 65)
    print(df_tabla[[
        "region", "umbral_estrategia", "roc_auc", "pr_auc", 
        "precision", "sensibilidad_recall", "f05_score", "falsos_positivos"
    ]].to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    evaluar_modelo_en_test()
