"""
Script 07: Evaluación Ciega Out-of-Time y Generación de Métricas Científicas y Operativas.
- Evalúa los modelos regionales sobre el conjunto de testeo no visto (test_dataset.parquet).
- Genera curvas ROC-AUC, PR-AUC, Brier Skill Score y matrices de confusión en múltiples umbrales:
  * Umbral estándar (0.50)
  * Umbral óptimo de Youden J (balance teórico)
  * Umbral óptimo F0.5 (prioridad a la Precisión para minimizar falsas alarmas)
- Exporta tablas formales para el manuscrito científico:
  * tablas/tabla02_metricas_desempeno_modelos.csv
  * tablas/tabla02_metricas_desempeno_modelos.md
- Exporta predicciones individuales: datos/processed/evaluacion_test_predicciones.parquet.
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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
    calcular_umbral_optimo_f_beta
)

def evaluar_modelo_en_test():
    print("=" * 65)
    print("EVALUACIÓN CIEGA OUT-OF-TIME (TEST SET NO VISTO)")
    print("=" * 65)

    p_test = DIR_PROCESSED / "test_dataset.parquet"
    if not p_test.exists():
        raise FileNotFoundError(f"Dataset de test no encontrado en {p_test}. Ejecuta scripts/05_muestrear_ausencias.py primero.")

    df_test = pl.read_parquet(p_test)
    print(f"Cargado dataset de test: {len(df_test):,} registros.")
    print(f"Fuegos positivos en test: {(df_test['fuego'] == 1).sum():,}")

    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    regiones = ["Chaco", "Oriental"]
    
    filas_metricas = []
    dfs_predicciones = []

    for region in regiones:
        print(f"\n--- EVALUANDO REGIONAL: {region.upper()} ---")
        nombre_lower = region.lower()
        ruta_b = DIR_MODELOS / f"lightgbm_{nombre_lower}.booster"
        ruta_c = DIR_MODELOS / "calibradores" / f"calibrador_{nombre_lower}.joblib"

        if not ruta_b.exists() or not ruta_c.exists():
            print(f"Aviso: No se encontraron los archivos de modelo serializados para {region}.")
            continue

        # 1. Cargar modelo y calibrador
        booster, calibrador = cargar_modelo_y_calibrador(ruta_b, ruta_c)

        # 2. Extraer matriz X e y del conjunto de test
        df_reg_test = df_test.filter(pl.col("region") == region)
        X_test, y_test, feats = preparar_matrices_xy(df_test, region)

        n_pos = int(y_test.sum())
        n_neg = int(len(y_test) - n_pos)
        print(f" • Muestras evaluadas: {len(y_test):,} (Positivos: {n_pos}, Negativos: {n_neg})")

        # 3. Predicción probabilística calibrada
        probs = predecir_probabilidad(booster, calibrador, X_test)

        # Guardar predicciones enriquecidas
        df_pred_sub = df_reg_test.with_columns([
            pl.Series("probabilidad_calibrada", probs)
        ])
        dfs_predicciones.append(df_pred_sub)

        # 4. Métricas de discriminación y calibración global
        auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        bs = float(calcular_brier_score(y_test, probs))
        bss = float(calcular_brier_skill_score(y_test, probs))

        print(f" • ROC-AUC:            {auc:.4f}")
        print(f" • PR-AUC:             {pr_auc:.4f}")
        print(f" • Brier Score:        {bs:.4f}")
        print(f" • Brier Skill Score:  {bss:.4f} (Ganancia respecto a la climatología)")

        # 5. Evaluación en 3 umbrales operativos
        c_youden, max_j = calcular_umbral_optimo_youden(y_test, probs)
        c_f05, max_f = calcular_umbral_optimo_f_beta(y_test, probs, beta=0.5)

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
                "umbral_estrategia": desc_umbral,
                "cutoff": round(cutoff, 3),
                "roc_auc": round(auc, 4),
                "pr_auc": round(pr_auc, 4),
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

    # 6. Consolidar tabla de resultados
    df_tabla = pd.DataFrame(filas_metricas)
    
    path_csv = DIR_TABLAS / "tabla02_metricas_desempeno_modelos.csv"
    path_md = DIR_TABLAS / "tabla02_metricas_desempeno_modelos.md"
    df_tabla.to_csv(path_csv, index=False)

    # Formatear tabla Markdown con estilo académico
    md_content = "# Tabla 2. Métricas de Rendimiento y Discriminación Operativa en Test Ciego\n\n"
    md_content += df_tabla.to_markdown(index=False) + "\n"
    path_md.write_text(md_content, encoding="utf-8")

    # Guardar predicciones completas en Parquet
    if dfs_predicciones:
        df_preds_total = pl.concat(dfs_predicciones)
        out_preds = DIR_PROCESSED / "evaluacion_test_predicciones.parquet"
        df_preds_total.write_parquet(out_preds, compression="zstd")
        print(f"\n✓ Predicciones individuales guardadas en: {out_preds.name}")

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
