"""
Script 06: Entrenamiento y Calibración Isotónica de Modelos Regionales (LightGBM).
- Entrena 2 modelos especializados independientes: Modelo Chaco vs. Modelo Región Oriental.
- Ajuste de hiperparámetros con Early Stopping en validación.
- Calibración Isotónica de probabilidades P(Y=1) ∈ [0.0, 1.0].
- Serialización interoperable en modelos/*.booster (C++) y modelos/calibradores/*.joblib.
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score
from src.constantes import DIR_PROCESSED, DIR_MODELOS
from src.modelo import (
    preparar_matrices_xy,
    entrenar_modelo_lightgbm,
    calibrar_probabilidades_isotonica,
    predecir_probabilidad,
    guardar_modelo_y_calibrador
)
from src.metricas import (
    calcular_umbral_optimo_youden,
    calcular_umbral_optimo_f_beta,
    calcular_brier_skill_score
)

def entrenar_pipeline():
    print("=" * 65)
    print("ENTRENAMIENTO Y CALIBRACIÓN REGIONAL LIGHTGBM (CHACO & ORIENTAL)")
    print("=" * 65)

    p_train = DIR_PROCESSED / "train_dataset.parquet"
    p_val = DIR_PROCESSED / "val_dataset.parquet"

    for p in [p_train, p_val]:
        if not p.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {p}. Ejecuta scripts/05_muestrear_ausencias.py primero.")

    df_train = pl.read_parquet(p_train)
    df_val = pl.read_parquet(p_val)

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    dir_calibradores = DIR_MODELOS / "calibradores"
    dir_calibradores.mkdir(parents=True, exist_ok=True)

    regiones = ["Chaco", "Oriental"]
    modelos_resumen = {}

    for region in regiones:
        print(f"\n--- ENTRENANDO MODELO: {region.upper()} ---")
        
        # 1. Preparar matrices X e y
        X_train, y_train, feats = preparar_matrices_xy(df_train, region)
        X_val, y_val, _ = preparar_matrices_xy(df_val, region)

        n_pos_train = int(y_train.sum())
        n_pos_val = int(y_val.sum())
        print(f" • Muestras Train: {len(X_train):,} ({n_pos_train} fuegos positivos)")
        print(f" • Muestras Val:   {len(X_val):,} ({n_pos_val} fuegos positivos)")
        print(f" • Features activas: {len(feats)}")

        if len(X_train) == 0 or n_pos_train == 0:
            print(f"Aviso: Sin datos positivos suficientes para entrenar {region}.")
            continue

        # 2. Entrenar LightGBM
        print(" • Ajustando árboles de gradiente (Early Stopping)...")
        booster = entrenar_modelo_lightgbm(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            nombres_features=feats,
            scale_pos_weight=10.0,
            num_boost_round=150,
            early_stopping_rounds=15,
            semilla=42
        )
        print(f"   ✓ Mejor iteración alcanzada: {booster.best_iteration}")

        # 3. Calibración Isotónica
        print(" • Ajustando calibrador isotónico de probabilidades...")
        calibrador = calibrar_probabilidades_isotonica(booster, X_val, y_val)

        # 4. Inferencia y Evaluación en Validación
        probs_val = predecir_probabilidad(booster, calibrador, X_val)
        
        auc_val = float(roc_auc_score(y_val, probs_val)) if n_pos_val > 0 else 0.0
        pr_auc_val = float(average_precision_score(y_val, probs_val)) if n_pos_val > 0 else 0.0
        bss_val = float(calcular_brier_skill_score(y_val, probs_val)) if n_pos_val > 0 else 0.0

        c_youden, max_j = calcular_umbral_optimo_youden(y_val, probs_val)
        c_f05, max_f = calcular_umbral_optimo_f_beta(y_val, probs_val, beta=0.5)

        print(f" • Métricas en Validación:")
        print(f"   - ROC-AUC:            {auc_val:.4f}")
        print(f"   - PR-AUC:             {pr_auc_val:.4f}")
        print(f"   - Brier Skill Score:  {bss_val:.4f}")
        print(f"   - Umbral Youden (J):  {c_youden:.3f} (J={max_j:.3f})")
        print(f"   - Umbral F0.5 (Prec): {c_f05:.3f} (F0.5={max_f:.3f})")

        # 5. Serializar
        nombre_lower = region.lower()
        ruta_b = DIR_MODELOS / f"lightgbm_{nombre_lower}.booster"
        ruta_c = dir_calibradores / f"calibrador_{nombre_lower}.joblib"
        guardar_modelo_y_calibrador(booster, calibrador, ruta_b, ruta_c)
        print(f"   ✓ Modelo exportado: {ruta_b.name} ({ruta_b.stat().st_size / 1024:.1f} KB)")
        print(f"   ✓ Calibrador exportado: {ruta_c.name}")

        modelos_resumen[region] = {
            "roc_auc": auc_val,
            "pr_auc": pr_auc_val,
            "bss": bss_val,
            "c_youden": c_youden,
            "c_f05": c_f05,
            "n_train": len(X_train),
            "n_val": len(X_val)
        }

    print("\n" + "=" * 65)
    print("RESUMEN CONSOLIDADO DE MODELOS REGIONALES")
    print("=" * 65)
    for reg, m in modelos_resumen.items():
        print(f"[{reg.upper()}] ROC-AUC: {m['roc_auc']:.3f} | PR-AUC: {m['pr_auc']:.3f} | BSS: {m['bss']:.3f} | Corte F0.5: {m['c_f05']:.3f}")
    print("=" * 65)

if __name__ == "__main__":
    entrenar_pipeline()
