"""
Script 06: Training and Monotonic Platt Calibration of Regional Models (LightGBM).
- Trains 2 independent specialized models: Chaco Model vs. Oriental Region Model.
- Hyperparameter tuning with Early Stopping on validation.
- Monotonic Platt calibration of probabilities P(Y=1) ∈ [0.0, 1.0].
- Interoperable serialization in modelos/*.booster (C++) and modelos/calibradores/*.joblib.
"""

from pathlib import Path
import sys

# Ensure the project root is on the PYTHONPATH
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
    print("ENTRENAMIENTO Y CALIBRACIÓN DUAL (IGNICIÓN & PROPAGACIÓN TÁCTICA)")
    print("=" * 65)

    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    rutas = {
        "ignicion": (
            dir_targets / "train_ignicion.parquet",
            dir_targets / "val_ignicion.parquet",
        ),
        "propagacion": (
            dir_targets / "train_propagacion.parquet",
            dir_targets / "val_propagacion.parquet",
        ),
    }
    for modo, (p_train, p_val) in rutas.items():
        for p in [p_train, p_val]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Archivo requerido no encontrado: {p}. "
                    "Ejecuta scripts/05d_muestrear_poblacional_d2.py primero."
                )

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    dir_calibradores = DIR_MODELOS / "calibradores"
    dir_calibradores.mkdir(parents=True, exist_ok=True)

    regiones = ["Chaco", "Oriental"]
    modos = [
        ("ignicion", "IGNICIÓN PRIMARIA (FÍSICA PURA & ECOTONO)"),
        ("propagacion", "PROPAGACIÓN TÁCTICA (CONTAGIO SATELITAL t-1)")
    ]
    modelos_resumen = {}

    for modo_clave, modo_nombre in modos:
        print(f"\n=======================================================")
        print(f"CAPA MODELIZADA: {modo_nombre}")
        print(f"=======================================================")

        p_train, p_val = rutas[modo_clave]
        df_train = pl.read_parquet(p_train)
        df_val = pl.read_parquet(p_val)

        for region in regiones:
            print(f"\n--- ENTRENANDO MODELO: {region.upper()} [{modo_clave.upper()}] ---")

            # 1. Prepare X and y matrices
            X_train, y_train, feats = preparar_matrices_xy(df_train, region, modo=modo_clave)
            X_val, y_val, _ = preparar_matrices_xy(df_val, region, modo=modo_clave)

            n_pos_train = int(y_train.sum())
            n_pos_val = int(y_val.sum())
            print(f" • Muestras Train: {len(X_train):,} ({n_pos_train} fuegos positivos)")
            print(f" • Muestras Val:   {len(X_val):,} ({n_pos_val} fuegos positivos)")
            print(f" • Features activas: {len(feats)} ({', '.join(feats[:5])}...)")

            if len(X_train) == 0 or n_pos_train == 0:
                print(f"Aviso: Sin datos positivos suficientes para entrenar {region}.")
                continue

            # 2. Train LightGBM with hyperparameters optimized per ecoregion
            print(" • Ajustando árboles de gradiente...")
            # Same capacity as scripts/16 and scripts/24 for the final M1/M2/M3
            # comparison. Do not lower this without changing the ablation too.
            lr = 0.03
            leaves = 23 if region == "Chaco" else 31
            spw = 1.0 if region == "Chaco" else 2.0
            n_rounds = 45 if region == "Chaco" else 60
            min_samples = 20 if region == "Chaco" else 10

            booster = entrenar_modelo_lightgbm(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                nombres_features=feats,
                scale_pos_weight=spw,
                learning_rate=lr,
                num_leaves=leaves,
                min_child_samples=min_samples,
                num_boost_round=n_rounds,
                early_stopping_rounds=None,
                semilla=42
            )
            print(f"   ✓ Iteraciones completadas: {booster.current_iteration()}")

            # 3. Monotonic Platt Probabilistic Calibration
            print(" • Ajustando calibrador sigmoide de Platt (monótono)...")
            calibrador = calibrar_probabilidades_isotonica(booster, X_val, y_val, metodo="platt_monotono")

            # 4. Inference and Evaluation on Validation
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

            # 5. Serialize
            nombre_lower = region.lower()
            ruta_b_especifica = DIR_MODELOS / f"lightgbm_{modo_clave}_{nombre_lower}.booster"
            ruta_c_especifica = dir_calibradores / f"calibrador_{modo_clave}_{nombre_lower}.joblib"
            guardar_modelo_y_calibrador(booster, calibrador, ruta_b_especifica, ruta_c_especifica)
            print(f"   ✓ Guardado: {ruta_b_especifica.name}")

            # Save calibration metadata and optimal thresholds derived strictly on validation (Anti-Leakage)
            meta_umbrales = {
                "region": region,
                "modo": modo_clave,
                "c_youden": round(float(c_youden), 4),
                "c_f05": round(float(c_f05), 4),
                "max_j_val": round(float(max_j), 4),
                "max_f05_val": round(float(max_f), 4),
                "roc_auc_val": round(float(auc_val), 4),
                "pr_auc_val": round(float(pr_auc_val), 4)
            }
            ruta_meta = dir_calibradores / f"umbrales_{modo_clave}_{nombre_lower}.json"
            import json
            ruta_meta.write_text(json.dumps(meta_umbrales, indent=2), encoding="utf-8")

            # If ignicion, also save under the standard name
            if modo_clave == "ignicion":
                ruta_b_std = DIR_MODELOS / f"lightgbm_{nombre_lower}.booster"
                ruta_c_std = dir_calibradores / f"calibrador_{nombre_lower}.joblib"
                guardar_modelo_y_calibrador(booster, calibrador, ruta_b_std, ruta_c_std)

            clave_resumen = f"{region}_{modo_clave}"
            modelos_resumen[clave_resumen] = {
                "region": region,
                "modo": modo_clave,
                "roc_auc": auc_val,
                "pr_auc": pr_auc_val,
                "bss": bss_val,
                "c_youden": c_youden,
                "c_f05": c_f05,
                "n_train": len(X_train),
                "n_val": len(X_val)
            }

    print("\n" + "=" * 65)
    print("RESUMEN CONSOLIDADO DE MODELOS DUALES")
    print("=" * 65)
    for k, m in modelos_resumen.items():
        print(f"[{m['region'].upper()} | {m['modo'].upper()}] ROC-AUC: {m['roc_auc']:.3f} | PR-AUC: {m['pr_auc']:.3f} | BSS: {m['bss']:.3f} | Corte F0.5: {m['c_f05']:.3f}")
    print("=" * 65)

if __name__ == "__main__":
    entrenar_pipeline()
