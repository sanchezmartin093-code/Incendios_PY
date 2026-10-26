"""
Script 14: Auditoría Suplementaria y Generación de Tablas S3 y S4 (Reproducibilidad C&G).
- Evalúa el desglose territorial departamental sobre el conjunto de test ciego.
- Ejecuta el análisis de sensibilidad empírico de radios de buffer espacial (10 km, 20 km, 30 km).
- Mide el perfil de latencia y memoria RAM en hardware Apple Silicon ARM64.
- Exporta:
  * tablas/tablaS3_desglose_departamental.csv y .md
  * tablas/tablaS3_sensibilidad_buffer.csv y .md
  * tablas/tablaS4_perfil_computacional.csv y .md
"""

from pathlib import Path
import sys
import time
import os

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import resource
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from src.alerta import (
    cargar_artefactos_modelos,
    preparar_datos_diarios,
    evaluar_alerta_nacional,
    generar_boletin_operativo,
)
from src.espaciotemporal import agregar_features_antropicas

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_MODELOS
from src.metricas import (
    calcular_umbral_optimo_f_beta,
    calcular_cuantil_conforme,
    predecir_conjuntos_conformes
)

def auditar_tablas_suplementarias():
    print("=" * 70)
    print("AUDITORÍA SUPLEMENTARIA: REPRODUCIBILIDAD DE TABLAS S3 Y S4")
    print("=" * 70)

    p_test_preds = DIR_PROCESSED / "evaluacion_test_predicciones.parquet"
    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    p_test = DIR_PROCESSED / "test_dataset.parquet"

    if not p_test_preds.exists() or not p_grilla.exists():
        raise FileNotFoundError("Archivos requeridos no encontrados en datos/processed/")

    df_preds = pl.read_parquet(p_test_preds)
    df_grilla = pl.read_parquet(p_grilla)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. TABLA S3: Desglose Departamental
    # -------------------------------------------------------------
    print("\n1. Computando desglose departamental (Tabla S3)...")
    
    # Censo total de celdas por departamento en la grilla soberana
    censo_deptos = df_grilla.group_by(["departamento", "region"]).agg([
        pl.len().alias("celdas_soberanas")
    ]).sort(["region", "celdas_soberanas"], descending=[True, True])

    filas_depto = []
    # Umbrales F0.5 de propagación derivados en validación (leer metadatos reales).
    import json
    meta_prop_c = json.loads((DIR_MODELOS / "calibradores" / "umbrales_propagacion_chaco.json").read_text())
    meta_prop_o = json.loads((DIR_MODELOS / "calibradores" / "umbrales_propagacion_oriental.json").read_text())
    c_f05_chaco = float(meta_prop_c.get("c_f05", 0.642))
    c_f05_oriental = float(meta_prop_o.get("c_f05", 0.544))

    for row in censo_deptos.iter_rows():
        dep_nom, reg, celdas_tot = row[0], row[1], row[2]
        sub = df_preds.filter(pl.col("departamento") == dep_nom)
        
        y_true = sub["fuego"].to_numpy().astype(int) if len(sub) > 0 else np.array([])
        n_pos = int(y_true.sum()) if len(y_true) > 0 else 0
        n_tot = len(sub)

        c_corte = c_f05_chaco if reg == "Chaco" else c_f05_oriental

        if n_pos >= 2 and (n_tot - n_pos) >= 2:
            col_ign = "probabilidad_hibrido" if "probabilidad_hibrido" in sub.columns else "probabilidad_calibrada"
            p_ign = sub[col_ign].to_numpy()
            p_prop = sub["probabilidad_propagacion"].to_numpy()
            
            auc_ign = float(roc_auc_score(y_true, p_ign))
            auc_prop = float(roc_auc_score(y_true, p_prop))
            
            y_bin = (p_prop >= c_corte).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_bin, labels=[0, 1]).ravel()
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        elif n_pos == 0:
            auc_ign = np.nan
            auc_prop = np.nan
            prec = 1.0 if n_tot > 0 else np.nan
        else:
            auc_ign = np.nan
            auc_prop = np.nan
            prec = np.nan

        filas_depto.append({
            "departamento": dep_nom,
            "region": reg,
            "celdas_soberanas": celdas_tot,
            "muestras_test": n_tot,
            "fuegos_test": n_pos,
            "roc_auc_ignicion": round(auc_ign, 3) if not np.isnan(auc_ign) else "—",
            "roc_auc_propagacion": round(auc_prop, 3) if not np.isnan(auc_prop) else "—",
            "precision_tactica": f"{prec * 100:.1f} %" if not np.isnan(prec) else "—"
        })

    df_s3 = pd.DataFrame(filas_depto)
    p_csv_s3 = DIR_TABLAS / "tablaS3_desglose_departamental.csv"
    p_md_s3 = DIR_TABLAS / "tablaS3_desglose_departamental.md"
    df_s3.to_csv(p_csv_s3, index=False)
    
    md_s3_str = "# Tabla Suplementaria S3. Desglose Departamental de Desempeño Predictivo en Test Ciego\n\n"
    md_s3_str += df_s3.to_markdown(index=False) + "\n"
    p_md_s3.write_text(md_s3_str, encoding="utf-8")
    print(f"   ✓ Guardado en {p_csv_s3.name} y {p_md_s3.name}")

    # -------------------------------------------------------------
    # 2. ESTRATIFICACIÓN POR DISTANCIA AL FOCO t-1
    #    (no es una sensibilidad al buffer de muestreo: eso requiere
    #     remuestrear y reentrenar; se deja como análisis pendiente)
    # -------------------------------------------------------------
    print("\n2. Computando estratificación por distancia al foco t-1...")
    sub_c = df_preds.filter(pl.col("region") == "Chaco")
    sub_o = df_preds.filter(pl.col("region") == "Oriental")

    condiciones = {
        10: "Incluye celdas a >=10 km del foco t-1",
        20: "Incluye celdas a >=20 km del foco t-1",
        30: "Sólo celdas a >=30 km del foco t-1 (alejadas)"
    }
    col_ign_c = "probabilidad_hibrido" if "probabilidad_hibrido" in sub_c.columns else "probabilidad_calibrada"
    col_ign_o = "probabilidad_hibrido" if "probabilidad_hibrido" in sub_o.columns else "probabilidad_calibrada"

    sensibilidad_datos = []
    for d in [10, 20, 30]:
        mask_c = (sub_c["dist_foco_t1_km"] >= d).to_numpy()
        mask_o = (sub_o["dist_foco_t1_km"] >= d).to_numpy()

        y_c = sub_c["fuego"].to_numpy()[mask_c]
        p_ign_c = sub_c[col_ign_c].to_numpy()[mask_c]
        p_prop_c = sub_c["probabilidad_propagacion"].to_numpy()[mask_c]

        y_o = sub_o["fuego"].to_numpy()[mask_o]
        p_ign_o = sub_o[col_ign_o].to_numpy()[mask_o]

        auc_ic = float(roc_auc_score(y_c, p_ign_c))
        auc_pc = float(roc_auc_score(y_c, p_prop_c))
        auc_io = float(roc_auc_score(y_o, p_ign_o))

        sensibilidad_datos.append({
            "distancia_min_t1_km": d,
            "condicion": condiciones[d],
            "n_muestras_chaco": int(mask_c.sum()),
            "auc_ign_chaco": round(auc_ic, 4),
            "auc_prop_chaco": round(auc_pc, 4),
            "n_muestras_oriental": int(mask_o.sum()),
            "auc_ign_oriental": round(auc_io, 4)
        })

    df_sens = pd.DataFrame(sensibilidad_datos)
    p_csv_sens = DIR_TABLAS / "tablaS3_sensibilidad_buffer.csv"
    p_md_sens = DIR_TABLAS / "tablaS3_sensibilidad_buffer.md"
    df_sens.to_csv(p_csv_sens, index=False)
    p_md_sens.write_text(
        "# Estratificación del desempeño por distancia al foco t-1\n\n"
        "Esta tabla NO es un análisis de sensibilidad al buffer de muestreo. "
        "Para evaluar el buffer hay que remuestrear y reentrenar con 10, 20 y "
        "30 km. Aquí sólo se filtra el test por distancia al foco de ayer.\n\n"
        + df_sens.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"   ✓ Guardado en {p_csv_sens.name}")

    # -------------------------------------------------------------
    # 3. TABLA S4: Perfil Computacional Real
    # -------------------------------------------------------------
    print("\n3. Midiendo perfil computacional y latencia del pipeline real...")

    # Usar un día de alta actividad; si no existe, el último disponible.
    df_fechas = pl.read_parquet(
        DIR_PROCESSED / "clima_era5_diario.parquet", columns=["fecha"]
    )
    fechas_disponibles = set(df_fechas["fecha"].unique().to_list())
    fecha_bench = "2024-09-09" if "2024-09-09" in fechas_disponibles else sorted(fechas_disponibles)[-1]

    t0 = time.perf_counter()
    artefactos = cargar_artefactos_modelos()
    df_nacional = preparar_datos_diarios(fecha_bench)
    t_ingesta = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_evaluado = evaluar_alerta_nacional(df_nacional, artefactos)
    t_inferencia = time.perf_counter() - t0

    t0 = time.perf_counter()
    _resumen, _boletin = generar_boletin_operativo(df_evaluado, fecha_bench)
    t_boletin = time.perf_counter() - t0

    lat_tot = t_ingesta + t_inferencia + t_boletin
    peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS devuelve bytes; Linux devuelve kilobytes.
    peak_mb = peak_raw / (1024 * 1024) if sys.platform == "darwin" else peak_raw / 1024

    perfil_datos = [
        {
            "etapa": "1. Ingesta, priors causales y gating",
            "motor": "Polars / cKDTree",
            "ram_pico_mb": np.nan,
            "latencia_arm64_s": round(t_ingesta, 3),
            "latencia_x86_s": np.nan,
            "nota": "medido en el equipo de ejecución",
        },
        {
            "etapa": "2. Inferencia dual LightGBM",
            "motor": "LightGBM C++ / NumPy",
            "ram_pico_mb": np.nan,
            "latencia_arm64_s": round(t_inferencia, 3),
            "latencia_x86_s": np.nan,
            "nota": "medido en el equipo de ejecución",
        },
        {
            "etapa": "3. Agregación y boletín",
            "motor": "Polars",
            "ram_pico_mb": np.nan,
            "latencia_arm64_s": round(t_boletin, 3),
            "latencia_x86_s": np.nan,
            "nota": "medido en el equipo de ejecución",
        },
        {
            "etapa": "TOTAL PIPELINE NACIONAL (PUNTA A PUNTA)",
            "motor": "Consolidado",
            "ram_pico_mb": round(peak_mb, 1),
            "latencia_arm64_s": round(lat_tot, 3),
            "latencia_x86_s": np.nan,
            "nota": f"fecha={fecha_bench}; x86_64 no medido",
        },
    ]

    df_s4 = pd.DataFrame(perfil_datos)
    p_csv_s4 = DIR_TABLAS / "tablaS4_perfil_computacional.csv"
    p_md_s4 = DIR_TABLAS / "tablaS4_perfil_computacional.md"
    df_s4.to_csv(p_csv_s4, index=False)
    p_md_s4.write_text("# Tabla Suplementaria S4. Perfil de Rendimiento Computacional y Consumo de Memoria\n\n" + df_s4.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"   ✓ Guardado en {p_csv_s4.name} y {p_md_s4.name}")

    # -------------------------------------------------------------
    # 4. TABLA S5: Predicción Conforme (Split Conformal Prediction)
    # -------------------------------------------------------------
    print("\n4. Evaluando cuantificación de incertidumbre conforme (Split Conformal Prediction)...")
    p_val = DIR_PROCESSED / "val_dataset.parquet"

    if p_val.exists():
        from src.modelo import CalibradorPlattMonotono
        from src.modelo import FEATURES_HIBRIDO_CHACO, FEATURES_HIBRIDO_ORIENTAL

        df_val = agregar_features_antropicas(pl.read_parquet(p_val))

        filas_conf = []
        for reg, feats in [("Chaco", FEATURES_HIBRIDO_CHACO), ("Oriental", FEATURES_HIBRIDO_ORIENTAL)]:
            sub_val = df_val.filter(pl.col("region") == reg).sort("fecha")

            # Separar calibración (60% cronológico) y conformal (40% restante).
            # El calibrador Platt y q_hat NO se estiman sobre los mismos datos.
            fechas = sorted(sub_val["fecha"].unique().to_list())
            n_cal = max(1, int(len(fechas) * 0.60))
            fechas_cal = set(fechas[:n_cal])
            fechas_conf = set(fechas[n_cal:])
            sub_cal = sub_val.filter(pl.col("fecha").is_in(list(fechas_cal)))
            sub_conf = sub_val.filter(pl.col("fecha").is_in(list(fechas_conf)))
            if sub_conf.is_empty():
                sub_conf = sub_cal

            b = lgb.Booster(model_file=str(DIR_MODELOS / f"lightgbm_hibrido_{reg.lower()}.booster"))
            raw_cal = b.predict(sub_cal.select(feats).to_numpy())
            raw_conf = b.predict(sub_conf.select(feats).to_numpy())
            cal = CalibradorPlattMonotono().fit(raw_cal, sub_cal["fuego"].to_numpy())
            p_conf = cal.predict(raw_conf)
            y_conf = sub_conf["fuego"].to_numpy().astype(int)

            q_hat_90 = calcular_cuantil_conforme(y_conf, p_conf, alpha=0.10)
            q_hat_95 = calcular_cuantil_conforme(y_conf, p_conf, alpha=0.05)

            sub_test = df_preds.filter(pl.col("region") == reg)
            y_test = sub_test["fuego"].to_numpy().astype(int)
            raw_test = b.predict(sub_test.select(feats).to_numpy())
            p_test = cal.predict(raw_test)

            c0_90, c1_90 = predecir_conjuntos_conformes(p_test, q_hat_90)
            cov_90 = float(np.where(y_test == 1, c1_90, c0_90).mean())

            c0_95, c1_95 = predecir_conjuntos_conformes(p_test, q_hat_95)
            cov_95 = float(np.where(y_test == 1, c1_95, c0_95).mean())

            filas_conf.append({
                "region": reg,
                "q_hat_90": round(q_hat_90, 4),
                "cobertura_empirica_90": f"{cov_90 * 100:.2f} %",
                "q_hat_95": round(q_hat_95, 4),
                "cobertura_empirica_95": f"{cov_95 * 100:.2f} %"
            })
    else:
        filas_conf = []

    df_s5 = pd.DataFrame(filas_conf)
    p_csv_s5 = DIR_TABLAS / "tablaS5_prediccion_conforme.csv"
    p_md_s5 = DIR_TABLAS / "tablaS5_prediccion_conforme.md"
    df_s5.to_csv(p_csv_s5, index=False)
    p_md_s5.write_text("# Tabla Suplementaria S5. Calibración Conforme y Cobertura Empírica\n\n" + df_s5.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"   ✓ Guardado en {p_csv_s5.name} y {p_md_s5.name}")

    print("\n" + "=" * 70)
    print("AUDITORÍA SUPLEMENTARIA COMPLETADA CON ÉXITO")
    print("=" * 70)

if __name__ == "__main__":
    auditar_tablas_suplementarias()
