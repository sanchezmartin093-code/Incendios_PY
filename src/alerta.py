"""
Módulo alerta.py: Servicio Operativo de Alerta Temprana de Incendios Forestales para Paraguay.
Filosofía de Producción:
- Arquitectura Two-Stage Bio-physical Gating:
  * Etapa 1: Filtro biofísico O(1). Descarta de plano celdas no combustibles (agua, urbano),
    precipitaciones extintoras (P_48h >= 10 mm) o atmósferas saturadas (VPD < 0.8 kPa).
  * Etapa 2: Inferencia de ensamble discriminante regional (LightGBM Booster C++) calibrado
    monótonamente (Platt) P(Fuego|X) ∈ [0.0, 1.0].
  * Fallback M1: en celdas sin historia de fuego (`freq_hist_1km = 0`) se usa el
    modelo biofísico M1; M3 se reserva para celdas con recurrencia observada.
- Clasificación de riesgo operativa según umbrales de máxima precisión (F0.5):
  * NULO: Sin material combustible vegetal (agua o casco urbano consolidado).
  * BAJO: Condiciones subcríticas (P < c*).
  * MODERADO: Ignición probable bajo fuentes activas (c* <= P < c* + 0.25*(1 - c*)).
  * ALTO: Propagación rápida sostenida (c* + 0.25*(1 - c*) <= P < c* + 0.55*(1 - c*)).
  * EXTREMO: Condiciones críticas de tormenta de fuego (P >= c* + 0.55*(1 - c*)).
- Despacho y serialización de alertas en Apache Arrow / Parquet, JSON y Markdown.
"""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import polars as pl
import lightgbm as lgb
import joblib

from src.constantes import (
    DIR_PROCESSED,
    DIR_MODELOS,
    DIR_OUTPUTS,
    UMBRAL_VPD_MINIMO_KPA,
    UMBRAL_LLUVIA_48H_MM
)
from src.modelo import (
    FEATURES_PROPAGACION_CHACO,
    FEATURES_PROPAGACION_ORIENTAL,
    predecir_probabilidad,
)
from src.metricas import corregir_prior_probabilidades
from src.espaciotemporal import (
    agregar_features_antropicas,
    enriquecer_dataset_espaciotemporal,
)

# Prevalencia de la muestra case-control 1:10 usada para calibrar Platt.
PREVALENCIA_MUESTREO = 1.0 / 11.0

# Umbrales óptimos derivados estrictamente en validación out-of-time (F0.5 de máxima precisión anti-falsas alarmas)
UMBRAL_CORTE_CHACO = 0.399       # Ignición Chaco
UMBRAL_CORTE_ORIENTAL = 0.273    # Ignición Oriental
UMBRAL_PROP_CHACO = 0.642        # Propagación Chaco
UMBRAL_PROP_ORIENTAL = 0.544     # Propagación Oriental

def _cargar_prevalencias_poblacionales() -> dict:
    """
    Carga la prevalencia poblacional estimada sobre celdas-día candidatas.

    El archivo lo genera scripts/19_estimar_prevalencias.py. Si no existe, se
    devuelve un diccionario vacío y el servicio mantiene las probabilidades
    case-control (con una advertencia explícita).
    """
    ruta_d2 = DIR_MODELOS / "calibradores" / "prevalencias_poblacionales_d2.json"
    if ruta_d2.exists():
        try:
            data = json.loads(ruta_d2.read_text(encoding="utf-8"))
            tasks = data.get("tasks", {})
            return {
                str(task).lower(): {
                    str(k).lower(): v for k, v in regiones.items()
                    if isinstance(v, dict)
                }
                for task, regiones in tasks.items()
            }
        except Exception:
            pass

    ruta = DIR_MODELOS / "calibradores" / "prevalencias_poblacionales.json"
    if not ruta.exists():
        return {}
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        regiones = data.get("regiones", data)
        norm = {
            str(k).lower(): v for k, v in regiones.items()
            if isinstance(v, dict)
        }
        return {"ignicion": norm, "propagacion": norm}
    except Exception:
        return {}


def _corregir_cutoff(cutoff_muestreo: float, prevalencia_poblacion: float | None) -> float:
    if not prevalencia_poblacion:
        return float(cutoff_muestreo)
    return float(
        corregir_prior_probabilidades(
            np.array([cutoff_muestreo]),
            PREVALENCIA_MUESTREO,
            prevalencia_poblacion,
        )[0]
    )


def cargar_artefactos_modelos() -> dict:
    """Carga los ensambles nativos C++ y los calibradores duales (Ignición/Híbrido y Propagación)."""
    artefactos = {}
    prevalencias = _cargar_prevalencias_poblacionales()
    for region in ["chaco", "oriental"]:
        # 1. Capa de Ignición Primaria (Prioriza Modelo Híbrido M3 si existe, con fallback a M2)
        ruta_b_hibrido = DIR_MODELOS / f"lightgbm_hibrido_{region}.booster"
        ruta_c_hibrido = DIR_MODELOS / "calibradores" / f"calibrador_hibrido_{region}.joblib"
        
        ruta_b_ign = DIR_MODELOS / f"lightgbm_ignicion_{region}.booster"
        ruta_c_ign = DIR_MODELOS / "calibradores" / f"calibrador_ignicion_{region}.joblib"
        
        es_hibrido = ruta_b_hibrido.exists() and ruta_c_hibrido.exists()
        if es_hibrido:
            ruta_booster_final = ruta_b_hibrido
            ruta_cal_final = ruta_c_hibrido
        elif ruta_b_ign.exists() and ruta_c_ign.exists():
            ruta_booster_final = ruta_b_ign
            ruta_cal_final = ruta_c_ign
        else:
            ruta_booster_final = DIR_MODELOS / f"lightgbm_{region}.booster"
            ruta_cal_final = DIR_MODELOS / "calibradores" / f"calibrador_{region}.joblib"

        # 2. Capa de Propagación Táctica
        ruta_b_prop = DIR_MODELOS / f"lightgbm_propagacion_{region}.booster"
        ruta_c_prop = DIR_MODELOS / "calibradores" / f"calibrador_propagacion_{region}.joblib"

        booster_ign = lgb.Booster(model_file=str(ruta_booster_final))
        calibrador_ign = joblib.load(ruta_cal_final)

        booster_prop = lgb.Booster(model_file=str(ruta_b_prop)) if ruta_b_prop.exists() else None
        calibrador_prop = joblib.load(ruta_c_prop) if ruta_c_prop.exists() else None

        ruta_b_m1 = DIR_MODELOS / f"lightgbm_m1_{region}.booster"
        ruta_c_m1 = DIR_MODELOS / "calibradores" / f"calibrador_m1_{region}.joblib"
        booster_m1 = lgb.Booster(model_file=str(ruta_b_m1)) if ruta_b_m1.exists() else None
        calibrador_m1 = joblib.load(ruta_c_m1) if ruta_c_m1.exists() else None

        # Cargar umbrales dinámicos si existen en json
        if es_hibrido:
            ruta_meta_hibrido = DIR_MODELOS / "calibradores" / f"umbrales_hibrido_{region}.json"
            cutoff_ign = UMBRAL_CORTE_CHACO if region == "chaco" else UMBRAL_CORTE_ORIENTAL
            if ruta_meta_hibrido.exists():
                try:
                    meta_h = json.loads(ruta_meta_hibrido.read_text(encoding="utf-8"))
                    cutoff_ign = meta_h.get("c_f05", cutoff_ign)
                except Exception:
                    pass
        else:
            ruta_meta_ign = DIR_MODELOS / "calibradores" / f"umbrales_ignicion_{region}.json"
            cutoff_ign = UMBRAL_CORTE_CHACO if region == "chaco" else UMBRAL_CORTE_ORIENTAL
            if ruta_meta_ign.exists():
                try:
                    meta_j = json.loads(ruta_meta_ign.read_text(encoding="utf-8"))
                    cutoff_ign = meta_j.get("c_f05", cutoff_ign)
                except Exception:
                    pass

        ruta_meta_prop = DIR_MODELOS / "calibradores" / f"umbrales_propagacion_{region}.json"
        cutoff_prop = UMBRAL_PROP_CHACO if region == "chaco" else UMBRAL_PROP_ORIENTAL
        if ruta_meta_prop.exists():
            try:
                meta_p = json.loads(ruta_meta_prop.read_text(encoding="utf-8"))
                cutoff_prop = meta_p.get("c_f05", cutoff_prop)
            except Exception:
                pass

        # Pasar los umbrales (y las probabilidades) de la escala case-control a
        # la escala poblacional. Sin esta corrección, un corte de 0,40 en una
        # muestra con 9,09 % de prevalencia marca una fracción enorme del país.
        prev_ign = prevalencias.get("ignicion", {}).get(region, {}).get("global")
        prev_prop = prevalencias.get("propagacion", {}).get(region, {}).get("global")
        if not prev_ign or not prev_prop:
            print(
                f"[ADVERTENCIA] Faltan prevalencias D2 para {region}. "
                "Las probabilidades y umbrales quedan en escala case-control; "
                "ejecute scripts/19_estimar_prevalencias.py."
            )
        cutoff_ign = _corregir_cutoff(cutoff_ign, prev_ign)
        cutoff_prop = _corregir_cutoff(cutoff_prop, prev_prop)

        cutoff_m1 = cutoff_ign
        if booster_m1 is not None:
            ruta_meta_m1 = DIR_MODELOS / "calibradores" / f"umbrales_m1_{region}.json"
            if ruta_meta_m1.exists():
                try:
                    meta_m1 = json.loads(ruta_meta_m1.read_text(encoding="utf-8"))
                    cutoff_m1 = meta_m1.get("c_f05", cutoff_m1)
                except Exception:
                    pass
            cutoff_m1 = _corregir_cutoff(cutoff_m1, prev_ign)

        features_modelo = booster_ign.feature_name()

        artefactos[region] = {
            "booster_ign": booster_ign,
            "calibrador_ign": calibrador_ign,
            "features_ign": features_modelo,
            "cutoff_ign": cutoff_ign,
            "booster_prop": booster_prop,
            "calibrador_prop": calibrador_prop,
            "booster_m1": booster_m1,
            "calibrador_m1": calibrador_m1,
            "features_m1": booster_m1.feature_name() if booster_m1 is not None else None,
            "cutoff_m1": cutoff_m1,
            "features_prop": FEATURES_PROPAGACION_CHACO if region == "chaco" else FEATURES_PROPAGACION_ORIENTAL,
            "cutoff_prop": cutoff_prop,
            "es_hibrido": es_hibrido,
            "prevalencia_muestreo": PREVALENCIA_MUESTREO,
            "prevalencia_ignicion": prev_ign,
            "prevalencia_propagacion": prev_prop,
            # Compatibilidad
            "booster": booster_ign,
            "calibrador": calibrador_ign,
            "features": features_modelo,
            "cutoff_f05": cutoff_ign
        }
    return artefactos

def categorizar_riesgo(
    prob: float,
    region: str,
    es_combustible: bool,
    gating_motivo: str | None = None,
    cutoff: float | None = None
) -> str:
    """Asigna la categoría de riesgo operativa basada en la probabilidad calibrada y la física."""
    if not es_combustible:
        return "NULO"
    if gating_motivo is not None and gating_motivo != "":
        return "BAJO"

    if cutoff is None:
        cutoff = UMBRAL_CORTE_CHACO if region.lower() == "chaco" else UMBRAL_CORTE_ORIENTAL
    
    if prob < cutoff:
        return "BAJO"
    else:
        # Escalonamiento dinámico sobre el margen activo [cutoff, 1.0]
        margen = 1.0 - cutoff
        umbral_mod = cutoff + margen * 0.25
        umbral_alto = cutoff + margen * 0.55
        if prob < umbral_mod:
            return "MODERADO"
        elif prob < umbral_alto:
            return "ALTO"
        else:
            return "EXTREMO"

def categorizar_riesgo_vectorizado(
    probs: np.ndarray,
    cutoff: float,
    modo: str = "ignicion"
) -> np.ndarray:
    """
    Categorización vectorizada ultra-rápida de riesgo operativo O(1) vía NumPy.
    Escala dinámicamente sobre el margen activo [cutoff, 1.0] garantizando
    coherencia total con la función categorizar_riesgo() escalar.
    """
    margen = max(1.0 - cutoff, 1e-6)
    if modo == "propagacion":
        umbral_extremo = cutoff + margen * 0.40
        condiciones = [
            probs < cutoff,
            (probs >= cutoff) & (probs < umbral_extremo),
            probs >= umbral_extremo
        ]
        elecciones = ["BAJO", "ALTO", "EXTREMO"]
        return np.select(condiciones, elecciones, default="BAJO")
    else:
        umbral_mod = cutoff + margen * 0.25
        umbral_alto = cutoff + margen * 0.55
        condiciones = [
            probs < cutoff,
            (probs >= cutoff) & (probs < umbral_mod),
            (probs >= umbral_mod) & (probs < umbral_alto),
            probs >= umbral_alto
        ]
        elecciones = ["BAJO", "MODERADO", "ALTO", "EXTREMO"]
        return np.select(condiciones, elecciones, default="BAJO")

def consolidar_nivel_alerta(nivel_ign: np.ndarray, nivel_prop: np.ndarray) -> np.ndarray:
    """Consolida la jerarquía de alerta operativa máxima: EXTREMO > ALTO > MODERADO > BAJO."""
    es_extremo = (nivel_ign == "EXTREMO") | (nivel_prop == "EXTREMO")
    es_alto = ((nivel_ign == "ALTO") | (nivel_prop == "ALTO")) & ~es_extremo
    es_moderado = ((nivel_ign == "MODERADO") | (nivel_prop == "MODERADO")) & ~es_extremo & ~es_alto
    
    condiciones = [es_extremo, es_alto, es_moderado]
    elecciones = ["EXTREMO", "ALTO", "MODERADO"]
    return np.select(condiciones, elecciones, default="BAJO")

def preparar_datos_diarios(fecha: str) -> pl.DataFrame:
    """
    Ensambla la matriz nacional de 1 km² para la fecha especificada uniendo
    grilla geográfica, biomasa MapBiomas y meteorología diaria ERA5.
    """
    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    p_biomasa = DIR_PROCESSED / "combustible_mapbiomas_1km.parquet"
    p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"

    if not p_grilla.exists() or not p_biomasa.exists() or not p_clima.exists():
        raise FileNotFoundError("Archivos base procesados no encontrados en datos/processed/")

    grilla = pl.read_parquet(p_grilla)
    biomasa = pl.read_parquet(p_biomasa)
    clima = pl.read_parquet(p_clima).filter(pl.col("fecha") == fecha)

    if clima.height == 0:
        raise ValueError(f"No hay registros climáticos para la fecha {fecha} en clima_era5_diario.parquet")

    # 1. Unir grilla y biomasa por cell_id
    df_nacional = grilla.join(biomasa, on="cell_id", how="left")
    if "origen_datos" in df_nacional.columns:
        origenes = set(df_nacional["origen_datos"].drop_nulls().unique().to_list())
        if origenes == {"sintetico_referencia"}:
            print(
                "[ADVERTENCIA] El perfil de combustible es SINTÉTICO "
                "(sintetico_referencia). No use estas alertas como producto "
                "biofísico operativo hasta sustituirlo por MapBiomas real."
            )

    # 2. Unir clima por nodo_clima_id
    df_nacional = df_nacional.join(clima, on="nodo_clima_id", how="left")

    # 3. Derivar variables de calendario y biofísicas no lineales
    dt = datetime.strptime(fecha, "%Y-%m-%d")
    df_nacional = df_nacional.with_columns([
        pl.lit(dt.weekday()).alias("dia_semana"),
        pl.lit(1 if dt.weekday() in [5, 6] else 0).alias("es_fin_de_semana"),
        pl.lit(dt.month).alias("mes"),
        pl.lit(dt.timetuple().tm_yday).alias("dia_del_anio"),
        (pl.col("temp_max_c") - pl.col("temp_min_c")).alias("rango_termico_diario"),
        (pl.col("vpd_max_kpa") * (pl.col("dias_secos_consecutivos") + 1.0)).alias("deficit_hidrico_acumulado"),
        ((pl.col("viento_max_kmh") / 10.0).pow(2)).alias("potencial_viento_cuadratico")
    ])

    # 4. Enriquecimiento espaciotemporal causal.
    # Los priors se calculan con los focos estrictamente anteriores al primer
    # día del mes de la fecha de alerta. No se usa el archivo estático
    # priors_socioecologicos_1km.parquet, que podía quedar desactualizado.
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    if not path_focos.exists():
        raise FileNotFoundError(
            f"Archivo de focos no encontrado: {path_focos}. "
            "Ejecuta scripts/01b_descargar_firms_api.py y scripts/05_muestrear_ausencias.py."
        )
    df_focos = pl.read_parquet(path_focos)
    fecha_corte = f"{dt.year:04d}-{dt.month:02d}-01"
    df_nacional = enriquecer_dataset_espaciotemporal(
        df_nacional,
        df_focos,
        usar_priors_causales=True,
        fecha_corte=fecha_corte,
    )

    # Variables de calendario estacional armónico y temporal para el modelo híbrido (M3)
    df_nacional = agregar_features_antropicas(df_nacional)

    return df_nacional

def evaluar_alerta_nacional(df_nacional: pl.DataFrame, artefactos: dict) -> pl.DataFrame:
    """
    Aplica el Two-Stage Bio-physical Gating y la inferencia discriminante calibrada
    a todas las 400.006 celdas del territorio nacional.
    """
    # 1. Etapa 1: Filtro Biofísico (Gating)
    # Condiciones de descarte inmediato
    cond_sin_combustible = ~pl.col("es_combustible")
    cond_lluvia_extintora = pl.col("precipitacion_48h_mm") >= UMBRAL_LLUVIA_48H_MM
    cond_atmosfera_humeda = pl.col("vpd_max_kpa") < UMBRAL_VPD_MINIMO_KPA

    df_gating = df_nacional.with_columns([
        pl.when(cond_sin_combustible).then(pl.lit("SIN_COMBUSTIBLE"))
        .when(cond_lluvia_extintora).then(pl.lit("LLUVIA_EXTINTORA_48H"))
        .when(cond_atmosfera_humeda).then(pl.lit("VPD_INSUFICIENTE"))
        .otherwise(pl.lit("CANDIDATO_ACTIVO"))
        .alias("estado_gating")
    ])

    # Separar celdas activas vs. descartadas
    df_activos = df_gating.filter(pl.col("estado_gating") == "CANDIDATO_ACTIVO")
    df_descartados = df_gating.filter(pl.col("estado_gating") != "CANDIDATO_ACTIVO")

    dfs_predichos = []

    # 2. Etapa 2: Inferencia Dual (Ignición Primaria y Propagación Táctica) en Celdas Activas
    for region_nombre in ["Chaco", "Oriental"]:
        reg_key = region_nombre.lower()
        art = artefactos[reg_key]

        df_reg = df_activos.filter(pl.col("region") == region_nombre)
        if df_reg.height == 0:
            continue

        # Inferencia de Ignición Primaria (M3) y fallback M1
        X_ign = df_reg.select(art["features_ign"]).to_numpy()
        p_m3 = predecir_probabilidad(art["booster_ign"], art["calibrador_ign"], X_ign)
        if art.get("booster_m1") is not None:
            X_m1 = df_reg.select(art["features_m1"]).to_numpy()
            p_m1 = predecir_probabilidad(art["booster_m1"], art["calibrador_m1"], X_m1)
            fallback_m1 = (df_reg["freq_hist_1km"].to_numpy() <= 0)
        else:
            p_m1 = p_m3
            fallback_m1 = np.zeros(df_reg.height, dtype=bool)
        p_ign = np.where(fallback_m1, p_m1, p_m3)

        # Inferencia de Propagación Táctica
        if art["booster_prop"] is not None and art["calibrador_prop"] is not None:
            X_prop = df_reg.select(art["features_prop"]).to_numpy()
            p_prop = predecir_probabilidad(art["booster_prop"], art["calibrador_prop"], X_prop)
        else:
            p_prop = p_ign

        # Guardar la probabilidad case-control para diagnóstico y corregir a
        # prevalencia poblacional antes de clasificar/alertar.
        p_m3_muestreo = p_m3.copy()
        p_m1_muestreo = p_m1.copy()
        p_ign_muestreo = p_ign.copy()
        p_prop_muestreo = p_prop.copy()
        prev_ign = art.get("prevalencia_ignicion")
        prev_prop = art.get("prevalencia_propagacion")
        if prev_ign:
            p_m3 = corregir_prior_probabilidades(p_m3, PREVALENCIA_MUESTREO, prev_ign)
            p_m1 = corregir_prior_probabilidades(p_m1, PREVALENCIA_MUESTREO, prev_ign)
            p_ign = corregir_prior_probabilidades(p_ign, PREVALENCIA_MUESTREO, prev_ign)
        if prev_prop:
            p_prop = corregir_prior_probabilidades(p_prop, PREVALENCIA_MUESTREO, prev_prop)

        en_propagacion = df_reg["focos_25km_t1"].to_numpy() > 0
        p_fuego = np.where(en_propagacion, p_prop, p_ign)

        cutoff_ign = art.get("cutoff_ign", UMBRAL_CORTE_CHACO if reg_key == "chaco" else UMBRAL_CORTE_ORIENTAL)
        cutoff_prop = art.get("cutoff_prop", UMBRAL_PROP_CHACO if reg_key == "chaco" else UMBRAL_PROP_ORIENTAL)

        niveles_m3 = categorizar_riesgo_vectorizado(p_m3, cutoff=cutoff_ign, modo="ignicion")
        niveles_m1 = categorizar_riesgo_vectorizado(p_m1, cutoff=art.get("cutoff_m1", cutoff_ign), modo="ignicion")
        niveles_ign = np.where(fallback_m1, niveles_m1, niveles_m3)
        niveles_prop = categorizar_riesgo_vectorizado(p_prop, cutoff=cutoff_prop, modo="propagacion")
        niveles_tot = np.where(en_propagacion, niveles_prop, niveles_ign)

        df_reg = df_reg.with_columns([
            pl.Series("probabilidad_ignicion", p_ign.astype(np.float32)),
            pl.Series("probabilidad_m3", p_m3.astype(np.float32)),
            pl.Series("probabilidad_m1", p_m1.astype(np.float32)),
            pl.Series("fallback_m1", fallback_m1),
            pl.Series("probabilidad_propagacion", p_prop.astype(np.float32)),
            pl.Series("probabilidad_fuego", p_fuego.astype(np.float32)),
            pl.Series("probabilidad_cruda", (np.where(fallback_m1, p_m1_muestreo, p_m3_muestreo)).astype(np.float32)),
            pl.Series("probabilidad_ignicion_muestreo", p_ign_muestreo.astype(np.float32)),
            pl.Series("probabilidad_propagacion_muestreo", p_prop_muestreo.astype(np.float32)),
            pl.Series("nivel_alerta_ignicion", niveles_ign),
            pl.Series("nivel_alerta_propagacion", niveles_prop),
            pl.Series("nivel_alerta", niveles_tot)
        ])
        dfs_predichos.append(df_reg)

    # 3. Asignar probabilidad 0.0 y riesgo a las celdas descartadas por el filtro biofísico
    if df_descartados.height > 0:
        niveles_desc = [
            "NULO" if motivo == "SIN_COMBUSTIBLE" else "BAJO"
            for motivo in df_descartados["estado_gating"]
        ]
        df_descartados = df_descartados.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_ignicion"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_m3"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_m1"),
            pl.lit(False).alias("fallback_m1"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_propagacion"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_fuego"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_cruda"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_ignicion_muestreo"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_propagacion_muestreo"),
            pl.Series("nivel_alerta_ignicion", niveles_desc),
            pl.Series("nivel_alerta_propagacion", niveles_desc),
            pl.Series("nivel_alerta", niveles_desc)
        ])
        dfs_predichos.append(df_descartados)

    df_resultado = pl.concat(dfs_predichos, how="vertical").sort("cell_id")
    return df_resultado

def generar_boletin_operativo(df_evaluado: pl.DataFrame, fecha: str) -> tuple[dict, str]:
    """
    Genera el resumen ejecutivo departamental y el boletín de alerta temprana (Markdown y dict).
    """
    total_celdas = df_evaluado.height
    conteo_niveles = df_evaluado["nivel_alerta"].value_counts().to_dicts()
    dist_niveles = {item["nivel_alerta"]: item["count"] for item in conteo_niveles}

    # Desglose departamental
    resumen_depto = (
        df_evaluado.group_by(["departamento", "region"])
        .agg([
            pl.len().alias("total_km2"),
            pl.col("nivel_alerta").filter(pl.col("nivel_alerta") == "EXTREMO").count().alias("km2_extremo"),
            pl.col("nivel_alerta").filter(pl.col("nivel_alerta") == "ALTO").count().alias("km2_alto"),
            pl.col("nivel_alerta").filter(pl.col("nivel_alerta") == "MODERADO").count().alias("km2_moderado"),
            pl.col("probabilidad_fuego").mean().alias("prob_media")
        ])
        .sort(["km2_extremo", "km2_alto"], descending=True)
    )

    # Identificar las 10 celdas más críticas a nivel nacional
    top10_criticos = (
        df_evaluado.sort("probabilidad_fuego", descending=True)
        .head(10)
        .select([
            "cell_id", "departamento", "region", "lat", "lon",
            "probabilidad_fuego", "nivel_alerta", "isi", "vpd_max_kpa", "viento_max_kmh", "rafaga_max_kmh"
        ])
        .to_dicts()
    )

    resumen = {
        "fecha": fecha,
        "cobertura_nacional_km2": total_celdas,
        "distribucion_riesgo": dist_niveles,
        "deptos_criticos": resumen_depto.to_dicts(),
        "top_10_celdas_criticas": top10_criticos
    }

    # Construir boletín en formato Markdown oficial
    lineas = [
        f"# BOLETÍN TÉCNICO DIARIO DE ALERTA DE INCENDIOS FORESTALES - PARAGUAY",
        f"**Sistema de Alerta Temprana Biofísico GeoAI (SAT-Incendios PY)**  ",
        f"**Fecha de Operación:** {fecha} | **Resolución:** Grilla Nacional 1 km² (400.006 celdas)",
        "",
        "---",
        "## 1. RESUMEN EJECUTIVO NACIONAL",
        f"- **Área total analizada:** {total_celdas:,} km²",
        f"- **Nivel Extremo (Alerta Roja):** {dist_niveles.get('EXTREMO', 0):,} km² ({dist_niveles.get('EXTREMO', 0)/total_celdas*100:.2f}%)",
        f"- **Nivel Alto (Alerta Naranja):** {dist_niveles.get('ALTO', 0):,} km² ({dist_niveles.get('ALTO', 0)/total_celdas*100:.2f}%)",
        f"- **Nivel Moderado (Alerta Amarilla):** {dist_niveles.get('MODERADO', 0):,} km² ({dist_niveles.get('MODERADO', 0)/total_celdas*100:.2f}%)",
        f"- **Nivel Bajo / Seguro:** {dist_niveles.get('BAJO', 0):,} km²",
        f"- **Nivel Nulo (Sin combustible/Agua):** {dist_niveles.get('NULO', 0):,} km²",
        "",
        "---",
        "## 2. DESGLOSE DE RIESGO POR DEPARTAMENTO",
        "",
        "| Departamento | Región | Área Crítica (Extremo + Alto) | km² Extremo | km² Alto | Prob. Media |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for d in resumen_depto.to_dicts():
        area_critica = d["km2_extremo"] + d["km2_alto"]
        lineas.append(
            f"| {d['departamento']} | {d['region']} | {area_critica:,} km² | {d['km2_extremo']:,} | {d['km2_alto']:,} | {d['prob_media']:.3f} |"
        )

    lineas.extend([
        "",
        "---",
        "## 3. TOP 10 CELDAS DE MÁXIMA CRITICIDAD NACIONAL",
        "",
        "| Cell ID | Departamento | Latitud | Longitud | P(Fuego) | Alerta | FWI ISI | VPD (kPa) | Ráfaga (km/h) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for c in top10_criticos:
        lineas.append(
            f"| `{c['cell_id']}` | {c['departamento']} | {c['lat']:.4f}° | {c['lon']:.4f}° | **{c['probabilidad_fuego']:.3f}** | {c['nivel_alerta']} | {c['isi']:.1f} | {c['vpd_max_kpa']:.2f} | {c['rafaga_max_kmh']:.1f} |"
        )

    lineas.extend([
        "",
        "---",
        "## 4. PROTOCOLO DE INTERVENCIÓN INSTITUCIONAL RECOMENDADO",
        "> [!IMPORTANT]",
        "> **Sectores en Nivel EXTREMO (Alerta Roja):** Prohibición absoluta de quemas prescriptas (Ley 4014/10). Despliegue de patrullas de monitoreo y alistamiento de brigadas forestales de la Secretaría de Emergencia Nacional (SEN), Cuerpos de Bomberos Voluntarios (CBVP) e INFONA.",
        ">",
        "> **Sectores en Nivel ALTO (Alerta Naranja):** Restricción de quemas de pastura. Monitoreo satelital activo en tiempo real cada 3 horas (VIIRS 375m Suomi-NPP / NOAA-20).",
        "",
        "_Documento generado automáticamente por el Pipeline Operativo de Incendios_PY._"
    ])

    boletin_md = "\n".join(lineas)
    return resumen, boletin_md
