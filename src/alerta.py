"""
Módulo alerta.py: Servicio Operativo de Alerta Temprana de Incendios Forestales para Paraguay.
Filosofía de Producción:
- Arquitectura Two-Stage Bio-physical Gating:
  * Etapa 1: Filtro biofísico O(1). Descarta de plano celdas no combustibles (agua, urbano),
    precipitaciones extintoras (P_48h >= 10 mm) o atmósferas saturadas (VPD < 0.8 kPa).
  * Etapa 2: Inferencia de ensamble discriminante regional (LightGBM Booster C++) calibrado
    isotónicamente P(Fuego|X) ∈ [0.0, 1.0].
- Clasificación de riesgo operativa según umbrales de máxima precisión (F0.5):
  * NULO: Sin material combustible vegetal.
  * BAJO: Condiciones subcríticas (P < c*).
  * MODERADO: Ignición probable bajo fuentes activas (c* <= P < 0.25).
  * ALTO: Propagación rápida sostenida (0.25 <= P < 0.40).
  * EXTREMO: Condiciones críticas de tormenta de fuego (P >= 0.40).
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
    FEATURES_CHACO,
    FEATURES_ORIENTAL,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
    FEATURES_PROPAGACION_CHACO,
    FEATURES_PROPAGACION_ORIENTAL,
    predecir_probabilidad
)
from src.espaciotemporal import enriquecer_dataset_espaciotemporal

# Umbrales óptimos derivados de calibración en validación out-of-time (F0.5 de máxima precisión)
UMBRAL_CORTE_CHACO = 0.092
UMBRAL_CORTE_ORIENTAL = 0.162

def cargar_artefactos_modelos() -> dict:
    """Carga los ensambles nativos C++ y los calibradores duales (Ignición y Propagación)."""
    artefactos = {}
    for region in ["chaco", "oriental"]:
        # 1. Capa de Ignición Primaria
        ruta_b_ign = DIR_MODELOS / f"lightgbm_ignicion_{region}.booster"
        ruta_c_ign = DIR_MODELOS / "calibradores" / f"calibrador_ignicion_{region}.joblib"
        if not ruta_b_ign.exists():
            ruta_b_ign = DIR_MODELOS / f"lightgbm_{region}.booster"
            ruta_c_ign = DIR_MODELOS / "calibradores" / f"calibrador_{region}.joblib"

        # 2. Capa de Propagación Táctica
        ruta_b_prop = DIR_MODELOS / f"lightgbm_propagacion_{region}.booster"
        ruta_c_prop = DIR_MODELOS / "calibradores" / f"calibrador_propagacion_{region}.joblib"

        booster_ign = lgb.Booster(model_file=str(ruta_b_ign))
        calibrador_ign = joblib.load(ruta_c_ign)

        booster_prop = lgb.Booster(model_file=str(ruta_b_prop)) if ruta_b_prop.exists() else None
        calibrador_prop = joblib.load(ruta_c_prop) if ruta_c_prop.exists() else None

        artefactos[region] = {
            "booster_ign": booster_ign,
            "calibrador_ign": calibrador_ign,
            "features_ign": FEATURES_IGNICION_CHACO if region == "chaco" else FEATURES_IGNICION_ORIENTAL,
            "cutoff_ign": 0.092 if region == "chaco" else 0.162,
            "booster_prop": booster_prop,
            "calibrador_prop": calibrador_prop,
            "features_prop": FEATURES_PROPAGACION_CHACO if region == "chaco" else FEATURES_PROPAGACION_ORIENTAL,
            "cutoff_prop": 0.805 if region == "chaco" else 0.708,
            # Compatibilidad
            "booster": booster_ign,
            "calibrador": calibrador_ign,
            "features": FEATURES_IGNICION_CHACO if region == "chaco" else FEATURES_IGNICION_ORIENTAL,
            "cutoff_f05": 0.092 if region == "chaco" else 0.162
        }
    return artefactos

def categorizar_riesgo(prob: float, region: str, es_combustible: bool, gating_motivo: str | None = None) -> str:
    """Asigna la categoría de riesgo operativa basada en la probabilidad calibrada y la física."""
    if not es_combustible:
        return "NULO"
    if gating_motivo is not None and gating_motivo != "":
        return "BAJO"

    cutoff = UMBRAL_CORTE_CHACO if region.lower() == "chaco" else UMBRAL_CORTE_ORIENTAL
    
    if prob < cutoff:
        return "BAJO"
    elif prob < 0.25:
        return "MODERADO"
    elif prob < 0.40:
        return "ALTO"
    else:
        return "EXTREMO"

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

    # 4. Enriquecimiento espaciotemporal (Ecotono, Contagio t-1 y Prior histórico)
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    if path_focos.exists():
        df_focos = pl.read_parquet(path_focos)
    else:
        df_focos = pl.DataFrame()

    ruta_coords_hist = DIR_MODELOS / "fuegos_train_coords.npy"
    if ruta_coords_hist.exists():
        fuegos_train_coords = np.load(ruta_coords_hist)
    else:
        fuegos_train_coords = None

    df_nacional = enriquecer_dataset_espaciotemporal(
        df_nacional,
        df_focos,
        fuegos_train_coords
    )

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

        # Inferencia de Ignición Primaria
        X_ign = df_reg.select(art["features_ign"]).to_numpy()
        p_ign = predecir_probabilidad(art["booster_ign"], art["calibrador_ign"], X_ign)

        # Inferencia de Propagación Táctica
        if art["booster_prop"] is not None and art["calibrador_prop"] is not None:
            X_prop = df_reg.select(art["features_prop"]).to_numpy()
            p_prop = predecir_probabilidad(art["booster_prop"], art["calibrador_prop"], X_prop)
        else:
            p_prop = p_ign

        # Probabilidad de fuego consolidada (máximo riesgo entre ambas capas)
        p_fuego = np.maximum(p_ign, p_prop)

        # Asignar niveles de alerta
        niveles_ign = [categorizar_riesgo(p, region_nombre, True, None) for p in p_ign]
        niveles_prop = [categorizar_riesgo(p, region_nombre, True, None) for p in p_prop]
        niveles_tot = [categorizar_riesgo(p, region_nombre, True, None) for p in p_fuego]

        df_reg = df_reg.with_columns([
            pl.Series("probabilidad_ignicion", p_ign.astype(np.float32)),
            pl.Series("probabilidad_propagacion", p_prop.astype(np.float32)),
            pl.Series("probabilidad_fuego", p_fuego.astype(np.float32)),
            pl.Series("probabilidad_cruda", p_ign.astype(np.float32)),
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
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_propagacion"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_fuego"),
            pl.lit(0.0).cast(pl.Float32).alias("probabilidad_cruda"),
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
