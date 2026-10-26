"""
Módulo firms.py: Ingesta, filtrado de calidad y agregación espacial de focos NASA FIRMS (VIIRS 375m).
Aplica los principios de rigor de la teledetección:
- Transformación y asignación a cell_id O(1) en UTM 21S.
- Filtro de distorsión óptica / efecto Bow-Tie (scan footprint <= 1.5 km).
- Filtrado por nivel de confianza y potencia radiativa (FRP).
- Agregación diaria por celda de 1 km².
"""

import pandas as pd
import numpy as np
import polars as pl
from pyproj import Transformer
from pathlib import Path

from src.constantes import (
    BBOX_PARAGUAY,
    CRS_GEOGRAFICO,
    CRS_PROYECTADO,
    RESOLUCION_GRILLA_METROS
)
from src.geometria import coordenadas_a_cell_id

def procesar_focos_firms_a_grilla(
    df_raw: pd.DataFrame,
    df_grilla: pl.DataFrame,
    max_scan_km: float = 1.5,
    excluir_baja_confianza: bool = True
) -> pl.DataFrame:
    """
    Procesa un DataFrame crudo de FIRMS, aplica filtros de teledetección y realiza
    el spatial join O(1) con la grilla nacional.
    Retorna un DataFrame de Polars con los focos georreferenciados a nivel de celda.
    """
    # 1. Filtro rápido de Bounding Box continental
    df_bbox = df_raw[
        (df_raw["latitude"] >= BBOX_PARAGUAY["min_lat"]) &
        (df_raw["latitude"] <= BBOX_PARAGUAY["max_lat"]) &
        (df_raw["longitude"] >= BBOX_PARAGUAY["min_lon"]) &
        (df_raw["longitude"] <= BBOX_PARAGUAY["max_lon"])
    ].copy()

    if len(df_bbox) == 0:
        return pl.DataFrame()

    # 2. Filtro Bow-Tie: descartar píxeles hiper-deformados en el extremo orbital
    if "scan" in df_bbox.columns:
        df_bbox = df_bbox[df_bbox["scan"] <= max_scan_km]

    # 3. Filtro de confianza
    if excluir_baja_confianza and "confidence" in df_bbox.columns:
        confianza = df_bbox["confidence"].astype(str).str.strip().str.lower()
        df_bbox = df_bbox[~confianza.isin({"l", "low"})]

    # 4. Sólo fuegos de vegetación (type=0). Excluye fuentes industriales,
    #    volcanes y detecciones offshore (type=2/3 en FIRMS).
    if "type" in df_bbox.columns:
        df_bbox = df_bbox[df_bbox["type"] == 0]

    if len(df_bbox) == 0:
        return pl.DataFrame()

    # 4. Proyección a UTM 21S y cálculo de cell_id determinístico O(1)
    transformer = Transformer.from_crs(CRS_GEOGRAFICO, CRS_PROYECTADO, always_xy=True)
    x_utm, y_utm = transformer.transform(df_bbox["longitude"].values, df_bbox["latitude"].values)
    
    df_bbox["x_utm"] = x_utm
    df_bbox["y_utm"] = y_utm
    df_bbox["cell_id"] = coordenadas_a_cell_id(x_utm, y_utm, res_m=RESOLUCION_GRILLA_METROS)

    # Convertir a Polars para inner join de alta velocidad
    pl_focos = pl.from_pandas(df_bbox[[
        "cell_id", "latitude", "longitude", "acq_date", "acq_time", 
        "frp", "confidence", "daynight", "scan", "track"
    ]])

    # 5. Inner Join con la grilla nacional (descarta puntos en países limítrofes)
    pl_grilla_sub = df_grilla.select(["cell_id", "departamento", "region"])
    pl_resultado = pl_focos.join(pl_grilla_sub, on="cell_id", how="inner")

    return pl_resultado

def agregar_focos_diarios(pl_focos_grilla: pl.DataFrame) -> pl.DataFrame:
    """
    Agrega los focos térmicos por celda y por fecha (acq_date):
    Calcula: total_focos, max_frp, suma_frp y clasifica fuego = 1.
    """
    if pl_focos_grilla.is_empty():
        return pl.DataFrame()

    return (
        pl_focos_grilla
        .group_by(["cell_id", "acq_date", "departamento", "region"])
        .agg([
            pl.len().alias("num_focos_dia"),
            pl.col("frp").max().alias("max_frp_dia"),
            pl.col("frp").sum().alias("sum_frp_dia"),
            pl.lit(1).alias("fuego")
        ])
    )
