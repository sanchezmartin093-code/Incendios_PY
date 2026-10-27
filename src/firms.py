"""
Module firms.py: Ingestion, quality filtering and spatial aggregation of NASA FIRMS hotspots (VIIRS 375m).
Applies remote sensing rigor principles:
- Transformation and O(1) cell_id assignment in UTM 21S.
- Optical distortion / Bow-Tie effect filter (scan footprint <= 1.5 km).
- Filtering by confidence level and radiative power (FRP).
- Daily aggregation per 1 km² cell.
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
    Processes a raw FIRMS DataFrame, applies remote sensing filters and performs
    the O(1) spatial join with the national grid.
    Returns a Polars DataFrame with the hotspots georeferenced at cell level.
    """
    # 1. Fast continental Bounding Box filter
    df_bbox = df_raw[
        (df_raw["latitude"] >= BBOX_PARAGUAY["min_lat"]) &
        (df_raw["latitude"] <= BBOX_PARAGUAY["max_lat"]) &
        (df_raw["longitude"] >= BBOX_PARAGUAY["min_lon"]) &
        (df_raw["longitude"] <= BBOX_PARAGUAY["max_lon"])
    ].copy()

    if len(df_bbox) == 0:
        return pl.DataFrame()

    # 2. Bow-Tie filter: discard hyper-deformed pixels at the orbital edge
    if "scan" in df_bbox.columns:
        df_bbox = df_bbox[df_bbox["scan"] <= max_scan_km]

    # 3. Confidence filter
    if excluir_baja_confianza and "confidence" in df_bbox.columns:
        confianza = df_bbox["confidence"].astype(str).str.strip().str.lower()
        df_bbox = df_bbox[~confianza.isin({"l", "low"})]

    # 4. Vegetation fires only (type=0). Excludes industrial sources,
    #    volcanoes and offshore detections (type=2/3 in FIRMS).
    if "type" in df_bbox.columns:
        df_bbox = df_bbox[df_bbox["type"] == 0]

    if len(df_bbox) == 0:
        return pl.DataFrame()

    # 4. Projection to UTM 21S and deterministic O(1) cell_id computation
    transformer = Transformer.from_crs(CRS_GEOGRAFICO, CRS_PROYECTADO, always_xy=True)
    x_utm, y_utm = transformer.transform(df_bbox["longitude"].values, df_bbox["latitude"].values)
    
    df_bbox["x_utm"] = x_utm
    df_bbox["y_utm"] = y_utm
    df_bbox["cell_id"] = coordenadas_a_cell_id(x_utm, y_utm, res_m=RESOLUCION_GRILLA_METROS)

    # Convert to Polars for high-speed inner join
    pl_focos = pl.from_pandas(df_bbox[[
        "cell_id", "latitude", "longitude", "acq_date", "acq_time", 
        "frp", "confidence", "daynight", "scan", "track"
    ]])

    # 5. Inner Join with the national grid (discards points in neighboring countries)
    pl_grilla_sub = df_grilla.select(["cell_id", "departamento", "region"])
    pl_resultado = pl_focos.join(pl_grilla_sub, on="cell_id", how="inner")

    return pl_resultado

def agregar_focos_diarios(pl_focos_grilla: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates thermal hotspots by cell and by date (acq_date):
    Computes: total_focos, max_frp, suma_frp and classifies fuego = 1.
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
