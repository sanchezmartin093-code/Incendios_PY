"""
Módulo raster.py: Conversión de alertas tabulares GeoParquet a coberturas ráster estándar GeoTIFF (EPSG:32721).
Permite la interoperabilidad directa con SIG de escritorio (QGIS, ArcGIS) y plataformas en la nube (Google Earth Engine).
- Resolución espacial: 1.000 m x 1.000 m (1 km²).
- Proyección: UTM Zona 21 Sur (EPSG:32721).
- Bandas exportadas:
  * Banda 1: Probabilidad calibrada de fuego P(Fuego|X) ∈ [0.0, 1.0] (Float32, NoData = -9999.0).
  * Banda 2: Nivel operativo de alerta (Byte / UInt8, NoData = 255):
    0 = NULO, 1 = BAJO, 2 = MODERADO, 3 = ALTO, 4 = EXTREMO.
"""

from pathlib import Path
import numpy as np
import polars as pl
import rasterio
from rasterio.transform import from_origin

from src.constantes import CRS_PROYECTADO

MAPA_NIVELES_INT = {
    "NULO": 0,
    "BAJO": 1,
    "MODERADO": 2,
    "ALTO": 3,
    "EXTREMO": 4
}

def exportar_alerta_a_geotiff(ruta_parquet_alerta: Path, ruta_salida_tif: Path) -> Path:
    """
    Convierte una capa Parquet diaria de alerta generada por el pipeline en un GeoTIFF multi-banda.
    """
    if not ruta_parquet_alerta.exists():
        raise FileNotFoundError(f"Archivo Parquet de alerta no encontrado en {ruta_parquet_alerta}")

    df = pl.read_parquet(ruta_parquet_alerta)
    fecha = ruta_parquet_alerta.stem.replace("alerta_nacional_", "")

    min_x = -81500.0
    max_x = 777500.0
    min_y = 6948500.0
    max_y = 7865500.0

    res = 1000.0
    cols = int(round((max_x - min_x) / res)) + 1
    rows = int(round((max_y - min_y) / res)) + 1

    # Crear matrices 2D inicializadas con NoData
    nodata_prob = -9999.0
    nodata_nivel = 255

    matriz_prob = np.full((rows, cols), nodata_prob, dtype=np.float32)
    matriz_nivel = np.full((rows, cols), nodata_nivel, dtype=np.uint8)

    # Coordenadas a índices de fila/columna
    x_coords = df["x_utm"].to_numpy()
    y_coords = df["y_utm"].to_numpy()
    r_idx = np.round((max_y - y_coords) / res).astype(int)
    c_idx = np.round((x_coords - min_x) / res).astype(int)

    # Valores
    probs = df["probabilidad_fuego"].to_numpy().astype(np.float32)
    niveles_str = df["nivel_alerta"].to_list()
    niveles_int = np.array([MAPA_NIVELES_INT.get(n, nodata_nivel) for n in niveles_str], dtype=np.uint8)

    # Asignar a la matriz raster
    matriz_prob[r_idx, c_idx] = probs
    matriz_nivel[r_idx, c_idx] = niveles_int

    # Configurar geotransformación (esquina superior izquierda de la celda de origen)
    transform = from_origin(min_x - res / 2.0, max_y + res / 2.0, res, res)

    ruta_salida_tif.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 2,
        "dtype": rasterio.float32,
        "crs": CRS_PROYECTADO,
        "transform": transform,
        "compress": "deflate",
        "nodata": nodata_prob,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    with rasterio.open(ruta_salida_tif, "w", **profile) as dst:
        dst.write(matriz_prob, 1)
        dst.write(matriz_nivel.astype(np.float32), 2)
        dst.set_band_description(1, "Probabilidad_Calibrada_Fuego")
        dst.set_band_description(2, "Nivel_Alerta_Operativo_0a4")
        dst.update_tags(
            FECHA_ALERTA=fecha,
            SISTEMA="SAT-Incendios PY",
            RESOLUCION_METROS="1000",
            LICENCIA="MIT Open Source",
            DESCRIPCION="Riesgo de incendios forestales estimado con LightGBM y calibracion isotonica"
        )

    return ruta_salida_tif
