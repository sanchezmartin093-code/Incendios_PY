"""
Script 00: Generación de la Grilla Nacional de 1 km² para Paraguay (SOTA GeoAI).
- Proyección: UTM Zona 21 Sur (EPSG:32721).
- Indexación determinística O(1): cell_id entero.
- Clasificación regional: Chaco (Occidental) vs. Región Oriental.
- Formato de salida: datos/processed/grilla_nacional_1km.parquet (Zero-Copy).
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import geopandas as gpd
import polars as pl
from shapely.geometry import shape, Point
from pyproj import Transformer

from src.constantes import (
    DIR_PROCESSED,
    DIR_RAW,
    CRS_GEOGRAFICO,
    CRS_PROYECTADO,
    RESOLUCION_GRILLA_METROS
)
from src.geometria import coordenadas_a_cell_id, cell_id_a_coordenadas_centro

DEPARTAMENTOS_CHACO = {"BOQUERON", "ALTO PARAGUAY", "PRESIDENTE HAYES"}

def main():
    print("=" * 60)
    print("INICIANDO GENERACIÓN DE GRILLA NACIONAL 1 km² (PARAGUAY)")
    print("=" * 60)

    path_adm1 = DIR_RAW / "vector" / "paraguay_adm1_departamentos.geojson"
    if not path_adm1.exists():
        raise FileNotFoundError(f"No se encontró el archivo de límites en: {path_adm1}")

    # 1. Cargar geometrías departamentales
    print(f"Cargando límites desde: {path_adm1}")
    gdf_adm1 = gpd.read_file(path_adm1)
    
    # Estandarizar nombre de departamento
    gdf_adm1["departamento"] = gdf_adm1["shapeName"].str.upper().str.strip()
    gdf_adm1["region"] = gdf_adm1["departamento"].apply(
        lambda d: "Chaco" if d in DEPARTAMENTOS_CHACO else "Oriental"
    )

    # 2. Reproyectar a UTM Zona 21 Sur (EPSG:32721)
    print(f"Reproyectando de {CRS_GEOGRAFICO} a {CRS_PROYECTADO}...")
    gdf_utm = gdf_adm1.to_crs(CRS_PROYECTADO)

    # 3. Determinar límites envolventes (Bounding Box en metros UTM)
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    print(f"Bounds UTM: X=[{minx:.1f}, {maxx:.1f}], Y=[{miny:.1f}, {maxy:.1f}]")

    # Alinear a la resolución de grilla
    x_coords = np.arange(np.floor(minx / RESOLUCION_GRILLA_METROS) * RESOLUCION_GRILLA_METROS + (RESOLUCION_GRILLA_METROS / 2),
                         maxx + RESOLUCION_GRILLA_METROS,
                         RESOLUCION_GRILLA_METROS)
    y_coords = np.arange(np.floor(miny / RESOLUCION_GRILLA_METROS) * RESOLUCION_GRILLA_METROS + (RESOLUCION_GRILLA_METROS / 2),
                         maxy + RESOLUCION_GRILLA_METROS,
                         RESOLUCION_GRILLA_METROS)

    xx, yy = np.meshgrid(x_coords, y_coords)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    print(f"Total de celdas candidatas en el bounding box: {len(x_flat):,}")

    # 4. Crear GeoDataFrame de puntos centroides
    print("Construyendo puntos centroides y filtrando por polígonos de Paraguay...")
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_flat, y_flat, crs=CRS_PROYECTADO)
    )

    # Spatial Join con los departamentos para asignar departamento y región
    # Utiliza spatial index (STRtree) de GEOS/Shapely 2.0 (extremadamente rápido)
    pts_py = gpd.sjoin(pts, gdf_utm[["departamento", "region", "geometry"]], how="inner", predicate="within")
    print(f"Celdas efectivas dentro del territorio paraguayo: {len(pts_py):,}")

    # 5. Calcular cell_id determinístico O(1) y coordenadas geográficas WGS84
    x_final = pts_py.geometry.x.values
    y_final = pts_py.geometry.y.values
    cell_ids = coordenadas_a_cell_id(x_final, y_final, res_m=RESOLUCION_GRILLA_METROS)

    transformer = Transformer.from_crs(CRS_PROYECTADO, CRS_GEOGRAFICO, always_xy=True)
    lons, lats = transformer.transform(x_final, y_final)

    # 6. Estructurar en Polars / Apache Arrow para guardado Zero-Copy
    df_grilla = pl.DataFrame({
        "cell_id": cell_ids,
        "x_utm": x_final.astype(np.float32),
        "y_utm": y_final.astype(np.float32),
        "lon": lons.astype(np.float32),
        "lat": lats.astype(np.float32),
        "departamento": pts_py["departamento"].astype(str).to_list(),
        "region": pts_py["region"].astype(str).to_list()
    })

    # Resumen regional
    resumen = df_grilla.group_by("region").len()
    print("\nDistribución de celdas por región:")
    for row in resumen.iter_rows():
        print(f" - {row[0]}: {row[1]:,} celdas (~{row[1]} km²)")

    # 7. Guardar en formato GeoParquet / Parquet comprimido ZSTD
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    df_grilla.write_parquet(out_parquet, compression="zstd")
    print(f"\nGrilla nacional guardada con éxito en: {out_parquet}")
    print(f"Tamaño de archivo: {out_parquet.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
