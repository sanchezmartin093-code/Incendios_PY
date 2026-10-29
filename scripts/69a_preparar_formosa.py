"""G0-Formosa: boundary, 1-km grid, climate nodes and border buffer.

Writes only under datos/processed/formosa_g0/. The national Paraguay grid is
not modified.
"""

from pathlib import Path
import json
import sys

import geopandas as gpd
import numpy as np
import polars as pl
from pyproj import Transformer
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clima import asignar_nodos_a_grilla, generar_malla_nodos_clima
from src.constantes import DIR_PROCESSED, DIR_RAW

OUT = DIR_PROCESSED / "formosa_g0"
OUT.mkdir(parents=True, exist_ok=True)
BUFFER_KM = 25.0


def main() -> None:
    adm1 = gpd.read_file(DIR_RAW / "vector" / "argentina_adm1.geojson")
    formosa = adm1.loc[adm1["shapeName"] == "Formosa"].copy()
    if len(formosa) != 1:
        raise RuntimeError("No se encontró exactamente una geometría de Formosa")
    formosa = formosa.to_crs("EPSG:32721")
    geom = formosa.geometry.iloc[0]
    if geom.geom_type != "Polygon":
        geom = geom.buffer(0)
    minx, miny, maxx, maxy = geom.bounds

    x_centers = np.arange(np.floor(minx / 1000) * 1000 + 500, maxx, 1000)
    y_centers = np.arange(np.floor(miny / 1000) * 1000 + 500, maxy, 1000)
    xx, yy = np.meshgrid(x_centers, y_centers)
    x = xx.ravel(); y = yy.ravel()
    inside = np.array([geom.contains(Point(float(a), float(b))) for a, b in zip(x, y)])
    x = x[inside]; y = y[inside]
    if len(x) < 1000:
        raise RuntimeError(f"Grilla Formosa sospechosamente pequeña: {len(x)}")

    cell_id = (np.floor(x / 1000).astype(np.int64) * 10_000_000
               + np.floor(y / 1000).astype(np.int64))
    if len(np.unique(cell_id)) != len(cell_id):
        raise RuntimeError("cell_id duplicado en la grilla Formosa")

    transformer = Transformer.from_crs("EPSG:32721", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    grid = pl.DataFrame({
        "cell_id": cell_id.astype(np.int64),
        "x_utm": x.astype(np.float64),
        "y_utm": y.astype(np.float64),
        "lon": lon.astype(np.float32),
        "lat": lat.astype(np.float32),
        "departamento": ["FORMOSA"] * len(x),
        "region": ["Chaco"] * len(x),
    })
    nodos = generar_malla_nodos_clima(resolucion_deg=0.25)
    grid = asignar_nodos_a_grilla(grid, nodos)

    # Paraguay cells inside Formosa + 25 km, used to exclude training leakage.
    py = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    py_x = py["x_utm"].to_numpy(); py_y = py["y_utm"].to_numpy()
    py_inside = np.array([geom.contains(Point(float(a), float(b))) for a, b in zip(py_x, py_y)])
    py_buffer = np.array([geom.distance(Point(float(a), float(b))) <= BUFFER_KM * 1000 for a, b in zip(py_x, py_y)])
    excluir = py.filter(pl.Series(py_buffer)).select("cell_id")
    excluir.write_parquet(OUT / "py_cells_excluir_buffer_formosa.parquet")

    grid.write_parquet(OUT / "grilla_formosa_1km.parquet", compression="zstd")
    formosa.to_crs("EPSG:4326").to_file(OUT / "formosa_adm1.geojson", driver="GeoJSON")

    meta = {
        "fuente_limite": "geoBoundaries ARG ADM1 simplified (auditoría)",
        "buffer_km": BUFFER_KM,
        "celdas_formosa": grid.height,
        "celdas_paraguay_en_buffer": int(py_buffer.sum()),
        "area_km2_shapely": round(float(geom.area) / 1e6, 2),
        "nodos_clima": int(grid["nodo_clima_id"].n_unique()),
    }
    (OUT / "grid_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
