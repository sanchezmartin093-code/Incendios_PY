"""
Script 23: benchmark computacional de lookup O(1), spatial join y pipeline.
"""

from pathlib import Path
import sys
import time
import resource

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
from shapely.geometry import Point

from src.constantes import DIR_PROCESSED, DIR_RAW, DIR_TABLAS
from src.geometria import coordenadas_a_cell_id


def mediana_lookup(x, y, rep=100):
    tiempos = []
    for _ in range(rep):
        t = time.perf_counter()
        coordenadas_a_cell_id(x, y)
        tiempos.append(time.perf_counter() - t)
    return float(np.median(tiempos))


def main():
    g = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    x = g["x_utm"].to_numpy(); y = g["y_utm"].to_numpy()
    t_lookup = mediana_lookup(x, y)

    adm = gpd.read_file(DIR_RAW / "vector" / "paraguay_adm1_departamentos.geojson")
    adm = adm.to_crs("EPSG:32721")
    pts = gpd.GeoDataFrame(geometry=[Point(xi, yi) for xi, yi in zip(x[:50000], y[:50000])], crs="EPSG:32721")
    t0 = time.perf_counter()
    pts.sjoin(adm[["shapeName", "geometry"]], how="left", predicate="within")
    t_sjoin_50k = time.perf_counter() - t0

    s4 = pd.read_csv(DIR_TABLAS / "tablaS4_perfil_computacional.csv")
    total = s4[s4["etapa"].str.contains("TOTAL", na=False)].iloc[0]
    filas = [
        {"etapa": "Lookup cell_id O(1)", "metodo": "NumPy int64", "n": len(g), "tiempo_s": round(t_lookup, 6), "ram_mb": np.nan, "nota": "mediana 100 repeticiones"},
        {"etapa": "Point-in-Polygon", "metodo": "GeoPandas sjoin", "n": 50000, "tiempo_s": round(t_sjoin_50k, 4), "ram_mb": np.nan, "nota": "50k puntos; extrapolación no lineal"},
        {"etapa": "Ingesta + priors + gating", "metodo": "Polars/cKDTree", "n": 400006, "tiempo_s": float(s4.iloc[0]["latencia_arm64_s"]), "ram_mb": np.nan, "nota": "2024-09-09"},
        {"etapa": "Inferencia dual", "metodo": "LightGBM", "n": 400006, "tiempo_s": float(s4.iloc[1]["latencia_arm64_s"]), "ram_mb": np.nan, "nota": "2024-09-09"},
        {"etapa": "Agregación/boletín", "metodo": "Polars", "n": 400006, "tiempo_s": float(s4.iloc[2]["latencia_arm64_s"]), "ram_mb": np.nan, "nota": "2024-09-09"},
        {"etapa": "Total pipeline", "metodo": "Consolidado", "n": 400006, "tiempo_s": float(total["latencia_arm64_s"]), "ram_mb": float(total["ram_pico_mb"]), "nota": "ARM64; x86_64 no medido"},
    ]
    out = pd.DataFrame(filas)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIR_TABLAS / "tabla12_rendimiento_computacional.csv", index=False)
    (DIR_TABLAS / "tabla12_rendimiento_computacional.md").write_text(
        "# Tabla 12. Rendimiento computacional\n\n" + out.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
