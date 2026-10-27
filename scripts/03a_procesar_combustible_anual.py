"""
Script 03a: annual MapBiomas fuel table with causal prior-year assignment.

For each modelling year Y (2018-2026), the fuel features come from the
MapBiomas Paraguay Collection 2 map of Y-1. Because Collection 2 ends in 2023:

- 2018-2024 use maps 2017-2023 (lag 1 year);
- 2025 and 2026 reuse the 2023 map (lag 2 and 3 years).

The output is a long table keyed by ``cell_id, anio_fila``. ``anio_mapa`` and
``fuel_lag`` are kept for auditing and for the manuscript.
"""

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.biomasa import procesar_geotiff_mapbiomas_a_grilla
from src.constantes import DIR_PROCESSED, DIR_RAW

ANIO_MIN_FILA = 2018
ANIO_MAX_FILA = 2026
ULTIMO_MAPA_C2 = 2023


def _mapas_disponibles(dir_mapbiomas: Path) -> dict[int, Path]:
    mapas: dict[int, Path] = {}
    for ruta in sorted(dir_mapbiomas.glob("mapbiomas_paraguay_collection2_*.tif")):
        m = re.search(r"collection2_(\d{4})\.tif$", ruta.name)
        if m:
            mapas[int(m.group(1))] = ruta
    return mapas


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabla anual de combustible MapBiomas")
    parser.add_argument("--anio-min", type=int, default=ANIO_MIN_FILA)
    parser.add_argument("--anio-max", type=int, default=ANIO_MAX_FILA)
    args = parser.parse_args()

    dir_mapbiomas = DIR_RAW / "mapbiomas"
    mapas = _mapas_disponibles(dir_mapbiomas)
    requeridos = set(range(2017, ULTIMO_MAPA_C2 + 1))
    faltantes = sorted(requeridos - set(mapas))
    if faltantes:
        raise FileNotFoundError(f"Faltan mapas MapBiomas anuales: {faltantes}")

    df_grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    cache: dict[int, pl.DataFrame] = {}
    tablas: list[pl.DataFrame] = []

    for anio_fila in range(args.anio_min, args.anio_max + 1):
        anio_mapa = min(anio_fila - 1, ULTIMO_MAPA_C2)
        if anio_mapa not in cache:
            print(f"\nProcesando mapa {anio_mapa}: {mapas[anio_mapa].name}")
            cache[anio_mapa] = procesar_geotiff_mapbiomas_a_grilla(
                df_grilla, [mapas[anio_mapa]]
            )
        df = cache[anio_mapa].with_columns([
            pl.lit(anio_fila, dtype=pl.Int32).alias("anio_fila"),
            pl.lit(anio_mapa, dtype=pl.Int32).alias("anio_mapa"),
            pl.lit(anio_fila - anio_mapa, dtype=pl.Int16).alias("fuel_lag"),
        ])
        tablas.append(df)
        print(
            f"  {anio_fila}: mapa {anio_mapa} "
            f"(lag={anio_fila - anio_mapa})"
        )

    salida = pl.concat(tablas, how="vertical")
    if not salida.select(["cell_id", "anio_fila"]).is_unique().all():
        raise ValueError("La tabla anual tiene duplicados cell_id/anio_fila")
    if salida["fuel_lag"].min() < 1:
        raise ValueError("Hay filas con fuel_lag < 1: se estaría usando el mapa del mismo año")

    componentes = salida.select([
        (
            pl.col("pct_pastura")
            - (
                pl.col("pct_pastura_ganadera")
                + pl.col("pct_agricultura_anual")
                + pl.col("pct_plantacion_forestal")
            )
        ).abs().max().alias("max_error_pastura"),
        pl.col("borde_bosque_agro_pct").max().alias("borde_max"),
    ]).row(0, named=True)
    if componentes["max_error_pastura"] > 1e-3:
        raise ValueError(f"pct_pastura no suma componentes: {componentes['max_error_pastura']}")
    if componentes["borde_max"] > 100.0:
        raise ValueError(f"borde > 100: {componentes['borde_max']}")

    out = DIR_PROCESSED / "combustible_mapbiomas_1km_anual.parquet"
    salida.write_parquet(out, compression="zstd")
    meta = {
        "fuente": "MapBiomas Paraguay Collection 2",
        "anio_fila": [args.anio_min, args.anio_max],
        "anio_mapa": [2017, ULTIMO_MAPA_C2],
        "regla": "anio_mapa = min(anio_fila - 1, 2023)",
        "filas": salida.height,
        "celdas": salida["cell_id"].n_unique(),
        "archivo": out.name,
        "mapas": {str(k): v.name for k, v in sorted(mapas.items()) if k <= ULTIMO_MAPA_C2},
    }
    (DIR_PROCESSED / "combustible_mapbiomas_1km_anual.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✓ Tabla anual: {out} ({salida.height:,} filas, {salida['cell_id'].n_unique():,} celdas)")
    print(f"  Rango de filas: {salida['anio_fila'].min()}–{salida['anio_fila'].max()}")
    print(f"  fuel_lag máximo: {salida['fuel_lag'].max()}")


if __name__ == "__main__":
    main()
