"""
Script 03: Procesamiento de Cobertura de Suelo y Tipificación de Combustibles MapBiomas.
- Cuantifica porcentajes de pastura, sabana, bosque seco/húmedo y palmares a 1 km².
- Aplica la máscara dura de exclusión biofísica (agua permanente y cascos urbanos).
- Genera el dataset estructurado en datos/processed/combustible_mapbiomas_1km.parquet.

IMPORTANTE: este repositorio no incluye los GeoTIFF de MapBiomas. Si están
presentes en datos/raw/mapbiomas/, se extraen fracciones reales con
`procesar_geotiff_mapbiomas_a_grilla`. Si no, el perfil `sintetico_referencia`
sólo existe para demos/tests y NO debe usarse para afirmar resultados biofísicos.
"""

from pathlib import Path
import sys
import argparse

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.constantes import DIR_PROCESSED, DIR_RAW
from src.biomasa import (
    ORIGEN_PERFIL_SINTETICO,
    generar_perfil_combustible_base,
    procesar_geotiff_mapbiomas_a_grilla,
)


def main(permitir_sintetico: bool = False):
    print("=" * 65)
    print("PROCESAMIENTO DE COMBUSTIBLE Y BIOMASA MAPBIOMAS - PARAGUAY")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if not path_grilla.exists():
        raise FileNotFoundError(
            f"Grilla no encontrada en {path_grilla}. "
            "Ejecuta scripts/00_generar_grilla_nacional.py primero."
        )

    print(f"Cargando grilla nacional de 1 km² desde: {path_grilla.name}...")
    df_grilla = pl.read_parquet(path_grilla)
    total_celdas = len(df_grilla)
    print(f"Total celdas soberanas: {total_celdas:,}")

    dir_mapbiomas_raw = DIR_RAW / "mapbiomas"
    dir_mapbiomas_raw.mkdir(parents=True, exist_ok=True)
    tifs_locales = sorted(dir_mapbiomas_raw.glob("*.tif"))

    if tifs_locales:
        print(f"\nExtrayendo fracciones reales de MapBiomas desde {len(tifs_locales)} GeoTIFF...")
        df_combustible = procesar_geotiff_mapbiomas_a_grilla(df_grilla, tifs_locales)
    else:
        if not permitir_sintetico:
            raise FileNotFoundError(
                "No hay GeoTIFF de MapBiomas en datos/raw/mapbiomas/. "
                "Para una demo explícita ejecute con --permitir-sintetico; el "
                "archivo resultante quedará marcado como sintético y no debe "
                "usarse en el manuscrito."
            )
        print("\n[ADVERTENCIA] Generando perfil SINTÉTICO de referencia.")
        print("Este archivo NO representa cobertura real de MapBiomas.")
        df_combustible = generar_perfil_combustible_base(df_grilla)

    # 1. Análisis de exclusión biofísica (Etapa 1 Gating)
    no_combustibles = df_combustible.filter(~pl.col("es_combustible"))
    combustibles = df_combustible.filter(pl.col("es_combustible"))

    print(
        f"\nCeldas clasificadas como COMBUSTIBLES: {len(combustibles):,} "
        f"({len(combustibles)/total_celdas*100:.1f}%)"
    )
    print(
        f"Celdas EXCLUIDAS (No combustibles: agua permanente / urbano): "
        f"{len(no_combustibles):,} ({len(no_combustibles)/total_celdas*100:.1f}%)"
    )

    # 2. Guardar dataset en Parquet ZSTD
    out_parquet = DIR_PROCESSED / "combustible_mapbiomas_1km.parquet"
    df_combustible.write_parquet(out_parquet, compression="zstd")
    print(f"\n✓ Dataset de combustible guardado en: {out_parquet}")
    print(f"  Tamaño en disco: {out_parquet.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Origen declarado: {df_combustible["origen_datos"].unique().to_list()}")

    # 3. Resumen estadístico por ecorregión
    df_join = df_combustible.join(df_grilla.select(["cell_id", "region"]), on="cell_id")
    resumen = df_join.group_by("region").agg([
        pl.col("pct_pastura").mean().alias("media_pastura_pct"),
        pl.col("pct_sabana_pastizal").mean().alias("media_sabana_pct"),
        pl.col("pct_bosque_seco").mean().alias("media_bosque_seco_pct"),
        pl.col("pct_bosque_humedo").mean().alias("media_bosque_humedo_pct"),
        pl.col("pct_humedal_palmar").mean().alias("media_palmar_pct")
    ])

    print("\nComposición media de combustible vegetal por región (perfil sintético):")
    for row in resumen.iter_rows():
        print(f"\n--- {row[0].upper()} ---")
        print(f" • Pasturas ganaderas/agrícolas: {row[1]:.1f}%")
        print(f" • Sabanas y pastizales nativos: {row[2]:.1f}%")
        print(f" • Bosque seco chaqueño:         {row[3]:.1f}%")
        print(f" • Bosque Atlántico (BAAPA):     {row[4]:.1f}%")
        print(f" • Palmares de karanda'y:        {row[5]:.1f}%")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento de combustible MapBiomas")
    parser.add_argument(
        "--permitir-sintetico",
        action="store_true",
        help="Generar perfil sintético de referencia si no hay GeoTIFF reales",
    )
    args = parser.parse_args()
    main(permitir_sintetico=args.permitir_sintetico)
