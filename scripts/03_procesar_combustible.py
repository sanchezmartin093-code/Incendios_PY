"""
Script 03: Land Cover Processing and MapBiomas Fuel Typing.
- Quantifies percentages of pasture, savanna, dry/humid forest and palm groves at 1 km².
- Applies the hard biophysical exclusion mask (permanent water and urban areas).
- Generates the structured dataset at datos/processed/combustible_mapbiomas_1km.parquet.

IMPORTANT: this repository does not include the MapBiomas GeoTIFFs. If they are
present in datos/raw/mapbiomas/, real fractions are extracted with
`procesar_geotiff_mapbiomas_a_grilla`. Otherwise, the `sintetico_referencia`
profile only exists for demos/tests and must NOT be used to claim biophysical results.
"""

from pathlib import Path
import sys
import argparse

# Ensure the project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.constantes import DIR_PROCESSED, DIR_RAW
from src.biomasa import (
    ORIGEN_PERFIL_SINTETICO,
    generar_perfil_combustible_base,
    procesar_geotiff_mapbiomas_a_grilla,
    seleccionar_tif_estatico,
)


def main(permitir_sintetico: bool = False, anio_mapa: int = 2023):
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

    try:
        ruta_estatica = seleccionar_tif_estatico(dir_mapbiomas_raw, anio_mapa)
        tifs_locales = [ruta_estatica]
    except FileNotFoundError:
        tifs_locales = []

    if tifs_locales:
        print(
            f"\nExtrayendo fracciones reales de MapBiomas desde el mapa estático "
            f"{anio_mapa}: {ruta_estatica.name}"
        )
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

    # 1. Biophysical exclusion analysis (Stage 1 Gating)
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

    # 2. Save dataset in ZSTD Parquet
    out_parquet = DIR_PROCESSED / "combustible_mapbiomas_1km.parquet"
    df_combustible.write_parquet(out_parquet, compression="zstd")
    print(f"\n✓ Dataset de combustible guardado en: {out_parquet}")
    print(f"  Tamaño en disco: {out_parquet.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Origen declarado: {df_combustible['origen_datos'].unique().to_list()}")

    # 3. Statistical summary by ecoregion
    df_join = df_combustible.join(df_grilla.select(["cell_id", "region"]), on="cell_id")
    resumen = df_join.group_by("region").agg([
        pl.col("pct_pastura").mean().alias("media_pastura_pct"),
        pl.col("pct_pastura_ganadera").mean().alias("media_pastura_gan_pct"),
        pl.col("pct_agricultura_anual").mean().alias("media_agricultura_pct"),
        pl.col("pct_plantacion_forestal").mean().alias("media_plantacion_pct"),
        pl.col("pct_sabana_pastizal").mean().alias("media_sabana_pct"),
        pl.col("pct_bosque_seco").mean().alias("media_bosque_seco_pct"),
        pl.col("pct_bosque_humedo").mean().alias("media_bosque_humedo_pct"),
        pl.col("pct_humedal_palmar").mean().alias("media_palmar_pct"),
        pl.col("borde_bosque_agro_pct").mean().alias("media_borde_pct"),
    ])

    print("\nComposición media de combustible vegetal por región:")
    for row in resumen.iter_rows():
        print(f"\n--- {row[0].upper()} ---")
        print(f" • Pastura total (compatibilidad): {row[1]:.1f}%")
        print(f" • Pastura ganadera (15):          {row[2]:.1f}%")
        print(f" • Agricultura anual (18):         {row[3]:.1f}%")
        print(f" • Plantación forestal (9):        {row[4]:.1f}%")
        print(f" • Sabanas y pastizales nativos:   {row[5]:.1f}%")
        print(f" • Bosque seco chaqueño:           {row[6]:.1f}%")
        print(f" • Bosque Atlántico (BAAPA):       {row[7]:.1f}%")
        print(f" • Estero / humedal (11):          {row[8]:.1f}%")
        print(f" • Borde bosque-agro:              {row[9]:.2f}%")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento de combustible MapBiomas")
    parser.add_argument(
        "--permitir-sintetico",
        action="store_true",
        help="Generar perfil sintético de referencia si no hay GeoTIFF reales",
    )
    parser.add_argument(
        "--anio-mapa",
        type=int,
        default=2023,
        help="Año único del mapa MapBiomas para el perfil estático (por defecto 2023)",
    )
    args = parser.parse_args()
    main(permitir_sintetico=args.permitir_sintetico, anio_mapa=args.anio_mapa)
