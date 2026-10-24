"""
Script 02: Ingesta meteorológica ERA5-Land y cálculo biofísico de FWI / VPD para Paraguay.
- Genera la malla de nodos meteorológicos a 0.25° (~25 km, resolución nativa del reanálisis).
- Asigna nodo_clima_id a las 400.006 celdas de la grilla nacional en O(1) con cKDTree.
- Descarga variables meteorológicas diarias por lotes desde Open-Meteo Historical Archive API.
- Calcula VPD máximo diario, lluvia 48h, días secos consecutivos (CDD), FWI FFMC e ISI.
- Exporta en formato columnar ZSTD: datos/processed/clima_era5_diario.parquet.
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
from src.clima import (
    generar_malla_nodos_clima,
    asignar_nodos_a_grilla,
    descargar_clima_openmeteo_batch,
    calcular_indices_fwi_y_sequedad
)

def ejecutar_pipeline_clima(start_date: str = "2024-09-01", end_date: str = "2024-09-10", limit_nodos: int | None = None):
    print("=" * 65)
    print(f"INGESTA Y PROCESAMIENTO METEOROLÓGICO ERA5-LAND: {start_date} a {end_date}")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if not path_grilla.exists():
        raise FileNotFoundError(f"Grilla no encontrada en {path_grilla}. Ejecuta scripts/00_generar_grilla_nacional.py primero.")

    print(f"1. Cargando grilla nacional de 1 km² desde: {path_grilla.name}...")
    df_grilla = pl.read_parquet(path_grilla)

    # 1. Generar malla de nodos meteorológicos
    print("2. Generando malla de nodos meteorológicos a resolución nativa de 0.25° (~25 km)...")
    df_nodos = generar_malla_nodos_clima(resolucion_deg=0.25)
    
    # Asignar nodo más cercano a cada celda de 1 km
    df_grilla_actualizada = asignar_nodos_a_grilla(df_grilla, df_nodos)
    df_grilla_actualizada.write_parquet(path_grilla, compression="zstd")
    print(f"   ✓ Asignados {df_grilla_actualizada['nodo_clima_id'].n_unique()} nodos representativos a 400.006 celdas.")

    # Filtrar solo los nodos que efectivamente cubren celdas dentro de Paraguay
    nodos_activos = df_grilla_actualizada["nodo_clima_id"].unique().to_list()
    df_nodos_filtrados = df_nodos.filter(pl.col("nodo_clima_id").is_in(nodos_activos))
    
    if limit_nodos:
        print(f"   (Modo acotado de prueba: limitando a {limit_nodos} nodos)")
        df_nodos_filtrados = df_nodos_filtrados.head(limit_nodos)

    lats = df_nodos_filtrados["lat"].to_list()
    lons = df_nodos_filtrados["lon"].to_list()
    ids = df_nodos_filtrados["nodo_clima_id"].to_list()

    print(f"\n3. Descargando series diarias ERA5-Land para {len(ids)} nodos meteorológicos...")
    dir_era5_raw = DIR_RAW / "era5"
    dir_era5_raw.mkdir(parents=True, exist_ok=True)

    df_clima_raw = descargar_clima_openmeteo_batch(
        lats=lats,
        lons=lons,
        nodo_ids=ids,
        start_date=start_date,
        end_date=end_date,
        batch_size=50,
        delay_seg=0.2
    )
    print(f"   ✓ Registros climáticos diarios descargados: {len(df_clima_raw):,}")

    # Guardar copia cruda inmutable
    raw_file = dir_era5_raw / f"era5_land_crudo_{start_date}_{end_date}.parquet"
    df_clima_raw.write_parquet(raw_file, compression="zstd")

    # 4. Cálculo de variables biofísicas (FWI FFMC, ISI, CDD, VPD, lluvia 48h)
    print("\n4. Calculando índices biofísicos de fuego (VPD, CDD, FWI FFMC, ISI e interacción viento)...")
    df_clima_procesado = calcular_indices_fwi_y_sequedad(df_clima_raw)

    # 5. Guardar en datos/processed/clima_era5_diario.parquet
    out_parquet = DIR_PROCESSED / "clima_era5_diario.parquet"
    df_clima_procesado.write_parquet(out_parquet, compression="zstd")
    print(f"\n✓ Dataset meteorológico procesado guardado en: {out_parquet}")
    print(f"  Tamaño en disco: {out_parquet.stat().st_size / 1024:.2f} KB")

    # Mostrar resumen estadístico de variables críticas
    print("\nResumen Estadístico de Variables Físicas de Peligro de Fuego:")
    print(df_clima_procesado.select([
        "temp_max_c", "vpd_max_kpa", "viento_max_kmh", 
        "dias_secos_consecutivos", "ffmc", "isi"
    ]).describe())

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y procesamiento meteorológico ERA5-Land")
    parser.add_argument("--start", default="2024-09-01", help="Fecha inicial AAAA-MM-DD")
    parser.add_argument("--end", default="2024-09-10", help="Fecha final AAAA-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Limitar cantidad de nodos para pruebas")
    args = parser.parse_args()

    ejecutar_pipeline_clima(start_date=args.start, end_date=args.end, limit_nodos=args.limit)
