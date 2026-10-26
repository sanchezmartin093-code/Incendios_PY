"""
Script 01: Ingesta, filtrado de calidad y armonización espacial de focos NASA FIRMS (VIIRS 375m).
- Ingesta de fuentes NRT abiertas (Suomi-NPP y NOAA-20) y archivos anuales históricos.
- Filtrado por distorsión óptica / Bow-Tie (scan <= 1.5 km) y calidad.
- Spatial Join O(1) con la grilla nacional de 1 km² (datos/processed/grilla_nacional_1km.parquet).
- Exportación en formato columnar comprimido ZSTD: datos/processed/focos_viirs_diarios.parquet.
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import zipfile
import glob
import pandas as pd
import polars as pl
from src.constantes import DIR_RAW, DIR_PROCESSED
from src.firms import procesar_focos_firms_a_grilla, agregar_focos_diarios

# URLs abiertas de NASA FIRMS para pases de Sudamérica (Ventana NRT 7 días como fallback operativo)
URL_VIIRS_SNPP_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_America_7d.csv"
URL_VIIRS_NOAA20_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_America_7d.csv"
URL_VIIRS_NOAA21_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_South_America_7d.csv"

SENSORES_FALLBACK_NRT = [
    ("Suomi-NPP", URL_VIIRS_SNPP_7D, "viirs_suomi-npp_nrt_7d.csv"),
    ("NOAA-20", URL_VIIRS_NOAA20_7D, "viirs_noaa-20_nrt_7d.csv"),
    ("NOAA-21", URL_VIIRS_NOAA21_7D, "viirs_noaa-21_nrt_7d.csv")
]

def descargar_e_ingestar_firms():
    print("=" * 65)
    print("INGESTA Y ARMONIZACIÓN DE NASA FIRMS (VIIRS 375m) - PARAGUAY")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if not path_grilla.exists():
        raise FileNotFoundError(f"Grilla nacional no encontrada en {path_grilla}. Ejecuta scripts/00_generar_grilla_nacional.py primero.")

    print(f"Cargando grilla nacional de 1 km² desde: {path_grilla.name}")
    df_grilla = pl.read_parquet(path_grilla)
    print(f"Total celdas soberanas disponibles: {len(df_grilla):,}")

    dir_firms_raw = DIR_RAW / "firms"
    dir_firms_raw.mkdir(parents=True, exist_ok=True)

    # 1. Descomprimir automáticamente cualquier paquete ZIP histórico presente
    archivos_zip = list(dir_firms_raw.glob("*.zip"))
    if archivos_zip:
        print(f"\nDetectados {len(archivos_zip)} paquete(s) ZIP en {dir_firms_raw.name}:")
        for p_zip in archivos_zip:
            print(f" • Descomprimiendo: {p_zip.name}...")
            with zipfile.ZipFile(p_zip, "r") as zf:
                zf.extractall(dir_firms_raw)

    # 2. Buscar archivos CSV históricos o NRT en el directorio
    archivos_csv = list(dir_firms_raw.glob("*.csv"))
    dfs_crudos = []

    # Priorizar archivos históricos si existen
    csvs_historicos = [c for c in archivos_csv if "nrt" not in c.name.lower()]
    csvs_a_cargar = csvs_historicos if csvs_historicos else archivos_csv

    if csvs_a_cargar:
        print(f"\nCargando {len(csvs_a_cargar)} archivo(s) CSV satelitales:")
        for ruta_csv in csvs_a_cargar:
            print(f" - Leyendo: {ruta_csv.name} ({ruta_csv.stat().st_size / (1024*1024):.2f} MB)...")
            try:
                # Lectura flexible
                df_sub = pd.read_csv(ruta_csv, low_memory=False)
                dfs_crudos.append(df_sub)
                print(f"   ✓ {len(df_sub):,} registros cargados.")
            except Exception as e:
                print(f"   ✗ Error leyendo {ruta_csv.name}: {e}")
    else:
        # Fallback a descarga NRT si no hay ningún CSV
        print("\nNo se encontraron archivos CSV históricos locales. Descargando feed operativo NRT...")
        for nombre_sensor, url, archivo_nom in SENSORES_FALLBACK_NRT:
            ruta_archivo = dir_firms_raw / archivo_nom
            try:
                df_temp = pd.read_csv(url)
                print(f"   ✓ {nombre_sensor}: {len(df_temp):,} detecciones recibidas.")
                df_temp.to_csv(ruta_archivo, index=False)
                dfs_crudos.append(df_temp)
            except Exception as e:
                print(f"   ✗ Error descargando {nombre_sensor}: {e}")

    if not dfs_crudos:
        print("No se pudieron cargar datos de focos térmicos.")
        return

    # 3. Concatenar y deduplicar registros crudos
    df_total_raw = pd.concat(dfs_crudos, ignore_index=True)
    cols_dedup = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df_total_raw.columns]
    if cols_dedup:
        len_antes = len(df_total_raw)
        df_total_raw = df_total_raw.drop_duplicates(subset=cols_dedup)
        print(f"\nTotal consolidado crudo: {len_antes:,} -> Deduplicados: {len(df_total_raw):,}")
    else:
        print(f"\nTotal consolidado de detecciones térmicas crudas: {len(df_total_raw):,}")

    if "acq_date" in df_total_raw.columns:
        print(f"Ventana temporal cubierta: {df_total_raw['acq_date'].min()} a {df_total_raw['acq_date'].max()}")

    # 4. Procesar, filtrar Bow-Tie y mapear a cell_id O(1)
    print("\nAplicando filtros de teledetección (Bow-Tie scan<=1.5km, confianza!=low) y spatial join O(1)...")
    pl_focos_filtrados = procesar_focos_firms_a_grilla(
        df_raw=df_total_raw,
        df_grilla=df_grilla,
        max_scan_km=1.5,
        excluir_baja_confianza=True
    )
    print(f"Focos validados y espacializados dentro de Paraguay: {len(pl_focos_filtrados):,}")

    # 5. Agregación diaria por celda
    print("Agregando focos diarios por celda de 1 km²...")
    df_diario = agregar_focos_diarios(pl_focos_filtrados)
    print(f"Total de registros celda-día con evento de fuego: {len(df_diario):,}")

    # 6. Guardar en Parquet comprimido ZSTD
    out_parquet = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    df_diario.write_parquet(out_parquet, compression="zstd")
    print(f"\nDataset consolidado guardado en: {out_parquet}")
    print(f"Tamaño en disco: {out_parquet.stat().st_size / 1024:.2f} KB")

    # Resumen regional
    resumen_fuego = df_diario.group_by("region").agg([
        pl.len().alias("celda_dias_quemados"),
        pl.col("max_frp_dia").max().alias("frp_maximo_mw"),
        pl.col("sum_frp_dia").mean().alias("frp_promedio_mw")
    ])
    print("\nResumen de actividad de fuego por región:")
    for row in resumen_fuego.iter_rows():
        print(f" - {row[0]}: {row[1]:,} eventos | FRP Máx: {row[2]:.1f} MW | FRP Medio: {row[3]:.1f} MW")

    print("=" * 65)

if __name__ == "__main__":
    descargar_e_ingestar_firms()
