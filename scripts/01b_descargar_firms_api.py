"""
Script 01b: Ingesta masiva automatizada de focos NASA FIRMS (VIIRS 375m) 2018–2025.
- Utiliza la NASA FIRMS Area API oficial con la MAP_KEY autenticada.
- Descarga en paralelo por ventanas de 5 días para Suomi-NPP y NOAA-20.
- Checkpointing en disco por mes en datos/raw/firms/batches/.
- Aplica filtros de teledetección: Bow-Tie (scan <= 1.5 km), exclusión de confianza low.
- Spatial Join determinista O(1) con la grilla nacional de 400.006 celdas.
- Genera el dataset maestro en datos/processed/focos_viirs_diarios.parquet.
"""

from pathlib import Path
import sys
import os
import argparse
from datetime import datetime, timedelta
import concurrent.futures
import io
import requests
import pandas as pd
import polars as pl

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_RAW, DIR_PROCESSED, BBOX_PARAGUAY
from src.firms import procesar_focos_firms_a_grilla, agregar_focos_diarios

# Obtener clave de API desde variable de entorno de forma segura
MAP_KEY_DEFAULT = os.environ.get("FIRMS_MAP_KEY", "")
BBOX_STR = f"{BBOX_PARAGUAY['min_lon']},{BBOX_PARAGUAY['min_lat']},{BBOX_PARAGUAY['max_lon']},{BBOX_PARAGUAY['max_lat']}"

def _descargar_intervalo(
    map_key: str,
    source: str,
    fecha_str: str,
    dias_rango: int = 5,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Descarga un rango de 5 días de focos activos para Paraguay desde NASA FIRMS API.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{BBOX_STR}/{dias_rango}/{fecha_str}"
    
    for intento in range(max_retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                if len(lines) > 1:
                    return pd.read_csv(io.StringIO(r.text))
                return pd.DataFrame()
            elif r.status_code == 429:
                import time
                time.sleep(2.0 * (intento + 1))
            else:
                return pd.DataFrame()
        except Exception:
            if intento == max_retries - 1:
                return pd.DataFrame()
    return pd.DataFrame()

def descargar_mes_satelital(
    map_key: str,
    source: str,
    anio: int,
    mes: int,
    cache_dir: Path
) -> pd.DataFrame:
    """
    Descarga y consolida un mes completo para una fuente satelital dada con checkpointing.
    """
    archivo_cache = cache_dir / f"focos_{source}_{anio}_{mes:02d}.parquet"
    if archivo_cache.exists():
        try:
            return pd.read_parquet(archivo_cache)
        except Exception:
            archivo_cache.unlink(missing_ok=True)

    # Generar ventanas de 5 días para el mes
    inicio_mes = datetime(anio, mes, 1)
    if mes == 12:
        fin_mes = datetime(anio + 1, 1, 1) - timedelta(days=1)
    else:
        fin_mes = datetime(anio, mes + 1, 1) - timedelta(days=1)

    # Si es fecha futura respecto a hoy, truncar
    hoy = datetime.now()
    if inicio_mes > hoy:
        return pd.DataFrame()
    if fin_mes > hoy:
        fin_mes = hoy

    fechas_pasos = []
    curr = inicio_mes
    while curr <= fin_mes:
        fechas_pasos.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=5)

    dfs_intervalos = []
    for f_str in fechas_pasos:
        df_int = _descargar_intervalo(map_key, source, f_str, dias_rango=5)
        if not df_int.empty:
            dfs_intervalos.append(df_int)

    if dfs_intervalos:
        df_mes = pd.concat(dfs_intervalos, ignore_index=True)
        # Deduplicar si se solaparon días
        cols_dedup = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df_mes.columns]
        if cols_dedup:
            df_mes = df_mes.drop_duplicates(subset=cols_dedup)
        # Guardar checkpoint
        df_mes.to_parquet(archivo_cache, index=False)
        return df_mes

    return pd.DataFrame()

def ejecutar_descarga_masiva_firms(
    start_year: int = 2018,
    end_year: int = 2025,
    map_key: str = MAP_KEY_DEFAULT,
    max_workers: int = 4
):
    print("=" * 65)
    print(f"INGESTA MASIVA NASA FIRMS (VIIRS 375m) PARAGUAY: {start_year} a {end_year}")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if not path_grilla.exists():
        raise FileNotFoundError(f"Grilla no encontrada en {path_grilla}. Ejecuta scripts/00_generar_grilla_nacional.py primero.")

    print(f"1. Cargando grilla nacional de 1 km² ({path_grilla.name})...")
    df_grilla = pl.read_parquet(path_grilla)
    print(f"   ✓ {len(df_grilla):,} celdas soberanas listas.")

    dir_cache = DIR_RAW / "firms" / "batches"
    dir_cache.mkdir(parents=True, exist_ok=True)

    # Definir fuentes satelitales según el período histórico vs NRT
    # Suomi-NPP: Standard Processing (SP) hasta mediados 2026, NRT después
    # NOAA-20: SP desde 2018-04, NRT después
    tareas = []
    for anio in range(start_year, end_year + 1):
        for mes in range(1, 13):
            # Suomi-NPP
            tareas.append((map_key, "VIIRS_SNPP_SP", anio, mes, dir_cache))
            # NOAA-20 (operativo desde abril de 2018)
            if not (anio == 2018 and mes < 4):
                tareas.append((map_key, "VIIRS_NOAA20_SP", anio, mes, dir_cache))

    print(f"\n2. Descargando {len(tareas)} meses satelitales con {max_workers} trabajadores paralelos...")

    dfs_consolidados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(descargar_mes_satelital, *t): t
            for t in tareas
        }

        completados = 0
        total = len(tareas)
        for futuro in concurrent.futures.as_completed(futuros):
            t = futuros[futuro]
            completados += 1
            try:
                df_res = futuro.result()
                if not df_res.empty:
                    dfs_consolidados.append(df_res)
                if completados % 10 == 0 or completados == total:
                    print(f" • Progreso: {completados}/{total} meses procesados ({completados*100/total:.1f}%)")
            except Exception as e:
                print(f"Error procesando {t[1]} {t[2]}-{t[3]:02d}: {e}")

    if not dfs_consolidados:
        print("No se obtuvieron registros de focos satelitales.")
        return

    print("\n3. Consolidando y deduplicando archivo histórico...")
    df_raw_total = pd.concat(dfs_consolidados, ignore_index=True)
    cols_dedup = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df_raw_total.columns]
    if cols_dedup:
        antes = len(df_raw_total)
        df_raw_total = df_raw_total.drop_duplicates(subset=cols_dedup)
        print(f"   ✓ Detecciones crudas totales: {antes:,} -> Deduplicadas: {len(df_raw_total):,}")

    archivo_raw_parquet = DIR_RAW / "firms" / f"viirs_historico_{start_year}_{end_year}_crudo.parquet"
    df_raw_total.to_parquet(archivo_raw_parquet, index=False)
    print(f"   ✓ Archivo crudo inmutable guardado: {archivo_raw_parquet.name}")

    print("\n4. Aplicando filtros de teledetección (Bow-Tie scan<=1.5km, confianza!=low) y spatial join O(1)...")
    pl_focos_filtrados = procesar_focos_firms_a_grilla(
        df_raw=df_raw_total,
        df_grilla=df_grilla,
        max_scan_km=1.5,
        excluir_baja_confianza=True
    )
    print(f"   ✓ Focos validados y espacializados dentro de Paraguay: {len(pl_focos_filtrados):,}")

    print("\n5. Agregación diaria por celda de 1 km²...")
    df_diario = agregar_focos_diarios(pl_focos_filtrados)
    print(f"   ✓ Registros celda-día con evento de fuego positivo: {len(df_diario):,}")
    print(f"   ✓ Ventana temporal cubierta: {df_diario['acq_date'].min()} a {df_diario['acq_date'].max()}")

    out_parquet = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    df_diario.write_parquet(out_parquet, compression="zstd")
    print(f"\n✓ Dataset consolidado guardado en: {out_parquet}")
    print(f"  Tamaño en disco: {out_parquet.stat().st_size / (1024*1024):.2f} MB")

    resumen_fuego = df_diario.group_by("region").agg([
        pl.len().alias("celda_dias_quemados"),
        pl.col("max_frp_dia").max().alias("frp_maximo_mw"),
        pl.col("sum_frp_dia").mean().alias("frp_promedio_mw")
    ])
    print("\nResumen consolidado por macro-región:")
    for row in resumen_fuego.iter_rows():
        print(f" • {row[0].upper()}: {row[1]:,} celda-días | FRP Máx: {row[2]:.1f} MW | FRP Medio: {row[3]:.1f} MW")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta masiva NASA FIRMS API 2018-2025")
    parser.add_argument("--start", type=int, default=2018, help="Año inicial")
    parser.add_argument("--end", type=int, default=2025, help="Año final")
    parser.add_argument("--workers", type=int, default=4, help="Hilos paralelos")
    args = parser.parse_args()

    ejecutar_descarga_masiva_firms(start_year=args.start, end_year=args.end, max_workers=args.workers)
