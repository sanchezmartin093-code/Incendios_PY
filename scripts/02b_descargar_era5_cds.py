"""
Script 02b: Massive ERA5-Land meteorological ingestion (2018–2025) from Copernicus CDS.
- Downloads yearly NetCDF files for the Paraguay grid from ECMWF CDS.
- Extracts and processes the 640 weather nodes in xarray/Polars matrix format.
- Computes biophysical fire indices: daily VPD (Tetens), CDD, 48h rainfall, FFMC, ISI.
- Exports in ZSTD columnar format: datos/processed/clima_era5_diario.parquet.
"""

from pathlib import Path
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cdsapi
import polars as pl
from src.constantes import DIR_PROCESSED, DIR_RAW
from src.clima import generar_malla_nodos_clima, asignar_nodos_a_grilla, calcular_indices_fwi_y_sequedad
from src.clima_cds import descargar_trimestre_cds, procesar_nc_a_nodos_diarios

def procesar_un_trimestre(
    anio: int,
    q: int,
    dir_cds_raw: Path,
    df_nodos_filtrados: pl.DataFrame,
    forzar: bool = False,
) -> pl.DataFrame | None:
    archivo_parquet_q = dir_cds_raw / f"lote_clima_{anio}_Q{q}.parquet"
    if archivo_parquet_q.exists() and not forzar:
        print(f" ✓ Lote en caché: {archivo_parquet_q.name}")
        return pl.read_parquet(archivo_parquet_q)

    print(f"\n--- PROCESANDO: AÑO {anio} - TRIMESTRE Q{q} ---")
    client = cdsapi.Client()
    try:
        ruta_nc = descargar_trimestre_cds(client, anio, q, dir_cds_raw, forzar=forzar)
        df_q = procesar_nc_a_nodos_diarios(ruta_nc, df_nodos_filtrados)
        print(f"   ✓ {len(df_q):,} registros diarios procesados para {anio} Q{q}.")
        df_q.write_parquet(archivo_parquet_q, compression="zstd")
        return df_q
    except Exception as e:
        print(f"   ✗ Error procesando {anio} Q{q}: {e}")
        return None

def ejecutar_pipeline_cds(anios: list[int], max_workers: int = 2, forzar: bool = False):
    print("=" * 65)
    print(f"INGESTA ERA5-LAND DESDE COPERNICUS CDS: AÑOS {anios[0]} a {anios[-1]}")
    print(f"Workers concurrentes: {max_workers}")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if not path_grilla.exists():
        raise FileNotFoundError(f"Grilla no encontrada en {path_grilla}. Ejecuta scripts/00_generar_grilla_nacional.py primero.")

    print(f"1. Cargando grilla nacional y nodos meteorológicos...")
    df_grilla = pl.read_parquet(path_grilla)
    df_nodos = generar_malla_nodos_clima(resolucion_deg=0.25)
    df_grilla = asignar_nodos_a_grilla(df_grilla, df_nodos)
    # Persist the assignment so later scripts (04) can perform
    # the join by nodo_clima_id even if script 02 has not been run.
    df_grilla.write_parquet(path_grilla, compression="zstd")

    nodos_activos = df_grilla["nodo_clima_id"].unique().to_list()
    df_nodos_filtrados = df_nodos.filter(pl.col("nodo_clima_id").is_in(nodos_activos))
    print(f"   ✓ {len(df_nodos_filtrados)} nodos meteorológicos activos sobre Paraguay.")

    dir_cds_raw = DIR_RAW / "era5" / "cds"
    dir_cds_raw.mkdir(parents=True, exist_ok=True)

    tareas = []
    for anio in anios:
        for q in [1, 2, 3, 4]:
            tareas.append((anio, q))

    dfs_trimestrales = []
    fallidos = []
    print(f"\nIniciando cola de procesamiento ({len(tareas)} trimestres)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(
                procesar_un_trimestre,
                anio, q, dir_cds_raw, df_nodos_filtrados, forzar,
            ): (anio, q)
            for anio, q in tareas
        }
        for fut in as_completed(futuros):
            anio, q = futuros[fut]
            try:
                res = fut.result()
                if res is None:
                    fallidos.append((anio, q))
                else:
                    dfs_trimestrales.append(res)
            except Exception as e:
                fallidos.append((anio, q))
                print(f"   ✗ Excepción en tarea {anio} Q{q}: {e}")

    if fallidos:
        detalle = ", ".join(f"{a}-Q{q}" for a, q in sorted(fallidos))
        raise RuntimeError(
            f"Fallaron {len(fallidos)} trimestre(s) de ERA5-Land: {detalle}. "
            "No se actualiza el clima para evitar datos parciales."
        )
    if len(dfs_trimestrales) != len(tareas):
        raise RuntimeError(
            f"Se esperaban {len(tareas)} trimestres y se procesaron "
            f"{len(dfs_trimestrales)}. No se actualiza el clima."
        )

    print("\n2. Consolidando serie temporal completa...")
    df_total_crudo = pl.concat(dfs_trimestrales).sort(["nodo_clima_id", "fecha"])
    cols_dedup = [c for c in ["nodo_clima_id", "fecha"] if c in df_total_crudo.columns]
    if cols_dedup:
        df_total_crudo = df_total_crudo.unique(subset=cols_dedup, keep="first")
    print(f"   ✓ Total de observaciones nodo-día consolidadas: {len(df_total_crudo):,}")
    print(f"   ✓ Fechas: {df_total_crudo['fecha'].min()} a {df_total_crudo['fecha'].max()}")

    print("\n3. Calculando índices biofísicos de fuego (VPD, CDD, FWI FFMC, ISI e interacción viento)...")
    df_clima_procesado = calcular_indices_fwi_y_sequedad(df_total_crudo)

    out_parquet = DIR_PROCESSED / "clima_era5_diario.parquet"
    df_clima_procesado.write_parquet(out_parquet, compression="zstd")
    print(f"\n✓ Dataset meteorológico consolidado guardado en: {out_parquet}")
    print(f"  Tamaño en disco: {out_parquet.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga ERA5-Land desde Copernicus CDS")
    parser.add_argument("--anios", nargs="+", type=int, default=list(range(2018, 2026)), help="Lista de años AAAA")
    parser.add_argument("--workers", type=int, default=2, help="Número de descargas concurrentes en CDS (máx 2 recomendado)")
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Ignorar los parquet trimestrales en caché y reprocesar desde NetCDF/CDS.",
    )
    args = parser.parse_args()

    ejecutar_pipeline_cds(
        anios=args.anios,
        max_workers=args.workers,
        forzar=args.forzar,
    )
