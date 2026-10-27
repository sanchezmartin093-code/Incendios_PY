"""
Script 02c: Corrects daily precipitation and completes gusts/ET0 using
Open-Meteo Historical Archive in UTC.

Reason: the previous CDS version took the maximum of 3-hourly `tp` accumulations,
which produced impossible values (>1 000 mm/day) and truncated the
last hours of the day. This script keeps temperature/VPD/wind from the CDS
file and replaces precipitation, gust, ET0 and 0-7 cm soil moisture with
Open-Meteo daily series. Then it recalculates 48 h, CDD, FFMC, ISI, the
wind-VPD interaction, frost severity and the 60-day water deficit.
"""

from pathlib import Path
import fcntl
import sys
import shutil
import json
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl

from src.constantes import DIR_RAW, DIR_PROCESSED
from src.clima import (
    generar_malla_nodos_clima,
    asignar_nodos_a_grilla,
    descargar_clima_openmeteo_batch,
    calcular_indices_fwi_y_sequedad,
)


def main():
    lock_file = open(DIR_PROCESSED / ".02c_corregir_clima.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Ya hay otra corrida de 02c en curso; se omite este lanzamiento.")
        return

    print("=" * 70)
    print("CORRECCIÓN DE CLIMA DIARIO CON OPEN-METEO (UTC)")
    print("=" * 70)

    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    if not p_grilla.exists() or not p_clima.exists():
        raise FileNotFoundError("Faltan grilla o clima procesado")

    grilla = pl.read_parquet(p_grilla)
    nodos = generar_malla_nodos_clima(resolucion_deg=0.25)
    grilla = asignar_nodos_a_grilla(grilla, nodos)
    activos = grilla["nodo_clima_id"].unique().to_list()
    nodos = nodos.filter(pl.col("nodo_clima_id").is_in(activos)).sort("nodo_clima_id")
    print(f"Nodos activos: {nodos.height}")

    cache = DIR_RAW / "era5" / "openmeteo_corregido"
    om = descargar_clima_openmeteo_batch(
        lats=nodos["lat"].to_list(),
        lons=nodos["lon"].to_list(),
        nodo_ids=nodos["nodo_clima_id"].to_list(),
        start_date="2018-01-01",
        end_date="2025-12-31",
        batch_size=10,
        max_workers=1,
        delay_seg=10.0,
        cache_dir=cache,
    )
    print(f"Registros Open-Meteo: {om.height:,}")
    if om.is_empty():
        raise RuntimeError("Open-Meteo no devolvió datos")

    nuevas = om.select([
        "nodo_clima_id", "fecha",
        pl.col("precipitacion_mm").alias("precipitacion_mm_om"),
        pl.col("rafaga_max_kmh").alias("rafaga_max_kmh_om"),
        pl.col("et0_mm").alias("et0_mm_om"),
    ])

    base = pl.read_parquet(p_clima)
    if "swvl1_m3m3" not in base.columns:
        print(
            "[ADVERTENCIA] clima_era5_diario.parquet no tiene swvl1_m3m3. "
            "Regenerá 02b con volumetric_soil_water_layer_1; mientras tanto "
            "se usará el balance simple como fallback."
        )
    else:
        nulos_base_swvl1 = int(base["swvl1_m3m3"].null_count())
        if nulos_base_swvl1:
            print(f"[ADVERTENCIA] {nulos_base_swvl1:,} filas CDS sin swvl1_m3m3.")
    merged = base.join(nuevas, on=["nodo_clima_id", "fecha"], how="left")
    faltantes = int(merged["precipitacion_mm_om"].is_null().sum())
    if faltantes:
        print(f"[ADVERTENCIA] {faltantes:,} filas sin dato Open-Meteo; se usa el valor CDS previo.")

    columnas_drop = [
        "precipitacion_mm_om", "rafaga_max_kmh_om", "et0_mm_om",
        "precipitacion_48h_mm", "dias_secos_consecutivos",
        "ffmc", "isi", "interaccion_viento_vpd",
        "fdd_heladas_60d", "deficit_hidro_60d",
    ]

    merged = merged.with_columns([
        pl.coalesce([pl.col("precipitacion_mm_om"), pl.col("precipitacion_mm")]).alias("precipitacion_mm"),
        pl.coalesce([pl.col("rafaga_max_kmh_om"), pl.col("rafaga_max_kmh")]).alias("rafaga_max_kmh"),
        pl.coalesce([pl.col("et0_mm_om"), pl.col("et0_mm")]).alias("et0_mm"),
    ]).drop([c for c in columnas_drop if c in merged.columns])

    procesado = calcular_indices_fwi_y_sequedad(merged)

    backup = DIR_PROCESSED / "clima_era5_diario_old.parquet"
    if not backup.exists():
        shutil.copy2(p_clima, backup)
    procesado.write_parquet(p_clima, compression="zstd")

    meta = {
        "fuente": "Open-Meteo Historical Archive (UTC) + ERA5-Land CDS",
        "variables_reemplazadas": ["precipitacion_mm", "rafaga_max_kmh", "et0_mm"],
        "fecha_correccion": datetime.now(timezone.utc).isoformat(),
        "filas": procesado.height,
        "precipitacion_max_mm": float(procesado["precipitacion_mm"].max()),
        "precipitacion_48h_max_mm": float(procesado["precipitacion_48h_mm"].max()),
        "swvl1_fuente": "Copernicus CDS ERA5-Land volumetric_soil_water_layer_1 (0-7 cm daily mean); fallback balance simple",
        "swvl1_nulos": int(procesado["swvl1_m3m3"].null_count()),
        "swvl1_min": float(procesado["swvl1_m3m3"].min()),
        "swvl1_max": float(procesado["swvl1_m3m3"].max()),
    }
    (DIR_PROCESSED / "clima_era5_diario.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"✓ Clima corregido: {p_clima}")
    print(f"  Precipitación máxima diaria: {meta['precipitacion_max_mm']:.2f} mm")
    print(f"  Precipitación máxima 48 h: {meta['precipitacion_48h_max_mm']:.2f} mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
