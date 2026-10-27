"""
Script 04: Assembly and Construction of the Unified Tabular Dataset (DuckDB Zero-Copy).
- Integrates in memory: 1 km² Grid + MapBiomas Fuel + ERA5 Climate + VIIRS Hotspots.
- Applies Stage 1 Biophysical Gating: Exclusion of non-fuel (water/urban) and
  filtering of cells with meteorological inhibition (VPD < 0.8 kPa or 48-h rainfall > 10 mm).
- Generates anthropogenic calendar variables (day of the week, weekend).
- Generates the multiscale label: binary fire (Y ∈ {0, 1}) and severe fire (FRP ≥ 10 MW).
- Serializes to datos/processed/dataset_unificado_candidatos/dataset_YYYY.parquet,
  one deterministic, sorted partition per modelling year.
"""

from pathlib import Path
import sys
import argparse
import hashlib
import json

# Ensure the project root is on the PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import polars as pl
from src.constantes import (
    DIR_PROCESSED,
    UMBRAL_VPD_MINIMO_KPA,
    UMBRAL_LLUVIA_48H_MM
)

def construir_dataset_unificado(
    path_salida: Path | None = None,
    aplicar_gating: bool = True
):
    print("=" * 65)
    print("ENSAMBLE TABULAR ZERO-COPY DEL DATASET UNIFICADO (DUCKDB)")
    print("=" * 65)

    path_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    path_combustible = DIR_PROCESSED / "combustible_mapbiomas_1km_anual.parquet"
    path_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"

    for p in [path_grilla, path_combustible, path_clima, path_focos]:
        if not p.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {p}")

    if path_salida is None:
        path_salida = DIR_PROCESSED / "dataset_unificado_candidatos"
    path_salida = Path(path_salida)
    if path_salida.suffix == ".parquet":
        path_salida = path_salida.with_suffix("")
    path_salida.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # The annual-fuel join produces a wider 915M-row table than the old static
    # profile. Keep DuckDB within the 16 GB machine and allow disk spilling.
    con.execute("SET threads=2")
    con.execute("SET memory_limit='9GB'")
    con.execute("SET preserve_insertion_order=false")
    tmp_dir = DIR_PROCESSED / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET max_temp_directory_size='150GB'")

    filtro_gating = ""
    if aplicar_gating:
        # Pure biophysical gating: no leakage of fire labels.
        # Strictly conditions on kinematic viability (fuel present, atmospheric dryness, and absence of recent rainfall)
        filtro_gating = f"""
            (
                gcf.vpd_max_kpa >= {UMBRAL_VPD_MINIMO_KPA}
                AND gcf.precipitacion_48h_mm < {UMBRAL_LLUVIA_48H_MM}
                AND gcf.es_combustible = true
            )
        """
    else:
        filtro_gating = ""

    print("Ejecutando cruce relacional y enriquecimiento de variables...")
    query = f"""
        WITH grilla_clima AS (
            SELECT
                g.cell_id,
                g.nodo_clima_id,
                g.departamento,
                g.region,
                g.x_utm,
                g.y_utm,
                g.lon,
                g.lat,
                cli.fecha,
                cli.temp_max_c,
                cli.temp_min_c,
                cli.precipitacion_mm,
                cli.viento_max_kmh,
                cli.rafaga_max_kmh,
                cli.et0_mm,
                cli.vpd_max_kpa,
                cli.swvl1_m3m3,
                cli.precipitacion_48h_mm,
                cli.dias_secos_consecutivos,
                cli.ffmc,
                cli.isi,
                cli.interaccion_viento_vpd,
                cli.fdd_heladas_60d,
                cli.deficit_hidro_60d
            FROM '{path_grilla}' g
            JOIN '{path_clima}' cli
                ON g.nodo_clima_id = cli.nodo_clima_id
        ),
        grilla_fuel AS (
            SELECT
                gc.*,
                c.pct_pastura,
                c.pct_pastura_ganadera,
                c.pct_agricultura_anual,
                c.pct_plantacion_forestal,
                c.pct_sabana_pastizal,
                c.pct_bosque_seco,
                c.pct_bosque_humedo,
                c.pct_humedal_palmar,
                c.borde_bosque_agro_pct,
                c.pct_no_combustible,
                c.es_combustible,
                c.anio_fila,
                c.anio_mapa,
                c.fuel_lag
            FROM grilla_clima gc
            JOIN '{path_combustible}' c
                ON gc.cell_id = c.cell_id
                AND c.anio_fila = YEAR(CAST(gc.fecha AS DATE))
        )
        SELECT
            gcf.*,
            -- Variables temporales y de calendario antrópico
            ISODOW(CAST(gcf.fecha AS DATE)) - 1 AS dia_semana,
            CASE
                WHEN (ISODOW(CAST(gcf.fecha AS DATE)) - 1) IN (5, 6) THEN 1
                ELSE 0
            END AS es_fin_de_semana,
            MONTH(CAST(gcf.fecha AS DATE)) AS mes,
            DAYOFYEAR(CAST(gcf.fecha AS DATE)) AS dia_del_anio,
            (gcf.temp_max_c - gcf.temp_min_c) AS rango_termico_diario,
            (gcf.vpd_max_kpa * (gcf.dias_secos_consecutivos + 1.0)) AS deficit_hidrico_acumulado,
            POW(gcf.viento_max_kmh / 10.0, 2) AS potencial_viento_cuadratico,
            -- Etiquetas de fuego
            COALESCE(f.fuego, 0) AS fuego,
            COALESCE(f.num_focos_dia, 0) AS num_focos_dia,
            COALESCE(f.max_frp_dia, 0.0) AS max_frp_dia,
            COALESCE(f.sum_frp_dia, 0.0) AS sum_frp_dia,
            CASE
                WHEN f.fuego = 1 AND f.max_frp_dia >= 10.0 THEN 1
                ELSE 0
            END AS fuego_severo
        FROM grilla_fuel gcf
        LEFT JOIN '{path_focos}' f
            ON gcf.cell_id = f.cell_id AND gcf.fecha = f.acq_date
        __FILTRO_ANIO__
        ORDER BY gcf.cell_id, gcf.fecha
    """

    anios = [
        int(r[0]) for r in con.execute(
            f"SELECT DISTINCT YEAR(CAST(fecha AS DATE)) AS anio FROM '{path_clima}' ORDER BY 1"
        ).fetchall()
    ]
    print(f"Particiones anuales: {anios}")

    partes = []
    for anio in anios:
        condicion = (
            f"({filtro_gating}) AND YEAR(CAST(gcf.fecha AS DATE)) = {anio}"
            if aplicar_gating else f"YEAR(CAST(gcf.fecha AS DATE)) = {anio}"
        )
        filtro_anio = f"WHERE {condicion}"
        query_anio = query.replace("__FILTRO_ANIO__", filtro_anio)
        ruta_parte = path_salida / f"dataset_{anio}.parquet"
        print(f"Exportando {ruta_parte.name}...")
        con.execute(
            f"COPY ({query_anio}) TO '{ruta_parte}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        filas = int(con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{ruta_parte}')"
        ).fetchone()[0])
        h = hashlib.sha256()
        with ruta_parte.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        partes.append({"anio": anio, "archivo": ruta_parte.name, "filas": filas, "sha256": h.hexdigest()})
        print(f"  {ruta_parte.name}: {filas:,} filas")

    # Remove the old monolithic file only after all partitions were written.
    legacy = DIR_PROCESSED / "dataset_unificado_candidatos.parquet"
    if legacy.exists():
        legacy.unlink()
        print(f"Eliminado el archivo monolítico obsoleto: {legacy.name}")

    patron = path_salida / "dataset_*.parquet"
    stats = con.execute(f"""
        SELECT
            COUNT(*) as total_filas,
            SUM(fuego) as total_fuegos,
            SUM(fuego_severo) as total_fuegos_severos,
            COUNT(DISTINCT cell_id) as celdas_unicas,
            COUNT(DISTINCT fecha) as fechas_unicas,
            AVG(temp_max_c) as t_max_media,
            AVG(vpd_max_kpa) as vpd_medio,
            AVG(ffmc) as ffmc_medio
        FROM read_parquet('{patron}')
    """).df()
    tamano_mb = sum(p.stat().st_size for p in path_salida.glob("dataset_*.parquet")) / (1024 * 1024)
    manifest = {
        "descripcion": "Dataset unificado particionado por año, ordenado por cell_id/fecha",
        "patron": "dataset_*.parquet",
        "anios": anios,
        "partes": partes,
        "total_filas": int(stats["total_filas"][0]),
        "tamano_mb": round(tamano_mb, 2),
    }
    (path_salida / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n✓ Ensamble completado con éxito:")
    print(f" • Total de observaciones celda-día candidatas: {stats['total_filas'][0]:,}")
    print(f" • Celdas únicas en riesgo analizadas:           {stats['celdas_unicas'][0]:,}")
    print(f" • Días únicos procesados:                       {stats['fechas_unicas'][0]}")
    print(f" • Eventos de fuego activos etiquetados:         {stats['total_fuegos'][0]:,}")
    print(f" • Eventos de fuego severo (FRP ≥ 10 MW):        {stats['total_fuegos_severos'][0]:,}")
    print(f" • Temperatura máxima media:                    {stats['t_max_media'][0]:.1f} °C")
    print(f" • VPD medio:                                    {stats['vpd_medio'][0]:.2f} kPa")
    print(f" • FFMC medio:                                   {stats['ffmc_medio'][0]:.1f}")
    print(f"\nTamaño de las particiones: {tamano_mb:.2f} MB")
    print(f"Manifiesto: {path_salida / 'manifest.json'}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensamble del dataset unificado en DuckDB")
    parser.add_argument("--no-gating", action="store_true", help="Desactivar filtrado de Etapa 1")
    parser.add_argument("--salida", type=Path, default=None, help="Directorio de salida alternativo")
    args = parser.parse_args()

    construir_dataset_unificado(path_salida=args.salida, aplicar_gating=not args.no_gating)
