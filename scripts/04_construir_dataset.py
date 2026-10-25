"""
Script 04: Ensamble y Construcción del Dataset Tabular Unificado (DuckDB Zero-Copy).
- Integra en memoria: Grilla 1 km² + Combustible MapBiomas + Clima ERA5 + Focos VIIRS.
- Aplica el Gating Biofísico de Etapa 1: Exclusión de no combustibles (agua/urbano) y
  filtrado de celdas con inhibición meteorológica (VPD < 0.8 kPa o Lluvia 48h > 10 mm).
- Genera variables de calendario antrópico (día de la semana, fin de semana).
- Genera la etiqueta multiescala: fuego binario (Y ∈ {0, 1}) y fuego severo (FRP ≥ 10 MW).
- Serializa a datos/processed/dataset_unificado_candidatos.parquet.
"""

from pathlib import Path
import sys
import argparse

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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
    path_combustible = DIR_PROCESSED / "combustible_mapbiomas_1km.parquet"
    path_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"

    for p in [path_grilla, path_combustible, path_clima, path_focos]:
        if not p.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {p}")

    if path_salida is None:
        path_salida = DIR_PROCESSED / "dataset_unificado_candidatos.parquet"

    con = duckdb.connect()

    filtro_gating = ""
    if aplicar_gating:
        # Gating biofísico puro: sin leakage de etiquetas de fuego.
        # Condiciona estrictamente a viabilidad cinemática (combustible presente, sequedad atmosférica y ausencia de lluvia reciente)
        filtro_gating = f"""
            WHERE (
                cli.vpd_max_kpa >= {UMBRAL_VPD_MINIMO_KPA} 
                AND cli.precipitacion_48h_mm < {UMBRAL_LLUVIA_48H_MM}
                AND gc.es_combustible = true
            )
        """
    else:
        filtro_gating = ""

    print("Ejecutando cruce relacional y enriquecimiento de variables...")
    query = f"""
        CREATE OR REPLACE TABLE dataset_unificado AS
        WITH grilla_comb AS (
            SELECT 
                g.cell_id,
                g.nodo_clima_id,
                g.departamento,
                g.region,
                g.x_utm,
                g.y_utm,
                g.lon,
                g.lat,
                c.pct_pastura,
                c.pct_sabana_pastizal,
                c.pct_bosque_seco,
                c.pct_bosque_humedo,
                c.pct_humedal_palmar,
                c.es_combustible
            FROM '{path_grilla}' g
            JOIN '{path_combustible}' c
                ON g.cell_id = c.cell_id
        )
        SELECT 
            gc.cell_id,
            gc.departamento,
            gc.region,
            gc.x_utm,
            gc.y_utm,
            gc.lon,
            gc.lat,
            gc.pct_pastura,
            gc.pct_sabana_pastizal,
            gc.pct_bosque_seco,
            gc.pct_bosque_humedo,
            gc.pct_humedal_palmar,
            cli.fecha,
            -- Variables temporales y de calendario antrópico (estrictamente alineado con ISO y Python dt.weekday)
            ISODOW(CAST(cli.fecha AS DATE)) - 1 AS dia_semana, -- 0=Lunes, 1=Martes, ..., 6=Domingo
            CASE 
                WHEN (ISODOW(CAST(cli.fecha AS DATE)) - 1) IN (5, 6) THEN 1 
                ELSE 0 
            END AS es_fin_de_semana,
            MONTH(CAST(cli.fecha AS DATE)) AS mes,
            DAYOFYEAR(CAST(cli.fecha AS DATE)) AS dia_del_anio,
            -- Variables biofísicas y climáticas
            cli.temp_max_c,
            cli.temp_min_c,
            cli.precipitacion_mm,
            cli.precipitacion_48h_mm,
            cli.viento_max_kmh,
            cli.rafaga_max_kmh,
            cli.et0_mm,
            cli.vpd_max_kpa,
            cli.dias_secos_consecutivos,
            cli.ffmc,
            cli.isi,
            cli.interaccion_viento_vpd,
            -- Nuevas variables biofísicas de desecación no lineal
            (cli.temp_max_c - cli.temp_min_c) AS rango_termico_diario,
            (cli.vpd_max_kpa * (cli.dias_secos_consecutivos + 1.0)) AS deficit_hidrico_acumulado,
            POW(cli.viento_max_kmh / 10.0, 2) AS potencial_viento_cuadratico,
            -- Etiquetas de fuego
            COALESCE(f.fuego, 0) AS fuego,
            COALESCE(f.num_focos_dia, 0) AS num_focos_dia,
            COALESCE(f.max_frp_dia, 0.0) AS max_frp_dia,
            COALESCE(f.sum_frp_dia, 0.0) AS sum_frp_dia,
            CASE 
                WHEN f.fuego = 1 AND f.max_frp_dia >= 10.0 THEN 1 
                ELSE 0 
            END AS fuego_severo
        FROM grilla_comb gc
        JOIN '{path_clima}' cli
            ON gc.nodo_clima_id = cli.nodo_clima_id
        LEFT JOIN '{path_focos}' f
            ON gc.cell_id = f.cell_id AND cli.fecha = f.acq_date
        {filtro_gating}
    """

    con.execute(query)

    # 1. Exportar a Parquet ZSTD
    print(f"Exportando matriz integrada a: {path_salida.name}...")
    con.execute(f"COPY dataset_unificado TO '{path_salida}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # 2. Métricas y estadísticas descriptivas
    stats = con.execute("""
        SELECT 
            COUNT(*) as total_filas,
            SUM(fuego) as total_fuegos,
            SUM(fuego_severo) as total_fuegos_severos,
            COUNT(DISTINCT cell_id) as celdas_unicas,
            COUNT(DISTINCT fecha) as fechas_unicas,
            AVG(temp_max_c) as t_max_media,
            AVG(vpd_max_kpa) as vpd_medio,
            AVG(ffmc) as ffmc_medio
        FROM dataset_unificado
    """).df()

    print("\n✓ Ensamble completado con éxito:")
    print(f" • Total de observaciones celda-día candidatas: {stats['total_filas'][0]:,}")
    print(f" • Celdas únicas en riesgo analizadas:           {stats['celdas_unicas'][0]:,}")
    print(f" • Días únicos procesados:                       {stats['fechas_unicas'][0]}")
    print(f" • Eventos de fuego activos etiquetados:         {stats['total_fuegos'][0]:,}")
    print(f" • Eventos de fuego severo (FRP ≥ 10 MW):        {stats['total_fuegos_severos'][0]:,}")
    print(f" • Temperatura máxima media:                    {stats['t_max_media'][0]:.1f} °C")
    print(f" • VPD medio:                                    {stats['vpd_medio'][0]:.2f} kPa")
    print(f" • FFMC medio:                                   {stats['ffmc_medio'][0]:.1f}")
    print(f"\nTamaño de la matriz en disco: {path_salida.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensamble del dataset unificado en DuckDB")
    parser.add_argument("--no-gating", action="store_true", help="Desactivar filtrado de Etapa 1")
    args = parser.parse_args()

    construir_dataset_unificado(aplicar_gating=not args.no_gating)
