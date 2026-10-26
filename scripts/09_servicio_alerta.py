"""
Script 09: Servicio Operativo Diario de Alerta Temprana de Incendios Forestales.
- Ejecuta la inferencia biofísica y discriminante sobre las 400.006 celdas de 1 km² de Paraguay.
- Aplica el Two-Stage Bio-physical Gating y la calibración monótona de Platt.
- Genera:
  * Capa Parquet georreferenciada: datos/outputs/raster_riesgo_diario/alerta_nacional_{fecha}.parquet
  * Boletín técnico de alerta (Markdown): datos/outputs/boletin_alerta_{fecha}.md
  * Payload de despacho (JSON): datos/outputs/boletin_alerta_{fecha}.json
- Imprime resumen ejecutivo en consola.
"""

from pathlib import Path
import sys
import argparse
import json
import time

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.constantes import DIR_OUTPUTS, DIR_PROCESSED
from src.alerta import (
    cargar_artefactos_modelos,
    preparar_datos_diarios,
    evaluar_alerta_nacional,
    generar_boletin_operativo
)

def ejecutar_servicio_alerta(fecha: str | None = None):
    inicio_t = time.time()
    print("=" * 70)
    print("SISTEMA DE ALERTA TEMPRANA DE INCENDIOS FORESTALES (SAT-INCENDIOS PY)")
    print("=" * 70)

    # Determinar fecha objetivo
    if fecha is None:
        p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
        if not p_clima.exists():
            raise FileNotFoundError(f"Archivo de clima no encontrado en {p_clima}")
        df_c = pl.read_parquet(p_clima)
        fechas_disp = sorted(df_c["fecha"].unique().to_list())
        fecha = fechas_disp[-1]
        print(f"[*] Fecha no especificada. Utilizando fecha más reciente disponible: {fecha}")
    else:
        print(f"[*] Fecha seleccionada: {fecha}")

    print("\n1. Cargando modelos nativos C++ y calibradores de Platt...")
    artefactos = cargar_artefactos_modelos()
    print("   ✓ Modelos Chaco y Oriental cargados correctamente.")

    print("\n2. Ensamblando grilla nacional de 1 km² (400.006 celdas)...")
    df_nacional = preparar_datos_diarios(fecha)
    print(f"   ✓ Grilla integrada: {df_nacional.height:,} celdas.")

    print("\n3. Ejecutando Two-Stage Bio-physical Gating e inferencia calibrada...")
    t_inf_ini = time.time()
    df_evaluado = evaluar_alerta_nacional(df_nacional, artefactos)
    duracion_inf = time.time() - t_inf_ini
    print(f"   ✓ Inferencia completada en {duracion_inf:.2f} segundos.")

    print("\n4. Generando boletín y analítica departamental...")
    resumen, boletin_md = generar_boletin_operativo(df_evaluado, fecha)

    # 5. Exportar resultados
    dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
    dir_raster.mkdir(parents=True, exist_ok=True)
    p_parquet = dir_raster / f"alerta_nacional_{fecha}.parquet"
    p_md = DIR_OUTPUTS / f"boletin_alerta_{fecha}.md"
    p_json = DIR_OUTPUTS / f"boletin_alerta_{fecha}.json"

    # Seleccionar columnas clave para la capa geoespacial de salida
    columnas_export = [
        "cell_id", "x_utm", "y_utm", "lon", "lat", "departamento", "region",
        "probabilidad_fuego", "probabilidad_cruda", "probabilidad_ignicion",
        "probabilidad_m3", "probabilidad_m1", "fallback_m1", "nivel_alerta", "estado_gating",
        "isi", "ffmc", "vpd_max_kpa", "viento_max_kmh", "rafaga_max_kmh", "temp_max_c"
    ]
    df_evaluado.select(columnas_export).write_parquet(p_parquet)
    p_md.write_text(boletin_md, encoding="utf-8")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Capa Parquet exportada: {p_parquet.name} ({p_parquet.stat().st_size / (1024*1024):.1f} MB)")
    print(f"✓ Boletín Markdown exportado: {p_md.name}")
    print(f"✓ Resumen JSON exportado: {p_json.name}")

    tiempo_total = time.time() - inicio_t
    print("\n" + "=" * 70)
    print(f"RESUMEN OPERATIVO - FECHA: {fecha} (Tiempo total: {tiempo_total:.2f} s)")
    print("=" * 70)
    dist = resumen["distribucion_riesgo"]
    for nivel in ["EXTREMO", "ALTO", "MODERADO", "BAJO", "NULO"]:
        cnt = dist.get(nivel, 0)
        pct = cnt / df_evaluado.height * 100
        print(f"  * {nivel:<10}: {cnt:>8,} km² ({pct:>5.1f}%)")

    print("\nTop 3 Departamentos con Mayor Área en Alerta (Extremo + Alto):")
    for d in resumen["deptos_criticos"][:3]:
        crit = d["km2_extremo"] + d["km2_alto"]
        print(f"  - {d['departamento']:<20}: {crit:>6,} km² críticos (Extremo: {d['km2_extremo']}, Alto: {d['km2_alto']})")

    print("=" * 70)
    return p_parquet, p_md, p_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servicio de Alerta Temprana de Incendios Forestales.")
    parser.add_argument("--fecha", type=str, default=None, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()
    ejecutar_servicio_alerta(args.fecha)
