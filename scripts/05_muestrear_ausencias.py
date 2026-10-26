"""
Script 05: Muestreo de Pseudo-Ausencias con Dead-Zone Buffer (20 km) y Partición Anti-Leakage.
- Aplica el Dead-Zone Buffer de 20 km para anular la autocorrelación espacial entre presencias y ausencias.
- Muestrea una relación controlada 1:10 estratificada por fecha y ecorregión (Chaco vs. Oriental).
- Genera las particiones temporales estrictas out-of-time: Train, Val y Test.
- Serializa: train_dataset.parquet, val_dataset.parquet y test_dataset.parquet.
"""

from pathlib import Path
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.constantes import DIR_PROCESSED, BUFFER_ZONA_MUERTA_KM, RATIO_AUSENCIAS_PRESENCIAS
from src.muestreo import (
    muestrear_dataset_unificado_por_meses,
    particionar_dataset_out_of_time
)
from src.espaciotemporal import (
    agregar_features_antropicas,
    enriquecer_dataset_espaciotemporal,
)
from src.targets import definir_targets_d2

def main():
    print("=" * 65)
    print("MUESTREO DE PSEUDO-AUSENCIAS & DEAD-ZONE BUFFER (20 km)")
    print("=" * 65)

    path_dataset = DIR_PROCESSED / "dataset_unificado_candidatos.parquet"
    if not path_dataset.exists():
        raise FileNotFoundError(f"Dataset unificado no encontrado en {path_dataset}. Ejecuta scripts/04_construir_dataset.py primero.")

    # 1. Ejecutar muestreo estratificado escalable mes a mes con Dead-Zone Buffer
    print(f"\nAplicando Dead-Zone Spatial Buffer de {BUFFER_ZONA_MUERTA_KM} km y ratio 1:{RATIO_AUSENCIAS_PRESENCIAS}...")
    df_balanceado = muestrear_dataset_unificado_por_meses(
        path_dataset,
        ratio_neg_pos=RATIO_AUSENCIAS_PRESENCIAS,
        buffer_km=BUFFER_ZONA_MUERTA_KM,
        semilla=42
    )

    n_pos = int((df_balanceado["fuego"] == 1).sum())
    n_neg = int((df_balanceado["fuego"] == 0).sum())
    print(f"\n✓ Muestreo completado:")
    print(f" • Presencias reales (Y=1):        {n_pos:,}")
    print(f" • Pseudo-ausencias seguras (Y=0): {n_neg:,}")
    print(f" • Tasa de prevalencia efectiva:   {100.0 * n_pos / len(df_balanceado):.2f}% (Ratio 1:{n_neg // max(1, n_pos)})")

    # 2. Partición temporal out-of-time
    print("\nGenerando partición temporal out-of-time (Anti-Data-Leakage)...")
    df_train, df_val, df_test = particionar_dataset_out_of_time(df_balanceado)

    # 3. Enriquecimiento espaciotemporal causal:
    #    - Contagio t-1 (sólo fuegos de ayer)
    #    - Priors históricos con corte mensual: sólo fuegos anteriores al
    #      primer día del mes de cada fila. Nunca se incluye el evento actual
    #      ni fuegos futuros.
    print("\nAplicando ingeniería de características espaciotemporales (Ecotono, Contagio t-1 y Priors causales)...")
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    if not path_focos.exists():
        raise FileNotFoundError(f"Archivo de focos no encontrado: {path_focos}")
    df_focos = pl.read_parquet(path_focos)

    print(" • Enriqueciendo Train...")
    df_train = definir_targets_d2(
        agregar_features_antropicas(
            enriquecer_dataset_espaciotemporal(df_train, df_focos, usar_priors_causales=True)
        ),
        df_focos,
    )
    print(" • Enriqueciendo Val...")
    df_val = definir_targets_d2(
        agregar_features_antropicas(
            enriquecer_dataset_espaciotemporal(df_val, df_focos, usar_priors_causales=True)
        ),
        df_focos,
    )
    print(" • Enriqueciendo Test...")
    df_test = definir_targets_d2(
        agregar_features_antropicas(
            enriquecer_dataset_espaciotemporal(df_test, df_focos, usar_priors_causales=True)
        ),
        df_focos,
    )
    print(" ✓ Particiones enriquecidas exitosamente.")

    p_train = DIR_PROCESSED / "train_dataset.parquet"
    p_val = DIR_PROCESSED / "val_dataset.parquet"
    p_test = DIR_PROCESSED / "test_dataset.parquet"

    df_train.write_parquet(p_train, compression="zstd")
    df_val.write_parquet(p_val, compression="zstd")
    df_test.write_parquet(p_test, compression="zstd")

    print("\nDistribución de particiones generadas:")
    for nombre, df_part, p_file in [("Entrenamiento (Train)", df_train, p_train),
                                     ("Validación (Val)", df_val, p_val),
                                     ("Testeo Ciego (Test)", df_test, p_test)]:
        fuegos_part = int((df_part["fuego"] == 1).sum())
        fechas = sorted(df_part["fecha"].unique().to_list())
        rango_f = f"{fechas[0]} a {fechas[-1]}" if fechas else "N/A"
        print(f" • {nombre}: {len(df_part):,} filas ({fuegos_part} fuegos) | {rango_f} | Archivo: {p_file.name}")

    print("=" * 65)

if __name__ == "__main__":
    main()
