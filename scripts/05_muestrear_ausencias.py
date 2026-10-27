"""
Script 05: Pseudo-Absence Sampling with a 20 km Dead-Zone Buffer and Anti-Leakage Partitioning.
- Applies a 20 km Dead-Zone Buffer to cancel spatial autocorrelation between presences and absences.
- Samples a controlled 1:10 ratio stratified by date and ecoregion (Chaco vs. Oriental).
- Generates strict out-of-time temporal splits: Train, Val, and Test.
- Serializes: train_dataset.parquet, val_dataset.parquet, and test_dataset.parquet.
"""

from pathlib import Path
import sys

# Ensure the project root is on the PYTHONPATH
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

    path_dataset = DIR_PROCESSED / "dataset_unificado_candidatos" / "dataset_*.parquet"
    if not list((DIR_PROCESSED / "dataset_unificado_candidatos").glob("dataset_*.parquet")):
        raise FileNotFoundError(f"Dataset unificado no encontrado en {path_dataset}. Ejecuta scripts/04_construir_dataset.py primero.")

    # 1. Run scalable month-by-month stratified sampling with the Dead-Zone Buffer
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

    # 2. Out-of-time temporal split
    print("\nGenerando partición temporal out-of-time (Anti-Data-Leakage)...")
    df_train, df_val, df_test = particionar_dataset_out_of_time(df_balanceado)

    # 3. Causal spatiotemporal enrichment:
    #    - t-1 contagion (only yesterday's fires)
    #    - Historical priors with a monthly cutoff: only fires prior to the
    #      first day of the month of each row. The current event is never
    #      included, nor are future fires.
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
