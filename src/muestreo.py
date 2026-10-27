"""
muestreo.py module: Stratified sampling of pseudo-absences with Dead-Zone Spatial Buffer (20 km)
and anti-data-leakage out-of-time temporal partitioning.
"""

from pathlib import Path
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from src.constantes import (
    BUFFER_ZONA_MUERTA_KM,
    RATIO_AUSENCIAS_PRESENCIAS,
    YEARS_TRAIN,
    YEAR_VAL,
    YEARS_TEST,
    YEAR_TEST
)

def filtrar_candidatos_con_buffer_muerto(
    df_positivos_dia: pl.DataFrame,
    df_candidatos_dia: pl.DataFrame,
    buffer_km: float = BUFFER_ZONA_MUERTA_KM
) -> pl.DataFrame:
    """
    Applies the Dead-Zone Spatial Buffer: Excludes any negative candidate cell located
    less than `buffer_km` from an active fire recorded on the same date.
    Mathematically cancels the spatial autocorrelation between presences and control absences.
    """
    if df_positivos_dia.is_empty() or df_candidatos_dia.is_empty():
        return df_candidatos_dia

    coords_positivos = df_positivos_dia.select(["x_utm", "y_utm"]).to_numpy()
    coords_candidatos = df_candidatos_dia.select(["x_utm", "y_utm"]).to_numpy()

    # Spatial tree in UTM 21S metric meters with C++ multithreaded acceleration
    tree = cKDTree(coords_positivos)
    distancias, _ = tree.query(coords_candidatos, k=1, workers=-1)

    buffer_metros = buffer_km * 1000.0
    mascara_segura = distancias >= buffer_metros

    return df_candidatos_dia.filter(pl.Series("seguro", mascara_segura))

def muestrear_dataset_unificado_por_meses(
    path_parquet: Path,
    ratio_neg_pos: int = RATIO_AUSENCIAS_PRESENCIAS,
    buffer_km: float = BUFFER_ZONA_MUERTA_KM,
    semilla: int = 42
) -> pl.DataFrame:
    """
    Scalable massive stratified sampling with constant O(1) memory:
    - Extracts presences (Y=1) via DuckDB without loading the entire file.
    - Iterates month by month loading only that month's candidates into memory.
    - Applies a 20 km Dead-Zone Spatial Buffer for each stratum (date, region).
    - Samples ratio_neg_pos absences per presence.
    """
    import duckdb
    rng = np.random.default_rng(semilla)
    con = duckdb.connect()
    path_sql = (
        f"read_parquet('{path_parquet}')"
        if "*" in str(path_parquet) or Path(path_parquet).is_dir()
        else f"'{path_parquet}'"
    )

    print("Extrayendo todas las presencias de fuego (Y=1) con DuckDB...")
    df_positivos = con.sql(f"SELECT * FROM {path_sql} WHERE fuego = 1").pl()
    n_pos_total = len(df_positivos)
    print(f" ✓ Presencias cargadas: {n_pos_total:,}")

    if n_pos_total == 0:
        raise ValueError("No se encontraron eventos de fuego en el dataset.")

    # Unique months with fire
    meses = sorted(
        df_positivos.select(pl.col("fecha").str.slice(0, 7).alias("mes"))["mes"]
        .unique()
        .to_list()
    )
    print(f"Procesando {len(meses)} meses con actividad de fuego para el muestreo...")

    muestras_negativas = []
    total_neg_muestreados = 0

    for idx, mes in enumerate(meses, 1):
        pos_mes = df_positivos.filter(pl.col("fecha").str.slice(0, 7) == mes)
        if pos_mes.is_empty():
            continue

        # Load only the negative candidates for the corresponding month
        df_cand_mes = con.sql(f"""
            SELECT * FROM {path_sql}
            WHERE fecha >= '{mes}-01' AND fecha <= '{mes}-31' AND fuego = 0
        """).pl()

        if df_cand_mes.is_empty():
            continue

        estratos_mes = pos_mes.select(["fecha", "region"]).unique().sort(["fecha", "region"])

        for row in estratos_mes.iter_rows():
            f_str, reg_str = row[0], row[1]
            pos_strato = pos_mes.filter((pl.col("fecha") == f_str) & (pl.col("region") == reg_str))
            n_pos = len(pos_strato)
            n_neg_deseados = n_pos * ratio_neg_pos

            cand_strato = df_cand_mes.filter((pl.col("fecha") == f_str) & (pl.col("region") == reg_str))
            if cand_strato.is_empty():
                continue

            cand_seguros = filtrar_candidatos_con_buffer_muerto(pos_strato, cand_strato, buffer_km=buffer_km)
            n_disp = len(cand_seguros)
            if n_disp == 0:
                continue

            n_tomar = min(n_neg_deseados, n_disp)
            indices = rng.choice(n_disp, size=n_tomar, replace=False)
            muestras_negativas.append(cand_seguros[indices])
            total_neg_muestreados += n_tomar

        if idx % 12 == 0 or idx == len(meses):
            print(f"   [{idx:02d}/{len(meses)}] Mes {mes} procesado | Ausencias acumuladas: {total_neg_muestreados:,}")

    con.close()

    print(f"\nConsolidando {n_pos_total:,} presencias y {total_neg_muestreados:,} pseudo-ausencias...")
    if muestras_negativas:
        df_neg_final = pl.concat(muestras_negativas)
        df_balanceado = pl.concat([df_positivos, df_neg_final]).sort(["fecha", "region", "cell_id"])
    else:
        df_balanceado = df_positivos

    return df_balanceado

def muestrear_pseudo_ausencias_estratificadas(
    df_unificado: pl.DataFrame,
    ratio_neg_pos: int = RATIO_AUSENCIAS_PRESENCIAS,
    buffer_km: float = BUFFER_ZONA_MUERTA_KM,
    semilla: int = 42
) -> pl.DataFrame:
    """
    Performs spatiotemporal stratified sampling of pseudo-absences:
    - For each stratum (date, region):
      1. Extracts all fire presences (Y=1).
      2. Filters negative candidates (Y=0) outside the 20 km Dead-Zone Buffer.
      3. Samples exactly (ratio_neg_pos * N_positives) random pseudo-absences.
    """
    rng = np.random.default_rng(semilla)
    
    # Separate positive fires
    df_positivos = df_unificado.filter(pl.col("fuego") == 1)
    df_negativos = df_unificado.filter(pl.col("fuego") == 0)

    if df_positivos.is_empty():
        print("Aviso: No se encontraron registros de fuego positivo en el dataset.")
        return df_unificado.sample(n=min(len(df_unificado), 1000), seed=semilla)

    estratos = df_positivos.select(["fecha", "region"]).unique().sort(["fecha", "region"])
    
    muestras_negativas = []

    for row in estratos.iter_rows():
        f_str, reg_str = row[0], row[1]

        pos_strato = df_positivos.filter(
            (pl.col("fecha") == f_str) & (pl.col("region") == reg_str)
        )
        n_pos = len(pos_strato)
        n_neg_deseados = n_pos * ratio_neg_pos

        cand_strato = df_negativos.filter(
            (pl.col("fecha") == f_str) & (pl.col("region") == reg_str)
        )

        # Apply the 20 km spatial Dead-Zone Buffer
        cand_seguros = filtrar_candidatos_con_buffer_muerto(
            pos_strato, cand_strato, buffer_km=buffer_km
        )

        n_disponibles = len(cand_seguros)
        if n_disponibles == 0:
            continue

        n_tomar = min(n_neg_deseados, n_disponibles)
        # Sampling without replacement
        indices_elegidos = rng.choice(n_disponibles, size=n_tomar, replace=False)
        muestra_k = cand_seguros[indices_elegidos]
        muestras_negativas.append(muestra_k)

    if muestras_negativas:
        df_neg_final = pl.concat(muestras_negativas)
        df_balanceado = pl.concat([df_positivos, df_neg_final]).sort(["fecha", "region", "cell_id"])
    else:
        df_balanceado = df_positivos

    return df_balanceado

def particionar_dataset_out_of_time(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits the dataset into Train, Validation, and Test strictly following the temporal dimension:
    - If it contains multiple years:
      * Train: 2018 to 2022
      * Val: 2023
      * Test: 2024
    - If the dataset corresponds to a bounded range:
      * Initial chronological 70% -> Train
      * Middle chronological 15%   -> Val
      * Final chronological 15%    -> Test
    """
    # Extract years present
    anios_presentes = (
        df.select(pl.col("fecha").str.slice(0, 4).cast(pl.Int32).alias("anio"))
        ["anio"]
        .unique()
        .to_list()
    )

    tiene_anios_historicos = any(a in YEARS_TRAIN for a in anios_presentes) and any(a in YEARS_TEST for a in anios_presentes)

    if tiene_anios_historicos:
        df_train = df.filter(pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_in(YEARS_TRAIN))
        df_val = df.filter(pl.col("fecha").str.slice(0, 4).cast(pl.Int32) == YEAR_VAL)
        df_test = df.filter(pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_in(YEARS_TEST))
    else:
        # Sequential chronological partition
        fechas_unicas = sorted(df["fecha"].unique().to_list())
        n_fechas = len(fechas_unicas)
        
        idx_train = max(1, int(n_fechas * 0.70))
        idx_val = max(idx_train + 1, int(n_fechas * 0.85))

        fechas_train = set(fechas_unicas[:idx_train])
        fechas_val = set(fechas_unicas[idx_train:idx_val])
        fechas_test = set(fechas_unicas[idx_val:])

        # If there are very few dates, ensure test does not remain empty
        if not fechas_test:
            fechas_test = {fechas_unicas[-1]}

        df_train = df.filter(pl.col("fecha").is_in(fechas_train))
        df_val = df.filter(pl.col("fecha").is_in(fechas_val))
        df_test = df.filter(pl.col("fecha").is_in(fechas_test))

    return df_train, df_val, df_test
