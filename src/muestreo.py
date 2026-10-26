"""
Módulo muestreo.py: Muestreo estratificado de pseudo-ausencias con Dead-Zone Spatial Buffer (20 km)
y partición temporal out-of-time anti-fuga de datos (Anti-Data-Leakage).
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
    Aplica el Dead-Zone Spatial Buffer: Excluye cualquier celda candidata negativa que se encuentre
    a menos de `buffer_km` de un fuego activo registrado en la misma fecha.
    Anula matemáticamente la autocorrelación espacial entre presencias y ausencias de control.
    """
    if df_positivos_dia.is_empty() or df_candidatos_dia.is_empty():
        return df_candidatos_dia

    coords_positivos = df_positivos_dia.select(["x_utm", "y_utm"]).to_numpy()
    coords_candidatos = df_candidatos_dia.select(["x_utm", "y_utm"]).to_numpy()

    # Árbol espacial en metros métricos UTM 21S con aceleración multihilo C++
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
    Muestreo estratificado masivo escalable con memoria constante O(1):
    - Extrae presencias (Y=1) mediante DuckDB sin cargar todo el archivo.
    - Itera mes a mes cargando únicamente los candidatos del mes en memoria.
    - Aplica Dead-Zone Spatial Buffer de 20 km para cada estrato (fecha, region).
    - Muestrea ratio_neg_pos ausencias por presencia.
    """
    import duckdb
    rng = np.random.default_rng(semilla)
    con = duckdb.connect()

    print("Extrayendo todas las presencias de fuego (Y=1) con DuckDB...")
    df_positivos = con.sql(f"SELECT * FROM '{path_parquet}' WHERE fuego = 1").pl()
    n_pos_total = len(df_positivos)
    print(f" ✓ Presencias cargadas: {n_pos_total:,}")

    if n_pos_total == 0:
        raise ValueError("No se encontraron eventos de fuego en el dataset.")

    # Meses únicos con fuego
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

        # Cargar únicamente los candidatos negativos del mes correspondiente
        df_cand_mes = con.sql(f"""
            SELECT * FROM '{path_parquet}'
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
    Ejecuta el muestreo estratificado espaciotemporal de pseudo-ausencias:
    - Para cada estrato (fecha, region):
      1. Extrae todas las presencias de fuego (Y=1).
      2. Filtra candidatos negativos (Y=0) fuera del Dead-Zone Buffer de 20 km.
      3. Muestrea exactamente (ratio_neg_pos * N_positivos) pseudo-ausencias aleatorias.
    """
    rng = np.random.default_rng(semilla)
    
    # Separar fuegos positivos
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

        # Aplicar Dead-Zone Buffer espacial de 20 km
        cand_seguros = filtrar_candidatos_con_buffer_muerto(
            pos_strato, cand_strato, buffer_km=buffer_km
        )

        n_disponibles = len(cand_seguros)
        if n_disponibles == 0:
            continue

        n_tomar = min(n_neg_deseados, n_disponibles)
        # Muestreo sin reemplazo
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
    Divide el dataset en Train, Validation y Test siguiendo estrictamente la dimensión temporal:
    - Si contiene múltiples años:
      * Train: 2018 a 2022
      * Val: 2023
      * Test: 2024
    - Si el dataset corresponde a un rango acotado:
      * 70% cronológico inicial -> Train
      * 15% cronológico medio   -> Val
      * 15% cronológico final   -> Test
    """
    # Extraer años presentes
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
        # Partición cronológica secuencial
        fechas_unicas = sorted(df["fecha"].unique().to_list())
        n_fechas = len(fechas_unicas)
        
        idx_train = max(1, int(n_fechas * 0.70))
        idx_val = max(idx_train + 1, int(n_fechas * 0.85))

        fechas_train = set(fechas_unicas[:idx_train])
        fechas_val = set(fechas_unicas[idx_train:idx_val])
        fechas_test = set(fechas_unicas[idx_val:])

        # Si hay muy pocas fechas, asegurar que test no quede vacío
        if not fechas_test:
            fechas_test = {fechas_unicas[-1]}

        df_train = df.filter(pl.col("fecha").is_in(fechas_train))
        df_val = df.filter(pl.col("fecha").is_in(fechas_val))
        df_test = df.filter(pl.col("fecha").is_in(fechas_test))

    return df_train, df_val, df_test
