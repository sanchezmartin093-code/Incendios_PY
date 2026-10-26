"""
Módulo espaciotemporal.py: Ingeniería de características de contagio rezagado (t-1),
prior espacial histórico de ignición e índice de fragmentación de ecotono.
Fundamento Teórico (2024-2026):
- Contagio espacial causal: La presencia y potencia radiativa (FRP) de incendios activos en t-1
  dentro de un radio de 25 km gobierna la propagación sostenida y los clusters de quema.
- Efecto Borde / Ecotono: Las igniciones antrópicas se concentran en la interfaz bosque-pastura
  (desmontes y acordonamientos).
- Prior de propensión antrópica: Densidad espacial histórica calculada sobre el set de entrenamiento
  para modelar la recurrencia geográfica sin data leakage.
"""

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from src.geometria import cell_id_a_coordenadas_centro

def calcular_indice_ecotono(df: pl.DataFrame) -> pl.Series:
    """
    Calcula el índice normalizado de ecotono bosque-pastura [0.0, 1.0].
    Alcanza su valor máximo de 1.0 cuando la celda tiene 50% bosque y 50% pastura (interfaz crítica).
    """
    regiones = df["region"].to_list()
    p_past = (df["pct_pastura"] / 100.0).to_numpy()
    p_b_seco = (df["pct_bosque_seco"] / 100.0).to_numpy()
    p_b_humedo = (df["pct_bosque_humedo"] / 100.0).to_numpy()

    p_bosque = np.where(np.array(regiones) == "Chaco", p_b_seco, p_b_humedo)
    ecotono = 4.0 * p_bosque * p_past
    ecotono = np.clip(ecotono, 0.0, 1.0).astype(np.float32)
    return pl.Series("indice_ecotono", ecotono)

def asegurar_coordenadas_utm(df: pl.DataFrame) -> np.ndarray:
    """Extrae o reconstruye coordenadas métricas UTM (X, Y) en O(1) desde cell_id."""
    if "x_utm" in df.columns and "y_utm" in df.columns:
        return np.column_stack([df["x_utm"].to_numpy(), df["y_utm"].to_numpy()])
    else:
        c_ids = df["cell_id"].to_numpy()
        xs, ys = cell_id_a_coordenadas_centro(c_ids)
        return np.column_stack([xs, ys])

def calcular_variables_contagio_lag(
    df: pl.DataFrame,
    focos_df: pl.DataFrame,
    radio_m: float = 25000.0,
    dist_max_km: float = 100.0
) -> pl.DataFrame:
    """
    Calcula variables de contagio espacial estrictamente causales en t-1:
    - dist_foco_t1_km: Distancia al foco activo más cercano de ayer (km).
    - focos_25km_t1: Conteo de focos activos en un radio de 25 km ayer.
    - frp_25km_t1: Suma de potencia radiativa de fuego (MW) en 25 km ayer.
    """
    if focos_df.is_empty():
        return df.with_columns([
            pl.lit(dist_max_km).cast(pl.Float32).alias("dist_foco_t1_km"),
            pl.lit(0.0).cast(pl.Float32).alias("focos_25km_t1"),
            pl.lit(0.0).cast(pl.Float32).alias("frp_25km_t1")
        ])

    coords_df = asegurar_coordenadas_utm(df)

    fechas_df = [str(f)[:10] for f in df["fecha"].to_list()]
    fechas_unicas = sorted(list(set(fechas_df)))

    # Determinar exactamente qué fechas de focos se necesitan (fechas de ayer)
    fechas_ayer_necesarias = set()
    for f_act in fechas_unicas:
        dt_act = datetime.strptime(f_act, "%Y-%m-%d").date()
        fechas_ayer_necesarias.add((dt_act - timedelta(days=1)).strftime("%Y-%m-%d"))

    # Pre-filtrar focos exclusivamente a las fechas de ayer requeridas para O(1) hashing y evitar construir miles de árboles
    focos_filtrados = focos_df.filter(pl.col("acq_date").cast(pl.String).str.slice(0, 10).is_in(list(fechas_ayer_necesarias)))
    if focos_filtrados.is_empty():
        return df.with_columns([
            pl.lit(dist_max_km).cast(pl.Float32).alias("dist_foco_t1_km"),
            pl.lit(0.0).cast(pl.Float32).alias("focos_25km_t1"),
            pl.lit(0.0).cast(pl.Float32).alias("frp_25km_t1")
        ])

    coords_focos_filtrados = asegurar_coordenadas_utm(focos_filtrados)
    focos_fechas = np.array([str(f)[:10] for f in focos_filtrados["acq_date"].to_list()])
    focos_frp = focos_filtrados["sum_frp_dia"].to_numpy()

    trees_por_fecha = {}
    frp_por_fecha = {}
    for f in np.unique(focos_fechas):
        mask = (focos_fechas == f)
        trees_por_fecha[f] = cKDTree(coords_focos_filtrados[mask])
        frp_por_fecha[f] = focos_frp[mask]

    n_rows = len(df)
    dist_t1_km = np.full(n_rows, dist_max_km, dtype=np.float32)
    focos_t1_count = np.zeros(n_rows, dtype=np.float32)
    frp_t1_sum = np.zeros(n_rows, dtype=np.float32)

    fechas_df_arr = np.array(fechas_df)

    for f_actual in fechas_unicas:
        dt_actual = datetime.strptime(f_actual, "%Y-%m-%d").date()
        fecha_ayer = (dt_actual - timedelta(days=1)).strftime("%Y-%m-%d")

        if fecha_ayer not in trees_por_fecha:
            # No se registraron focos ayer (o es el inicio de la serie)
            continue

        tree_ayer = trees_por_fecha[fecha_ayer]
        frp_ayer = frp_por_fecha[fecha_ayer]

        idx_df_fecha = np.where(fechas_df_arr == f_actual)[0]
        coords_consulta = coords_df[idx_df_fecha]

        # 1. Distancia al foco más cercano (multiprocesamiento C++ vectorizado)
        dists, _ = tree_ayer.query(coords_consulta, k=1, workers=-1)
        dist_t1_km[idx_df_fecha] = np.clip(dists / 1000.0, 0.0, dist_max_km).astype(np.float32)

        # 2. Conteo y suma de FRP en radio de 25 km
        vecinos_por_punto = tree_ayer.query_ball_point(coords_consulta, r=radio_m, workers=-1)
        counts = np.array([len(v) for v in vecinos_por_punto], dtype=np.float32)
        frp_sums = np.array([np.sum(frp_ayer[v]) if len(v) > 0 else 0.0 for v in vecinos_por_punto], dtype=np.float32)

        focos_t1_count[idx_df_fecha] = counts
        frp_t1_sum[idx_df_fecha] = frp_sums

    return df.with_columns([
        pl.Series("dist_foco_t1_km", dist_t1_km),
        pl.Series("focos_25km_t1", focos_t1_count),
        pl.Series("frp_25km_t1", frp_t1_sum)
    ])

def calcular_prior_espacial_historico(
    df: pl.DataFrame,
    fuegos_train_coords: np.ndarray,
    radio_m: float = 25000.0,
    dist_max_km: float = 100.0
) -> pl.DataFrame:
    """
    Calcula variables de susceptibilidad geográfica histórica a partir de las coordenadas
    de focos del conjunto de entrenamiento:
    - dist_foco_hist_km: Distancia al fuego histórico más cercano.
    - densidad_hist_25km: Conteo de fuegos históricos en radio de 25 km.
    """
    coords_df = asegurar_coordenadas_utm(df)
    tree_hist = cKDTree(fuegos_train_coords)

    # Distancia al fuego histórico más cercano (k=1, multithreaded)
    dists, _ = tree_hist.query(coords_df, k=1, workers=-1)
    dist_hist_km = np.clip(dists / 1000.0, 0.0, dist_max_km).astype(np.float32)

    # Densidad en radio de 25 km con return_length vectorizado O(N)
    densidad_hist = tree_hist.query_ball_point(coords_df, r=radio_m, return_length=True, workers=-1).astype(np.float32)

    return df.with_columns([
        pl.Series("dist_foco_hist_km", dist_hist_km),
        pl.Series("densidad_hist_25km", densidad_hist)
    ])

def enriquecer_dataset_espaciotemporal(
    df: pl.DataFrame,
    focos_df: pl.DataFrame,
    fuegos_train_coords: np.ndarray | None = None
) -> pl.DataFrame:
    """Aplica de forma secuencial todo el paquete de ingeniería espaciotemporal."""
    # 1. Índice de ecotono
    df = df.with_columns(calcular_indice_ecotono(df))

    # 2. Contagio causal t-1
    df = calcular_variables_contagio_lag(df, focos_df)

    # 3. Prior histórico espacial
    if fuegos_train_coords is not None and len(fuegos_train_coords) > 0:
        df = calcular_prior_espacial_historico(df, fuegos_train_coords)
    else:
        df = df.with_columns([
            pl.lit(50.0).cast(pl.Float32).alias("dist_foco_hist_km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_25km")
        ])

    return df
