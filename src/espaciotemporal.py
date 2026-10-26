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
    [Deprecado] Prior estático calculado con todas las coordenadas de train.

    Se conserva sólo para compatibilidad y tests. No usar en el pipeline:
    incluye los fuegos del propio periodo de entrenamiento (fuga de etiqueta).
    Use ``calcular_priors_causales``.
    """
    coords_df = asegurar_coordenadas_utm(df)
    tree_hist = cKDTree(fuegos_train_coords)

    dists, _ = tree_hist.query(coords_df, k=1, workers=-1)
    dist_hist_km = np.clip(dists / 1000.0, 0.0, dist_max_km).astype(np.float32)
    densidad_hist = tree_hist.query_ball_point(
        coords_df, r=radio_m, return_length=True, workers=-1
    ).astype(np.float32)

    return df.with_columns([
        pl.Series("dist_foco_hist_km", dist_hist_km),
        pl.Series("densidad_hist_25km", densidad_hist),
        pl.lit(0.0).cast(pl.Float32).alias("freq_hist_1km"),
        pl.lit(float(dist_max_km)).cast(pl.Float32).alias("dist_corredor_antropico_km"),
        pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_5km"),
        pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_10km"),
        pl.lit(0.0).cast(pl.Float32).alias("focalizacion_antropica_local"),
    ])


def _asegurar_fecha_focos(focos_df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza acq_date a string ISO y valida columnas mínimas."""
    if focos_df.is_empty():
        return focos_df
    cols = set(focos_df.columns)
    if "cell_id" not in cols:
        raise ValueError("focos_df debe contener la columna cell_id")
    if "acq_date" not in cols:
        raise ValueError("focos_df debe contener la columna acq_date")
    return focos_df.with_columns(
        pl.col("acq_date").cast(pl.String).str.slice(0, 10).alias("_fecha")
    )


def _priors_para_corte(
    sub: pl.DataFrame,
    focos_prev: pl.DataFrame,
    dist_max_km: float,
    radio_m: float,
) -> pl.DataFrame:
    """Calcula priors espaciales con los focos estrictamente anteriores al corte."""
    coords_df = asegurar_coordenadas_utm(sub)
    n = len(sub)

    if focos_prev.is_empty():
        return sub.with_columns([
            pl.lit(float(dist_max_km)).cast(pl.Float32).alias("dist_foco_hist_km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_25km"),
            pl.lit(0.0).cast(pl.Float32).alias("freq_hist_1km"),
            pl.lit(float(dist_max_km)).cast(pl.Float32).alias("dist_corredor_antropico_km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_5km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_10km"),
            pl.lit(0.0).cast(pl.Float32).alias("focalizacion_antropica_local"),
        ])

    coords_focos = asegurar_coordenadas_utm(focos_prev)
    tree_hist = cKDTree(coords_focos)

    dists, _ = tree_hist.query(coords_df, k=1, workers=-1)
    dist_hist_km = np.clip(dists / 1000.0, 0.0, dist_max_km).astype(np.float32)

    dens_25 = tree_hist.query_ball_point(
        coords_df, r=radio_m, return_length=True, workers=-1
    ).astype(np.float32)
    dens_10 = tree_hist.query_ball_point(
        coords_df, r=10000.0, return_length=True, workers=-1
    ).astype(np.float32)
    dens_5 = tree_hist.query_ball_point(
        coords_df, r=5000.0, return_length=True, workers=-1
    ).astype(np.float32)

    # Frecuencia histórica por celda (días-fuego, no años)
    freq = (
        focos_prev.group_by("cell_id")
        .len()
        .rename({"len": "freq_hist_1km"})
    )
    sub = sub.join(freq, on="cell_id", how="left").with_columns(
        pl.col("freq_hist_1km").fill_null(0).cast(pl.Float32)
    )

    # Corredores: celdas con recurrencia crónica (>= 3 días-fuego previos).
    # Se usa el mismo criterio que la definición operativa original; el
    # manuscrito debe describirlo como días-fuego, no como "temporadas".
    corredores = freq.filter(pl.col("freq_hist_1km") >= 3)
    if corredores.is_empty():
        dist_corredor = np.full(n, dist_max_km, dtype=np.float32)
    else:
        xs, ys = cell_id_a_coordenadas_centro(corredores["cell_id"].to_numpy())
        tree_corr = cKDTree(np.column_stack([xs, ys]))
        dists_corr, _ = tree_corr.query(coords_df, k=1, workers=-1)
        dist_corredor = np.clip(dists_corr / 1000.0, 0.0, dist_max_km).astype(np.float32)

    focalizacion = (dens_5 / (dens_25 + 1.0)).astype(np.float32)

    return sub.with_columns([
        pl.Series("dist_foco_hist_km", dist_hist_km),
        pl.Series("densidad_hist_25km", dens_25),
        pl.Series("dist_corredor_antropico_km", dist_corredor),
        pl.Series("densidad_hist_5km", dens_5),
        pl.Series("densidad_hist_10km", dens_10),
        pl.Series("focalizacion_antropica_local", focalizacion),
    ])


def agregar_features_antropicas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade variables derivadas de calendario y presión antrópica usadas por M3.

    Se centraliza aquí para que train, val, test y el servicio de alerta usen
    exactamente la misma definición.
    """
    return df.with_columns([
        pl.when(pl.col("mes").is_in([8, 9, 10])).then(1.0).otherwise(0.0).cast(pl.Float32).alias("temporada_quema_ganadera"),
        (np.sin(2.0 * np.pi * pl.col("dia_del_anio") / 365.25)).cast(pl.Float32).alias("doy_sin"),
        (np.cos(2.0 * np.pi * pl.col("dia_del_anio") / 365.25)).cast(pl.Float32).alias("doy_cos"),
        (
            (pl.col("pct_pastura") / 100.0) * pl.col("vpd_max_kpa")
            * pl.when(pl.col("mes").is_in([8, 9, 10])).then(1.5).otherwise(0.5)
        ).cast(pl.Float32).alias("presion_quema_pastura"),
        (
            np.exp(-pl.col("dist_foco_hist_km") / 5.0) * pl.col("vpd_max_kpa")
        ).cast(pl.Float32).alias("interaccion_acceso_vpd"),
    ])


def calcular_priors_causales(
    df: pl.DataFrame,
    focos_df: pl.DataFrame,
    fecha_corte: str | None = None,
    radio_m: float = 25000.0,
    dist_max_km: float = 100.0,
    ventana_anios: int | None = 3,
    excluir_cell_ids: list[int] | None = None,
) -> pl.DataFrame:
    """
    Calcula priors de susceptibilidad histórica estrictamente causales.

    Por defecto usa como corte el primer día de cada mes: para una fila del
    mes M sólo se usan focos con ``acq_date`` anterior al día 1 de M. Así se
    excluyen el evento actual, los eventos futuros y los del mismo mes. En
    inferencia operativa se puede pasar ``fecha_corte`` (p. ej. el primer día
    del mes de la alerta) para mantener exactamente la misma definición.

    ``ventana_anios`` limita la historia a los últimos N años antes del corte
    (por defecto 3). Esto evita que las filas de 2025 tengan una densidad
    histórica mucho mayor que las de 2018 por pura acumulación de años.

    Columnas generadas:
    - dist_foco_hist_km, densidad_hist_25km
    - freq_hist_1km, dist_corredor_antropico_km
    - densidad_hist_5km, densidad_hist_10km, focalizacion_antropica_local
    """
    if df.is_empty():
        return df
    if "fecha" not in df.columns:
        raise ValueError("df debe contener la columna fecha")

    cols_prior = [
        "dist_foco_hist_km", "densidad_hist_25km", "freq_hist_1km",
        "dist_corredor_antropico_km", "densidad_hist_5km",
        "densidad_hist_10km", "focalizacion_antropica_local",
    ]
    df = df.drop([c for c in cols_prior if c in df.columns])

    focos = _asegurar_fecha_focos(focos_df)
    df = df.with_row_index("_orden_priors")
    df = df.with_columns(pl.col("fecha").cast(pl.String).str.slice(0, 10).alias("_fecha_df"))

    if fecha_corte is not None:
        cortes = [str(fecha_corte)[:10]]
        grupos = [df]
    else:
        df = df.with_columns(pl.col("_fecha_df").str.slice(0, 7).alias("_mes"))
        meses = sorted(df["_mes"].unique().to_list())
        cortes = [f"{m}-01" for m in meses]
        grupos = [df.filter(pl.col("_mes") == m) for m in meses]

    resultados = []
    for corte, sub in zip(cortes, grupos):
        if sub.is_empty():
            continue
        if focos.is_empty():
            focos_prev = focos
        elif ventana_anios:
            dt_corte = datetime.strptime(corte, "%Y-%m-%d")
            dt_ini = dt_corte.replace(year=dt_corte.year - int(ventana_anios))
            fecha_ini = dt_ini.strftime("%Y-%m-%d")
            focos_prev = focos.filter(
                (pl.col("_fecha") >= fecha_ini) & (pl.col("_fecha") < corte)
            )
        else:
            focos_prev = focos.filter(pl.col("_fecha") < corte)

        if excluir_cell_ids is not None and not focos_prev.is_empty():
            focos_prev = focos_prev.filter(
                ~pl.col("cell_id").is_in(excluir_cell_ids)
            )

        resultados.append(_priors_para_corte(sub, focos_prev, dist_max_km, radio_m))

    if not resultados:
        return df.drop(["_orden_priors", "_fecha_df", "_mes"], strict=False)

    out = pl.concat(resultados, how="diagonal_relaxed").sort("_orden_priors")
    cols_drop = [c for c in ["_orden_priors", "_fecha_df", "_mes"] if c in out.columns]
    return out.drop(cols_drop)


def enriquecer_dataset_espaciotemporal(
    df: pl.DataFrame,
    focos_df: pl.DataFrame,
    fuegos_train_coords: np.ndarray | None = None,
    usar_priors_causales: bool = True,
    fecha_corte: str | None = None,
    excluir_cell_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Aplica ecotono, contagio causal t-1 y priors históricos causales."""
    df = df.with_columns(calcular_indice_ecotono(df))
    df = calcular_variables_contagio_lag(df, focos_df)

    if usar_priors_causales:
        df = calcular_priors_causales(
            df, focos_df, fecha_corte=fecha_corte,
            excluir_cell_ids=excluir_cell_ids,
        )
    elif fuegos_train_coords is not None and len(fuegos_train_coords) > 0:
        df = calcular_prior_espacial_historico(df, fuegos_train_coords)
    else:
        df = df.with_columns([
            pl.lit(50.0).cast(pl.Float32).alias("dist_foco_hist_km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_25km"),
            pl.lit(0.0).cast(pl.Float32).alias("freq_hist_1km"),
            pl.lit(50.0).cast(pl.Float32).alias("dist_corredor_antropico_km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_5km"),
            pl.lit(0.0).cast(pl.Float32).alias("densidad_hist_10km"),
            pl.lit(0.0).cast(pl.Float32).alias("focalizacion_antropica_local"),
        ])

    return df
