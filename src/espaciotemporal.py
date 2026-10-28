"""
Module espaciotemporal.py: Feature engineering for lagged contagion (t-1),
historical spatial ignition prior and ecotone fragmentation index.
Theoretical Basis (2024-2026):
- Causal spatial contagion: The presence and radiative power (FRP) of active fires at t-1
  within a 25 km radius governs sustained propagation and burn clusters.
- Edge / Ecotone Effect: Anthropogenic ignitions concentrate at the forest-pasture interface
  (clearings and windrows).
- Anthropogenic propensity prior: Historical spatial density computed on the training set
  to model geographic recurrence without data leakage.
"""

from __future__ import annotations


from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from src.geometria import cell_id_a_coordenadas_centro

def calcular_indice_ecotono(df: pl.DataFrame) -> pl.Series:
    """
    Computes the normalized forest-pasture ecotone index [0.0, 1.0].
    Reaches its maximum value of 1.0 when the cell has 50% forest and 50% pasture (critical interface).
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
    """Extracts or reconstructs UTM metric coordinates (X, Y) in O(1) from cell_id."""
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
    Computes strictly causal spatial contagion variables at t-1:
    - dist_foco_t1_km: Distance to the nearest active hotspot from yesterday (km).
    - focos_25km_t1: Count of active hotspots within a 25 km radius yesterday.
    - frp_25km_t1: Sum of fire radiative power (MW) within 25 km yesterday.
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

    # Determine exactly which hotspot dates are needed (yesterday's dates)
    fechas_ayer_necesarias = set()
    for f_act in fechas_unicas:
        dt_act = datetime.strptime(f_act, "%Y-%m-%d").date()
        fechas_ayer_necesarias.add((dt_act - timedelta(days=1)).strftime("%Y-%m-%d"))

    # Pre-filter hotspots exclusively to the required yesterday dates for O(1) hashing and to avoid building thousands of trees
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
            # No hotspots were recorded yesterday (or it is the start of the series)
            continue

        tree_ayer = trees_por_fecha[fecha_ayer]
        frp_ayer = frp_por_fecha[fecha_ayer]

        idx_df_fecha = np.where(fechas_df_arr == f_actual)[0]
        coords_consulta = coords_df[idx_df_fecha]

        # 1. Distance to the nearest hotspot (vectorized C++ multiprocessing)
        dists, _ = tree_ayer.query(coords_consulta, k=1, workers=-1)
        dist_t1_km[idx_df_fecha] = np.clip(dists / 1000.0, 0.0, dist_max_km).astype(np.float32)

        # 2. Count and FRP sum within a 25 km radius
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
    [Deprecated] Static prior computed with all training coordinates.

    Kept only for compatibility and tests. Do not use in the pipeline:
    it includes fires from the training period itself (label leakage).
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
    """Normalizes acq_date to an ISO string and validates minimum columns."""
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
    """Computes spatial priors with hotspots strictly prior to the cutoff."""
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

    # Historical frequency per cell (fire-days, not years)
    freq = (
        focos_prev.group_by("cell_id")
        .len()
        .rename({"len": "freq_hist_1km"})
    )
    sub = sub.join(freq, on="cell_id", how="left").with_columns(
        pl.col("freq_hist_1km").fill_null(0).cast(pl.Float32)
    )

    # Corridors: cells with chronic recurrence (>= 3 previous fire-days).
    # The same criterion as the original operational definition is used; the
    # manuscript must describe it as fire-days, not as "seasons".
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
    Adds calendar-, land-use- and climate-interaction variables.

    Defaults exist only for tests and legacy partitions. New production
    datasets must provide them explicitly; ``scripts/05e`` validates that.
    """
    defaults = {
        "pct_desmonte_t1": 0.0,
        "pct_desmonte_t2": 0.0,
        "pct_desmonte_2a": 0.0,
        "swvl1_m3m3": 0.25,
        "fdd_heladas_60d": 0.0,
        "deficit_hidro_60d": 0.0,
        "pct_pastura_ganadera": None,
        "pct_agricultura_anual": 0.0,
        "pct_plantacion_forestal": 0.0,
        "pct_sabana_pastizal": 0.0,
        "pct_humedal_palmar": 0.0,
        "borde_bosque_agro_pct": 0.0,
    }
    exprs_default = []
    for col_name, val in defaults.items():
        if col_name in df.columns:
            continue
        if col_name == "pct_pastura_ganadera":
            exprs_default.append(
                pl.col("pct_pastura").cast(pl.Float32).alias(col_name)
            )
        else:
            exprs_default.append(
                pl.lit(float(val)).cast(pl.Float32).alias(col_name)
            )
    if exprs_default:
        df = df.with_columns(exprs_default)

    return df.with_columns([
        pl.when(pl.col("mes").is_in([8, 9, 10])).then(1.0).otherwise(0.0).cast(pl.Float32).alias("temporada_quema_ganadera"),
        (np.sin(2.0 * np.pi * pl.col("dia_del_anio") / 365.25)).cast(pl.Float32).alias("doy_sin"),
        (np.cos(2.0 * np.pi * pl.col("dia_del_anio") / 365.25)).cast(pl.Float32).alias("doy_cos"),
        (
            (pl.col("pct_pastura_ganadera") / 100.0) * pl.col("vpd_max_kpa")
            * pl.when(pl.col("mes").is_in([8, 9, 10])).then(1.5).otherwise(0.5)
        ).cast(pl.Float32).alias("presion_quema_pastura"),
        (
            np.exp(-pl.col("dist_foco_hist_km") / 5.0) * pl.col("vpd_max_kpa")
        ).cast(pl.Float32).alias("interaccion_acceso_vpd"),
        (
            (pl.col("pct_desmonte_2a") / 100.0) * pl.col("vpd_max_kpa")
        ).cast(pl.Float32).alias("presion_desmonte_vpd"),
        (
            (pl.col("pct_humedal_palmar") / 100.0)
            * pl.when(pl.col("deficit_hidro_60d") > 0.0)
            .then(pl.col("deficit_hidro_60d"))
            .otherwise(0.0)
        ).cast(pl.Float32).alias("riesgo_estero_seco"),
        (
            ((pl.col("pct_pastura_ganadera") + pl.col("pct_sabana_pastizal")) / 100.0)
            * pl.col("fdd_heladas_60d")
        ).cast(pl.Float32).alias("combustible_curado_helada"),
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
    Computes strictly causal historical susceptibility priors.

    By default it uses the first day of each month as the cutoff: for a row of
    month M only hotspots with ``acq_date`` prior to day 1 of M are used. This
    excludes the current event, future events and those from the same month. In
    operational inference, ``fecha_corte`` can be passed (e.g. the first day
    of the alert's month) to keep exactly the same definition.

    ``ventana_anios`` limits the history to the last N years before the cutoff
    (3 by default). This prevents rows from 2025 from having a much higher
    historical density than those from 2018 purely due to the accumulation of years.

    Generated columns:
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
    """Applies ecotone, causal t-1 contagion and causal historical priors."""
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
