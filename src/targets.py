"""
targets.py module: D2 definition of primary ignition and tactical propagation.

- Primary ignition: fire at t, no fire in the same cell in the last
  N days and no fire at t-1 within R km.
- Propagation: fire at t within R km of a fire at t-1.
"""

import polars as pl

from src.constantes import DIAS_IGNICION_SIN_FUEGO, RADIO_PROPAGACION_KM


def agregar_dias_desde_fuego_celda(
    df: pl.DataFrame,
    focos: pl.DataFrame,
) -> pl.DataFrame:
    """Adds the days since the last previous fire in the same cell."""
    if df.is_empty():
        return df

    base = df.drop([c for c in ["dias_desde_fuego_celda"] if c in df.columns])
    base = base.with_columns(
        pl.col("fecha").cast(pl.String).str.slice(0, 10).str.to_date().alias("_fecha")
    )

    if focos.is_empty():
        return base.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("dias_desde_fuego_celda")
        ).drop("_fecha")

    fuegos = (
        focos.select(["cell_id", "acq_date"])
        .unique()
        .with_columns(
            pl.col("acq_date").cast(pl.String).str.slice(0, 10).str.to_date().alias("_fecha_foco")
        )
        .with_columns(
            (pl.col("_fecha_foco") + pl.duration(days=1)).alias("_fecha_prev")
        )
        .sort(["cell_id", "_fecha_prev"])
    )

    out = (
        base.sort(["cell_id", "_fecha"])
        .join_asof(
            fuegos,
            left_on="_fecha",
            right_on="_fecha_prev",
            by="cell_id",
            strategy="backward",
        )
        .with_columns(
            (pl.col("_fecha") - pl.col("_fecha_foco"))
            .dt.total_days()
            .cast(pl.Int64)
            .alias("dias_desde_fuego_celda")
        )
        .drop(["_fecha", "_fecha_foco", "_fecha_prev"], strict=False)
    )
    return out


def definir_targets_d2(
    df: pl.DataFrame,
    focos: pl.DataFrame,
    dias_ignicion: int = DIAS_IGNICION_SIN_FUEGO,
    radio_km: float = RADIO_PROPAGACION_KM,
) -> pl.DataFrame:
    """Adds ``target_ignicion`` and ``target_propagacion`` with disjoint domains.

    ``target_ignicion`` is null for fires that are not primary (continuations
    or fires near t-1). ``target_propagacion`` is null outside the domain of
    t-1.
    """
    if "focos_25km_t1" not in df.columns:
        raise ValueError("Se requiere la columna focos_25km_t1 para definir D2")
    if int(radio_km) != int(RADIO_PROPAGACION_KM):
        raise ValueError("El radio D2 debe coincidir con el kernel focos_25km_t1")

    out = agregar_dias_desde_fuego_celda(df, focos)

    sin_fuego_t1 = pl.col("focos_25km_t1") <= 0
    con_fuego_t1 = pl.col("focos_25km_t1") > 0
    fuego = pl.col("fuego") == 1
    no_fuego = pl.col("fuego") == 0
    sin_fuego_reciente = (
        pl.col("dias_desde_fuego_celda").is_null()
        | (pl.col("dias_desde_fuego_celda") > int(dias_ignicion))
    )

    out = out.with_columns([
        pl.when(fuego & sin_fuego_t1 & sin_fuego_reciente)
        .then(1)
        .when(no_fuego & sin_fuego_t1)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
        .alias("target_ignicion"),
        pl.when(fuego & con_fuego_t1)
        .then(1)
        .when(no_fuego & con_fuego_t1)
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
        .alias("target_propagacion"),
    ])
    return out
