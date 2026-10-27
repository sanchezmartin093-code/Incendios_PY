"""Spatial confirmation holdout utilities.

The confirmation holdout is defined from the grid only. It is intentionally
independent of the development holdout used by script 20 and must not be used
for model selection.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

BLOQUE_M = 50_000.0
BUFFER_M = 20_000.0
FRACCION_CONFIRMACION = 0.20
SAL_CONFIRMACION = "fase2-confirmacion-2026-10-27"

# Same deterministic hash used by the development holdout in script 20.
_HASH_A = 738_956_093
_HASH_B = 19_349_663


def block_id(x_utm: np.ndarray, y_utm: np.ndarray) -> np.ndarray:
    bx = np.floor(np.asarray(x_utm, dtype=float) / BLOQUE_M).astype(np.int64)
    by = np.floor(np.asarray(y_utm, dtype=float) / BLOQUE_M).astype(np.int64)
    return bx * 1_000_003 + by


def development_holdout_blocks(grid: pl.DataFrame) -> set[int]:
    """Reproduces the block hash used by script 20 on the full grid."""
    ids = np.unique(block_id(grid["x_utm"].to_numpy(), grid["y_utm"].to_numpy()))
    h = ((ids * _HASH_A) ^ (ids * _HASH_B)).astype(np.int64)
    return set(int(v) for v in ids[(h % 5) == 0].tolist())


def _salt_hash(salt: str, block: int) -> str:
    return hashlib.sha256(f"{salt}:{int(block)}".encode("utf-8")).hexdigest()


def design_confirmation_holdout(
    grid: pl.DataFrame,
    fraccion: float = FRACCION_CONFIRMACION,
    buffer_m: float = BUFFER_M,
    sal: str = SAL_CONFIRMACION,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Returns blocks, holdout cells and buffer cells for confirmation.

    The ranking is based only on block identifiers and a salt, never on labels
    or model outputs. Blocks from the development holdout are excluded.
    """
    if not (0.0 < fraccion < 1.0):
        raise ValueError("fraccion debe estar en (0, 1)")
    if not {"cell_id", "x_utm", "y_utm"}.issubset(grid.columns):
        raise ValueError("grid debe contener cell_id, x_utm, y_utm")

    g = grid.select(["cell_id", "x_utm", "y_utm"]).unique("cell_id")
    g = g.with_columns(pl.Series("block_id", block_id(g["x_utm"].to_numpy(), g["y_utm"].to_numpy())))

    current = development_holdout_blocks(g)
    blocks = (
        g.select("block_id").unique().sort("block_id")
        .filter(~pl.col("block_id").is_in(list(current)))
        .with_columns(pl.col("block_id").map_elements(lambda b: _salt_hash(sal, int(b)), return_dtype=pl.String).alias("_hash"))
        .sort("_hash")
    )
    n_new = max(1, int(round(len(blocks) * fraccion)))
    nuevos = set(blocks.head(n_new)["block_id"].to_list())
    if not nuevos.isdisjoint(current):
        raise RuntimeError("El holdout de confirmación se solapa con el de desarrollo")

    holdout = g.filter(pl.col("block_id").is_in(list(nuevos))).sort("cell_id")
    holdout_ids = set(holdout["cell_id"].to_list())

    ref = holdout.select(["x_utm", "y_utm"]).to_numpy()
    tree = cKDTree(ref)
    dist, _ = tree.query(g.select(["x_utm", "y_utm"]).to_numpy(), k=1, workers=-1)
    buffer = (
        g.with_columns(pl.Series("_dist", dist))
        .filter(
            (pl.col("_dist") <= buffer_m)
            & (~pl.col("cell_id").is_in(list(holdout_ids)))
        )
        .drop("_dist")
        .sort("cell_id")
    )

    blocks_out = (
        g.select("block_id").unique().sort("block_id")
        .with_columns([
            pl.col("block_id").is_in(list(current)).alias("desarrollo"),
            pl.col("block_id").is_in(list(nuevos)).alias("confirmacion"),
        ])
    )
    return blocks_out, holdout, buffer


ROLES_CONFIRMACION = ("desarrollo", "buffer_confirmacion", "confirmacion")


def etiquetar_particion_confirmacion(
    df: pl.DataFrame,
    holdout: pl.DataFrame,
    buffer: pl.DataFrame,
) -> pl.DataFrame:
    """Labels rows as development, confirmation buffer or confirmation holdout.

    This is a read-only partition helper for Phase 5. It never uses labels or
    model outputs. Buffer cells must be excluded from training and calibration
    to avoid spatial leakage from the confirmation blocks.
    """
    if "cell_id" not in df.columns:
        raise ValueError("df debe contener cell_id")

    holdout_ids = holdout.select("cell_id").unique()
    buffer_ids = buffer.select("cell_id").unique()
    if holdout_ids.height == 0:
        raise ValueError("holdout no puede estar vacío")
    if buffer_ids.height == 0:
        raise ValueError("buffer no puede estar vacío")
    if holdout_ids.join(buffer_ids, on="cell_id", how="inner").height:
        raise ValueError("holdout y buffer se solapan")

    return df.with_columns(
        pl.when(pl.col("cell_id").is_in(holdout_ids["cell_id"].to_list()))
        .then(pl.lit("confirmacion"))
        .when(pl.col("cell_id").is_in(buffer_ids["cell_id"].to_list()))
        .then(pl.lit("buffer_confirmacion"))
        .otherwise(pl.lit("desarrollo"))
        .alias("rol_confirmacion")
    )


def filtrar_particion_confirmacion(
    df: pl.DataFrame,
    holdout: pl.DataFrame,
    buffer: pl.DataFrame,
    rol: str,
) -> pl.DataFrame:
    """Returns rows for a Phase 5 role: desarrollo, buffer_confirmacion or confirmacion."""
    if rol not in ROLES_CONFIRMACION:
        raise ValueError(f"rol inválido: {rol}. Opciones: {ROLES_CONFIRMACION}")
    etiquetado = etiquetar_particion_confirmacion(df, holdout, buffer)
    return etiquetado.filter(pl.col("rol_confirmacion") == rol).drop("rol_confirmacion")
