"""
remuestreo.py module: stratified 1:N resampling by D2 target.
"""

import polars as pl

from src.constantes import RATIO_AUSENCIAS_PRESENCIAS


def remuestrear_por_target(
    df: pl.DataFrame,
    target_col: str,
    ratio: int = RATIO_AUSENCIAS_PRESENCIAS,
    seed: int = 42,
) -> pl.DataFrame:
    """Resamples by (fecha, region) to obtain up to ``ratio`` negatives per positive.

    If a stratum does not have enough negatives, all negatives are kept and
    the positives are subsampled to maintain the ratio.
    """
    if target_col not in df.columns:
        raise ValueError(f"Falta la columna {target_col}")

    datos = df.filter(pl.col(target_col).is_not_null())
    if datos.is_empty():
        return datos

    muestras = []
    for (fecha, region), sub in datos.group_by(["fecha", "region"], maintain_order=True):
        pos = sub.filter(pl.col(target_col) == 1)
        neg = sub.filter(pl.col(target_col) == 0)
        if pos.is_empty() or neg.is_empty():
            continue

        n_pos = pos.height
        n_neg = neg.height
        if n_neg >= ratio * n_pos:
            pos_sel = pos
            neg_sel = neg.sample(n=ratio * n_pos, seed=seed)
        else:
            n_pos_max = n_neg // ratio
            if n_pos_max == 0:
                continue
            pos_sel = pos.sample(n=n_pos_max, seed=seed)
            neg_sel = neg

        muestras.append(pl.concat([pos_sel, neg_sel]))

    if not muestras:
        return datos.head(0)
    return pl.concat(muestras).sort(["fecha", "region", "cell_id"])
