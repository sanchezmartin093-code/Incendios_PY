"""G0-Formosa: assemble features, causal priors and train/val/test partitions."""

from pathlib import Path
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_PROCESSED
from src.espaciotemporal import agregar_features_antropicas, enriquecer_dataset_espaciotemporal

OUT = DIR_PROCESSED / "formosa_g0"
TAG = os.environ.get("G0_TAG", "")
REQUIRED = [
    "pct_pastura_ganadera", "pct_agricultura_anual", "pct_plantacion_forestal",
    "pct_sabana_pastizal", "pct_bosque_seco", "pct_bosque_humedo",
    "pct_humedal_palmar", "borde_bosque_agro_pct", "pct_no_combustible",
    "pct_desmonte_t1", "pct_desmonte_t2", "pct_desmonte_2a",
    "vpd_max_kpa", "precipitacion_48h_mm", "swvl1_m3m3",
]


def _base() -> pl.DataFrame:
    plan = pl.read_parquet(OUT / f"plan_formosa_d2{TAG}.parquet").with_columns(
        pl.col("fecha").str.slice(0, 4).cast(pl.Int32).alias("anio"),
        pl.col("fecha").str.slice(0, 4).cast(pl.Int32).alias("anio_fila"),
    )
    grid = pl.read_parquet(OUT / "grilla_formosa_1km.parquet")
    fuel = pl.read_parquet(OUT / "combustible_formosa_anual.parquet")
    clim = pl.read_parquet(OUT / "clima_era5_diario.parquet")
    hansen = pl.read_parquet(OUT / "desmonte_hansen_formosa.parquet")
    focos = pl.read_parquet(OUT / "focos_formosa_diarios.parquet")

    df = plan.join(grid, on="cell_id", how="left")
    df = df.join(clim, on=["nodo_clima_id", "fecha"], how="left")
    df = df.join(fuel, on=["cell_id", "anio_fila"], how="left")
    df = df.join(hansen, on=["cell_id", "anio"], how="left")
    df = df.join(
        focos.select(["cell_id", "acq_date", "num_focos_dia", "max_frp_dia", "sum_frp_dia"]),
        left_on=["cell_id", "fecha"], right_on=["cell_id", "acq_date"], how="left",
    )
    df = df.with_columns([
        pl.col("target").cast(pl.Int8).alias("fuego"),
        pl.col("target").cast(pl.Int8).alias("target_ignicion"),
        pl.col("num_focos_dia").fill_null(0).cast(pl.Int32),
        pl.col("max_frp_dia").fill_null(0.0).cast(pl.Float32),
        pl.col("sum_frp_dia").fill_null(0.0).cast(pl.Float32),
        (pl.col("fecha").str.to_date("%Y-%m-%d").dt.weekday().cast(pl.Int32) - 1).alias("dia_semana"),
        pl.col("fecha").str.to_date("%Y-%m-%d").dt.month().cast(pl.Int32).alias("mes"),
        pl.col("fecha").str.to_date("%Y-%m-%d").dt.ordinal_day().cast(pl.Int32).alias("dia_del_anio"),
    ])
    df = df.with_columns([
        (pl.col("dia_semana").is_in([5, 6])).cast(pl.Int8).alias("es_fin_de_semana"),
        (pl.col("temp_max_c") - pl.col("temp_min_c")).cast(pl.Float32).alias("rango_termico_diario"),
        (pl.col("vpd_max_kpa") * (pl.col("dias_secos_consecutivos") + 1.0)).cast(pl.Float32).alias("deficit_hidrico_acumulado"),
        ((pl.col("viento_max_kmh") / 10.0) ** 2).cast(pl.Float32).alias("potencial_viento_cuadratico"),
        (pl.col("max_frp_dia") >= 10.0).cast(pl.Int8).alias("fuego_severo"),
        pl.col("target").cast(pl.Int8).alias("target_propagacion"),
    ])
    falt = [c for c in REQUIRED if c not in df.columns or df[c].null_count() > 0]
    if falt:
        raise RuntimeError(f"Features faltantes o con nulos: {falt}")
    return df


def main() -> None:
    base = _base()
    focos = pl.read_parquet(OUT / "focos_formosa_diarios.parquet")
    train_val = base.filter(pl.col("anio") <= 2023)
    test = base.filter(pl.col("anio") >= 2024)
    print(f"train_val={train_val.height:,} test={test.height:,}", flush=True)

    train_val = enriquecer_dataset_espaciotemporal(train_val, focos, usar_priors_causales=True)
    test = enriquecer_dataset_espaciotemporal(test, focos, usar_priors_causales=True, fecha_corte="2023-12-01")
    data = pl.concat([train_val, test], how="diagonal_relaxed")
    data = agregar_features_antropicas(data).sort(["fecha", "cell_id"])

    # No silent defaults: all feature columns must be present and finite.
    from src.modelo import FEATURES_HIBRIDO_CHACO, FEATURES_IGNICION_CHACO
    for feats in (FEATURES_IGNICION_CHACO, FEATURES_HIBRIDO_CHACO):
        miss = [c for c in feats if c not in data.columns]
        if miss:
            raise RuntimeError(f"Faltan features: {miss}")
        for c in feats:
            n = data[c].null_count()
            if n:
                raise RuntimeError(f"{c} tiene {n} nulos")

    data.write_parquet(OUT / f"formosa_dataset{TAG}.parquet", compression="zstd")
    splits = {
        "train": data.filter(pl.col("anio").is_in([2018, 2019, 2020, 2021, 2022]) & (pl.col("fecha") >= "2018-03-02")),
        "val": data.filter(pl.col("anio") == 2023),
        "test": data.filter(pl.col("anio").is_in([2024, 2025])),
    }
    for name, sub in splits.items():
        sub.write_parquet(OUT / f"formosa_{name}_ignicion{TAG}.parquet", compression="zstd")
        print(name, sub.height, int(sub["target_ignicion"].sum()))
    meta = {
        "rows": data.height,
        "pos": int(data["target_ignicion"].sum()),
        "features_m2": len(FEATURES_IGNICION_CHACO),
        "features_m3": len(FEATURES_HIBRIDO_CHACO),
        "splits": {k: {"rows": v.height, "pos": int(v["target_ignicion"].sum())} for k, v in splits.items()},
        "priors_test": "frozen 2023-12-01",
    }
    (OUT / f"features_meta{TAG}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
