"""G0-Formosa: VIIRS active-fire labels from the existing FIRMS bbox archive."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.path import Path as MplPath
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_PROCESSED, DIR_RAW

OUT = DIR_PROCESSED / "formosa_g0"


def main() -> None:
    grid = pl.read_parquet(OUT / "grilla_formosa_1km.parquet")
    ids = set(grid["cell_id"].to_list())
    gj = __import__("json").load(open(DIR_RAW / "vector" / "argentina_adm1.geojson"))
    feat = [f for f in gj["features"] if f["properties"]["shapeName"] == "Formosa"][0]
    ring = feat["geometry"]["coordinates"][0]
    path = MplPath(np.asarray(ring))

    raw = pd.read_parquet(DIR_RAW / "firms" / "viirs_historico_2018_2025_crudo.parquet")
    raw = raw[(raw["longitude"] >= ring_min(ring, 0)) & (raw["longitude"] <= ring_max(ring, 0))
              & (raw["latitude"] >= ring_min(ring, 1)) & (raw["latitude"] <= ring_max(ring, 1))].copy()
    inside = path.contains_points(np.column_stack([raw["longitude"], raw["latitude"]]))
    raw = raw.loc[inside].copy()
    q = raw[
        (raw["scan"] <= 1.5)
        & (~raw["confidence"].astype(str).str.lower().isin(["l", "low"]))
        & (raw["type"] == 0)
    ].copy()
    q["acq_date"] = pd.to_datetime(q["acq_date"]).dt.strftime("%Y-%m-%d")
    q = q[(q["acq_date"] >= "2018-01-01") & (q["acq_date"] <= "2025-12-31")]
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
    x, y = tr.transform(q["longitude"].values, q["latitude"].values)
    q["cell_id"] = (np.floor(x / 1000).astype(np.int64) * 10_000_000
                    + np.floor(y / 1000).astype(np.int64))
    q = q[q["cell_id"].isin(ids)]
    q["frp"] = pd.to_numeric(q["frp"], errors="coerce").fillna(0.0)
    daily = (
        q.groupby(["cell_id", "acq_date"], as_index=False)
        .agg(num_focos_dia=("frp", "size"), max_frp_dia=("frp", "max"), sum_frp_dia=("frp", "sum"))
    )
    daily["departamento"] = "FORMOSA"
    daily["region"] = "Chaco"
    daily["fuego"] = 1
    out = pl.from_pandas(daily).with_columns(pl.col("acq_date").alias("fecha"))
    out.write_parquet(OUT / "focos_formosa_diarios.parquet", compression="zstd")

    meta = {
        "fuente": "NASA FIRMS VIIRS SNPP + NOAA-20 (bbox Paraguay, contiene Formosa)",
        "detecciones_crudas_2018_2025": int(len(raw)),
        "detecciones_filtradas": int(len(q)),
        "celdas_dia": int(out.height),
        "fechas": [str(out["acq_date"].min()), str(out["acq_date"].max())],
        "positivos_2024_2025": int(out.filter(pl.col("acq_date").str.slice(0, 4).is_in(["2024", "2025"])).height),
    }
    (OUT / "focos_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def ring_min(ring, axis):
    return min(p[axis] for p in ring)


def ring_max(ring, axis):
    return max(p[axis] for p in ring)


if __name__ == "__main__":
    main()
