"""G0-Formosa: MapBiomas Argentina C3 to 1-km fuel fractions.

The crosswalk is H-A from `auditoria/g0_externa/crosswalk_mapbiomas_argentina.csv`.
Class 6 is assigned to forest (same operational rule used for Paraguay C2 in
the Chaco); its fraction is preserved separately for H-B/H-C sensitivities.
"""

from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_PROCESSED, DIR_RAW

OUT = DIR_PROCESSED / "formosa_g0"
RAW = DIR_RAW / "mapbiomas_argentina"
CODES = [3, 4, 6, 9, 10, 11, 12, 14, 15, 18, 19, 21, 22, 24, 25, 26, 27, 33, 34, 36, 63, 66, 73, 77]
C2I = {c: i for i, c in enumerate(CODES)}
FOREST = {3, 4, 6}
AGRO = {15, 18, 19, 21, 36}
VALID = set(CODES) - {27}

def _counts_for_map(path: Path, grid: pl.DataFrame) -> pl.DataFrame:
    x = grid["x_utm"].to_numpy(); y = grid["y_utm"].to_numpy()
    res = 1000.0
    minx = float(x.min()) - res / 2.0
    maxx = float(x.max()) + res / 2.0
    miny = float(y.min()) - res / 2.0
    maxy = float(y.max()) + res / 2.0
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    c_idx = np.floor((x - minx) / res).astype(int)
    r_idx = np.floor((maxy - y) / res).astype(int)
    if c_idx.min() < 0 or c_idx.max() >= width or r_idx.min() < 0 or r_idx.max() >= height:
        raise ValueError("La grilla Formosa no cabe en el destino")

    counts = np.zeros((height * width, len(CODES)), dtype=np.int32)
    edge_counts = np.zeros(height * width, dtype=np.int32)
    valid_counts = np.zeros(height * width, dtype=np.int32)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)

    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(
            "EPSG:32721", src.crs, minx, miny, maxx, maxy, densify_pts=21
        )
        win = src.window(west, south, east, north)
        row0 = max(0, int(np.floor(win.row_off)))
        col0 = max(0, int(np.floor(win.col_off)))
        row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
        col1 = min(src.width, int(np.ceil(win.col_off + win.width)))
        print(f"  {path.name}: ventana {row1-row0}x{col1-col0}", flush=True)
        lon_cols = src.transform.c + (np.arange(col0, col1) + 0.5) * src.transform.a

        for rr in range(row0, row1, 512):
            ee = min(rr + 512, row1)
            # one-pixel halo for the edge mask
            hr0 = max(row0, rr - 1); hr1 = min(row1, ee + 1)
            hc0 = max(col0, col0 - 1); hc1 = min(col1, col1 + 1)
            data_h = src.read(1, window=((hr0, hr1), (hc0, hc1)))
            data = data_h[rr - hr0:ee - hr0, :]
            lat_rows = src.transform.f + (np.arange(rr, ee) + 0.5) * src.transform.e
            lon2, lat2 = np.meshgrid(lon_cols, lat_rows)
            xx, yy = tr.transform(lon2.ravel(), lat2.ravel())
            ci = np.floor((xx - minx) / res).astype(np.int64)
            ri = np.floor((maxy - yy) / res).astype(np.int64)
            ok = (ci >= 0) & (ci < width) & (ri >= 0) & (ri < height)
            dest = (ri[ok] * width + ci[ok]).astype(np.int64)
            vals = data.ravel()[ok]

            valid = np.isin(vals, list(VALID))
            if valid.any():
                d = dest[valid]; v = vals[valid]
                code_idx = np.array([C2I[int(c)] for c in v], dtype=np.int64)
                bc = np.bincount(d * len(CODES) + code_idx, minlength=width * height * len(CODES))
                counts += bc.reshape(width * height, len(CODES)).astype(np.int32)
                valid_counts += np.bincount(d, minlength=width * height).astype(np.int32)

            # Edge mask: forest pixel with an 8-neighbour agro pixel.
            es_bosque = np.isin(data_h, list(FOREST))
            es_agro = np.isin(data_h, list(AGRO))
            if es_bosque.any() and es_agro.any():
                pad = np.pad(es_agro, 1, mode="constant", constant_values=False)
                neigh = np.zeros_like(es_agro)
                hh, ww = es_agro.shape
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        neigh |= pad[1 + dr:1 + dr + hh, 1 + dc:1 + dc + ww]
                edge_h = es_bosque & neigh
                edge = edge_h[rr - hr0:ee - hr0, :]
                ev = edge.ravel()[ok]
                if ev.any():
                    edge_counts += np.bincount(dest[ev], minlength=width * height).astype(np.int32)
            if (rr - row0) % (512 * 8) == 0:
                print(f"    filas {rr-row0}/{row1-row0}", flush=True)

    # Reorder to the grid row order.
    flat_idx = r_idx * width + c_idx
    c = counts[flat_idx]
    vc = valid_counts[flat_idx].astype(np.float64)
    denom = np.where(vc > 0, vc, np.nan)
    frac = c / denom[:, None] * 100.0
    edge = np.where(vc > 0, edge_counts[flat_idx] / denom * 100.0, 0.0)

    def f(codes):
        return np.nansum(frac[:, [C2I[k] for k in codes]], axis=1).astype(np.float32)

    pasture = f([15, 21, 14])
    agriculture = f([18, 19, 36])
    plantation = f([9])
    pct_no_comb = np.clip(f([22, 24, 25, 26, 33, 34]), 0, 100).astype(np.float32)
    return pl.DataFrame({
        "cell_id": grid["cell_id"],
        "pct_pastura": (pasture + agriculture + plantation).astype(np.float32),
        "pct_pastura_ganadera": pasture.astype(np.float32),
        "pct_agricultura_anual": agriculture.astype(np.float32),
        "pct_plantacion_forestal": plantation.astype(np.float32),
        "pct_sabana_pastizal": f([10, 12, 63, 66, 77]).astype(np.float32),
        "pct_bosque_seco": f([3, 4, 6]).astype(np.float32),
        "pct_bosque_humedo": np.zeros(len(grid), dtype=np.float32),
        "pct_humedal_palmar": f([11, 73]).astype(np.float32),
        "borde_bosque_agro_pct": np.clip(edge, 0, 100).astype(np.float32),
        "pct_no_combustible": pct_no_comb,
        "es_combustible": (pct_no_comb < 80.0),
        "pct_clase6": f([6]).astype(np.float32),
        "pct_arbustal": f([63, 66, 77]).astype(np.float32),
        "pct_clase21": f([21]).astype(np.float32),
        "pct_clase27": f([27]).astype(np.float32),
        "pct_clase14": f([14]).astype(np.float32),
        "pct_validos": vc.astype(np.float32),
    })


def main() -> None:
    grid = pl.read_parquet(OUT / "grilla_formosa_1km.parquet")
    cache = {}
    tables = []
    for anio_mapa in range(2017, 2025):
        path = RAW / f"integration-argentina_classification_{anio_mapa}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"MapBiomas {anio_mapa}", flush=True)
        df = _counts_for_map(path, grid)
        cache[anio_mapa] = df.with_columns(pl.lit(anio_mapa, dtype=pl.Int32).alias("anio_mapa"))
    for anio_fila in range(2018, 2026):
        anio_mapa = anio_fila - 1
        df = cache[anio_mapa].with_columns([
            pl.lit(anio_fila, dtype=pl.Int32).alias("anio_fila"),
            pl.lit(anio_fila - anio_mapa, dtype=pl.Int16).alias("fuel_lag"),
        ])
        tables.append(df)
    out = pl.concat(tables, how="vertical")
    out.write_parquet(OUT / "combustible_formosa_anual.parquet", compression="zstd")
    meta = {
        "fuente": "MapBiomas Argentina Colección 3",
        "crosswalk": "H-A (auditoria/g0_externa/crosswalk_mapbiomas_argentina.csv)",
        "anio_fila": [2018, 2025],
        "anio_mapa": [2017, 2024],
        "filas": out.height,
        "celdas": out["cell_id"].n_unique(),
        "max_clase6_pct": float(out["pct_clase6"].max()),
        "max_clase27_pct": float(out["pct_clase27"].max()),
    }
    (OUT / "combustible_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
