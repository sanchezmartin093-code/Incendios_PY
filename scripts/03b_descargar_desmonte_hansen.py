"""
Script 03b: Hansen Global Forest Change annual tree-cover loss.

Computes ``pct_desmonte_t1`` (loss in Y-1), ``pct_desmonte_t2`` (loss in Y-2)
and ``pct_desmonte_2a`` (the sum of both) on the national 1 km² grid.

Data source
-----------
Hansen/UMD/Google/USGS/NASA Global Forest Change, version 1.13 (2025).
The tiles are 10x10 degrees and use the north-west corner in their name.
Paraguay requires four tiles:

    10S_060W, 10S_070W, 20S_060W, 20S_070W

``30S_*`` tiles start at -30 degrees and do not intersect continental Paraguay.

Important
---------
``lossyear`` is tree-cover loss, not a cadastral "desmonte" layer. It includes
fire-driven loss and plantation harvest. It is used here as a transparent
annual proxy of recent woody-cover loss. The feature is strictly causal: a row
from year Y only uses loss years Y-1 and Y-2.
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_PROCESSED, DIR_RAW

VERSION_HANSEN = "GFC-2025-v1.13"
TILES_HANSEN = ["10S_060W", "10S_070W", "20S_060W", "20S_070W"]
BASE_URL_HANSEN = (
    "https://storage.googleapis.com/earthenginepartners-hansen/"
    "{version}/Hansen_{version}_lossyear_{tile}.tif"
)

# lossyear codes: 16 = 2016, ..., 25 = 2025.
CODIGOS_ANIO = list(range(16, 26))
ANIO_MIN = 2018
ANIO_MAX = 2026


def _url_tile(tile: str, version: str = VERSION_HANSEN) -> str:
    return BASE_URL_HANSEN.format(version=version, tile=tile)


def descargar_tiles_hansen(
    dir_destino: Path,
    forzar: bool = False,
    max_reintentos: int = 3,
    version: str = VERSION_HANSEN,
) -> list[Path]:
    """Downloads the four Hansen tiles with atomic writes and retries."""
    dir_destino.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []

    for tile in TILES_HANSEN:
        ruta = dir_destino / f"hansen_lossyear_{version}_{tile}.tif"
        if ruta.exists() and not forzar and ruta.stat().st_size > 1_000_000:
            print(f" ✓ Hansen en caché: {ruta.name}")
            rutas.append(ruta)
            continue

        url = _url_tile(tile, version=version)
        ruta_parcial = ruta.with_suffix(ruta.suffix + ".part")
        ultimo_error: Exception | None = None

        for intento in range(1, max_reintentos + 1):
            try:
                print(f" Descargando {ruta.name} (intento {intento}/{max_reintentos})...")
                with requests.get(
                    url, stream=True, timeout=(30, 300), headers={"User-Agent": "Incendios_PY/1.0"}
                ) as resp:
                    resp.raise_for_status()
                    with open(ruta_parcial, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if ruta_parcial.stat().st_size < 1_000_000:
                    raise RuntimeError("archivo descargado demasiado pequeño")
                os.replace(ruta_parcial, ruta)
                ultimo_error = None
                break
            except Exception as exc:  # noqa: BLE001 - download retry loop
                ultimo_error = exc
                ruta_parcial.unlink(missing_ok=True)
                if intento < max_reintentos:
                    time.sleep(5.0 * intento)

        if ultimo_error is not None:
            raise RuntimeError(f"No se pudo descargar {url}: {ultimo_error}") from ultimo_error

        rutas.append(ruta)

    return rutas


def _ventana_intersectada(src: rasterio.DatasetReader, bounds_utm: tuple[float, float, float, float]) -> Window | None:
    """Returns the source window intersecting the UTM destination bounds."""
    try:
        west, south, east, north = transform_bounds(
            "EPSG:32721", src.crs, *bounds_utm, densify_pts=21
        )
    except Exception:
        return None

    win = src.window(west, south, east, north)
    row0 = max(0, int(np.floor(win.row_off)))
    col0 = max(0, int(np.floor(win.col_off)))
    row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
    col1 = min(src.width, int(np.ceil(win.col_off + win.width)))
    if row1 <= row0 or col1 <= col0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def procesar_desmonte_anual(
    df_grilla: pl.DataFrame,
    rutas_tif: list[Path],
    resolucion_m: int = 1000,
    ventana_px: int = 2048,
) -> pl.DataFrame:
    """Computes annual loss percentages per cell without looking into the future."""
    x = df_grilla["x_utm"].to_numpy()
    y = df_grilla["y_utm"].to_numpy()
    min_x = float(np.min(x)) - resolucion_m / 2.0
    max_y = float(np.max(y)) + resolucion_m / 2.0
    max_x = float(np.max(x)) + resolucion_m / 2.0
    min_y = float(np.min(y)) - resolucion_m / 2.0
    width = int(np.ceil((max_x - min_x) / resolucion_m - 1e-9))
    height = int(np.ceil((max_y - min_y) / resolucion_m - 1e-9))
    dst_transform = from_origin(min_x, max_y, resolucion_m, resolucion_m)
    bounds_utm = (min_x, min_y, max_x, max_y)

    c_idx = np.round((x - min_x - resolucion_m / 2.0) / resolucion_m).astype(int)
    r_idx = np.round((max_y - resolucion_m / 2.0 - y) / resolucion_m).astype(int)
    if c_idx.min() < 0 or c_idx.max() >= width or r_idx.min() < 0 or r_idx.max() >= height:
        raise ValueError("La grilla no cabe en el raster destino")

    acumulado = {cod: np.zeros(width * height, dtype=np.float32) for cod in CODIGOS_ANIO}

    for ruta in rutas_tif:
        print(f"  Procesando Hansen: {ruta.name}")
        with rasterio.open(ruta) as src:
            win_aoi = _ventana_intersectada(src, bounds_utm)
            if win_aoi is None:
                print(f"    [aviso] {ruta.name} no intersecta la grilla nacional; se omite.")
                continue

            destinos = {cod: np.zeros((height, width), dtype=np.float32) for cod in CODIGOS_ANIO}
            row_start = int(win_aoi.row_off)
            col_start = int(win_aoi.col_off)
            row_stop = int(win_aoi.row_off + win_aoi.height)
            col_stop = int(win_aoi.col_off + win_aoi.width)

            for row in range(row_start, row_stop, ventana_px):
                for col in range(col_start, col_stop, ventana_px):
                    h = min(ventana_px, row_stop - row)
                    w = min(ventana_px, col_stop - col)
                    window = Window(col, row, w, h)
                    data = src.read(1, window=window)
                    if data.size == 0 or int(data.max()) == 0:
                        continue
                    src_transform = src.window_transform(window)
                    for cod in CODIGOS_ANIO:
                        mascara = (data == cod).astype(np.float32)
                        if mascara.max() == 0:
                            continue
                        reproject(
                            source=mascara,
                            destination=destinos[cod],
                            src_transform=src_transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs="EPSG:32721",
                            resampling=Resampling.average,
                            init_dest_nodata=False,
                        )

            for cod in CODIGOS_ANIO:
                acumulado[cod] += destinos[cod].ravel()

    frac_anio = {
        2000 + cod: np.clip(
            acumulado[cod][r_idx * width + c_idx] * 100.0, 0.0, 100.0
        ).astype(np.float32)
        for cod in CODIGOS_ANIO
    }

    n = len(df_grilla)
    tablas = []
    for anio in range(ANIO_MIN, ANIO_MAX + 1):
        anio_t1 = anio - 1
        anio_t2 = anio - 2
        if anio_t1 in frac_anio:
            desm_t1 = frac_anio[anio_t1]
        else:
            desm_t1 = np.full(n, np.nan, dtype=np.float32)
        if anio_t2 in frac_anio:
            desm_t2 = frac_anio[anio_t2]
        else:
            desm_t2 = np.full(n, np.nan, dtype=np.float32)

        desm_2a = np.where(
            np.isfinite(desm_t1) & np.isfinite(desm_t2),
            desm_t1 + desm_t2,
            np.nan,
        )
        tablas.append(pl.DataFrame({
            "cell_id": df_grilla["cell_id"],
            "anio": np.full(n, anio, dtype=np.int32),
            "pct_desmonte_t1": desm_t1.astype(np.float32),
            "pct_desmonte_t2": desm_t2.astype(np.float32),
            "pct_desmonte_2a": np.clip(desm_2a, 0.0, 100.0).astype(np.float32),
        }))

    return pl.concat(tablas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Desmonte anual Hansen a 1 km²")
    parser.add_argument("--forzar", action="store_true", help="Volver a descargar las teselas")
    parser.add_argument("--version", default=VERSION_HANSEN, help="Versión Hansen (por defecto GFC-2025-v1.13)")
    args = parser.parse_args()

    df_grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    rutas = descargar_tiles_hansen(DIR_RAW / "hansen", forzar=args.forzar, version=args.version)
    df_desmonte = procesar_desmonte_anual(df_grilla, rutas)

    out = DIR_PROCESSED / "desmonte_hansen_1km.parquet"
    df_desmonte.write_parquet(out, compression="zstd")

    meta = {
        "fuente": "Hansen Global Forest Change",
        "version": args.version,
        "teselas": TILES_HANSEN,
        "anios": [ANIO_MIN, ANIO_MAX],
        "filas": df_desmonte.height,
        "columnas": df_desmonte.columns,
        "archivo": out.name,
    }
    (DIR_PROCESSED / "desmonte_hansen_1km.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✓ Desmonte anual guardado en: {out} ({df_desmonte.height:,} filas)")


if __name__ == "__main__":
    main()
