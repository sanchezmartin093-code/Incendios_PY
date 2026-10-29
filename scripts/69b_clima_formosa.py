"""G0-Formosa: daily climate from the existing CDS ERA5-Land bbox.

Temperature, VPD and wind come from the CDS t2m/d2m/u10/v10 files already
downloaded for the Paraguay bbox (which fully contains Formosa). Precipitation,
gusts and ET0 use the same Open-Meteo correction as the Paraguay pipeline.
Soil moisture comes from the CDS swvl1 files. All indices are recomputed with
``calcular_indices_fwi_y_sequedad``.
"""

from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clima import (
    calcular_indices_fwi_y_sequedad,
    descargar_clima_openmeteo_batch,
)
from src.clima_cds import procesar_nc_a_nodos_diarios
from src.constantes import DIR_PROCESSED, DIR_RAW

OUT = DIR_PROCESSED / "formosa_g0"
DIR_CDS = DIR_RAW / "era5" / "cds"
DIR_SWVL = DIR_RAW / "era5" / "cds_swvl1_pev"
YEARS = list(range(2018, 2026))


def _swvl_nodos(path: Path, nodos: pl.DataFrame) -> pl.DataFrame:
    ds = xr.open_dataset(path)
    try:
        var = next(v for v in ds.data_vars if "swvl1" in str(v).lower() or "volumetric_soil_water" in str(v).lower())
        lat_dim = "latitude" if "latitude" in ds.coords else "lat"
        lon_dim = "longitude" if "longitude" in ds.coords else "lon"
        time_dim = "valid_time" if "valid_time" in ds.coords else "time"
        sub = ds.sel({
            lat_dim: xr.DataArray(nodos["lat"].to_numpy(), dims="nodo"),
            lon_dim: xr.DataArray(nodos["lon"].to_numpy(), dims="nodo"),
        }, method="nearest")
        arr = sub[var].values
        times = sub[time_dim].values
        ids = nodos["nodo_clima_id"].to_numpy()
    finally:
        ds.close()
    dates = np.array([str(t)[:10] for t in times])
    uniq, inv = np.unique(dates, return_inverse=True)
    rows = []
    for d, fecha in enumerate(uniq):
        rows.append(pl.DataFrame({
            "nodo_clima_id": ids.astype(np.int64),
            "fecha": [str(fecha)] * len(ids),
            "swvl1_m3m3": np.nanmean(arr[inv == d, :], axis=0).astype(np.float32),
        }))
    return pl.concat(rows)


def main() -> None:
    grid = pl.read_parquet(OUT / "grilla_formosa_1km.parquet")
    nodos = (
        grid.select(["nodo_clima_id", "lat", "lon"]).unique("nodo_clima_id")
        .sort("nodo_clima_id")
    )
    print(f"Nodos Formosa: {nodos.height}")

    base_parts = []
    swvl_parts = []
    for year in YEARS:
        for q in (1, 2, 3, 4):
            f = DIR_CDS / f"era5_land_py_{year}_Q{q}.nc"
            if not f.exists():
                raise FileNotFoundError(f)
            base_parts.append(procesar_nc_a_nodos_diarios(f, nodos))
            fs = DIR_SWVL / f"era5_land_swvl1_pev_{year}_Q{q}.nc"
            if not fs.exists():
                raise FileNotFoundError(fs)
            swvl_parts.append(_swvl_nodos(fs, nodos))
        print(f"  CDS {year} listo", flush=True)

    base = pl.concat(base_parts).unique(["nodo_clima_id", "fecha"], keep="first").sort(["nodo_clima_id", "fecha"])
    swvl = pl.concat(swvl_parts).unique(["nodo_clima_id", "fecha"], keep="first")
    base = base.drop("swvl1_m3m3").join(swvl, on=["nodo_clima_id", "fecha"], how="left")
    print(f"CDS nodo-día: {base.height:,}; swvl nulos: {base['swvl1_m3m3'].null_count()}")

    om = None
    try:
        om = descargar_clima_openmeteo_batch(
            lats=nodos["lat"].to_list(), lons=nodos["lon"].to_list(),
            nodo_ids=nodos["nodo_clima_id"].to_list(),
            start_date="2018-01-01", end_date="2025-12-31",
            batch_size=10, max_workers=1, delay_seg=10.0,
            cache_dir=DIR_RAW / "era5" / "openmeteo_formosa",
        )
        print(f"Open-Meteo nodo-día: {om.height:,}")
        om = om.select([
            "nodo_clima_id", "fecha",
            pl.col("precipitacion_mm").alias("precipitacion_mm_om"),
            pl.col("rafaga_max_kmh").alias("rafaga_max_kmh_om"),
            pl.col("et0_mm").alias("et0_mm_om"),
        ])
        base = base.join(om, on=["nodo_clima_id", "fecha"], how="left")
        for col, omcol in (("precipitacion_mm", "precipitacion_mm_om"), ("rafaga_max_kmh", "rafaga_max_kmh_om"), ("et0_mm", "et0_mm_om")):
            if base[omcol].null_count():
                print(f"  [aviso] {col}: {base[omcol].null_count()} nulos Open-Meteo; se conserva CDS")
            base = base.with_columns(pl.coalesce([pl.col(omcol), pl.col(col)]).alias(col))
        base = base.drop(["precipitacion_mm_om", "rafaga_max_kmh_om", "et0_mm_om"])
        fuente_om = "completa"
    except Exception as exc:
        print(f"[aviso] Open-Meteo incompleto: {exc}; se usa CDS + ET0 Hargreaves")
        # Hargreaves ET0 from Tmax/Tmin/lat, as in 02f.
        j = base["fecha"].str.to_date("%Y-%m-%d").dt.ordinal_day().to_numpy().astype(float)
        tmax = base["temp_max_c"].to_numpy(); tmin = base["temp_min_c"].to_numpy()
        lat = np.radians(base.join(nodos.select(["nodo_clima_id", "lat"]), on="nodo_clima_id", how="left")["lat"].to_numpy())
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
        dec = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)
        ws = np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0))
        ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws))
        et0 = 0.0023 * ((tmax + tmin) / 2.0 + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)) * ra / 2.45
        base = base.with_columns(pl.Series("et0_mm", np.clip(et0, 0.0, None)))
        fuente_om = "fallback_cds_hargreaves"

    # Ensure daily continuity before recursive FFMC.
    for nid, sub in base.group_by("nodo_clima_id"):
        fechas = sub["fecha"].sort().to_list()
        if len(fechas) != 2922:
            raise RuntimeError(f"Nodo {nid}: {len(fechas)} días, se esperaban 2922 (2018-2025)")
        fechas_dt = [np.datetime64(x) for x in fechas]
        if not all((b - a).astype(int) == 1 for a, b in zip(fechas_dt, fechas_dt[1:])):
            raise RuntimeError(f"Nodo {nid}: serie no continua")

    clima = calcular_indices_fwi_y_sequedad(base.sort(["nodo_clima_id", "fecha"]))
    clima.write_parquet(OUT / "clima_era5_diario.parquet", compression="zstd")
    meta = {
        "fuente": "ERA5-Land CDS + Open-Meteo/Hargreaves + swvl1 CDS",
        "openmeteo": fuente_om,
        "nodos": nodos.height,
        "filas": clima.height,
        "fecha_min": clima["fecha"].min(),
        "fecha_max": clima["fecha"].max(),
        "precipitacion_max_mm": float(clima["precipitacion_mm"].max()),
        "swvl1_nulos": int(clima["swvl1_m3m3"].null_count()),
    }
    (OUT / "clima_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
