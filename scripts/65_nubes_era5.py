"""Script 65: total cloud cover ERA5 (proxy) para R.8.

Descarga tcc a las 15 UTC (0.25 grados) y lo asigna a los nodos climaticos.
No es una mascara de nubes FIRMS; es un proxy ERA5 documentado.
"""

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cdsapi
import polars as pl
import xarray as xr
from src.clima import generar_malla_nodos_clima
from src.constantes import DIR_PROCESSED, DIR_RAW


def descargar(anio, ruta):
    if ruta.exists():
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": "total_cloud_cover",
        "year": str(anio),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": "15:00",
        "area": [-19, -63, -28, -54],
        "grid": [0.25, 0.25],
        "format": "netcdf",
    }, str(ruta))


def main():
    nodos = generar_malla_nodos_clima(0.25).select(["nodo_clima_id", "lat", "lon"])
    lat = xr.DataArray(nodos["lat"].to_numpy(), dims="nodo")
    lon = xr.DataArray(nodos["lon"].to_numpy(), dims="nodo")
    ids = nodos["nodo_clima_id"].to_numpy()
    filas = []
    for anio in range(2018, 2026):
        ruta = DIR_RAW / "era5" / "tcc" / f"tcc_{anio}.nc"
        descargar(anio, ruta)
        ds = xr.open_dataset(ruta)
        lat_dim = "latitude" if "latitude" in ds.coords else "lat"
        lon_dim = "longitude" if "longitude" in ds.coords else "lon"
        t_dim = "valid_time" if "valid_time" in ds.coords else "time"
        sub = ds.sel({lat_dim: lat, lon_dim: lon}, method="nearest")
        vals = sub["tcc"].values
        fechas = [str(t)[:10] for t in sub[t_dim].values]
        for k, f in enumerate(fechas):
            for j, nid in enumerate(ids):
                v = float(vals[k, j])
                if v > 1.0:
                    v = v / 100.0
                filas.append({"nodo_clima_id": int(nid), "fecha": f, "tcc": v})
        ds.close()
        print(anio, len(fechas), flush=True)
    out = pl.DataFrame(filas).unique(subset=["nodo_clima_id", "fecha"]).sort(["fecha", "nodo_clima_id"])
    out.write_parquet(DIR_PROCESSED / "nubes_era5_diario.parquet", compression="zstd")
    print(out.shape, out["tcc"].min(), out["tcc"].max())


if __name__ == "__main__":
    main()
