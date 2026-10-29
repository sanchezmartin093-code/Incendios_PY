"""
Script 02d: Full Canadian FWI from ERA5-Land 3-hourly.

Extracts temperature, dew point and wind at 15 UTC (approximate local
noon), uses the daily rainfall from clima_era5_diario and computes
FFMC, DMC, DC, BUI, ISI and FWI per node-day.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
import xarray as xr

from src.constantes import DIR_PROCESSED, DIR_RAW
from src.fwi import (
    calcular_bui,
    calcular_dc,
    calcular_dmc,
    calcular_ffmc,
    calcular_fwi,
    calcular_isi,
)

HORA_MEDIO_DIA_UTC = 15


def _nodos_activos() -> pl.DataFrame:
    grilla = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet",
        columns=["nodo_clima_id"],
    )
    ids = sorted(grilla["nodo_clima_id"].unique().to_list())
    from src.clima import generar_malla_nodos_clima
    nodos = generar_malla_nodos_clima(0.25)
    return nodos.filter(pl.col("nodo_clima_id").is_in(ids)).sort("nodo_clima_id")


def _extraer_mediodia(nodos: pl.DataFrame) -> pl.DataFrame:
    registros = []
    archivos = sorted((DIR_RAW / "era5" / "cds").glob("era5_land_py_*_Q*.nc"))
    if not archivos:
        raise FileNotFoundError("No hay NetCDF ERA5-Land en datos/raw/era5/cds")

    lats = xr.DataArray(nodos["lat"].to_numpy(), dims="nodo")
    lons = xr.DataArray(nodos["lon"].to_numpy(), dims="nodo")
    ids = nodos["nodo_clima_id"].to_numpy()

    for ruta in archivos:
        ds = xr.open_dataset(ruta)
        lat_dim = "latitude" if "latitude" in ds.coords else "lat"
        lon_dim = "longitude" if "longitude" in ds.coords else "lon"
        t_dim = "valid_time" if "valid_time" in ds.coords else "time"
        sub = ds.sel({lat_dim: lats, lon_dim: lons}, method="nearest")
        horas = sub[t_dim].dt.hour.values
        mascara = horas == HORA_MEDIO_DIA_UTC
        if not mascara.any():
            ds.close()
            continue
        sub = sub.isel({t_dim: np.where(mascara)[0]})
        t2m = sub["t2m"].values - 273.15
        d2m = sub["d2m"].values - 273.15
        u10 = sub["u10"].values
        v10 = sub["v10"].values
        fechas = [str(t)[:10] for t in sub[t_dim].values]
        for k, fecha in enumerate(fechas):
            for j, nodo_id in enumerate(ids):
                t = float(t2m[k, j])
                td = float(d2m[k, j])
                es = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
                ea = 0.61078 * np.exp((17.27 * td) / (td + 237.3))
                rh = float(np.clip(100.0 * ea / es, 1.0, 100.0))
                wind = float(np.sqrt(u10[k, j] ** 2 + v10[k, j] ** 2) * 3.6)
                registros.append((int(nodo_id), fecha, t, rh, wind))
        ds.close()
    return pl.DataFrame(
        registros,
        schema={
            "nodo_clima_id": pl.Int64,
            "fecha": pl.String,
            "temp_mediodia_c": pl.Float64,
            "rh_mediodia_pct": pl.Float64,
            "viento_mediodia_kmh": pl.Float64,
        },
        orient="row",
    )


def main():
    nodos = _nodos_activos()
    met = _extraer_mediodia(nodos)
    lluvia = pl.read_parquet(
        DIR_PROCESSED / "clima_era5_diario.parquet",
        columns=["nodo_clima_id", "fecha", "precipitacion_mm"],
    )
    df = met.join(lluvia, on=["nodo_clima_id", "fecha"], how="inner").sort(
        ["nodo_clima_id", "fecha"]
    )

    salida = []
    for (nodo,), sub in df.group_by("nodo_clima_id", maintain_order=True):
        ffmc, dmc, dc = 85.0, 6.0, 15.0
        for r in sub.iter_rows(named=True):
            ffmc = calcular_ffmc(
                r["temp_mediodia_c"], r["rh_mediodia_pct"],
                r["viento_mediodia_kmh"], r["precipitacion_mm"], ffmc,
            )
            mes = int(r["fecha"][5:7])
            dmc = calcular_dmc(
                r["temp_mediodia_c"], r["rh_mediodia_pct"],
                r["viento_mediodia_kmh"], r["precipitacion_mm"], dmc, mes=mes,
            )
            dc = calcular_dc(r["temp_mediodia_c"], r["precipitacion_mm"], dc, mes=mes)
            bui = calcular_bui(dmc, dc)
            isi = calcular_isi(r["viento_mediodia_kmh"], ffmc)
            fwi = calcular_fwi(isi, bui)
            salida.append({
                "nodo_clima_id": int(nodo),
                "fecha": r["fecha"],
                "ffmc_full": ffmc,
                "dmc": dmc,
                "dc": dc,
                "bui": bui,
                "isi_full": isi,
                "fwi": fwi,
            })

    out = pl.DataFrame(salida)
    ruta = DIR_PROCESSED / "fwi_completo_diario.parquet"
    out.write_parquet(ruta, compression="zstd")
    print(out.describe())
    print(f"✓ Guardado: {ruta}")


if __name__ == "__main__":
    main()
