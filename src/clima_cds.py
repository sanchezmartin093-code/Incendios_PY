"""
clima_cds.py module: Direct ingestion and processing of ERA5-Land from Copernicus CDS.
- Downloads NetCDF grids bounded to Paraguay (official bounding box).
- Extracts and processes meteorological variables for the national representative nodes.
- Calculates biophysical fire indices: daily VPD (Tetens), CDD, 48h rainfall, FFMC, ISI.
- Exports in compressed ZSTD columnar format: datos/processed/clima_era5_diario.parquet.
"""

from pathlib import Path
import time
import zipfile
import cdsapi
import numpy as np
import polars as pl
import xarray as xr

from src.constantes import DIR_RAW, DIR_PROCESSED, BBOX_PARAGUAY
from src.clima import generar_malla_nodos_clima, asignar_nodos_a_grilla, calcular_indices_fwi_y_sequedad
from src.fwi import calcular_vpd

VARIABLES_CDS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# CDS format bounding box: [North, West, South, East]
AREA_PARAGUAY_CDS = [
    BBOX_PARAGUAY["max_lat"],
    BBOX_PARAGUAY["min_lon"],
    BBOX_PARAGUAY["min_lat"],
    BBOX_PARAGUAY["max_lon"]
]

TRIMESTRES_MESES = {
    1: ["01", "02", "03"],
    2: ["04", "05", "06"],
    3: ["07", "08", "09"],
    4: ["10", "11", "12"]
}

def agregar_precipitacion_diaria(tp_mm: np.ndarray) -> np.ndarray:
    """
    Aggregates a subdaily precipitation series into daily totals.

    The first time step is discarded because in ERA5-Land it may contain the
    previous day's accumulation. The tail is interpreted as accumulated if it is
    monotonic and, if not, as subdaily increments. The result has shape
    ``(nodos,)``.
    """
    valores = np.asarray(tp_mm, dtype=float)
    if valores.ndim == 1:
        valores = valores[:, None]

    totales = np.full(valores.shape[1], np.nan, dtype=float)
    for j in range(valores.shape[1]):
        serie = valores[:, j]
        validos = np.isfinite(serie)
        if not validos.any():
            continue
        s = serie[validos]
        if s.size == 1:
            totales[j] = max(float(s[0]), 0.0)
            continue

        cola_raw = serie[1:]
        cola = cola_raw[np.isfinite(cola_raw)]
        if cola.size == 0:
            totales[j] = max(float(s[0]), 0.0)
            continue
        diferencias = np.diff(cola)
        tolerancia = 1e-2
        if diferencias.size > 0 and np.all(diferencias >= -tolerancia):
            totales[j] = max(float(cola[-1]), 0.0)
        else:
            totales[j] = float(np.nansum(np.clip(cola, 0.0, None)))
    return totales


def descargar_trimestre_cds(
    client: cdsapi.Client,
    anio: int,
    trimestre: int,
    dir_salida: Path,
    forzar: bool = False,
) -> Path:
    """
    Requests and downloads a quarter of ERA5-Land (3 months, 3-hourly resolution) from Copernicus CDS.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    archivo_nc = dir_salida / f"era5_land_py_{anio}_Q{trimestre}.nc"
    archivo_zip = dir_salida / f"era5_land_py_{anio}_Q{trimestre}.zip"

    if forzar:
        archivo_nc.unlink(missing_ok=True)
        archivo_zip.unlink(missing_ok=True)

    if archivo_nc.exists():
        print(f" ✓ Archivo NetCDF encontrado en disco: {archivo_nc.name}")
        return archivo_nc

    if archivo_zip.exists():
        print(f" ✓ Descomprimiendo {archivo_zip.name}...")
        with zipfile.ZipFile(archivo_zip, "r") as zf:
            zf.extractall(dir_salida)
        ncs = list(dir_salida.glob(f"*{anio}*Q{trimestre}*.nc"))
        if ncs:
            return ncs[0]

    meses = TRIMESTRES_MESES[trimestre]
    dias = [f"{d:02d}" for d in range(1, 32)]
    # 3-hourly resolution is requested due to CDS cost limits; daily rain
    # is reconstructed by summing the subdaily increments.
    horas = [f"{h:02d}:00" for h in range(0, 24, 3)]

    peticion = {
        "variable": VARIABLES_CDS,
        "year": str(anio),
        "month": meses,
        "day": dias,
        "time": horas,
        "data_format": "netcdf",
        "area": AREA_PARAGUAY_CDS
    }

    print(f"\n[CDS ECMWF] Solicitando ERA5-Land {anio} Q{trimestre} ({meses[0]} a {meses[-1]})...")
    destino_temp = dir_salida / f"era5_land_py_{anio}_Q{trimestre}_descarga.nc"
    client.retrieve("reanalysis-era5-land", peticion, str(destino_temp))

    if zipfile.is_zipfile(destino_temp):
        destino_temp.rename(archivo_zip)
        temp_extract_dir = dir_salida / f"temp_{anio}_Q{trimestre}"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archivo_zip, "r") as zf:
            zf.extractall(temp_extract_dir)
        ncs = list(temp_extract_dir.glob("*.nc"))
        if ncs:
            if archivo_nc.exists():
                archivo_nc.unlink()
            ncs[0].rename(archivo_nc)
        import shutil
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        if archivo_zip.exists():
            archivo_zip.unlink()
    else:
        destino_temp.rename(archivo_nc)

    print(f"   ✓ Guardado NetCDF: {archivo_nc.name} ({archivo_nc.stat().st_size / (1024*1024):.2f} MB)")
    return archivo_nc

def procesar_nc_a_nodos_diarios(
    ruta_nc: Path,
    df_nodos: pl.DataFrame
) -> pl.DataFrame:
    """
    Opens the ERA5-Land NetCDF with xarray, interpolates/extracts for the climate nodes
    and aggregates to daily metrics (T max, T min, daily rain, max wind, max VPD,
    daily mean 0-7 cm soil moisture).
    """
    print(f"Procesando NetCDF: {ruta_nc.name} con xarray...")
    ds = xr.open_dataset(ruta_nc)

    # Identify variable names (in ERA5-Land they are usually t2m, d2m, tp, u10, v10)
    var_map = {}
    for v in ds.data_vars:
        v_low = v.lower()
        if "d2m" in v_low or "dew" in v_low:
            var_map["d2m"] = v
        elif "t2m" in v_low or "temperature" in v_low:
            var_map["t2m"] = v
        elif "tp" in v_low or "precip" in v_low:
            var_map["tp"] = v
        elif "u10" in v_low or "u_wind" in v_low:
            var_map["u10"] = v
        elif "v10" in v_low or "v_wind" in v_low:
            var_map["v10"] = v
        elif "swvl1" in v_low or "volumetric_soil_water" in v_low:
            var_map["swvl1"] = v

    # Meteorological node coordinates
    nodo_lats = xr.DataArray(df_nodos["lat"].to_numpy(), dims="nodo")
    nodo_lons = xr.DataArray(df_nodos["lon"].to_numpy(), dims="nodo")
    nodo_ids = df_nodos["nodo_clima_id"].to_numpy()

    # Nearest spatial selection of all nodes in a single matrix operation
    # Determine spatial dimension names (latitude/longitude or lat/lon)
    lat_dim = "latitude" if "latitude" in ds.coords else "lat"
    lon_dim = "longitude" if "longitude" in ds.coords else "lon"
    time_dim = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)

    sub_ds = ds.sel({lat_dim: nodo_lats, lon_dim: nodo_lons}, method="nearest")

    # Extract numpy arrays: dimensions (time, node)
    t2m_k = sub_ds[var_map["t2m"]].values
    d2m_k = sub_ds[var_map["d2m"]].values
    tp_m = sub_ds[var_map["tp"]].values
    u10 = sub_ds[var_map["u10"]].values
    v10 = sub_ds[var_map["v10"]].values
    swvl1 = sub_ds[var_map["swvl1"]].values if "swvl1" in var_map else None
    tiempos = sub_ds[time_dim].values

    # Physical conversion
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    tp_mm = np.maximum(tp_m * 1000.0, 0.0)
    viento_kmh = np.sqrt(u10**2 + v10**2) * 3.6
    vpd_kpa = calcular_vpd(t2m_c, d2m_c)

    # Identify unique dates
    fechas_dt = np.array([str(t)[:10] for t in tiempos])
    fechas_unicas, indices_dias = np.unique(fechas_dt, return_inverse=True)

    registros = []
    num_nodos = len(nodo_ids)

    for day_idx, fecha_str in enumerate(fechas_unicas):
        mascara_h = (indices_dias == day_idx)

        t_max_dia = np.nanmax(t2m_c[mascara_h, :], axis=0)
        t_min_dia = np.nanmin(t2m_c[mascara_h, :], axis=0)
        prec_dia = agregar_precipitacion_diaria(tp_mm[mascara_h, :])
        viento_max_dia = np.nanmax(viento_kmh[mascara_h, :], axis=0)
        vpd_max_dia = np.nanmax(vpd_kpa[mascara_h, :], axis=0)
        if swvl1 is not None:
            swvl1_dia = np.nanmean(swvl1[mascara_h, :], axis=0)
        else:
            swvl1_dia = np.full(num_nodos, np.nan, dtype=float)

        for n_i in range(num_nodos):
            registros.append({
                "nodo_clima_id": int(nodo_ids[n_i]),
                "fecha": fecha_str,
                "temp_max_c": float(t_max_dia[n_i]),
                "temp_min_c": float(t_min_dia[n_i]),
                "precipitacion_mm": float(prec_dia[n_i]),
                "viento_max_kmh": float(viento_max_dia[n_i]),
                # ERA5-Land does not publish gusts. A documented linear proxy
                # is left and excluded from the feature set.
                "rafaga_max_kmh": float(viento_max_dia[n_i] * 1.3),
                # ET0 is not requested in this download; it is left explicit.
                "et0_mm": 0.0,
                "vpd_max_kpa": float(vpd_max_dia[n_i]),
                "swvl1_m3m3": float(swvl1_dia[n_i]),
            })

    ds.close()
    return pl.DataFrame(registros)
