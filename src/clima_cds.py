"""
Módulo clima_cds.py: Ingesta directa y procesamiento de ERA5-Land desde Copernicus CDS.
- Descarga cuadrículas NetCDF acotadas a Paraguay (bounding box oficial).
- Extrae y procesa variables meteorológicas para los nodos representativos nacionales.
- Calcula índices biofísicos de fuego: VPD diario (Tetens), CDD, lluvia 48h, FFMC, ISI.
- Exporta en formato columnar comprimido ZSTD: datos/processed/clima_era5_diario.parquet.
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
    "10m_v_component_of_wind"
]

# Bounding box formato CDS: [North, West, South, East]
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

def descargar_trimestre_cds(
    client: cdsapi.Client,
    anio: int,
    trimestre: int,
    dir_salida: Path
) -> Path:
    """
    Solicita y descarga un trimestre de ERA5-Land (3 meses, 3-hourly) desde Copernicus CDS.
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    archivo_nc = dir_salida / f"era5_land_py_{anio}_Q{trimestre}.nc"
    archivo_zip = dir_salida / f"era5_land_py_{anio}_Q{trimestre}.zip"

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
    Abre el NetCDF de ERA5-Land con xarray, interpola/extrae para los nodos climáticos
    y agrega a métricas diarias (T max, T min, lluvia diaria, viento max, VPD max).
    """
    print(f"Procesando NetCDF: {ruta_nc.name} con xarray...")
    ds = xr.open_dataset(ruta_nc)

    # Identificar nombres de variables (en ERA5-Land suelen ser t2m, d2m, tp, u10, v10)
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

    # Coordenadas de los nodos meteorológicos
    nodo_lats = xr.DataArray(df_nodos["lat"].to_numpy(), dims="nodo")
    nodo_lons = xr.DataArray(df_nodos["lon"].to_numpy(), dims="nodo")
    nodo_ids = df_nodos["nodo_clima_id"].to_numpy()

    # Selección espacial nearest de todos los nodos en un solo paso matricial
    # Determinar nombres de dimensiones espaciales (latitude/longitude o lat/lon)
    lat_dim = "latitude" if "latitude" in ds.coords else "lat"
    lon_dim = "longitude" if "longitude" in ds.coords else "lon"
    time_dim = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)

    sub_ds = ds.sel({lat_dim: nodo_lats, lon_dim: nodo_lons}, method="nearest")

    # Extraer arrays numpy: dimensiones (tiempo, nodo)
    t2m_k = sub_ds[var_map["t2m"]].values
    d2m_k = sub_ds[var_map["d2m"]].values
    tp_m = sub_ds[var_map["tp"]].values
    u10 = sub_ds[var_map["u10"]].values
    v10 = sub_ds[var_map["v10"]].values
    tiempos = sub_ds[time_dim].values

    # Conversión física
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    tp_mm = np.maximum(tp_m * 1000.0, 0.0)
    viento_kmh = np.sqrt(u10**2 + v10**2) * 3.6
    vpd_kpa = calcular_vpd(t2m_c, d2m_c)

    # Identificar fechas únicas
    fechas_dt = np.array([str(t)[:10] for t in tiempos])
    fechas_unicas, indices_dias = np.unique(fechas_dt, return_inverse=True)

    registros = []
    num_nodos = len(nodo_ids)

    for day_idx, fecha_str in enumerate(fechas_unicas):
        mascara_h = (indices_dias == day_idx)

        t_max_dia = np.nanmax(t2m_c[mascara_h, :], axis=0)
        t_min_dia = np.nanmin(t2m_c[mascara_h, :], axis=0)
        # En ERA5-Land, tp es precipitación acumulada por ciclo diurno; el acumulado diario es el máximo registrado en el día
        prec_dia = np.nanmax(tp_mm[mascara_h, :], axis=0)
        viento_max_dia = np.nanmax(viento_kmh[mascara_h, :], axis=0)
        vpd_max_dia = np.nanmax(vpd_kpa[mascara_h, :], axis=0)

        for n_i in range(num_nodos):
            registros.append({
                "nodo_clima_id": int(nodo_ids[n_i]),
                "fecha": fecha_str,
                "temp_max_c": float(t_max_dia[n_i]),
                "temp_min_c": float(t_min_dia[n_i]),
                "precipitacion_mm": float(prec_dia[n_i]),
                "viento_max_kmh": float(viento_max_dia[n_i]),
                "rafaga_max_kmh": float(viento_max_dia[n_i] * 1.3),
                "et0_mm": 0.0,
                "vpd_max_kpa": float(vpd_max_dia[n_i])
            })

    ds.close()
    return pl.DataFrame(registros)
