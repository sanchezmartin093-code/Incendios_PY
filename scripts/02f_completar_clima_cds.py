"""
Script 02f: short CDS path for soil moisture + potential evaporation.

Why this script exists
----------------------
The full CDS rebuild (02b) re-downloads five meteorological variables and the
Open-Meteo correction (02c) is subject to hourly quotas. This script only adds
the two fields needed by the v3 contract:

- ``volumetric_soil_water_layer_1`` -> ``swvl1_m3m3`` (0-7 cm daily mean)
- ``potential_evaporation``        -> ``et0_mm`` (daily total, CDS proxy)

It merges them into the existing ``clima_era5_diario.parquet`` and recomputes
the antecedent frost/water-deficit features. It does NOT change temperature,
VPD, wind or precipitation.

Important: CDS ``potential_evaporation`` is a potential-evaporation field, not
the FAO-56 reference evapotranspiration from Open-Meteo. It is an explicit
proxy; the metadata records the source. All models using ``deficit_hidro_60d``
must be retrained and evaluated again.
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import numpy as np
import polars as pl
import xarray as xr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import BBOX_PARAGUAY, DIR_PROCESSED, DIR_RAW
from src.clima import (
    asignar_nodos_a_grilla,
    calcular_indices_fwi_y_sequedad,
    generar_malla_nodos_clima,
)

VARIABLES = ["volumetric_soil_water_layer_1", "potential_evaporation"]
AREA_PARAGUAY_CDS = [
    BBOX_PARAGUAY["max_lat"],
    BBOX_PARAGUAY["min_lon"],
    BBOX_PARAGUAY["min_lat"],
    BBOX_PARAGUAY["max_lon"],
]
HORAS = [f"{h:02d}:00" for h in range(0, 24, 3)]
MESES = [f"{m:02d}" for m in range(1, 13)]
DIAS = [f"{d:02d}" for d in range(1, 32)]


def _ruta_nc(dir_salida: Path, anio: int, trimestre: int | None) -> Path:
    if trimestre is None:
        return dir_salida / f"era5_land_swvl1_pev_{anio}.nc"
    return dir_salida / f"era5_land_swvl1_pev_{anio}_Q{trimestre}.nc"


def _extraer_nc_si_zip(destino: Path, archivo_descarga: Path) -> None:
    """CDS sometimes returns a zip renamed as .nc; extract its inner NetCDF."""
    if zipfile.is_zipfile(archivo_descarga):
        with zipfile.ZipFile(archivo_descarga) as zf:
            candidatos = [n for n in zf.namelist() if n.endswith(".nc")]
            if not candidatos:
                raise RuntimeError(f"El ZIP de CDS no contiene NetCDF: {archivo_descarga}")
            temporal = destino.with_name(destino.name + ".extract")
            with zf.open(candidatos[0]) as src, open(temporal, "wb") as dst:
                shutil.copyfileobj(src, dst)
        archivo_descarga.unlink(missing_ok=True)
        os.replace(temporal, destino)
    else:
        os.replace(archivo_descarga, destino)


def _tiene_variables(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            nombres = {str(v).lower() for v in ds.data_vars}
        return (
            any("swvl1" in v or "volumetric_soil_water" in v for v in nombres)
            and any("pev" in v or "potential_evaporation" in v for v in nombres)
        )
    except Exception:
        return False


def descargar_bloque(
    client: cdsapi.Client,
    anio: int,
    trimestre: int | None,
    dir_salida: Path,
    forzar: bool = False,
) -> Path:
    dir_salida.mkdir(parents=True, exist_ok=True)
    destino = _ruta_nc(dir_salida, anio, trimestre)
    if destino.exists() and not forzar and _tiene_variables(destino):
        print(f" ✓ CDS en caché: {destino.name}")
        return destino

    destino.unlink(missing_ok=True)
    destino.with_name(destino.name + ".download").unlink(missing_ok=True)
    if trimestre is None:
        meses = MESES
    else:
        meses = [f"{m:02d}" for m in range(3 * trimestre - 2, 3 * trimestre + 1)]

    peticion = {
        "variable": VARIABLES,
        "year": str(anio),
        "month": meses,
        "day": DIAS,
        "time": HORAS,
        "data_format": "netcdf",
        "area": AREA_PARAGUAY_CDS,
    }
    etiqueta = f"{anio}" if trimestre is None else f"{anio}-Q{trimestre}"
    print(f"\n[CDS] Solicitando swvl1 + potential_evaporation {etiqueta}...")
    temporal = destino.with_name(destino.name + ".download")
    client.retrieve("reanalysis-era5-land", peticion, str(temporal))
    _extraer_nc_si_zip(destino, temporal)
    print(f"   ✓ Guardado: {destino.name} ({destino.stat().st_size / (1024 * 1024):.2f} MB)")
    return destino


def _agregar_evaporacion_diaria(pev_m: np.ndarray) -> np.ndarray:
    """
    Converts 3-hourly potential evaporation (m, negative) to mm/day.

    ERA5-Land ``pev`` is accumulated from 00 UTC: discard the first step
    (which belongs to the previous day), then use the last value if the tail is
    monotonic; otherwise sum the non-negative increments. The sign convention
    is converted to positive evaporation.
    """
    valores = np.abs(np.asarray(pev_m, dtype=float)) * 1000.0
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
            totales[j] = float(s[0])
            continue
        cola = serie[1:]
        cola = cola[np.isfinite(cola)]
        if cola.size == 0:
            totales[j] = float(s[0])
            continue
        diffs = np.diff(cola)
        if diffs.size > 0 and np.all(diffs >= -1e-3):
            totales[j] = max(float(cola[-1]), 0.0)
        else:
            totales[j] = float(np.nansum(np.clip(cola, 0.0, None)))
    return totales


def _detectar_variable(ds: xr.Dataset, claves: tuple[str, ...]) -> str | None:
    for v in ds.data_vars:
        low = str(v).lower()
        if any(k in low for k in claves):
            return str(v)
    return None


def procesar_nc(path: Path, df_nodos: pl.DataFrame) -> pl.DataFrame:
    ds = xr.open_dataset(path)
    try:
        var_swvl1 = _detectar_variable(ds, ("swvl1", "volumetric_soil_water"))
        var_pev = _detectar_variable(ds, ("pev", "potential_evaporation"))
        if var_swvl1 is None or var_pev is None:
            raise ValueError(f"{path.name} no contiene swvl1 y pev: {list(ds.data_vars)}")

        lat_dim = "latitude" if "latitude" in ds.coords else "lat"
        lon_dim = "longitude" if "longitude" in ds.coords else "lon"
        time_dim = "valid_time" if "valid_time" in ds.coords else "time"
        nodo_lats = xr.DataArray(df_nodos["lat"].to_numpy(), dims="nodo")
        nodo_lons = xr.DataArray(df_nodos["lon"].to_numpy(), dims="nodo")
        sub = ds.sel({lat_dim: nodo_lats, lon_dim: nodo_lons}, method="nearest")

        swvl1 = sub[var_swvl1].values
        pev = sub[var_pev].values
        tiempos = sub[time_dim].values
        nodo_ids = df_nodos["nodo_clima_id"].to_numpy()
    finally:
        ds.close()

    fechas = np.array([str(t)[:10] for t in tiempos])
    fechas_unicas, indices = np.unique(fechas, return_inverse=True)

    registros = []
    for day_idx, fecha in enumerate(fechas_unicas):
        mask = indices == day_idx
        swvl1_dia = np.nanmean(swvl1[mask, :], axis=0)
        et0_dia = _agregar_evaporacion_diaria(pev[mask, :])
        for i, nodo_id in enumerate(nodo_ids):
            registros.append({
                "nodo_clima_id": int(nodo_id),
                "fecha": str(fecha),
                "swvl1_m3m3": float(swvl1_dia[i]),
                "et0_mm": float(et0_dia[i]),
            })
    return pl.DataFrame(registros)


def main() -> None:
    parser = argparse.ArgumentParser(description="CDS short path: swvl1 + potential_evaporation")
    parser.add_argument("--anios", nargs="+", type=int, default=list(range(2018, 2026)))
    parser.add_argument("--trimestre", action="store_true", help="Descargar por trimestres (más trabajos, archivos más chicos)")
    parser.add_argument("--forzar", action="store_true", help="Volver a descargar todos los bloques")
    parser.add_argument("--workers", type=int, default=1, help="Descargas CDS concurrentes; CDS puede rechazar >2")
    args = parser.parse_args()

    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    if not p_grilla.exists() or not p_clima.exists():
        raise FileNotFoundError("Faltan grilla_nacional_1km.parquet o clima_era5_diario.parquet")

    grilla = pl.read_parquet(p_grilla)
    nodos = generar_malla_nodos_clima(resolucion_deg=0.25)
    grilla = asignar_nodos_a_grilla(grilla, nodos)
    activos = grilla["nodo_clima_id"].unique().to_list()
    nodos = nodos.filter(pl.col("nodo_clima_id").is_in(activos)).sort("nodo_clima_id")
    print(f"Nodos activos: {nodos.height}")

    dir_cds = DIR_RAW / "era5" / "cds_swvl1_pev"

    tareas: list[tuple[int, int | None]] = []
    for anio in args.anios:
        if args.trimestre:
            for q in (1, 2, 3, 4):
                tareas.append((anio, q))
        else:
            tareas.append((anio, None))

    def _tarea(anio: int, q: int | None) -> Path:
        return descargar_bloque(cdsapi.Client(), anio, q, dir_cds, forzar=args.forzar)

    rutas: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futuros = {ex.submit(_tarea, anio, q): (anio, q) for anio, q in tareas}
        for fut in as_completed(futuros):
            anio, q = futuros[fut]
            try:
                rutas.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - report and preserve completed cache
                raise RuntimeError(f"Falló CDS {anio}-Q{q}: {exc}") from exc

    print("\nProcesando NetCDF de swvl1 + potential_evaporation...")
    aux = pl.concat([procesar_nc(ruta, nodos) for ruta in rutas]).sort(["nodo_clima_id", "fecha"])

    if not aux.select(["nodo_clima_id", "fecha"]).is_unique().all():
        raise ValueError("La tabla auxiliar tiene claves nodo/fecha duplicadas")
    if aux["swvl1_m3m3"].null_count() or aux["et0_mm"].null_count():
        raise ValueError("La tabla auxiliar contiene nulos en swvl1 o ET0")

    base = pl.read_parquet(p_clima)
    if not base.select(["nodo_clima_id", "fecha"]).is_unique().all():
        raise ValueError("clima_era5_diario.parquet tiene claves nodo/fecha duplicadas")

    columnas_reemplazadas = [
        "swvl1_m3m3", "et0_mm", "fdd_heladas_60d", "deficit_hidro_60d",
        "precipitacion_48h_mm", "dias_secos_consecutivos",
        "ffmc", "isi", "interaccion_viento_vpd",
    ]
    base = base.drop([c for c in columnas_reemplazadas if c in base.columns])
    merged = base.join(aux, on=["nodo_clima_id", "fecha"], how="left")

    nulos = int(merged["swvl1_m3m3"].null_count() + merged["et0_mm"].null_count())
    if merged.height != base.height or nulos:
        raise ValueError(
            f"La unión clima+CDS cambió filas ({base.height} -> {merged.height}) "
            f"o dejó {nulos} nulos. Revisá cobertura de fechas/nodos."
        )

    procesado = calcular_indices_fwi_y_sequedad(merged)

    backup = DIR_PROCESSED / "clima_era5_diario_pre_02f.parquet"
    if not backup.exists():
        shutil.copy2(p_clima, backup)

    ruta_tmp = p_clima.with_name(p_clima.name + ".tmp")
    procesado.write_parquet(ruta_tmp, compression="zstd")
    os.replace(ruta_tmp, p_clima)

    meta = {
        "fuente": "Copernicus CDS ERA5-Land",
        "variables": VARIABLES,
        "swvl1_fuente": "volumetric_soil_water_layer_1 (0-7 cm, media diaria)",
        "et0_fuente": "potential_evaporation (suma diaria; proxy, no FAO-56)",
        "anios": args.anios,
        "filas": procesado.height,
        "swvl1_min": float(procesado["swvl1_m3m3"].min()),
        "swvl1_max": float(procesado["swvl1_m3m3"].max()),
        "et0_min_mm": float(procesado["et0_mm"].min()),
        "et0_max_mm": float(procesado["et0_mm"].max()),
        "deficit_hidro_60d_min": float(procesado["deficit_hidro_60d"].min()),
        "deficit_hidro_60d_max": float(procesado["deficit_hidro_60d"].max()),
    }
    (DIR_PROCESSED / "clima_era5_diario.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✓ Clima actualizado con swvl1 + ET0 CDS: {p_clima}")
    print(f"  swvl1: {meta['swvl1_min']:.3f} a {meta['swvl1_max']:.3f} m3/m3")
    print(f"  ET0:   {meta['et0_min_mm']:.2f} a {meta['et0_max_mm']:.2f} mm/día")


if __name__ == "__main__":
    main()
