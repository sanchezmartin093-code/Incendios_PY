"""
clima.py module: Representative meteorological mesh, ERA5-Land ingestion, and biophysical calculation of FWI and VPD.
SOTA philosophy:
- Meteorological mesh at 0.25° (~25 km, native reanalysis resolution).
- Instantaneous cKDTree assignment to the 400 006 cells of 1 km² without overhead.
- Batch requests optimized for the open Open-Meteo API.
- Calculation of biophysical dryness variables: VPD (Tetens), Consecutive Dry Days (CDD),
  Fine Fuel Moisture Code (FFMC), and Initial Spread Index (ISI).
"""

from __future__ import annotations


from pathlib import Path
from datetime import datetime
import time
import requests
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from pyproj import Transformer
import concurrent.futures

from src.constantes import (
    CRS_GEOGRAFICO,
    CRS_PROYECTADO,
    BBOX_PARAGUAY,
    DIR_RAW
)
from src.fwi import (
    calcular_vpd,
    calcular_ffmc,
    calcular_isi,
    calcular_interaccion_viento_vpd
)

URL_OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Increment this whenever the downloaded variable set or its parser changes.
# Otherwise an old parquet cache with a different variable set could be reused.
CACHE_VERSION = "v1_openmeteo_core"


def _ruta_cache(cache_dir: Path, batch_idx: int, start_date: str, end_date: str) -> Path:
    versionado = cache_dir / f"lote_{batch_idx:04d}_{start_date}_{end_date}_{CACHE_VERSION}.parquet"
    legado = cache_dir / f"lote_{batch_idx:04d}_{start_date}_{end_date}.parquet"
    if versionado.exists() or not legado.exists():
        return versionado
    # Reuse the pre-v3 Open-Meteo cache: its variable set (temperature, rain,
    # wind, gusts and ET0) is exactly the one requested again after moving
    # soil moisture to CDS.
    return legado


class OpenMeteoRateLimitError(RuntimeError):
    pass


def generar_malla_nodos_clima(resolucion_deg: float = 0.25) -> pl.DataFrame:
    """
    Generates the discrete mesh of representative meteorological nodes over Paraguay.
    At 0.25°, approximately 568 nodes are obtained that cover the territory without redundancy.
    """
    min_lon = BBOX_PARAGUAY["min_lon"]
    max_lon = BBOX_PARAGUAY["max_lon"]
    min_lat = BBOX_PARAGUAY["min_lat"]
    max_lat = BBOX_PARAGUAY["max_lat"]

    lons = np.arange(np.floor(min_lon / resolucion_deg) * resolucion_deg, max_lon + resolucion_deg, resolucion_deg)
    lats = np.arange(np.floor(min_lat / resolucion_deg) * resolucion_deg, max_lat + resolucion_deg, resolucion_deg)
    
    xx, yy = np.meshgrid(lons, lats)
    flat_lons = xx.ravel().astype(np.float32)
    flat_lats = yy.ravel().astype(np.float32)

    # Deterministic meteorological node identifier
    nodo_ids = [int(i + 1) for i in range(len(flat_lons))]

    transformer = Transformer.from_crs(CRS_GEOGRAFICO, CRS_PROYECTADO, always_xy=True)
    x_utm, y_utm = transformer.transform(flat_lons, flat_lats)

    return pl.DataFrame({
        "nodo_clima_id": nodo_ids,
        "lon": flat_lons,
        "lat": flat_lats,
        "x_utm": x_utm.astype(np.float32),
        "y_utm": y_utm.astype(np.float32)
    })

def asignar_nodos_a_grilla(df_grilla: pl.DataFrame, df_nodos: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns the nearest meteorological node to each 1 km² cell using cKDTree in UTM metric space.
    Runs in ~0.05 seconds for 400 000 cells.
    """
    tree = cKDTree(df_nodos.select(["x_utm", "y_utm"]).to_numpy())
    puntos_grilla = df_grilla.select(["x_utm", "y_utm"]).to_numpy()
    
    distancias, indices = tree.query(puntos_grilla, k=1, workers=-1)
    nodos_asignados = df_nodos["nodo_clima_id"].to_numpy()[indices]

    return df_grilla.with_columns(
        pl.Series("nodo_clima_id", nodos_asignados)
    )

def _descargar_un_lote_con_reintentos(
    batch_idx: int,
    batch_nodos: list[int],
    batch_lats: list[float],
    batch_lons: list[float],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    max_reintentos: int = 4,
    timeout_seg: int = 90
) -> pl.DataFrame:
    """
    Downloads and processes a sub-batch of meteorological nodes with individual disk checkpointing.
    """
    archivo_cache = _ruta_cache(cache_dir, batch_idx, start_date, end_date)
    if archivo_cache.exists():
        try:
            return pl.read_parquet(archivo_cache)
        except Exception:
            archivo_cache.unlink(missing_ok=True)

    params = {
        "latitude": batch_lats,
        "longitude": batch_lons,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "et0_fao_evapotranspiration",
            "vapour_pressure_deficit_max",
        ],
        # FIRMS reports acq_date in UTC. Using UTC avoids shifting nighttime
        # events by one day when joining weather and fire detections.
        "timezone": "UTC"
    }

    resp_data = None
    ultimo_error = None
    for intento in range(1, max_reintentos + 1):
        try:
            resp = requests.get(URL_OPENMETEO_ARCHIVE, params=params, timeout=timeout_seg)
            if resp.status_code == 200:
                resp_data = resp.json()
                if not isinstance(resp_data, list):
                    resp_data = [resp_data]
                break
            if resp.status_code == 429:
                motivo = ""
                try:
                    motivo = resp.json().get("reason", "")
                except Exception:
                    pass
                detalle = f": {motivo}" if motivo else ""
                if "minutely" in motivo.lower() and intento < max_reintentos:
                    print(
                        f"Lote {batch_idx}: límite minucioso de Open-Meteo; "
                        f"reintento {intento}/{max_reintentos} en 65 s",
                        flush=True,
                    )
                    time.sleep(65)
                    continue
                raise OpenMeteoRateLimitError(
                    f"Open-Meteo rechazó el lote {batch_idx} por límite de cuota "
                    f"(HTTP 429){detalle}. No se reintenta: esperá el reset de la cuota "
                    "o configurá una API key."
                )
            if resp.status_code in [502, 503, 504]:
                espera = intento * 5.0
                print(
                    f"Lote {batch_idx}: HTTP {resp.status_code}, reintento "
                    f"{intento}/{max_reintentos} en {espera:.0f}s",
                    flush=True,
                )
                time.sleep(espera)
            else:
                ultimo_error = f"HTTP {resp.status_code}"
                time.sleep(2.0)
        except OpenMeteoRateLimitError:
            raise
        except Exception as e:
            ultimo_error = str(e)
            if intento == max_reintentos:
                print(f"Error persistente en lote {batch_idx}: {e}", flush=True)
            time.sleep(intento * 3.0)

    if resp_data is None:
        detalle = f" (último error: {ultimo_error})" if ultimo_error else ""
        raise RuntimeError(
            f"No se pudo descargar el lote {batch_idx} tras "
            f"{max_reintentos} intentos{detalle}"
        )

    registros = []
    for nodo_id, item in zip(batch_nodos, resp_data):
        daily = item.get("daily", {})

        fechas = daily.get("time", [])
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        prec = daily.get("precipitation_sum", [])
        wind_max = daily.get("wind_speed_10m_max", [])
        gust_max = daily.get("wind_gusts_10m_max", [])
        et0 = daily.get("et0_fao_evapotranspiration", [])
        vpd_max = daily.get("vapour_pressure_deficit_max", [])

        for day_idx, f_str in enumerate(fechas):
            vpd_val = float(vpd_max[day_idx]) if day_idx < len(vpd_max) and vpd_max[day_idx] is not None else 0.0

            registros.append({
                "nodo_clima_id": int(nodo_id),
                "fecha": f_str,
                "temp_max_c": float(t_max[day_idx]) if t_max[day_idx] is not None else np.nan,
                "temp_min_c": float(t_min[day_idx]) if t_min[day_idx] is not None else np.nan,
                "precipitacion_mm": float(prec[day_idx]) if prec[day_idx] is not None else 0.0,
                "viento_max_kmh": float(wind_max[day_idx]) if wind_max[day_idx] is not None else 0.0,
                "rafaga_max_kmh": float(gust_max[day_idx]) if gust_max[day_idx] is not None else 0.0,
                "et0_mm": float(et0[day_idx]) if et0[day_idx] is not None else 0.0,
                "vpd_max_kpa": vpd_val,
            })

    if registros:
        df_batch = pl.DataFrame(registros)
        df_batch.write_parquet(archivo_cache, compression="zstd")
        return df_batch

    raise RuntimeError(f"Open-Meteo devolvió un lote vacío para el lote {batch_idx}")

def descargar_clima_openmeteo_batch(
    lats: list[float],
    lons: list[float],
    nodo_ids: list[int],
    start_date: str,
    end_date: str,
    batch_size: int = 10,
    max_workers: int = 3,
    delay_seg: float = 0.2,
    cache_dir: Path | None = None
) -> pl.DataFrame:
    """
    Downloads daily climate variables from Open-Meteo for a list of coordinates
    via concurrent batches with fault tolerance and disk checkpointing.
    """
    if cache_dir is None:
        cache_dir = DIR_RAW / "era5" / "batches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_nodos = len(nodo_ids)
    batches = []
    batch_idx = 0
    for i in range(0, total_nodos, batch_size):
        batches.append((
            batch_idx,
            nodo_ids[i:i + batch_size],
            lats[i:i + batch_size],
            lons[i:i + batch_size]
        ))
        batch_idx += 1

    total_batches = len(batches)
    print(
        f"Dividido en {total_batches} lotes ({batch_size} nodos por lote) "
        f"con {max_workers} trabajadores paralelos...",
        flush=True,
    )

    dfs = [None] * total_batches

    # 1. Check which ones already exist in cache
    pendientes = []
    for b_idx, b_nodos, b_lats, b_lons in batches:
        f_cache = _ruta_cache(cache_dir, b_idx, start_date, end_date)
        if f_cache.exists():
            try:
                dfs[b_idx] = pl.read_parquet(f_cache)
            except Exception:
                pendientes.append((b_idx, b_nodos, b_lats, b_lons))
        else:
            pendientes.append((b_idx, b_nodos, b_lats, b_lons))

    completados = total_batches - len(pendientes)
    if completados > 0:
        print(
            f" ✓ {completados}/{total_batches} lotes encontrados en caché local.",
            flush=True,
        )

    if pendientes:
        print(
            f" Descargando {len(pendientes)} lotes pendientes desde Open-Meteo...",
            flush=True,
        )
        fallidos = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futuros = {}
            for i, (b_idx, b_nodos, b_lats, b_lons) in enumerate(pendientes):
                if i > 0 and delay_seg and delay_seg > 0:
                    time.sleep(float(delay_seg))
                futuros[executor.submit(
                    _descargar_un_lote_con_reintentos,
                    b_idx, b_nodos, b_lats, b_lons,
                    start_date, end_date, cache_dir
                )] = b_idx

            for futuro in concurrent.futures.as_completed(futuros):
                b_idx = futuros[futuro]
                try:
                    df_res = futuro.result()
                    if df_res.is_empty():
                        raise RuntimeError("Open-Meteo devolvió un lote vacío")
                    dfs[b_idx] = df_res
                    completados += 1
                    if completados % 5 == 0 or completados == total_batches:
                        print(
                            f"  • Progreso: {completados}/{total_batches} lotes "
                            f"listos ({completados * 100 / total_batches:.1f}%)",
                            flush=True,
                        )
                except concurrent.futures.CancelledError:
                    continue
                except OpenMeteoRateLimitError as e:
                    fallidos[b_idx] = str(e)
                    for pendiente in futuros:
                        if not pendiente.done():
                            pendiente.cancel()
                except Exception as e:
                    fallidos[b_idx] = str(e)
                    print(f"Error procesando lote {b_idx}: {e}", flush=True)

        if fallidos:
            detalle = "\n".join(
                f"  - lote {i}: {msg}" for i, msg in sorted(fallidos.items())
            )
            raise RuntimeError(
                f"{len(fallidos)} lote(s) de Open-Meteo fallaron; no se actualiza "
                f"el clima para evitar datos parciales.\n{detalle}"
            )

    dfs_validos = [d for d in dfs if d is not None and not d.is_empty()]
    if not dfs_validos:
        return pl.DataFrame()

    return pl.concat(dfs_validos)

def calcular_indices_fwi_y_sequedad(
    df_clima: pl.DataFrame,
    validar_continuidad: bool = True,
) -> pl.DataFrame:
    """
    Calculates sequentially and sorted by meteorological node:
    - Accumulated 48-hour precipitation.
    - Consecutive Dry Days (CDD, rainfall < 1.0 mm).
    - Fine Fuel Moisture Code (FFMC).
    - Initial Spread Index (ISI).
    - Convective Severity (Wind * VPD).
    - Topsoil moisture (0-7 cm), antecedent frost severity and 60-day water deficit.
      The soil-moisture fallback is a transparent bucket and is not ERA5 data.
    """
    if df_clima.is_empty():
        return df_clima

    # Strictly sort by node and date
    df_sorted = df_clima.sort(["nodo_clima_id", "fecha"])
    
    # Process by node to maintain the recursive FFMC state
    nodos = df_sorted["nodo_clima_id"].unique().to_list()
    resultados_nodos = []

    for nodo in nodos:
        sub = df_sorted.filter(pl.col("nodo_clima_id") == nodo)

        if validar_continuidad:
            fechas = [
                datetime.strptime(str(f)[:10], "%Y-%m-%d")
                for f in sub["fecha"].to_list()
            ]
            for anterior, actual in zip(fechas, fechas[1:]):
                if (actual - anterior).days != 1:
                    raise ValueError(
                        f"Serie climática no continua para el nodo {nodo}: "
                        f"{anterior.date()} -> {actual.date()}"
                    )

        n = len(sub)
        precs = np.nan_to_num(sub["precipitacion_mm"].to_numpy().astype(float), nan=0.0)
        temps = np.nan_to_num(sub["temp_max_c"].to_numpy().astype(float), nan=25.0)
        vientos = np.nan_to_num(sub["viento_max_kmh"].to_numpy().astype(float), nan=0.0)
        vpds = np.nan_to_num(sub["vpd_max_kpa"].to_numpy().astype(float), nan=0.0)
        if "temp_min_c" in sub.columns:
            tmins = np.nan_to_num(sub["temp_min_c"].to_numpy().astype(float), nan=15.0)
        else:
            tmins = np.full(n, 15.0, dtype=float)
        if "et0_mm" in sub.columns:
            et0s = np.nan_to_num(sub["et0_mm"].to_numpy().astype(float), nan=3.5)
        else:
            et0s = np.full(n, 3.5, dtype=float)

        if "swvl1_m3m3" in sub.columns:
            swvl1_input = sub["swvl1_m3m3"].to_numpy().astype(float)
            tiene_swvl1 = bool(np.isfinite(swvl1_input).any())
        else:
            swvl1_input = np.full(n, np.nan, dtype=float)
            tiene_swvl1 = False

        prec_48h = np.zeros(n, dtype=np.float32)
        cdd = np.zeros(n, dtype=np.int32)
        ffmc = np.zeros(n, dtype=np.float32)
        isi = np.zeros(n, dtype=np.float32)
        interaccion = np.zeros(n, dtype=np.float32)
        swvl1 = np.full(n, np.nan, dtype=np.float32)
        fdd_60d = np.zeros(n, dtype=np.float32)
        deficit_60d = np.zeros(n, dtype=np.float32)

        if tiene_swvl1:
            swvl1[:] = swvl1_input.astype(np.float32)

        # Frost severity (agronomic threshold: Tmin < 2 °C) and daily water
        # balance. Antecedent 60-day windows exclude the current day; the
        # current-day weather is already represented by VPD/rain/wind.
        helada_diaria = np.maximum(0.0, 2.0 - tmins)
        balance_diario = et0s - precs
        cum_fdd = np.concatenate([[0.0], np.cumsum(helada_diaria)])
        cum_balance = np.concatenate([[0.0], np.cumsum(balance_diario)])

        # Initialise the simple bucket from the first available measurement.
        primeros = swvl1[: min(7, n)]
        validos_iniciales = primeros[np.isfinite(primeros)]
        sw_actual = float(validos_iniciales.mean()) if validos_iniciales.size else 0.28

        dias_secos = 0
        ffmc_prev = 85.0  # Standard international reference initial value

        for k in range(n):
            p = float(precs[k])
            t = float(temps[k])
            w = float(vientos[k])
            vpd = float(vpds[k])
            et = float(et0s[k])

            # 48h rainfall
            prec_48h[k] = p + (float(precs[k - 1]) if k > 0 else 0.0)

            # CDD
            if p < 1.0:
                dias_secos += 1
            else:
                dias_secos = 0
            cdd[k] = dias_secos

            # Soil moisture: use ERA5/Open-Meteo when available; otherwise
            # integrate a transparent 0-7 cm bucket (provenance is documented
            # in the data metadata, not silently presented as ERA5).
            if np.isfinite(swvl1[k]):
                sw_actual = float(swvl1[k])
            else:
                infiltracion = min(max(p, 0.0), 25.0) / 70.0
                evaporacion = (et * max(sw_actual, 0.0) / 0.42) / 70.0
                sw_actual = float(np.clip(sw_actual + infiltracion - evaporacion, 0.04, 0.45))
                swvl1[k] = sw_actual

            # Antecedent 60-day windows (days k-60 .. k-1).
            ini_60 = max(0, k - 60)
            fdd_60d[k] = float(cum_fdd[k] - cum_fdd[ini_60])
            deficit_60d[k] = float(cum_balance[k] - cum_balance[ini_60])

            # Relative humidity estimation for FFMC from VPD
            # rh ~ 100 * ea / es = 100 * (es - vpd) / es
            es = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
            rh_est = float(np.clip(100.0 * (es - vpd) / es, 5.0, 100.0))

            # FWI FFMC & ISI
            ffmc_val = calcular_ffmc(temp=t, rh=rh_est, wind=w, prec=p, ffmc_prev=ffmc_prev)
            ffmc[k] = ffmc_val
            ffmc_prev = ffmc_val

            isi[k] = calcular_isi(wind_kmh=w, ffmc=ffmc_val)
            interaccion[k] = calcular_interaccion_viento_vpd(wind_kmh=w, vpd_kpa=vpd)

        sub_calc = sub.with_columns([
            pl.Series("precipitacion_48h_mm", prec_48h),
            pl.Series("dias_secos_consecutivos", cdd),
            pl.Series("ffmc", ffmc),
            pl.Series("isi", isi),
            pl.Series("interaccion_viento_vpd", interaccion),
            pl.Series("swvl1_m3m3", swvl1),
            pl.Series("fdd_heladas_60d", fdd_60d),
            pl.Series("deficit_hidro_60d", deficit_60d),
        ])
        resultados_nodos.append(sub_calc)

    return pl.concat(resultados_nodos)
