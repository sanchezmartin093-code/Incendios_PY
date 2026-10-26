"""
Módulo clima.py: Malla meteorológica representativa, ingesta ERA5-Land y cálculo biofísico de FWI y VPD.
Filosofía SOTA:
- Malla meteorológica a 0.25° (~25 km, resolución nativa del reanálisis).
- Asignación instantánea cKDTree a las 400.006 celdas de 1 km² sin overhead.
- Ingesta por lotes (Batch requests) optimizada a la API abierta de Open-Meteo.
- Cálculo de variables biofísicas de sequedad: VPD (Tetens), Días Secos Consecutivos (CDD),
  Índice de Combustible Fino (FFMC) e Índice de Propagación Inicial (ISI).
"""

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


class OpenMeteoRateLimitError(RuntimeError):
    pass


def generar_malla_nodos_clima(resolucion_deg: float = 0.25) -> pl.DataFrame:
    """
    Genera la malla discreta de nodos meteorológicos representativos sobre Paraguay.
    A 0.25°, se obtienen aproximadamente 568 nodos que cubren el territorio sin redundancia.
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

    # Identificador de nodo meteorológico determinista
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
    Asigna a cada celda de 1 km² el nodo meteorológico más cercano mediante cKDTree en espacio métrico UTM.
    Ejecución en ~0.05 segundos para 400.000 celdas.
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
    Descarga y procesa un sub-lote de nodos meteorológicos con checkpointing individual en disco.
    """
    archivo_cache = cache_dir / f"lote_{batch_idx:04d}_{start_date}_{end_date}.parquet"
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
            "vapour_pressure_deficit_max"
        ],
        # FIRMS reporta acq_date en UTC. Usar UTC evita desplazar un día
        # los eventos nocturnos al unir clima y focos.
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
                "vpd_max_kpa": vpd_val
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
    Descarga variables climáticas diarias desde Open-Meteo para una lista de coordenadas
    mediante lotes concurrentes con tolerancia a fallos y checkpointing en disco.
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

    # 1. Comprobar cuáles ya existen en cache
    pendientes = []
    for b_idx, b_nodos, b_lats, b_lons in batches:
        f_cache = cache_dir / f"lote_{b_idx:04d}_{start_date}_{end_date}.parquet"
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
    Calcula de forma secuencial y ordenada por nodo meteorológico:
    - Precipitacion acumulada 48 horas.
    - Días Secos Consecutivos (CDD, lluvia < 1.0 mm).
    - Fine Fuel Moisture Code (FFMC).
    - Initial Spread Index (ISI).
    - Severidad Convectiva (Viento * VPD).
    """
    if df_clima.is_empty():
        return df_clima

    # Ordenar estrictamente por nodo y fecha
    df_sorted = df_clima.sort(["nodo_clima_id", "fecha"])
    
    # Procesar por nodo para mantener el estado recursivo del FFMC
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

        precs = sub["precipitacion_mm"].to_numpy()
        temps = sub["temp_max_c"].to_numpy()
        vientos = sub["viento_max_kmh"].to_numpy()
        vpds = sub["vpd_max_kpa"].to_numpy()

        n = len(precs)
        prec_48h = np.zeros(n, dtype=np.float32)
        cdd = np.zeros(n, dtype=np.int32)
        ffmc = np.zeros(n, dtype=np.float32)
        isi = np.zeros(n, dtype=np.float32)
        interaccion = np.zeros(n, dtype=np.float32)

        dias_secos = 0
        ffmc_prev = 85.0  # Valor inicial estándar de referencia internacional

        for k in range(n):
            p = float(precs[k])
            t = float(temps[k]) if not np.isnan(temps[k]) else 25.0
            w = float(vientos[k])
            vpd = float(vpds[k])

            # 48h lluvia
            if k == 0:
                prec_48h[k] = p
            else:
                prec_48h[k] = p + float(precs[k - 1])

            # CDD
            if p < 1.0:
                dias_secos += 1
            else:
                dias_secos = 0
            cdd[k] = dias_secos

            # Estimación de HR relativa para FFMC a partir de VPD
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
            pl.Series("interaccion_viento_vpd", interaccion)
        ])
        resultados_nodos.append(sub_calc)

    return pl.concat(resultados_nodos)
