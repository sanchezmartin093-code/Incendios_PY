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
import time
import requests
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from pyproj import Transformer

from src.constantes import (
    CRS_GEOGRAFICO,
    CRS_PROYECTADO,
    BBOX_PARAGUAY
)
from src.fwi import (
    calcular_vpd,
    calcular_ffmc,
    calcular_isi,
    calcular_interaccion_viento_vpd
)

URL_OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

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
    
    distancias, indices = tree.query(puntos_grilla, k=1)
    nodos_asignados = df_nodos["nodo_clima_id"].to_numpy()[indices]

    return df_grilla.with_columns(
        pl.Series("nodo_clima_id", nodos_asignados)
    )

def descargar_clima_openmeteo_batch(
    lats: list[float],
    lons: list[float],
    nodo_ids: list[int],
    start_date: str,
    end_date: str,
    batch_size: int = 50,
    delay_seg: float = 0.5
) -> pl.DataFrame:
    """
    Descarga variables climáticas diarias desde Open-Meteo para una lista de coordenadas por lotes.
    """
    registros = []
    total_nodos = len(nodo_ids)

    for i in range(0, total_nodos, batch_size):
        batch_lats = lats[i:i + batch_size]
        batch_lons = lons[i:i + batch_size]
        batch_nodos = nodo_ids[i:i + batch_size]

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
                "et0_fao_evapotranspiration"
            ],
            "hourly": [
                "temperature_2m",
                "dew_point_2m"
            ],
            "timezone": "America/Asuncion"
        }

        try:
            resp = requests.get(URL_OPENMETEO_ARCHIVE, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"Aviso en lote {i}-{i+len(batch_nodos)}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            if not isinstance(data, list):
                data = [data]

            for nodo_id, item in zip(batch_nodos, data):
                daily = item.get("daily", {})
                hourly = item.get("hourly", {})
                
                fechas = daily.get("time", [])
                t_max = daily.get("temperature_2m_max", [])
                t_min = daily.get("temperature_2m_min", [])
                prec = daily.get("precipitation_sum", [])
                wind_max = daily.get("wind_speed_10m_max", [])
                gust_max = daily.get("wind_gusts_10m_max", [])
                et0 = daily.get("et0_fao_evapotranspiration", [])

                # Cálculo de VPD diario a partir de las 24 horas
                h_time = hourly.get("time", [])
                h_temp = np.array(hourly.get("temperature_2m", []), dtype=float)
                h_dew = np.array(hourly.get("dew_point_2m", []), dtype=float)
                
                # Calcular VPD para cada hora y extraer el máximo diario
                h_vpd = calcular_vpd(h_temp, h_dew)

                for day_idx, f_str in enumerate(fechas):
                    # Horas correspondientes al día (24 horas)
                    start_h = day_idx * 24
                    end_h = start_h + 24
                    vpd_max_dia = float(np.nanmax(h_vpd[start_h:end_h])) if len(h_vpd) >= end_h else 0.0

                    registros.append({
                        "nodo_clima_id": int(nodo_id),
                        "fecha": f_str,
                        "temp_max_c": float(t_max[day_idx]) if t_max[day_idx] is not None else np.nan,
                        "temp_min_c": float(t_min[day_idx]) if t_min[day_idx] is not None else np.nan,
                        "precipitacion_mm": float(prec[day_idx]) if prec[day_idx] is not None else 0.0,
                        "viento_max_kmh": float(wind_max[day_idx]) if wind_max[day_idx] is not None else 0.0,
                        "rafaga_max_kmh": float(gust_max[day_idx]) if gust_max[day_idx] is not None else 0.0,
                        "et0_mm": float(et0[day_idx]) if et0[day_idx] is not None else 0.0,
                        "vpd_max_kpa": vpd_max_dia
                    })

            if delay_seg > 0:
                time.sleep(delay_seg)

        except Exception as e:
            print(f"Error descargando lote {i}: {e}")

    return pl.DataFrame(registros)

def calcular_indices_fwi_y_sequedad(df_clima: pl.DataFrame) -> pl.DataFrame:
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
