"""
Módulo fwi.py: Formulación del Canadian Fire Weather Index (FWI) y variables microclimáticas avanzadas.
Incluye el cálculo de:
- Déficit de Presión de Vapor (VPD)
- Índices de combustible FWI (FFMC, DMC, DC, ISI, BUI, FWI)
- Días secos consecutivos (CDD)
- Interacción convectiva viento-sequedad
"""

import numpy as np

def calcular_presion_vapor_saturacion(temperatura_c: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula la presión de vapor de saturación (es) en kPa mediante la ecuación de Tetens.
    """
    return 0.61078 * np.exp((17.27 * temperatura_c) / (temperatura_c + 237.3))

def calcular_vpd(temperatura_c: float | np.ndarray, temperatura_rocio_c: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula el Déficit de Presión de Vapor (VPD) en kilopascales (kPa).
    VPD = es(T) - ea(Td)
    Valores > 2.5 kPa indican desecación atmosférica extrema y alta inflamabilidad.
    """
    es = calcular_presion_vapor_saturacion(temperatura_c)
    ea = calcular_presion_vapor_saturacion(temperatura_rocio_c)
    vpd = np.maximum(0.0, es - ea)
    return vpd

def calcular_ffmc(temp: float, rh: float, wind: float, prec: float, ffmc_prev: float = 85.0) -> float:
    """
    Calcula el Fine Fuel Moisture Code (FFMC) del sistema canadiense.
    Mide el contenido hídrico de la hojarasca y combustibles finos muertos (capa de 1-2 cm).
    Escala: 0 a 101. Valores > 89 indican inflamabilidad extrema.
    """
    # 1. Ajuste por precipitación
    if prec > 0.5:
        rf = prec - 0.5
        mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

        if rf <= 0.0:
            rf = 0.0

        if mo > 150.0:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + (0.0015 * (mo - 150.0) ** 2) * np.sqrt(rf)
            
        if mr > 250.0:
            mr = 250.0
        mo = mr
    else:
        mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # 2. Equilibrio higroscópico y secado por temperatura y viento
    ed = 0.942 * (rh ** 0.679) + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.614 * (rh ** 0.753) + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))

    if mo < ed:
        # Humidificación (wetting)
        ko = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * temp)
        m = ed + (mo - ed) * (10.0 ** (-kd))
    elif mo > ew:
        # Desecación (drying)
        kw = (0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)) * 0.581 * np.exp(0.0365 * temp)
        m = ew + (mo - ew) * (10.0 ** (-kw))
    else:
        m = mo

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(ffmc, 0.0, 101.0))

def calcular_isi(wind_kmh: float, ffmc: float) -> float:
    """
    Initial Spread Index (ISI): Potencial de propagación inicial del fuego.
    Combina la fuerza del viento con el contenido hídrico de la hojarasca fina (FFMC).
    """
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fu = np.exp(0.05039 * wind_kmh)
    fw = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / (4.93e07))
    isi = 0.208 * fu * fw
    return float(np.maximum(0.0, isi))

def calcular_interaccion_viento_vpd(wind_kmh: float, vpd_kpa: float) -> float:
    """
    Índice de severidad convectiva: Interacción no lineal entre la fuerza del viento
    y la demanda evaporativa de la atmósfera (VPD).
    """
    return float(wind_kmh * vpd_kpa)
