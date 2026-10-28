"""
fwi.py module: Canadian Fire Weather Index (FWI) formulation and advanced microclimatic variables.
Includes the calculation of:
- Vapor Pressure Deficit (VPD)
- FWI fuel indices (FFMC, DMC, DC, ISI, BUI, FWI)
- Consecutive dry days (CDD)
- Convective wind-dryness interaction
"""

from __future__ import annotations


import numpy as np

def calcular_presion_vapor_saturacion(temperatura_c: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the saturation vapor pressure (es) in kPa using the Tetens equation.
    """
    return 0.61078 * np.exp((17.27 * temperatura_c) / (temperatura_c + 237.3))

def calcular_vpd(temperatura_c: float | np.ndarray, temperatura_rocio_c: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the Vapor Pressure Deficit (VPD) in kilopascals (kPa).
    VPD = es(T) - ea(Td)
    Values > 2.5 kPa indicate extreme atmospheric desiccation and high flammability.
    """
    es = calcular_presion_vapor_saturacion(temperatura_c)
    ea = calcular_presion_vapor_saturacion(temperatura_rocio_c)
    vpd = np.maximum(0.0, es - ea)
    return vpd

def calcular_ffmc(temp: float, rh: float, wind: float, prec: float, ffmc_prev: float = 85.0) -> float:
    """
    Calculates the Fine Fuel Moisture Code (FFMC) of the Canadian system.
    Measures the moisture content of leaf litter and dead fine fuels (1-2 cm layer).
    Scale: 0 to 101. Values > 89 indicate extreme flammability.
    """
    # 1. Precipitation adjustment
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

    # 2. Hygroscopic equilibrium and drying driven by temperature and wind
    ed = 0.942 * (rh ** 0.679) + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.614 * (rh ** 0.753) + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))

    if mo < ed:
        # Humidification (wetting)
        ko = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * temp)
        m = ed + (mo - ed) * (10.0 ** (-kd))
    elif mo > ew:
        # Desiccation (drying)
        kw = (0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)) * 0.581 * np.exp(0.0365 * temp)
        m = ew + (mo - ew) * (10.0 ** (-kw))
    else:
        m = mo

    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(ffmc, 0.0, 101.0))

def calcular_isi(wind_kmh: float, ffmc: float) -> float:
    """
    Initial Spread Index (ISI): Initial fire propagation potential.
    Combines wind strength with the moisture content of fine leaf litter (FFMC).
    """
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fu = np.exp(0.05039 * wind_kmh)
    fw = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / (4.93e07))
    isi = 0.208 * fu * fw
    return float(np.maximum(0.0, isi))

def calcular_interaccion_viento_vpd(wind_kmh: float, vpd_kpa: float) -> float:
    """
    Convective severity index: Nonlinear interaction between wind strength
    and the evaporative demand of the atmosphere (VPD).
    """
    return float(wind_kmh * vpd_kpa)


def calcular_dmc(
    temp_c: float,
    rh_pct: float,
    wind_kmh: float,
    precip_mm: float,
    dmc_prev: float = 6.0,
) -> float:
    """Duff Moisture Code (DMC) with drying and rain recharge."""
    dmc = float(dmc_prev)
    if precip_mm > 1.5:
        re = 0.92 * precip_mm - 1.5
        mo = 20.0 + np.exp(5.6348 - dmc / 43.43)
        if dmc <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc)
        elif dmc <= 65.0:
            b = 14.0 - 1.3 * np.log(dmc)
        else:
            b = 6.2 * np.log(dmc) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        dmc = 244.72 - 43.43 * np.log(max(mr - 20.0, 1e-6))
        dmc = max(dmc, 0.0)

    t = max(float(temp_c), -1.1)
    rh = float(np.clip(rh_pct, 1.0, 100.0))
    le = np.exp(2.0 - 0.2 * rh / 100.0)
    k = 1.894 * (t + 1.1) * (100.0 - rh) * le * 1e-6
    return float(max(dmc * (10.0 ** k), 0.0))


def calcular_dc(
    temp_c: float,
    precip_mm: float,
    dc_prev: float = 15.0,
) -> float:
    """Drought Code (DC) with recharge and drying."""
    dc = float(dc_prev)
    if precip_mm > 2.8:
        rd = 0.83 * precip_mm - 2.8
        q = 800.0 * np.exp(-dc / 400.0) + 3.937 * rd
        q = min(q, 800.0)
        dc = max(400.0 * np.log(800.0 / max(q, 1e-6)), 0.0)
    dc += 0.5 * max(float(temp_c), -2.8)
    return float(max(dc, 0.0))


def calcular_bui(dmc: float, dc: float) -> float:
    """Buildup Index (BUI)."""
    dmc = max(float(dmc), 0.0)
    dc = max(float(dc), 0.0)
    if dmc <= 0.0:
        return 0.0
    if dmc <= 0.4 * dc:
        return float(0.8 * dmc * dc / (dmc + 0.4 * dc))
    return float(
        dmc
        - (1.0 - 0.8 * dc / (dmc + 0.4 * dc))
        * (0.92 + (0.0114 * dmc) ** 1.7)
    )


def calcular_fwi(isi: float, bui: float) -> float:
    """Final Fire Weather Index (FWI)."""
    if bui <= 80.0:
        fd = 0.626 * (max(float(bui), 0.0) ** 0.809) + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * float(bui)))
    b = 0.1 * float(isi) * fd
    if b > 1.0:
        return float(np.exp(2.72 * (0.434 * np.log(b)) ** 0.647))
    return float(b)
