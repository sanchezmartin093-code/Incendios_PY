"""
Módulo inpe.py: implementación del Risco de Fogo do INPE (Setzer et al., 2019).

Se implementa la versión pública 11 sin la corrección por focos de los últimos
3 días, para evitar fuga de etiquetas en el benchmark de ignición.
"""

import numpy as np
import polars as pl


def pse_desde_lluvia_diaria(precip_mm: np.ndarray, ventana_max: int = 120) -> np.ndarray:
    """Calcula los Días de Secura (PSE) para una serie diaria de lluvia."""
    p = np.asarray(precip_mm, dtype=float)
    n = len(p)
    cs = np.concatenate([[0.0], np.nancumsum(p)])
    salida = np.zeros(n, dtype=float)
    for t in range(n):
        def acumulado(k: int) -> float:
            ini = max(0, t - k)
            return float(cs[t] - cs[ini])
        p1, p2, p3, p4, p5 = (acumulado(k) for k in (1, 2, 3, 4, 5))
        p10, p15, p30, p60, p90, p120 = (acumulado(k) for k in (10, 15, 30, 60, 90, 120))
        fp1 = np.exp(-0.14 * p1)
        fp2 = np.exp(-0.07 * (p2 - p1))
        fp3 = np.exp(-0.04 * (p3 - p2))
        fp4 = np.exp(-0.03 * (p4 - p3))
        fp5 = np.exp(-0.02 * (p5 - p4))
        fp6_10 = np.exp(-0.01 * (p10 - p5))
        fp11_15 = np.exp(-0.008 * (p15 - p10))
        fp16_30 = np.exp(-0.004 * (p30 - p15))
        fp31_60 = np.exp(-0.002 * (p60 - p30))
        fp61_90 = np.exp(-0.001 * (p90 - p60))
        fp91_120 = np.exp(-0.0007 * (p120 - p90))
        salida[t] = 105.0 * np.prod([
            fp1, fp2, fp3, fp4, fp5, fp6_10, fp11_15,
            fp16_30, fp31_60, fp61_90, fp91_120,
        ])
    return salida


def riesgo_basico(pse: float, constante_a: float) -> float:
    """Riesgo de Fogo Básico senoidal (0 a 0,8)."""
    x = (float(constante_a) * float(pse) / 120.0) * (np.pi / 2.0) - (np.pi / 2.0)
    return float(0.8 * (1.0 + np.sin(x)) / 2.0)


def constante_a_desde_fracciones(
    pct_pastura: float,
    pct_sabana: float,
    pct_bosque_seco: float,
    pct_bosque_humedo: float,
    pct_palmar: float,
) -> float:
    """Constante A de INPE ponderada por las fracciones de combustible."""
    pesos = {
        "pastura": (float(pct_pastura), 6.0),
        "sabana": (float(pct_sabana), 3.0),
        "bosque_seco": (float(pct_bosque_seco), 1.72),
        "bosque_humedo": (float(pct_bosque_humedo), 1.72),
        "palmar": (float(pct_palmar), 1.5),
    }
    total = sum(w for w, _ in pesos.values())
    if total <= 0.0:
        return 0.0
    return sum(w * a for w, a in pesos.values()) / total


def riesgo_inpe(
    pse: float,
    constante_a: float,
    temp_max_c: float,
    rh_min_pct: float,
    latitud: float,
    elevacion_m: float = 0.0,
) -> float:
    """Riesgo de Fogo observado INPE, sin corrección por focos."""
    rb = riesgo_basico(pse, constante_a)
    fur = (-0.006 * float(rh_min_pct)) + 1.3
    ft = (0.02 * float(temp_max_c)) + 0.4
    flat = 1.0 + abs(float(latitud)) * 0.003
    felv = 1.0 + float(elevacion_m) * 0.00003
    return float(np.clip(rb * fur * ft * flat * felv, 0.0, 1.0))
