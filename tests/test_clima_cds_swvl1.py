"""Unit tests for the short CDS path (swvl1 + potential evaporation)."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


def _cargar_modulo():
    ruta = Path(__file__).resolve().parent.parent / "scripts" / "02f_completar_clima_cds.py"
    spec = importlib.util.spec_from_file_location("clima_cds_swvl1", ruta)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestEvaporacionCDS(unittest.TestCase):
    def test_pev_acumulada_se_convierte_a_mm_dia(self):
        mod = _cargar_modulo()
        # Real ERA5-Land sample: first value is the previous day's total;
        # the remaining values are cumulative from 00 UTC and the last is
        # the daily total (negative sign = evaporation).
        pev_m = np.array([
            -0.01135299,
            -0.00074060,
            -0.00145101,
            -0.00188305,
            -0.00260150,
            -0.00513120,
            -0.00849309,
            -0.01086723,
        ])[:, None]
        et0 = mod._agregar_evaporacion_diaria(pev_m)
        self.assertAlmostEqual(float(et0[0]), 10.86723, places=4)

    def test_pev_por_incrementos_se_suma(self):
        mod = _cargar_modulo()
        pev_m = np.array([-0.001, -0.002, -0.001, -0.003])[:, None]
        et0 = mod._agregar_evaporacion_diaria(pev_m)
        self.assertAlmostEqual(float(et0[0]), 6.0, places=4)


if __name__ == "__main__":
    unittest.main()
