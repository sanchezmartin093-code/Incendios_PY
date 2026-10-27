"""
Unit tests for VPD physical formulas and Canadian FWI components.
"""

import unittest
import numpy as np
from src.fwi import (
    calcular_presion_vapor_saturacion,
    calcular_vpd,
    calcular_ffmc,
    calcular_isi,
    calcular_interaccion_viento_vpd
)

class TestFWI(unittest.TestCase):
    def test_vpd_condiciones_extremas(self):
        # Saturated air (T = Td) -> VPD = 0
        vpd_saturado = calcular_vpd(25.0, 25.0)
        self.assertAlmostEqual(float(vpd_saturado), 0.0, places=4)

        # Chaco during an extreme heat wave (T = 42°C, Td = 12°C) -> VPD > 5 kPa
        vpd_chaco = calcular_vpd(42.0, 12.0)
        self.assertGreater(float(vpd_chaco), 5.0)

    def test_ffmc_rango_y_comportamiento(self):
        ffmc_seco = calcular_ffmc(temp=38.0, rh=20.0, wind=35.0, prec=0.0, ffmc_prev=85.0)
        self.assertTrue(85.0 <= ffmc_seco <= 101.0)

        ffmc_lluvia = calcular_ffmc(temp=20.0, rh=95.0, wind=10.0, prec=25.0, ffmc_prev=90.0)
        self.assertLess(ffmc_lluvia, 50.0)

    def test_isi_calculo(self):
        isi_calmo = calcular_isi(wind_kmh=5.0, ffmc=80.0)
        isi_temporal = calcular_isi(wind_kmh=45.0, ffmc=92.0)
        self.assertGreater(isi_temporal, isi_calmo * 3.0)

    def test_interaccion_viento_vpd(self):
        interaccion = calcular_interaccion_viento_vpd(wind_kmh=40.0, vpd_kpa=3.5)
        self.assertAlmostEqual(interaccion, 140.0, places=3)

if __name__ == "__main__":
    unittest.main()

class TestFWICompleto(unittest.TestCase):
    def test_componentes_fwi(self):
        from src.fwi import calcular_bui, calcular_dc, calcular_dmc, calcular_fwi
        dmc = calcular_dmc(30.0, 40.0, 20.0, 0.0, 6.0)
        dc = calcular_dc(30.0, 0.0, 15.0)
        bui = calcular_bui(dmc, dc)
        fwi = calcular_fwi(10.0, bui)
        self.assertGreaterEqual(dmc, 0.0)
        self.assertGreaterEqual(dc, 0.0)
        self.assertGreaterEqual(bui, 0.0)
        self.assertGreaterEqual(fwi, 0.0)
