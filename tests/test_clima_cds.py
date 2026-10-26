"""
Pruebas unitarias para la agregación diaria de precipitación ERA5-Land.
"""

import unittest
import numpy as np
from src.clima_cds import agregar_precipitacion_diaria


class TestClimaCDS(unittest.TestCase):
    def test_acumulado_desde_medianoche(self):
        tp_mm = np.array([
            [0.0, 0.0],
            [0.0, 0.2],
            [1.0, 0.3],
            [1.0, 0.3],
            [2.0, 0.6],
        ])
        totales = agregar_precipitacion_diaria(tp_mm)
        np.testing.assert_allclose(totales, [2.0, 0.6])

    def test_incrementos_horarios(self):
        tp_mm = np.array([
            [0.0, 0.0],
            [1.0, 0.2],
            [0.0, 0.3],
            [2.0, 0.0],
            [0.0, 0.1],
        ])
        totales = agregar_precipitacion_diaria(tp_mm)
        np.testing.assert_allclose(totales, [3.0, 0.6])

    def test_primer_paso_es_acumulado_previo(self):
        tp_mm = np.array([
            [43.364],
            [37.495],
            [90.479],
            [138.855],
            [150.584],
            [155.036],
            [160.161],
            [162.983],
        ])
        totales = agregar_precipitacion_diaria(tp_mm)
        np.testing.assert_allclose(totales, [162.983])

    def test_serie_con_nan(self):
        tp_mm = np.array([
            [np.nan, 0.0],
            [1.0, np.nan],
            [0.0, 0.5],
            [2.0, 0.0],
        ])
        totales = agregar_precipitacion_diaria(tp_mm)
        np.testing.assert_allclose(totales, [3.0, 0.5])


if __name__ == "__main__":
    unittest.main()
