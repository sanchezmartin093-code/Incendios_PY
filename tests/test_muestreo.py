"""
Pruebas unitarias para el módulo muestreo.py (Dead-Zone Spatial Buffer de 20 km y muestreo estratificado).
"""

import unittest
import numpy as np
import polars as pl
from src.muestreo import (
    filtrar_candidatos_con_buffer_muerto,
    muestrear_pseudo_ausencias_estratificadas,
    particionar_dataset_out_of_time
)

class TestMuestreo(unittest.TestCase):
    def test_dead_zone_buffer_exclusion(self):
        # Fuego positivo en (x=500.000, y=7.000.000)
        df_pos = pl.DataFrame({
            "x_utm": [500000.0],
            "y_utm": [7000000.0],
            "fuego": [1]
        })

        # Candidato 1: a 5 km (5.000 m) -> dentro de la zona muerta (<20 km) -> DEBE SER EXCLUIDO
        # Candidato 2: a 30 km (30.000 m) -> fuera de la zona muerta (>20 km) -> DEBE SOBREVIVIR
        df_cand = pl.DataFrame({
            "cell_id": [1, 2],
            "x_utm": [505000.0, 530000.0],
            "y_utm": [7000000.0, 7000000.0],
            "fuego": [0, 0]
        })

        seguros = filtrar_candidatos_con_buffer_muerto(df_pos, df_cand, buffer_km=20.0)
        self.assertEqual(len(seguros), 1)
        self.assertEqual(seguros["cell_id"][0], 2, "Solo el candidato a 30 km debe sobrevivir")

    def test_muestreo_estratificado_ratio(self):
        # Dataset sintético con 2 fuegos positivos y 100 negativos en la misma fecha y región
        np.random.seed(42)
        df_test = pl.DataFrame({
            "cell_id": list(range(102)),
            "fecha": ["2026-10-20"] * 102,
            "region": ["Chaco"] * 102,
            "x_utm": [500000.0, 501000.0] + list(np.linspace(530000.0, 600000.0, 100)),
            "y_utm": [7000000.0, 7001000.0] + list(np.linspace(7030000.0, 7100000.0, 100)),
            "fuego": [1, 1] + [0] * 100
        })

        # Con ratio=10, 2 fuegos * 10 = 20 negativos muestreados + 2 positivos = 22 total
        balanceado = muestrear_pseudo_ausencias_estratificadas(df_test, ratio_neg_pos=10, buffer_km=20.0)
        self.assertEqual(len(balanceado), 22)
        self.assertEqual((balanceado["fuego"] == 1).sum(), 2)
        self.assertEqual((balanceado["fuego"] == 0).sum(), 20)

    def test_particion_out_of_time(self):
        # Dataset con 5 fechas cronológicas
        df_fechas = pl.DataFrame({
            "fecha": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"],
            "val": [1, 2, 3, 4, 5]
        })
        train, val, test = particionar_dataset_out_of_time(df_fechas)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        # Asegurar no solapamiento temporal
        self.assertTrue(set(train["fecha"]).isdisjoint(set(val["fecha"])))
        self.assertTrue(set(val["fecha"]).isdisjoint(set(test["fecha"])))

if __name__ == "__main__":
    unittest.main()
