"""
Unit tests for the muestreo.py module (20 km Dead-Zone Spatial Buffer and stratified sampling).
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
        # Positive fire at (x=500.000, y=7.000.000)
        df_pos = pl.DataFrame({
            "x_utm": [500000.0],
            "y_utm": [7000000.0],
            "fuego": [1]
        })

        # Candidate 1: at 5 km (5,000 m) -> inside the dead zone (<20 km) -> MUST BE EXCLUDED
        # Candidate 2: at 30 km (30,000 m) -> outside the dead zone (>20 km) -> MUST SURVIVE
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
        # Synthetic dataset with 2 positive fires and 100 negatives on the same date and region
        np.random.seed(42)
        df_test = pl.DataFrame({
            "cell_id": list(range(102)),
            "fecha": ["2026-10-20"] * 102,
            "region": ["Chaco"] * 102,
            "x_utm": [500000.0, 501000.0] + list(np.linspace(530000.0, 600000.0, 100)),
            "y_utm": [7000000.0, 7001000.0] + list(np.linspace(7030000.0, 7100000.0, 100)),
            "fuego": [1, 1] + [0] * 100
        })

        # With ratio=10, 2 fires * 10 = 20 sampled negatives + 2 positives = 22 total
        balanceado = muestrear_pseudo_ausencias_estratificadas(df_test, ratio_neg_pos=10, buffer_km=20.0)
        self.assertEqual(len(balanceado), 22)
        self.assertEqual((balanceado["fuego"] == 1).sum(), 2)
        self.assertEqual((balanceado["fuego"] == 0).sum(), 20)

    def test_particion_out_of_time(self):
        # Dataset with 5 chronological dates
        df_fechas = pl.DataFrame({
            "fecha": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"],
            "val": [1, 2, 3, 4, 5]
        })
        train, val, test = particionar_dataset_out_of_time(df_fechas)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        # Ensure no temporal overlap
        self.assertTrue(set(train["fecha"]).isdisjoint(set(val["fecha"])))
        self.assertTrue(set(val["fecha"]).isdisjoint(set(test["fecha"])))

if __name__ == "__main__":
    unittest.main()
