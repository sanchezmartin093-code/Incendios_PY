"""
Unit tests for the D2 target definition.
"""

import unittest
import polars as pl
from src.targets import definir_targets_d2


class TestTargetsD2(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "cell_id": [1, 2, 3, 4, 5],
            "region": ["Chaco"] * 5,
            "fecha": ["2024-01-20", "2024-01-15", "2024-01-20", "2024-01-20", "2024-01-20"],
            "fuego": [1, 1, 1, 0, 0],
            "focos_25km_t1": [0.0, 0.0, 2.0, 0.0, 2.0],
        })
        self.focos = pl.DataFrame({
            "cell_id": [1, 2, 3, 5],
            "acq_date": ["2024-01-05", "2024-01-10", "2024-01-19", "2024-01-19"],
        })

    def test_definicion_d2(self):
        out = definir_targets_d2(self.df, self.focos)
        fila = {r["cell_id"]: r for r in out.to_dicts()}

        self.assertEqual(fila[1]["target_ignicion"], 1)
        self.assertIsNone(fila[1]["target_propagacion"])

        self.assertIsNone(fila[2]["target_ignicion"])
        self.assertIsNone(fila[2]["target_propagacion"])

        self.assertIsNone(fila[3]["target_ignicion"])
        self.assertEqual(fila[3]["target_propagacion"], 1)

        self.assertEqual(fila[4]["target_ignicion"], 0)
        self.assertIsNone(fila[4]["target_propagacion"])

        self.assertIsNone(fila[5]["target_ignicion"])
        self.assertEqual(fila[5]["target_propagacion"], 0)

    def test_dias_desde_fuego(self):
        out = definir_targets_d2(self.df, self.focos)
        fila = {r["cell_id"]: r for r in out.to_dicts()}
        self.assertEqual(fila[1]["dias_desde_fuego_celda"], 15)
        self.assertEqual(fila[2]["dias_desde_fuego_celda"], 5)
        self.assertEqual(fila[3]["dias_desde_fuego_celda"], 1)


if __name__ == "__main__":
    unittest.main()
