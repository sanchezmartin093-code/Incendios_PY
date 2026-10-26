"""
Pruebas unitarias para el remuestreo 1:N por target.
"""

import unittest
import polars as pl
from src.remuestreo import remuestrear_por_target


class TestRemuestreo(unittest.TestCase):
    def test_remuestreo_estratificado(self):
        df = pl.DataFrame({
            "cell_id": list(range(30)),
            "fecha": ["2024-01-01"] * 15 + ["2024-01-02"] * 15,
            "region": ["Chaco"] * 30,
            "target": [1] * 1 + [0] * 14 + [1] * 2 + [0] * 13,
        })
        out = remuestrear_por_target(df, "target", ratio=5)
        for _, sub in out.group_by(["fecha", "region"]):
            self.assertEqual(int((sub["target"] == 1).sum()), 1 if sub["fecha"][0] == "2024-01-01" else 2)
            self.assertEqual(int((sub["target"] == 0).sum()), 5 if sub["fecha"][0] == "2024-01-01" else 10)

    def test_estrato_sin_negativos(self):
        df = pl.DataFrame({
            "cell_id": [1, 2],
            "fecha": ["2024-01-01"] * 2,
            "region": ["Chaco"] * 2,
            "target": [1, 1],
        })
        out = remuestrear_por_target(df, "target", ratio=5)
        self.assertEqual(out.height, 0)


if __name__ == "__main__":
    unittest.main()
