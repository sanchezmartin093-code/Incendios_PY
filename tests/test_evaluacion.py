"""
Pruebas unitarias para el script de evaluación en test y cálculo de métricas operativas.
"""

import unittest
from pathlib import Path
from src.constantes import DIR_MODELOS, DIR_PROCESSED

class TestEvaluacion(unittest.TestCase):
    def test_archivos_modelo_y_test_existen(self):
        p_test = DIR_PROCESSED / "test_dataset.parquet"
        self.assertTrue(p_test.exists(), "El dataset de test debe existir")

        for reg in ["chaco", "oriental"]:
            b = DIR_MODELOS / f"lightgbm_{reg}.booster"
            c = DIR_MODELOS / "calibradores" / f"calibrador_{reg}.joblib"
            self.assertTrue(b.exists(), f"El modelo {b.name} debe existir")
            self.assertTrue(c.exists(), f"El calibrador {c.name} debe existir")

if __name__ == "__main__":
    unittest.main()
