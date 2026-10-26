import unittest
import numpy as np
from src.inpe import pse_desde_lluvia_diaria, riesgo_basico, riesgo_inpe

class TestINPE(unittest.TestCase):
    def test_pse_sin_lluvia(self):
        pse = pse_desde_lluvia_diaria(np.zeros(130))
        self.assertAlmostEqual(float(pse[-1]), 105.0, places=6)

    def test_riesgo_basico(self):
        self.assertAlmostEqual(riesgo_basico(120, 6.0), 0.8, places=6)
        self.assertGreaterEqual(riesgo_basico(60, 3.0), 0.0)

    def test_riesgo_en_rango(self):
        r = riesgo_inpe(100, 6.0, 38.0, 20.0, -22.0)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)

if __name__ == "__main__":
    unittest.main()
