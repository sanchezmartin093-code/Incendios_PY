import unittest
import numpy as np
from src.cost_loss import valor_cost_loss

class TestCostLoss(unittest.TestCase):
    def test_modelo_perfecto(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.001, 0.01, 0.8, 0.9])
        w = np.ones(4)
        v = valor_cost_loss(y, p, w, alpha=0.5)
        self.assertGreater(v["valor"], 0.0)

    def test_probabilidades_y_pesos(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        w = np.array([10.0, 1.0, 10.0, 1.0])
        v = valor_cost_loss(y, p, w, alpha=0.5)
        self.assertIn("valor", v)
        self.assertTrue(np.isfinite(v["valor"]))

if __name__ == "__main__":
    unittest.main()
