"""
Pruebas unitarias para cálculo de umbrales óptimos de Youden, F0.5 y Brier Skill Score.
"""

import unittest
import numpy as np
from src.metricas import (
    calcular_umbral_optimo_youden,
    calcular_umbral_optimo_f_beta,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
    calcular_bootstrap_ci_roc
)

class TestMetricas(unittest.TestCase):
    def test_optimizacion_youden_y_f_beta(self):
        np.random.seed(42)
        n = 1000
        y_true = np.zeros(n)
        y_true[:100] = 1.0  # 100 fuegos
        
        y_prob = np.random.uniform(0.0, 0.4, size=n)
        y_prob[:100] += np.random.uniform(0.3, 0.6, size=100)
        y_prob = np.clip(y_prob, 0.0, 1.0)
        
        # 1. Test Youden
        umbral_youden, max_j = calcular_umbral_optimo_youden(y_true, y_prob)
        self.assertTrue(0.0 < umbral_youden < 1.0)
        self.assertGreater(max_j, 0.5)
        
        # 2. Test F0.5 (Prioridad a precisión)
        umbral_f05, max_f = calcular_umbral_optimo_f_beta(y_true, y_prob, beta=0.5)
        self.assertTrue(0.0 < umbral_f05 < 1.0)
        self.assertGreaterEqual(umbral_f05, umbral_youden * 0.8)
        
        # 3. Test BSS
        bss = calcular_brier_skill_score(y_true, y_prob)
        self.assertGreater(bss, 0.0)

    def test_ece_y_bootstrap_ci(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        # Probabilidades perfectamente calibradas (0.0 para negativos, 1.0 para positivos)
        p_perfect = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        ece_perfect = calcular_expected_calibration_error(y_true, p_perfect)
        self.assertAlmostEqual(ece_perfect, 0.0, places=4, msg="ECE para predicción perfecta debe ser 0.0")

        # Probabilidades descalibradas (0.9 para todo)
        p_bad = np.full_like(y_true, 0.9)
        ece_bad = calcular_expected_calibration_error(y_true, p_bad)
        self.assertGreater(ece_bad, 0.3)

        # Bootstrap CI
        inf, sup = calcular_bootstrap_ci_roc(y_true, p_perfect, n_bootstrap=100)
        self.assertGreaterEqual(inf, 0.99)
        self.assertLessEqual(sup, 1.01)

if __name__ == "__main__":
    unittest.main()
