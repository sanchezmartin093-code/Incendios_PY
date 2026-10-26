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
    calcular_bootstrap_ci_roc,
    calcular_cuantil_conforme,
    predecir_conjuntos_conformes,
    evaluar_cobertura_conforme,
    corregir_prior_probabilidades,
    calcular_metricas_poblacionales,
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

    def test_split_conformal_prediction(self):
        rng = np.random.default_rng(42)
        n_val = 1000
        n_test = 500
        
        # Simular probabilidades y etiquetas
        p_val = rng.uniform(0.05, 0.95, size=n_val)
        y_val = (rng.uniform(0, 1, size=n_val) < p_val).astype(int)
        
        p_test = rng.uniform(0.05, 0.95, size=n_test)
        y_test = (rng.uniform(0, 1, size=n_test) < p_test).astype(int)
        
        # Calcular cuantil para 90% de cobertura marginal (alpha = 0.10)
        alpha = 0.10
        q_hat = calcular_cuantil_conforme(y_val, p_val, alpha=alpha)
        
        self.assertTrue(0.0 < q_hat <= 1.0)
        
        # Predecir conjuntos conformes sobre test
        c0, c1 = predecir_conjuntos_conformes(p_test, q_hat)
        self.assertEqual(len(c0), n_test)
        self.assertEqual(len(c1), n_test)
        
        # Verificar cobertura empírica marginal >= 1 - alpha - tolerancia de muestra finita
        cobertura = evaluar_cobertura_conforme(y_test, p_test, q_hat)
        self.assertGreaterEqual(cobertura, (1.0 - alpha) - 0.05)

    def test_empates_en_threshold(self):
        """El umbral debe reproducir la métrica al clasificar con >=."""
        from sklearn.metrics import fbeta_score
        y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
        cutoff, f_reportado = calcular_umbral_optimo_f_beta(y, p, beta=0.5)
        f_real = fbeta_score(y, (p >= cutoff).astype(int), beta=0.5, zero_division=0)
        self.assertAlmostEqual(f_reportado, f_real, places=6)

    def test_correccion_prevalencia(self):
        """La corrección case-control debe bajar probabilidades al prevalecer menos."""
        p = np.array([0.4, 0.2, 0.8])
        p_corr = corregir_prior_probabilidades(p, 0.0909, 0.0012)
        self.assertTrue(np.all(p_corr < p))
        self.assertTrue(np.all((p_corr > 0) & (p_corr < 1)))

    def test_metricas_poblacionales(self):
        y = np.array([0, 0, 1, 0, 1, 0])
        p = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.4])
        w = np.array([10.0, 10.0, 1.0, 10.0, 1.0, 10.0])
        m = calcular_metricas_poblacionales(y, p, w, cutoff=0.5)
        self.assertIn("precision_poblacional", m)
        self.assertIn("roc_auc_poblacional", m)
        self.assertGreaterEqual(m["precision_poblacional"], 0.0)
        self.assertLessEqual(m["precision_poblacional"], 1.0)

if __name__ == "__main__":
    unittest.main()
