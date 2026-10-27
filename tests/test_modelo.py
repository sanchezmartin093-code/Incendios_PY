"""
Unit tests for the modelo.py module (LightGBM training, isotonic calibration and serialization).
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from src.modelo import (
    entrenar_modelo_lightgbm,
    calibrar_probabilidades_isotonica,
    predecir_probabilidad,
    guardar_modelo_y_calibrador,
    cargar_modelo_y_calibrador,
    obtener_restricciones_monotonas
)

class TestModelo(unittest.TestCase):
    def setUp(self):
        # Synthetic test dataset
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 5
        self.feature_names = [f"feat_{i}" for i in range(self.n_features)]

        self.X_train = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        # Synthetic signal: if feat_0 > 0.5, higher fire probability
        prob = 1.0 / (1.0 + np.exp(-1.5 * self.X_train[:, 0]))
        self.y_train = (np.random.rand(self.n_samples) < prob).astype(np.float32)

        self.X_val = np.random.randn(50, self.n_features).astype(np.float32)
        self.y_val = (self.X_val[:, 0] > 0.0).astype(np.float32)

    def test_entrenamiento_y_calibracion(self):
        booster = entrenar_modelo_lightgbm(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            nombres_features=self.feature_names,
            num_boost_round=20,
            early_stopping_rounds=5
        )
        self.assertIsNotNone(booster)

        # Calibration
        calibrador = calibrar_probabilidades_isotonica(booster, self.X_val, self.y_val)
        self.assertIsNotNone(calibrador)

        # Inference
        probs = predecir_probabilidad(booster, calibrador, self.X_val)
        self.assertEqual(len(probs), len(self.X_val))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Las probabilidades deben estar en [0, 1]")

    def test_serializacion_y_recuperacion(self):
        booster = entrenar_modelo_lightgbm(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            nombres_features=self.feature_names,
            num_boost_round=10
        )
        calibrador = calibrar_probabilidades_isotonica(booster, self.X_val, self.y_val)

        with tempfile.TemporaryDirectory() as tmpdir:
            ruta_b = Path(tmpdir) / "modelo.booster"
            ruta_c = Path(tmpdir) / "calibrador.joblib"

            guardar_modelo_y_calibrador(booster, calibrador, ruta_b, ruta_c)
            self.assertTrue(ruta_b.exists())
            self.assertTrue(ruta_c.exists())

            # Load and predict
            b_cargado, c_cargado = cargar_modelo_y_calibrador(ruta_b, ruta_c)
            probs = predecir_probabilidad(b_cargado, c_cargado, self.X_val)
            self.assertEqual(len(probs), len(self.X_val))

    def test_restricciones_monotonas_biofisicas(self):
        """Verifies the consistency of monotonic signs and non-decreasing physical invariance."""
        feats = ["vpd_max_kpa", "precipitacion_mm", "mes"]
        restrs = obtener_restricciones_monotonas(feats)
        self.assertEqual(restrs, [1, -1, 0], "VPD debe ser +1, lluvia -1 y mes neutral 0")

        nuevos = [
            "swvl1_m3m3", "deficit_hidro_60d", "pct_desmonte_t1",
            "fdd_heladas_60d", "pct_agricultura_anual", "pct_pastura",
        ]
        self.assertEqual(
            obtener_restricciones_monotonas(nuevos),
            [-1, 1, 0, 0, 0, 0],
            "Solo la humedad y el déficit tienen signo físico inequívoco",
        )

        # Train synthetic physics-guided booster
        X_toy = np.array([
            [1.0, 30.0],  # low VPD, lots of rain -> no fire
            [1.5, 20.0],
            [4.0, 0.0],   # high VPD, zero rain -> fire
            [4.5, 0.0]
        ], dtype=np.float32)
        y_toy = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        booster = entrenar_modelo_lightgbm(
            X_train=X_toy, y_train=y_toy,
            X_val=X_toy, y_val=y_toy,
            nombres_features=["vpd_max_kpa", "precipitacion_mm"],
            num_boost_round=10,
            usar_restricciones_fisicas=True
        )

        # Evaluate strict monotonicity with respect to VPD (more dryness -> risk does not decrease)
        p_baja_sequedad = booster.predict(np.array([[1.0, 5.0]], dtype=np.float32))[0]
        p_alta_sequedad = booster.predict(np.array([[4.5, 5.0]], dtype=np.float32))[0]
        self.assertGreaterEqual(p_alta_sequedad, p_baja_sequedad, "A mayor VPD, el riesgo debe ser no-decreciente")

        # Evaluate strict monotonicity with respect to rain (more rain -> risk does not increase)
        p_cero_lluvia = booster.predict(np.array([[3.0, 0.0]], dtype=np.float32))[0]
        p_mucha_lluvia = booster.predict(np.array([[3.0, 40.0]], dtype=np.float32))[0]
        self.assertLessEqual(p_mucha_lluvia, p_cero_lluvia, "A mayor precipitación, el riesgo debe ser no-creciente")

if __name__ == "__main__":
    unittest.main()
