"""
Pruebas unitarias para el módulo modelo.py (entrenamiento LightGBM, calibración isotónica y serialización).
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
    cargar_modelo_y_calibrador
)

class TestModelo(unittest.TestCase):
    def setUp(self):
        # Dataset sintético de prueba
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 5
        self.feature_names = [f"feat_{i}" for i in range(self.n_features)]

        self.X_train = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        # Señal sintética: si feat_0 > 0.5, mayor probabilidad de fuego
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

        # Calibración
        calibrador = calibrar_probabilidades_isotonica(booster, self.X_val, self.y_val)
        self.assertIsNotNone(calibrador)

        # Inferencia
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

            # Cargar y predecir
            b_cargado, c_cargado = cargar_modelo_y_calibrador(ruta_b, ruta_c)
            probs = predecir_probabilidad(b_cargado, c_cargado, self.X_val)
            self.assertEqual(len(probs), len(self.X_val))

if __name__ == "__main__":
    unittest.main()
