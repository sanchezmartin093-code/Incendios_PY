import unittest
from datetime import datetime
from pathlib import Path
import json
import tempfile
import polars as pl
import numpy as np

from src.alerta import (
    _leer_umbral_validacion,
    cargar_artefactos_modelos,
    categorizar_riesgo,
    preparar_datos_diarios,
    evaluar_alerta_nacional,
    generar_boletin_operativo
)
from src.constantes import DIR_PROCESSED

class TestServicioAlerta(unittest.TestCase):
    def test_umbral_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "umbral.json"
            with self.assertRaises(FileNotFoundError):
                _leer_umbral_validacion(ruta)
            ruta.write_text(json.dumps({"c_youden": 0.1}), encoding="utf-8")
            with self.assertRaises(KeyError):
                _leer_umbral_validacion(ruta)
            ruta.write_text(json.dumps({"c_f05": 0.23}), encoding="utf-8")
            self.assertAlmostEqual(_leer_umbral_validacion(ruta), 0.23)

    def test_categorizar_riesgo(self):
        """Verifies the strict assignment of operational alert levels."""
        # No fuel -> NULO
        self.assertEqual(categorizar_riesgo(0.9, "Chaco", False), "NULO")
        # With gating reason (e.g. rain) -> BAJO
        self.assertEqual(categorizar_riesgo(0.8, "Chaco", True, "LLUVIA_EXTINTORA_48H"), "BAJO")
        # Subcritical Chaco (P < UMBRAL_CORTE_CHACO = 0.399)
        self.assertEqual(categorizar_riesgo(0.05, "Chaco", True), "BAJO")
        self.assertEqual(categorizar_riesgo(0.35, "Chaco", True), "BAJO")
        # Moderate Chaco (0.399 <= P < 0.399 + 0.601*0.25 = 0.549)
        self.assertEqual(categorizar_riesgo(0.45, "Chaco", True), "MODERADO")
        # High Chaco (0.549 <= P < 0.399 + 0.601*0.55 = 0.729)
        self.assertEqual(categorizar_riesgo(0.60, "Chaco", True), "ALTO")
        # Extreme Chaco (P >= 0.73)
        self.assertEqual(categorizar_riesgo(0.85, "Chaco", True), "EXTREMO")
        # Verification with explicit dynamic M3 cutoff (0.4045)
        self.assertEqual(categorizar_riesgo(0.40, "Chaco", True, cutoff=0.4045), "BAJO")
        self.assertEqual(categorizar_riesgo(0.42, "Chaco", True, cutoff=0.4045), "MODERADO")

    def test_carga_artefactos(self):
        """Verifies that the serialized models and calibrators exist and load."""
        from src.constantes import DIR_MODELOS
        if not (DIR_MODELOS / "lightgbm_chaco.booster").exists():
            self.skipTest("Modelos entrenados no disponibles en entorno CI")
        art = cargar_artefactos_modelos()
        self.assertIn("chaco", art)
        self.assertIn("oriental", art)
        self.assertIsNotNone(art["chaco"]["booster"])
        self.assertIsNotNone(art["oriental"]["calibrador"])
        self.assertIsNotNone(art["chaco"]["booster_m2"])
        self.assertEqual(len(art["chaco"]["features_m2"]), 35)

    def test_preparacion_e_inferencia_fecha(self):
        """Verifies the complete inference flow on a test date."""
        from src.constantes import DIR_MODELOS
        p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
        if not p_clima.exists():
            self.skipTest("Clima no disponible")
        if not (DIR_MODELOS / "lightgbm_chaco.booster").exists() and not (DIR_MODELOS / "lightgbm_hibrido_chaco.booster").exists():
            self.skipTest("Modelos entrenados no disponibles en entorno CI")

        df_clima = pl.read_parquet(p_clima)
        fecha_test = df_clima["fecha"].unique().to_list()[0]

        df_nac = preparar_datos_diarios(fecha_test)
        self.assertEqual(df_nac.height, 400006)
        self.assertIn("es_combustible", df_nac.columns)
        self.assertIn("vpd_max_kpa", df_nac.columns)

        artefactos = cargar_artefactos_modelos()
        # Test evaluation on a sample for speed
        df_sub = df_nac.head(1000)
        df_eval = evaluar_alerta_nacional(df_sub, artefactos)
        self.assertEqual(df_eval.height, 1000)
        self.assertIn("probabilidad_fuego", df_eval.columns)
        self.assertIn("fallback_m2", df_eval.columns)
        self.assertIn("nivel_alerta", df_eval.columns)

        resumen, md = generar_boletin_operativo(df_eval, fecha_test)
        self.assertIn("distribucion_riesgo", resumen)
        self.assertIn("BOLETÍN TÉCNICO DIARIO", md)

if __name__ == "__main__":
    unittest.main()
