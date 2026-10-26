import unittest
from datetime import datetime
import polars as pl
import numpy as np

from src.alerta import (
    cargar_artefactos_modelos,
    categorizar_riesgo,
    preparar_datos_diarios,
    evaluar_alerta_nacional,
    generar_boletin_operativo
)
from src.constantes import DIR_PROCESSED

class TestServicioAlerta(unittest.TestCase):
    def test_categorizar_riesgo(self):
        """Verifica la asignación estricta de niveles de alerta operativos."""
        # Sin combustible -> NULO
        self.assertEqual(categorizar_riesgo(0.9, "Chaco", False), "NULO")
        # Con motivo de gating (ej. lluvia) -> BAJO
        self.assertEqual(categorizar_riesgo(0.8, "Chaco", True, "LLUVIA_EXTINTORA_48H"), "BAJO")
        # Subcrítico Chaco (P < UMBRAL_CORTE_CHACO = 0.399)
        self.assertEqual(categorizar_riesgo(0.05, "Chaco", True), "BAJO")
        self.assertEqual(categorizar_riesgo(0.35, "Chaco", True), "BAJO")
        # Moderado Chaco (0.399 <= P < 0.399 + 0.601*0.25 = 0.549)
        self.assertEqual(categorizar_riesgo(0.45, "Chaco", True), "MODERADO")
        # Alto Chaco (0.549 <= P < 0.399 + 0.601*0.55 = 0.729)
        self.assertEqual(categorizar_riesgo(0.60, "Chaco", True), "ALTO")
        # Extremo Chaco (P >= 0.73)
        self.assertEqual(categorizar_riesgo(0.85, "Chaco", True), "EXTREMO")

    def test_carga_artefactos(self):
        """Verifica que los modelos serializados y calibradores existan y carguen."""
        from src.constantes import DIR_MODELOS
        if not (DIR_MODELOS / "lightgbm_chaco.booster").exists():
            self.skipTest("Modelos entrenados no disponibles en entorno CI")
        art = cargar_artefactos_modelos()
        self.assertIn("chaco", art)
        self.assertIn("oriental", art)
        self.assertIsNotNone(art["chaco"]["booster"])
        self.assertIsNotNone(art["oriental"]["calibrador"])

    def test_preparacion_e_inferencia_fecha(self):
        """Verifica el flujo completo de inferencia en una fecha de prueba."""
        p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
        if not p_clima.exists():
            self.skipTest("Clima no disponible")

        df_clima = pl.read_parquet(p_clima)
        fecha_test = df_clima["fecha"].unique().to_list()[0]

        df_nac = preparar_datos_diarios(fecha_test)
        self.assertEqual(df_nac.height, 400006)
        self.assertIn("es_combustible", df_nac.columns)
        self.assertIn("vpd_max_kpa", df_nac.columns)

        artefactos = cargar_artefactos_modelos()
        # Probar evaluación sobre una muestra para velocidad
        df_sub = df_nac.head(1000)
        df_eval = evaluar_alerta_nacional(df_sub, artefactos)
        self.assertEqual(df_eval.height, 1000)
        self.assertIn("probabilidad_fuego", df_eval.columns)
        self.assertIn("nivel_alerta", df_eval.columns)

        resumen, md = generar_boletin_operativo(df_eval, fecha_test)
        self.assertIn("distribucion_riesgo", resumen)
        self.assertIn("BOLETÍN TÉCNICO DIARIO", md)

if __name__ == "__main__":
    unittest.main()
