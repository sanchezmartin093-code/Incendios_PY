"""
Pruebas unitarias para el módulo biomasa.py (categorización de combustible y máscaras no combustibles).
"""

import unittest
import polars as pl
from src.biomasa import (
    categorizar_combustible,
    calcular_metricas_combustible_desde_conteo,
    generar_perfil_combustible_base
)

class TestBiomasa(unittest.TestCase):
    def test_categorizacion_por_region(self):
        # Código 3 (bosque) debe ser seco en Chaco y húmedo en Oriental
        self.assertEqual(categorizar_combustible(3, "Chaco"), "bosque_seco")
        self.assertEqual(categorizar_combustible(3, "Oriental"), "bosque_humedo")

        # Códigos de pasturas (15, 18, 21) deben ser pastura
        self.assertEqual(categorizar_combustible(15, "Chaco"), "pastura")
        self.assertEqual(categorizar_combustible(18, "Oriental"), "pastura")

        # Código de agua (33) o urbano (24) debe ser no_combustible
        self.assertEqual(categorizar_combustible(33, "Chaco"), "no_combustible")
        self.assertEqual(categorizar_combustible(24, "Oriental"), "no_combustible")

    def test_calculo_metricas_y_mascara_exclusion(self):
        # Celda 1: 90 píxeles de pastura, 10 de bosque seco en Chaco
        conteo_rural = {15: 90, 3: 10}
        res_rural = calcular_metricas_combustible_desde_conteo(conteo_rural, "Chaco")
        self.assertEqual(res_rural["pct_pastura"], 90.0)
        self.assertEqual(res_rural["pct_bosque_seco"], 10.0)
        self.assertTrue(res_rural["es_combustible"])

        # Celda 2: 95 píxeles de agua (cuerpo de agua mayor) -> NO combustible
        conteo_agua = {33: 95, 15: 5}
        res_agua = calcular_metricas_combustible_desde_conteo(conteo_agua, "Oriental")
        self.assertEqual(res_agua["pct_no_combustible"], 95.0)
        self.assertFalse(res_agua["es_combustible"], "Celda con 95% de agua debe ser excluida como no combustible")

    def test_generar_perfil_base(self):
        df_grilla_dummy = pl.DataFrame({
            "cell_id": [1000, 2000],
            "departamento": ["ALTO PARAGUAY", "ASUNCION"],
            "region": ["Chaco", "Oriental"],
            "lat": [-20.0, -25.3],
            "lon": [-58.0, -57.6]
        })
        perfil = generar_perfil_combustible_base(df_grilla_dummy)
        self.assertEqual(len(perfil), 2)
        # Asunción debe estar marcada como no combustible por casco urbano
        self.assertFalse(perfil.filter(pl.col("cell_id") == 2000)["es_combustible"][0])
        # Alto Paraguay debe tener alto bosque seco
        self.assertGreater(perfil.filter(pl.col("cell_id") == 1000)["pct_bosque_seco"][0], 50.0)

if __name__ == "__main__":
    unittest.main()
