"""
Pruebas unitarias para el módulo biomasa.py (categorización de combustible y máscaras no combustibles).
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import rasterio
from rasterio.transform import from_origin
from src.biomasa import (
    categorizar_combustible,
    calcular_metricas_combustible_desde_conteo,
    generar_perfil_combustible_base,
    procesar_geotiff_mapbiomas_a_grilla,
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

    def test_ingesta_mapbiomas_geotiff(self):
        """La extracción real debe devolver fracciones por celda, no aleatorios."""
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "mapbiomas_test.tif"
            ancho = alto = 100
            data = np.zeros((alto, ancho), dtype=np.uint8)
            data[:, :33] = 15   # pastura
            data[:, 33:] = 3    # bosque
            with rasterio.open(
                ruta, "w", driver="GTiff", height=alto, width=ancho,
                count=1, dtype="uint8", crs="EPSG:32721",
                transform=from_origin(499500.0, 7501500.0, 30.0, 30.0),
            ) as dst:
                dst.write(data, 1)

            grilla = pl.DataFrame({
                "cell_id": [1, 2, 3, 4],
                "x_utm": [500000.0, 501000.0, 500000.0, 501000.0],
                "y_utm": [7501000.0, 7501000.0, 7500000.0, 7500000.0],
                "region": ["Chaco"] * 4,
            })
            perfil = procesar_geotiff_mapbiomas_a_grilla(grilla, [ruta])
            self.assertEqual(perfil["origen_datos"].unique().to_list(), ["mapbiomas_real"])
            izq = perfil.filter(pl.col("cell_id") == 1)
            der = perfil.filter(pl.col("cell_id") == 2)
            self.assertGreater(float(izq["pct_pastura"][0]), 80.0)
            self.assertGreater(float(der["pct_bosque_seco"][0]), 80.0)

if __name__ == "__main__":
    unittest.main()
