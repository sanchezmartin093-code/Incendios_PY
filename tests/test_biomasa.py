"""
Unit tests for the biomasa.py module (fuel categorization and non-fuel masks).
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
    seleccionar_tif_estatico,
)

class TestBiomasa(unittest.TestCase):
    def test_categorizacion_por_region(self):
        # Code 3 (forest) must be dry in Chaco and humid in Oriental
        self.assertEqual(categorizar_combustible(3, "Chaco"), "bosque_seco")
        self.assertEqual(categorizar_combustible(3, "Oriental"), "bosque_humedo")

        # MapBiomas Paraguay C2/C3: 15 pasture, 18 annual agriculture,
        # 9 forest plantation. There is no class 21 in the Paraguayan legend.
        self.assertEqual(categorizar_combustible(15, "Chaco"), "pastura_ganadera")
        self.assertEqual(categorizar_combustible(18, "Oriental"), "agricultura_anual")
        self.assertEqual(categorizar_combustible(9, "Oriental"), "plantacion_forestal")

        # Water code (33) or urban/bare (24/25) must be no_combustible
        self.assertEqual(categorizar_combustible(33, "Chaco"), "no_combustible")
        self.assertEqual(categorizar_combustible(24, "Oriental"), "no_combustible")
        self.assertEqual(categorizar_combustible(25, "Oriental"), "no_combustible")

    def test_calculo_metricas_y_mascara_exclusion(self):
        # Cell 1: 90 pasture pixels, 10 dry forest in Chaco
        conteo_rural = {15: 90, 3: 10}
        res_rural = calcular_metricas_combustible_desde_conteo(conteo_rural, "Chaco")
        self.assertEqual(res_rural["pct_pastura"], 90.0)
        self.assertEqual(res_rural["pct_pastura_ganadera"], 90.0)
        self.assertEqual(res_rural["pct_agricultura_anual"], 0.0)
        self.assertEqual(res_rural["pct_bosque_seco"], 10.0)
        self.assertTrue(res_rural["es_combustible"])

        # Split pasture vs annual agriculture, and check the compatibility sum.
        conteo_mixto = {15: 60, 18: 30, 3: 10}
        res_mixto = calcular_metricas_combustible_desde_conteo(conteo_mixto, "Chaco")
        self.assertEqual(res_mixto["pct_pastura_ganadera"], 60.0)
        self.assertEqual(res_mixto["pct_agricultura_anual"], 30.0)
        self.assertAlmostEqual(
            res_mixto["pct_pastura"],
            res_mixto["pct_pastura_ganadera"] + res_mixto["pct_agricultura_anual"],
        )

        # Cell 2: 95 water pixels (major water body) -> NOT fuel
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
        # Asunción must be marked as non-fuel due to urban core
        self.assertFalse(perfil.filter(pl.col("cell_id") == 2000)["es_combustible"][0])
        # Alto Paraguay must have high dry forest
        self.assertGreater(perfil.filter(pl.col("cell_id") == 1000)["pct_bosque_seco"][0], 50.0)

    def test_seleccion_tif_estatico_unico(self):
        """El perfil estático no debe procesar los mapas anuales superpuestos."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for anio in (2017, 2022, 2023):
                (d / f"mapbiomas_paraguay_collection2_{anio}.tif").write_bytes(b"x")
            elegido = seleccionar_tif_estatico(d, 2023)
            self.assertEqual(elegido.name, "mapbiomas_paraguay_collection2_2023.tif")
            with self.assertRaises(FileNotFoundError):
                seleccionar_tif_estatico(d, 2024)

    def test_ingesta_mapbiomas_geotiff(self):
        """The real extraction must return per-cell fractions, not random values."""
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "mapbiomas_test.tif"
            ancho = alto = 100
            data = np.zeros((alto, ancho), dtype=np.uint8)
            data[:, :33] = 15   # pasture
            data[:, 33:] = 3    # forest
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
            self.assertGreater(float(izq["pct_pastura_ganadera"][0]), 80.0)
            self.assertEqual(float(izq["pct_agricultura_anual"][0]), 0.0)
            self.assertGreater(float(der["pct_bosque_seco"][0]), 80.0)
            self.assertGreaterEqual(float(izq["borde_bosque_agro_pct"][0]), 0.0)

if __name__ == "__main__":
    unittest.main()
