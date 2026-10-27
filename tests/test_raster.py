import unittest
from pathlib import Path
import rasterio
import numpy as np

from src.constantes import DIR_OUTPUTS, CRS_PROYECTADO
from src.raster import exportar_alerta_a_geotiff

class TestRasterExport(unittest.TestCase):
    def test_exportar_geotiff_valido(self):
        """Verifies that the exported GeoTIFF file has the correct CRS, 2 bands and metadata."""
        dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
        archivos = sorted(list(dir_raster.glob("alerta_nacional_*.parquet")))
        if not archivos:
            self.skipTest("No hay archivos parquet de alerta")

        p_parquet = archivos[0]
        p_tif_test = dir_raster / "test_temp_alerta.tif"

        try:
            exportar_alerta_a_geotiff(p_parquet, p_tif_test)
            self.assertTrue(p_tif_test.exists())

            with rasterio.open(p_tif_test) as src:
                self.assertEqual(src.crs.to_string(), CRS_PROYECTADO)
                self.assertEqual(src.count, 2)
                self.assertEqual(src.res, (1000.0, 1000.0))
                # Read band 1 (probability)
                b1 = src.read(1)
                self.assertEqual(b1.shape, (918, 860))
                # Valid values must be in [0, 1] or be nodata (-9999)
                validos = b1[b1 != -9999.0]
                self.assertGreater(len(validos), 10000)
                self.assertTrue(np.all((validos >= 0.0) & (validos <= 1.0)))

                # Metadata
                tags = src.tags()
                self.assertIn("SISTEMA", tags)
                self.assertEqual(tags["SISTEMA"], "SAT-Incendios PY")
        finally:
            if p_tif_test.exists():
                p_tif_test.unlink()

if __name__ == "__main__":
    unittest.main()
