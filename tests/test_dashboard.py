import unittest
import importlib
from pathlib import Path
from src.constantes import ROOT_DIR, DIR_OUTPUTS

class TestDashboardInteractivo(unittest.TestCase):
    def test_generacion_dashboard_html(self):
        """Verifies that the interactive dashboard is generated, is not empty, and contains the key components."""
        dir_raster = DIR_OUTPUTS / "raster_riesgo_diario"
        if not dir_raster.exists() or not list(dir_raster.glob("alerta_nacional_*.parquet")):
            self.skipTest("Archivos de alerta diaria no disponibles en entorno CI")

        mod = importlib.import_module("scripts.12_generar_dashboard_interactivo")
        generar_dashboard_interactivo = mod.generar_dashboard_interactivo

        p_html = ROOT_DIR / "entregables" / "web" / "dashboard_alerta_interactivo.html"
        res = generar_dashboard_interactivo()
        self.assertTrue(res.exists())
        self.assertEqual(res, p_html)
        self.assertGreater(p_html.stat().st_size, 50000)

        contenido = p_html.read_text(encoding="utf-8")
        self.assertIn("SAT-Incendios PY", contenido)
        self.assertIn("chartRiesgo", contenido)
        self.assertIn("leaflet", contenido.lower())

if __name__ == "__main__":
    unittest.main()
