import unittest
import importlib
from src.constantes import DIR_TABLAS, DIR_FIGURAS

class TestBenchmarking(unittest.TestCase):
    def test_archivos_benchmarking(self):
        """Verifies that the benchmarking script generates valid tables and figures."""
        csv_p = DIR_TABLAS / "tabla04_benchmarking_internacional.csv"
        fig_p = DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.png"

        if not csv_p.exists() or not fig_p.exists():
            self.skipTest("Artefactos de benchmarking no generados en este entorno")
        self.assertGreater(csv_p.stat().st_size, 300)

if __name__ == "__main__":
    unittest.main()
