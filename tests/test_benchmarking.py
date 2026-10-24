import unittest
import importlib
from src.constantes import DIR_TABLAS, DIR_FIGURAS

class TestBenchmarking(unittest.TestCase):
    def test_archivos_benchmarking(self):
        """Verifica que el script de benchmarking genere tablas y figuras válidas."""
        csv_p = DIR_TABLAS / "tabla04_benchmarking_internacional.csv"
        md_p = DIR_TABLAS / "tabla04_benchmarking_internacional.md"
        fig_p = DIR_FIGURAS / "fig05_benchmarking_curvas_roc_pr.png"

        self.assertTrue(csv_p.exists(), "Falta tabla04_benchmarking_internacional.csv")
        self.assertTrue(md_p.exists(), "Falta tabla04_benchmarking_internacional.md")
        self.assertTrue(fig_p.exists(), "Falta fig05_benchmarking_curvas_roc_pr.png")
        self.assertGreater(csv_p.stat().st_size, 300)

if __name__ == "__main__":
    unittest.main()
