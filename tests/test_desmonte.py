"""
Unit test for the causal Hansen annual-loss aggregation.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import rasterio
from rasterio.transform import from_origin


def _cargar_modulo():
    ruta = Path(__file__).resolve().parent.parent / "scripts" / "03b_descargar_desmonte_hansen.py"
    spec = importlib.util.spec_from_file_location("desmonte_hansen", ruta)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestDesmonteHansen(unittest.TestCase):
    def test_t1_y_t2_son_causales(self):
        mod = _cargar_modulo()

        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "hansen_test.tif"
            ancho = alto = 100
            data = np.zeros((alto, ancho), dtype=np.uint8)
            data[:, :33] = 23  # loss in 2023
            data[:, 33:] = 24  # loss in 2024
            with rasterio.open(
                ruta, "w", driver="GTiff", height=alto, width=ancho,
                count=1, dtype="uint8", crs="EPSG:32721",
                transform=from_origin(499500.0, 7501500.0, 30.0, 30.0),
            ) as dst:
                dst.write(data, 1)

            grilla = pl.DataFrame({
                "cell_id": [1, 2],
                "x_utm": [500000.0, 501000.0],
                "y_utm": [7501000.0, 7501000.0],
            })
            tabla = mod.procesar_desmonte_anual(grilla, [ruta])
            self.assertEqual(tabla.filter(pl.col("cell_id") == 1)["anio"].min(), 2018)
            self.assertEqual(tabla.filter(pl.col("cell_id") == 1)["anio"].max(), 2026)

            c1_2024 = tabla.filter((pl.col("cell_id") == 1) & (pl.col("anio") == 2024)).row(0, named=True)
            self.assertGreater(c1_2024["pct_desmonte_t1"], 90.0)
            self.assertEqual(c1_2024["pct_desmonte_t2"], 0.0)

            c2_2025 = tabla.filter((pl.col("cell_id") == 2) & (pl.col("anio") == 2025)).row(0, named=True)
            self.assertGreater(c2_2025["pct_desmonte_t1"], 90.0)

            c2_2024 = tabla.filter((pl.col("cell_id") == 2) & (pl.col("anio") == 2024)).row(0, named=True)
            self.assertEqual(c2_2024["pct_desmonte_t1"], 0.0)

            c2_2026 = tabla.filter((pl.col("cell_id") == 2) & (pl.col("anio") == 2026)).row(0, named=True)
            self.assertGreater(c2_2026["pct_desmonte_t2"], 90.0)


if __name__ == "__main__":
    unittest.main()
