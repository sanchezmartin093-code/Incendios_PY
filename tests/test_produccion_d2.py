"""Integration test for the production D2 sampling plan in scripts/05d."""

import importlib.util
from pathlib import Path
import unittest

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "05d_muestrear_poblacional_d2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("muestreo_d2_05d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProduccionD2(unittest.TestCase):
    def test_ignicion_y_propagacion_en_plan_real(self):
        module = _load_module()
        cells, xs, ys = [], [], []
        # A: fire at t-1, origin. B: fire at t, 10 km away -> propagation.
        # C: fire at t, 40 km away -> primary ignition.
        cells.append(1); xs.append(0.0); ys.append(0.0)          # A
        cells.append(2); xs.append(10000.0); ys.append(0.0)      # B
        cells.append(3); xs.append(40000.0); ys.append(0.0)      # C
        # 10 non-fire cells inside the 25 km buffer -> propagation negatives.
        for i in range(10):
            cells.append(10 + i); xs.append(15000.0 + i * 500.0); ys.append(5000.0)
        # 10 non-fire cells far outside the buffer -> ignition negatives.
        for i in range(10):
            cells.append(100 + i); xs.append(60000.0 + i * 1000.0); ys.append(0.0)

        grilla = pl.DataFrame({
            "cell_id": cells,
            "x_utm": xs,
            "y_utm": ys,
            "region": ["Chaco"] * len(cells),
            "nodo_clima_id": [0] * len(cells),
        })
        combustible = pl.DataFrame({
            "cell_id": cells,
            "anio_fila": [2024] * len(cells),
            "es_combustible": [True] * len(cells),
        })
        clima = pl.DataFrame({
            "nodo_clima_id": [0, 0],
            "fecha": ["2024-01-01", "2024-01-02"],
            "vpd_max_kpa": [5.0, 5.0],
            "precipitacion_48h_mm": [0.0, 0.0],
        })
        focos = pl.DataFrame({
            "cell_id": [1, 2, 3],
            "acq_date": ["2024-01-01", "2024-01-02", "2024-01-02"],
            "sum_frp_dia": [1.0, 1.0, 1.0],
        })

        plan, _ = module._planificar(grilla, combustible, clima, focos)
        day2_ign = plan.filter(
            (pl.col("fecha") == "2024-01-02")
            & (pl.col("task") == "ignicion")
            & (pl.col("target") == 1)
        )["cell_id"].to_list()
        day2_prop = plan.filter(
            (pl.col("fecha") == "2024-01-02")
            & (pl.col("task") == "propagacion")
            & (pl.col("target") == 1)
        )["cell_id"].to_list()
        self.assertEqual(day2_ign, [3])
        self.assertEqual(day2_prop, [2])

        neg_ign = plan.filter(
            (pl.col("fecha") == "2024-01-02")
            & (pl.col("task") == "ignicion")
            & (pl.col("target") == 0)
        )["cell_id"].to_list()
        neg_prop = plan.filter(
            (pl.col("fecha") == "2024-01-02")
            & (pl.col("task") == "propagacion")
            & (pl.col("target") == 0)
        )["cell_id"].to_list()
        self.assertTrue(all(c >= 100 for c in neg_ign))
        # Cell 1 (A) is the previous fire cell; on day 2 it is a valid
        # propagation-domain negative because it lies within 25 km of t-1.
        self.assertTrue(all(c == 1 or 10 <= c < 20 for c in neg_prop))


if __name__ == "__main__":
    unittest.main()
