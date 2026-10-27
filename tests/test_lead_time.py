"""Regression tests for the t-1 lead-time feature shift."""

import importlib.util
from pathlib import Path
import unittest

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "21_evaluacion_lead_time.py"


def _load_lead_time_module():
    spec = importlib.util.spec_from_file_location("lead_time_21", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLeadTimeNoLeakage(unittest.TestCase):
    def test_antecedent_windows_are_shifted_to_t1(self):
        module = _load_lead_time_module()
        weather = {
            "temp_max_c": 30.0, "temp_min_c": 10.0, "precipitacion_mm": 0.0,
            "precipitacion_48h_mm": 0.0, "viento_max_kmh": 10.0,
            "vpd_max_kpa": 2.0, "dias_secos_consecutivos": 4,
            "ffmc": 88.0, "isi": 5.0, "interaccion_viento_vpd": 10.0,
            "swvl1_m3m3": 0.2, "fdd_heladas_60d": 99.0,
            "deficit_hidro_60d": 99.0,
        }
        prev = dict(weather)
        prev.update({
            "fecha": "2024-01-01", "nodo_clima_id": 1,
            "fdd_heladas_60d": 11.0,
            "deficit_hidro_60d": 50.0, "vpd_max_kpa": 1.0,
        })
        current = dict(weather)
        current.update({"fecha": "2024-01-02", "nodo_clima_id": 1})
        clima = pl.DataFrame([prev, current])

        df = pl.DataFrame([{
            "fecha": "2024-01-02", "nodo_clima_id": 1,
            "mes": 1, "dia_del_anio": 2,
            "pct_pastura_ganadera": 40.0, "pct_sabana_pastizal": 10.0,
            "pct_humedal_palmar": 50.0, "pct_desmonte_2a": 0.0,
            "dist_foco_hist_km": 20.0,
            **weather,
        }])

        out = module._reemplazar_por_t1(df, clima)
        self.assertAlmostEqual(float(out["fdd_heladas_60d"][0]), 11.0)
        self.assertAlmostEqual(float(out["deficit_hidro_60d"][0]), 50.0)
        self.assertAlmostEqual(float(out["riesgo_estero_seco"][0]), 25.0)
        self.assertAlmostEqual(float(out["combustible_curado_helada"][0]), 5.5)


if __name__ == "__main__":
    unittest.main()
