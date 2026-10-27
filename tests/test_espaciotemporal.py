"""
Unit test for the src/espaciotemporal.py module.
Verifies:
1. Calculation of the forest-pasture ecotone index.
2. Extraction of UTM coordinates from cell_id.
3. Causal contagion variables at t-1 (distance, count, FRP).
4. Historical spatial propensity prior.
5. Comprehensive enrichment of datasets.
"""

import unittest
import numpy as np
import polars as pl
from src.espaciotemporal import (
    calcular_indice_ecotono,
    asegurar_coordenadas_utm,
    calcular_variables_contagio_lag,
    calcular_prior_espacial_historico,
    calcular_priors_causales,
    enriquecer_dataset_espaciotemporal
)

class TestEspaciotemporal(unittest.TestCase):

    def setUp(self):
        self.df_sample = pl.DataFrame({
            "cell_id": [1160007419, 5840007451],
            "region": ["Chaco", "Oriental"],
            "fecha": ["2026-10-18", "2026-10-18"],
            "pct_pastura": [50.0, 20.0],
            "pct_bosque_seco": [50.0, 0.0],
            "pct_bosque_humedo": [0.0, 80.0],
            "pct_sabana_pastizal": [0.0, 0.0],
            "pct_humedal_palmar": [0.0, 0.0]
        })

        self.focos_sample = pl.DataFrame({
            "cell_id": [1160007419],
            "acq_date": ["2026-10-17"],
            "sum_frp_dia": [15.5]
        })

    def test_calcular_indice_ecotono(self):
        s_ecotono = calcular_indice_ecotono(self.df_sample)
        # Chaco: 4 * 0.5 * 0.5 = 1.0 (maximum fragmentation)
        self.assertAlmostEqual(s_ecotono[0], 1.0, places=4)
        # Oriental: 4 * 0.8 * 0.2 = 0.64
        self.assertAlmostEqual(s_ecotono[1], 0.64, places=4)

    def test_asegurar_coordenadas_utm(self):
        coords = asegurar_coordenadas_utm(self.df_sample)
        self.assertEqual(coords.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(coords)))

    def test_calcular_variables_contagio_lag(self):
        df_enr = calcular_variables_contagio_lag(self.df_sample, self.focos_sample)
        self.assertIn("dist_foco_t1_km", df_enr.columns)
        self.assertIn("focos_25km_t1", df_enr.columns)
        self.assertIn("frp_25km_t1", df_enr.columns)

        # Cell 0 exactly matches yesterday's hotspot (distance 0 km)
        self.assertAlmostEqual(df_enr["dist_foco_t1_km"][0], 0.0, places=2)
        self.assertEqual(df_enr["focos_25km_t1"][0], 1.0)
        self.assertAlmostEqual(df_enr["frp_25km_t1"][0], 15.5, places=2)

    def test_calcular_prior_espacial_historico(self):
        coords_hist = np.array([[116500.0, 7419500.0]])
        df_prior = calcular_prior_espacial_historico(self.df_sample, coords_hist)
        self.assertIn("dist_foco_hist_km", df_prior.columns)
        self.assertIn("densidad_hist_25km", df_prior.columns)
        self.assertTrue((df_prior["dist_foco_hist_km"] >= 0.0).all())

    def test_enriquecer_dataset_espaciotemporal(self):
        coords_hist = np.array([[116500.0, 7419500.0]])
        df_full = enriquecer_dataset_espaciotemporal(self.df_sample, self.focos_sample, coords_hist)
        columnas_esperadas = [
            "indice_ecotono",
            "dist_foco_t1_km",
            "focos_25km_t1",
            "frp_25km_t1",
            "dist_foco_hist_km",
            "densidad_hist_25km"
        ]
        for col in columnas_esperadas:
            self.assertIn(col, df_full.columns)

    def test_priors_causales_excluyen_mismo_mes_y_futuro(self):
        df = pl.DataFrame({
            "cell_id": [1160007419, 5840007451],
            "region": ["Chaco", "Oriental"],
            "fecha": ["2026-10-18", "2026-10-18"],
            "pct_pastura": [50.0, 20.0],
            "pct_bosque_seco": [50.0, 0.0],
            "pct_bosque_humedo": [0.0, 80.0],
            "pct_sabana_pastizal": [0.0, 0.0],
            "pct_humedal_palmar": [0.0, 0.0],
        })
        # Cell 0: only fire from the same month/day (must be excluded).
        # Cell 1: fire from August (before the October monthly cutoff).
        focos = pl.DataFrame({
            "cell_id": [1160007419, 5840007451],
            "acq_date": ["2026-10-18", "2026-08-01"],
            "sum_frp_dia": [50.0, 10.0],
        })
        out = calcular_priors_causales(df, focos, ventana_anios=3)
        c0 = out.filter(pl.col("cell_id") == 1160007419)
        c1 = out.filter(pl.col("cell_id") == 5840007451)
        self.assertEqual(int(c0["freq_hist_1km"][0]), 0)
        self.assertGreater(float(c0["dist_foco_hist_km"][0]), 50.0)
        self.assertEqual(int(c1["freq_hist_1km"][0]), 1)
        self.assertAlmostEqual(float(c1["dist_foco_hist_km"][0]), 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
