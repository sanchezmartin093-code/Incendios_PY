"""
Unit tests for the clima.py module (meteorological mesh, cKDTree and FWI/CDD).
"""

import unittest
import numpy as np
import polars as pl
from src.clima import (
    generar_malla_nodos_clima,
    asignar_nodos_a_grilla,
    calcular_indices_fwi_y_sequedad
)

class TestClima(unittest.TestCase):
    def test_generar_malla_nodos(self):
        df_nodos = generar_malla_nodos_clima(resolucion_deg=0.5) # quick test resolution
        self.assertGreater(len(df_nodos), 100)
        self.assertIn("nodo_clima_id", df_nodos.columns)
        self.assertIn("x_utm", df_nodos.columns)
        self.assertIn("y_utm", df_nodos.columns)

    def test_asignar_nodos_ckdtree(self):
        # Synthetic 3-node mesh
        df_nodos = pl.DataFrame({
            "nodo_clima_id": [1, 2, 3],
            "x_utm": [100000.0, 200000.0, 300000.0],
            "y_utm": [7000000.0, 7000000.0, 7000000.0]
        })
        # Cell close to node 1
        df_grilla = pl.DataFrame({
            "cell_id": [10007000],
            "x_utm": [100200.0],
            "y_utm": [7000100.0]
        })

        grilla_asignada = asignar_nodos_a_grilla(df_grilla, df_nodos)
        self.assertEqual(grilla_asignada["nodo_clima_id"][0], 1)

    def test_calculo_cdd_y_fwi(self):
        # Synthetic 5-day series with drought and rain at the end
        df_clima_test = pl.DataFrame({
            "nodo_clima_id": [1, 1, 1, 1, 1],
            "fecha": ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05"],
            "temp_max_c": [35.0, 36.0, 38.0, 39.0, 22.0],
            "temp_min_c": [20.0, 21.0, 22.0, 24.0, 18.0],
            "precipitacion_mm": [0.0, 0.0, 0.0, 0.0, 25.0],
            "viento_max_kmh": [20.0, 25.0, 30.0, 40.0, 15.0],
            "rafaga_max_kmh": [30.0, 35.0, 45.0, 55.0, 20.0],
            "et0_mm": [5.0, 5.5, 6.0, 7.0, 2.0],
            "vpd_max_kpa": [3.0, 3.5, 4.0, 5.0, 0.5]
        })

        res = calcular_indices_fwi_y_sequedad(df_clima_test)
        
        # On day 4 (after 4 days without rain), CDD must be 4
        self.assertEqual(res["dias_secos_consecutivos"][3], 4)
        # On day 5 it rained 25 mm, CDD must reset to 0
        self.assertEqual(res["dias_secos_consecutivos"][4], 0)
        # 48h rain on day 5 must be 25.0 mm
        self.assertEqual(res["precipitacion_48h_mm"][4], 25.0)
        # FFMC must rise with heat and drought, and collapse on day 5
        self.assertGreater(res["ffmc"][3], res["ffmc"][0])
        self.assertLess(res["ffmc"][4], 60.0)
        # New hydrological/antecedent columns must exist and be finite.
        for col in ["swvl1_m3m3", "fdd_heladas_60d", "deficit_hidro_60d"]:
            self.assertIn(col, res.columns)
            self.assertTrue(
                np.isfinite(res[col].to_numpy().astype(float)).all(),
                f"{col} contiene valores no finitos",
            )
        self.assertTrue((res["fdd_heladas_60d"].to_numpy() >= 0.0).all())
        # The bucket fallback is used when no soil-moisture input is supplied.
        self.assertGreater(float(res["swvl1_m3m3"][0]), 0.0)

    def test_swvl1_observado_se_preserva(self):
        df = pl.DataFrame({
            "nodo_clima_id": [1, 1, 1],
            "fecha": ["2024-09-01", "2024-09-02", "2024-09-03"],
            "temp_max_c": [30.0, 31.0, 32.0],
            "temp_min_c": [18.0, 19.0, 20.0],
            "precipitacion_mm": [0.0, 0.0, 0.0],
            "viento_max_kmh": [10.0, 12.0, 15.0],
            "rafaga_max_kmh": [15.0, 18.0, 20.0],
            "et0_mm": [4.0, 4.5, 5.0],
            "vpd_max_kpa": [2.0, 2.5, 3.0],
            "swvl1_m3m3": [0.30, 0.28, np.nan],
        })
        res = calcular_indices_fwi_y_sequedad(df)
        self.assertAlmostEqual(float(res["swvl1_m3m3"][0]), 0.30, places=5)
        self.assertAlmostEqual(float(res["swvl1_m3m3"][1]), 0.28, places=5)
        self.assertTrue(np.isfinite(float(res["swvl1_m3m3"][2])))

    def test_continuidad_falla_con_gap(self):
        df_clima_test = pl.DataFrame({
            "nodo_clima_id": [1, 1, 1],
            "fecha": ["2024-09-01", "2024-09-02", "2024-09-04"],
            "temp_max_c": [35.0, 36.0, 38.0],
            "temp_min_c": [20.0, 21.0, 22.0],
            "precipitacion_mm": [0.0, 0.0, 0.0],
            "viento_max_kmh": [20.0, 25.0, 30.0],
            "rafaga_max_kmh": [30.0, 35.0, 45.0],
            "et0_mm": [5.0, 5.5, 6.0],
            "vpd_max_kpa": [3.0, 3.5, 4.0],
        })

        with self.assertRaises(ValueError):
            calcular_indices_fwi_y_sequedad(df_clima_test)

if __name__ == "__main__":
    unittest.main()
