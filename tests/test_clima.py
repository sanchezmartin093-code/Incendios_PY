"""
Pruebas unitarias para el módulo clima.py (malla meteorológica, cKDTree y FWI/CDD).
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
        df_nodos = generar_malla_nodos_clima(resolucion_deg=0.5) # resolución de prueba rápida
        self.assertGreater(len(df_nodos), 100)
        self.assertIn("nodo_clima_id", df_nodos.columns)
        self.assertIn("x_utm", df_nodos.columns)
        self.assertIn("y_utm", df_nodos.columns)

    def test_asignar_nodos_ckdtree(self):
        # Malla sintética de 3 nodos
        df_nodos = pl.DataFrame({
            "nodo_clima_id": [1, 2, 3],
            "x_utm": [100000.0, 200000.0, 300000.0],
            "y_utm": [7000000.0, 7000000.0, 7000000.0]
        })
        # Celda cercana al nodo 1
        df_grilla = pl.DataFrame({
            "cell_id": [10007000],
            "x_utm": [100200.0],
            "y_utm": [7000100.0]
        })

        grilla_asignada = asignar_nodos_a_grilla(df_grilla, df_nodos)
        self.assertEqual(grilla_asignada["nodo_clima_id"][0], 1)

    def test_calculo_cdd_y_fwi(self):
        # Serie sintética de 5 días con sequía y lluvia al final
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
        
        # En el día 4 (tras 4 días sin lluvia), CDD debe ser 4
        self.assertEqual(res["dias_secos_consecutivos"][3], 4)
        # En el día 5 llovió 25 mm, CDD debe reiniciarse a 0
        self.assertEqual(res["dias_secos_consecutivos"][4], 0)
        # Lluvia 48h en el día 5 debe ser 25.0 mm
        self.assertEqual(res["precipitacion_48h_mm"][4], 25.0)
        # FFMC debe subir con calor y sequía, y desplomarse el día 5
        self.assertGreater(res["ffmc"][3], res["ffmc"][0])
        self.assertLess(res["ffmc"][4], 60.0)

if __name__ == "__main__":
    unittest.main()
