"""
Pruebas unitarias para el módulo firms.py (filtrado de calidad y agregación espacial O(1)).
"""

import unittest
import pandas as pd
import polars as pl
from pyproj import Transformer as PyprojTransformer

from src.constantes import CRS_GEOGRAFICO, CRS_PROYECTADO
from src.firms import procesar_focos_firms_a_grilla, agregar_focos_diarios
from src.geometria import coordenadas_a_cell_id

class TestFIRMS(unittest.TestCase):
    def setUp(self):
        transformer = PyprojTransformer.from_crs(
            CRS_GEOGRAFICO, CRS_PROYECTADO, always_xy=True
        )
        x_utm, y_utm = transformer.transform(-57.63, -25.28)
        cell_id = int(coordenadas_a_cell_id(x_utm, y_utm))
        self.grilla_sintetica = pl.DataFrame({
            "cell_id": [cell_id],
            "departamento": ["ASUNCION"],
            "region": ["Oriental"]
        })

    def test_filtro_bow_tie_y_confianza(self):
        # Puntos de prueba sintéticos
        df_dummy = pd.DataFrame({
            "latitude": [-25.28, -25.29, -20.0, -25.28, -25.28],
            "longitude": [-57.63, -57.62, -58.0, -57.63, -57.63],
            "acq_date": ["2026-10-24"] * 5,
            "acq_time": [1400, 1400, 1400, 1400, 1400],
            "frp": [15.0, 5.0, 100.0, 8.0, 3.0],
            "confidence": ["n", "l", "h", "n", "low"],
            "daynight": ["D"] * 5,
            "scan": [0.4, 0.4, 2.5, 0.5, 0.5],
            "track": [0.4, 0.4, 2.0, 0.5, 0.5]
        })

        resultado = procesar_focos_firms_a_grilla(
            df_dummy,
            self.grilla_sintetica,
            max_scan_km=1.5,
            excluir_baja_confianza=True
        )

        self.assertEqual(len(resultado), 2)
        self.assertTrue((resultado["scan"] <= 1.5).all())
        self.assertEqual(set(resultado["confidence"].to_list()), {"n"})

if __name__ == "__main__":
    unittest.main()
