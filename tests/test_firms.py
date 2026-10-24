"""
Pruebas unitarias para el módulo firms.py (filtrado de calidad y agregación espacial O(1)).
"""

import unittest
import pandas as pd
import polars as pl
from src.firms import procesar_focos_firms_a_grilla, agregar_focos_diarios

class TestFIRMS(unittest.TestCase):
    def setUp(self):
        # Malla sintética con 2 celdas de prueba en UTM 21S (Asunción aprox)
        self.grilla_sintetica = pl.DataFrame({
            "cell_id": [4400007200, 4410007200],
            "departamento": ["ASUNCION", "ASUNCION"],
            "region": ["Oriental", "Oriental"]
        })

    def test_filtro_bow_tie_y_confianza(self):
        # Puntos de prueba sintéticos
        df_dummy = pd.DataFrame({
            "latitude": [-25.28, -25.29, -20.0, -25.28],
            "longitude": [-57.63, -57.62, -58.0, -57.63],
            "acq_date": ["2026-10-24", "2026-10-24", "2026-10-24", "2026-10-24"],
            "acq_time": [1400, 1400, 1400, 1400],
            "frp": [15.0, 5.0, 100.0, 8.0],
            "confidence": ["nominal", "low", "high", "nominal"],
            "daynight": ["D", "D", "D", "D"],
            "scan": [0.4, 0.4, 2.5, 0.5], # El 3ro tiene scan=2.5 (>1.5)
            "track": [0.4, 0.4, 2.0, 0.5]
        })

        # Al procesar con filtro de baja confianza y max_scan=1.5:
        # - Fila 1: pasa
        # - Fila 2: descartada por confidence='low'
        # - Fila 3: descartada por scan=2.5 (>1.5)
        # - Fila 4: pasa
        resultado = procesar_focos_firms_a_grilla(
            df_dummy,
            self.grilla_sintetica,
            max_scan_km=1.5,
            excluir_baja_confianza=True
        )

        # Solo deben sobrevivir las observaciones válidas dentro de la grilla
        self.assertLessEqual(len(resultado), 2)
        if len(resultado) > 0:
            self.assertTrue((resultado["scan"] <= 1.5).all())
            self.assertTrue((resultado["confidence"] != "low").all())

if __name__ == "__main__":
    unittest.main()
