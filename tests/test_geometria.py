"""
Consistency tests for the geometria.py module and O(1) spatial indexing.
"""

import unittest
from src.constantes import BBOX_PARAGUAY, CRS_PROYECTADO, RESOLUCION_GRILLA_METROS
from src.geometria import coordenadas_a_cell_id, cell_id_a_coordenadas_centro, generar_vecindad_3x3

class TestGeometria(unittest.TestCase):
    def test_bbox_paraguay_consistencia(self):
        self.assertLess(BBOX_PARAGUAY["min_lon"], BBOX_PARAGUAY["max_lon"])
        self.assertLess(BBOX_PARAGUAY["min_lat"], BBOX_PARAGUAY["max_lat"])
        self.assertLess(BBOX_PARAGUAY["min_lat"], -27.0)
        self.assertGreater(BBOX_PARAGUAY["max_lat"], -20.0)

    def test_parametros_grilla(self):
        self.assertEqual(RESOLUCION_GRILLA_METROS, 1000)
        self.assertEqual(CRS_PROYECTADO, "EPSG:32721")

    def test_indexacion_o1_bidireccional(self):
        x = 440350.0
        y = 7200780.0
        cell_id = coordenadas_a_cell_id(x, y, res_m=1000)
        cx, cy = cell_id_a_coordenadas_centro(cell_id, res_m=1000)
        self.assertAlmostEqual(cx, 440500.0, delta=1e-3)
        self.assertAlmostEqual(cy, 7200500.0, delta=1e-3)

    def test_vecindad_3x3_aritmetica(self):
        cell_id = coordenadas_a_cell_id(500000.0, 7000000.0, res_m=1000)
        vecinos = generar_vecindad_3x3(cell_id, res_m=1000)
        self.assertEqual(len(vecinos), 8)
        self.assertNotIn(cell_id, vecinos)

if __name__ == "__main__":
    unittest.main()
