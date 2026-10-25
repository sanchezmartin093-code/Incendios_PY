"""
Pruebas unitarias para el ensamble del dataset unificado y el Gating de Etapa 1.
"""

import unittest
from pathlib import Path
import duckdb
from src.constantes import DIR_PROCESSED

class TestDatasetUnificado(unittest.TestCase):
    def test_archivos_insumos_presentes(self):
        if not (DIR_PROCESSED / "grilla_nacional_1km.parquet").exists():
            self.skipTest("Insumos de datos procesados no disponibles en entorno CI")
        for arch in ["grilla_nacional_1km.parquet", "combustible_mapbiomas_1km.parquet", 
                     "clima_era5_diario.parquet", "focos_viirs_diarios.parquet"]:
            p = DIR_PROCESSED / arch
            self.assertTrue(p.exists(), f"El archivo insumo {arch} debe existir")

    def test_schema_duckdb_coherencia(self):
        if not (DIR_PROCESSED / "grilla_nacional_1km.parquet").exists():
            self.skipTest("Insumos de datos procesados no disponibles en entorno CI")
        con = duckdb.connect()
        # Verificar que podamos hacer un join de 1 fila sin errores de tipos
        test_query = """
            SELECT 
                g.cell_id, c.pct_pastura, cli.temp_max_c
            FROM 'datos/processed/grilla_nacional_1km.parquet' g
            JOIN 'datos/processed/combustible_mapbiomas_1km.parquet' c
                ON g.cell_id = c.cell_id
            JOIN 'datos/processed/clima_era5_diario.parquet' cli
                ON g.nodo_clima_id = cli.nodo_clima_id
            LIMIT 1
        """
        res = con.execute(test_query).df()
        self.assertEqual(len(res), 1)

if __name__ == "__main__":
    unittest.main()
