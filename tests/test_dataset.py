"""
Unit tests for unified dataset assembly and Stage 1 Gating.
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
        self.assertTrue(
            list((DIR_PROCESSED / "dataset_unificado_candidatos").glob("dataset_*.parquet")),
            "Debe existir al menos una particion del dataset unificado",
        )

    def test_schema_duckdb_coherencia(self):
        if not (DIR_PROCESSED / "grilla_nacional_1km.parquet").exists():
            self.skipTest("Insumos de datos procesados no disponibles en entorno CI")
        con = duckdb.connect()
        # Verify that we can perform a 1-row join without type errors
        test_query = """
            SELECT 
                g.cell_id, c.pct_pastura, cli.temp_max_c
            FROM 'datos/processed/grilla_nacional_1km.parquet' g
            JOIN 'datos/processed/combustible_mapbiomas_1km.parquet' c
                ON g.cell_id = c.cell_id
            JOIN 'datos/processed/clima_era5_diario.parquet' cli
                ON g.nodo_clima_id = cli.nodo_clima_id
            JOIN read_parquet('datos/processed/dataset_unificado_candidatos/dataset_*.parquet') d
                ON g.cell_id = d.cell_id
            LIMIT 1
        """
        res = con.execute(test_query).df()
        self.assertEqual(len(res), 1)

if __name__ == "__main__":
    unittest.main()
