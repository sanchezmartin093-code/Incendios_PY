import unittest
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS, DIR_FIGURAS
from src.modelo import preparar_matrices_xy

class TestAnalisisTreeSHAP(unittest.TestCase):
    def test_archivos_shap_generados(self):
        """Verifica que las tablas y figuras de SHAP hayan sido creadas y no estén vacías."""
        csv_shap = DIR_TABLAS / "tabla03_shap_ranking_variables.csv"
        md_shap = DIR_TABLAS / "tabla03_shap_ranking_variables.md"
        fig_comp = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.png"

        self.assertTrue(csv_shap.exists(), "Falta tabla03_shap_ranking_variables.csv")
        self.assertTrue(md_shap.exists(), "Falta tabla03_shap_ranking_variables.md")
        self.assertTrue(fig_comp.exists(), "Falta fig04_shap_comparativa_chaco_vs_oriental.png")
        self.assertGreater(csv_shap.stat().st_size, 500)

    def test_calculo_shap_nativo_booster(self):
        """Verifica que el booster C++ de LightGBM calcule contribuciones SHAP válidas."""
        ruta_b = DIR_MODELOS / "lightgbm_chaco.booster"
        p_val = DIR_PROCESSED / "val_dataset.parquet"
        if not ruta_b.exists() or not p_val.exists():
            self.skipTest("Booster o dataset no disponible")

        booster = lgb.Booster(model_file=str(ruta_b))
        df_val = pl.read_parquet(p_val)
        X_val, y_val, feats = preparar_matrices_xy(df_val, "Chaco")

        self.assertGreater(len(X_val), 0)
        contribuciones = booster.predict(X_val[:20], pred_contrib=True)
        # Debe tener dimensiones (N, P + 1)
        self.assertEqual(contribuciones.shape, (20, len(feats) + 1))
        # Debe contener valores finitos sin NaNs
        self.assertTrue(np.all(np.isfinite(contribuciones)))

if __name__ == "__main__":
    unittest.main()
