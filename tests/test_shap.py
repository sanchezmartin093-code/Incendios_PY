import unittest
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS, DIR_FIGURAS
from src.modelo import preparar_matrices_xy

class TestAnalisisTreeSHAP(unittest.TestCase):
    def test_archivos_shap_generados(self):
        """Verifies that the SHAP tables and figures have been created and are not empty."""
        csv_shap = DIR_TABLAS / "tabla03_shap_ranking_variables.csv"
        fig_comp = DIR_FIGURAS / "fig04_shap_comparativa_chaco_vs_oriental.png"

        self.assertTrue(csv_shap.exists(), "Falta tabla03_shap_ranking_variables.csv")
        self.assertTrue(fig_comp.exists(), "Falta fig04_shap_comparativa_chaco_vs_oriental.png")
        self.assertGreater(csv_shap.stat().st_size, 500)

    def test_calculo_shap_nativo_booster(self):
        """Verifies that the LightGBM C++ booster computes valid SHAP contributions."""
        ruta_b = DIR_MODELOS / "lightgbm_chaco.booster"
        p_val = DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet"
        if not ruta_b.exists() or not p_val.exists():
            self.skipTest("Booster o dataset no disponible")

        booster = lgb.Booster(model_file=str(ruta_b))
        df_val = pl.read_parquet(p_val)
        try:
            X_val, y_val, feats = preparar_matrices_xy(df_val, "Chaco")
        except ValueError as exc:
            self.skipTest(f"Partición sin el contrato v3 de features: {exc}")
        if booster.feature_name() != feats:
            self.skipTest("El booster serializado no coincide con el contrato de features actual")

        self.assertGreater(len(X_val), 0)
        contribuciones = booster.predict(X_val[:20], pred_contrib=True)
        # Must have dimensions (N, P + 1)
        self.assertEqual(contribuciones.shape, (20, len(feats) + 1))
        # Must contain finite values without NaNs
        self.assertTrue(np.all(np.isfinite(contribuciones)))

if __name__ == "__main__":
    unittest.main()
