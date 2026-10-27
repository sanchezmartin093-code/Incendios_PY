import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "31_congelar_candidato.py"

spec = importlib.util.spec_from_file_location("congelar_candidato", SCRIPT)
congelar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(congelar)


class TestArtefactosCongelados(unittest.TestCase):
    def test_incluye_modelos_operativos_y_diagnostico(self):
        rutas = {
            p.relative_to(ROOT).as_posix()
            for p in congelar.required_artifacts()
        }
        for region in ("chaco", "oriental"):
            for modo in ("hibrido", "ignicion", "propagacion", "m1"):
                self.assertIn(f"modelos/lightgbm_{modo}_{region}.booster", rutas)
                self.assertIn(
                    f"modelos/calibradores/calibrador_{modo}_{region}.joblib", rutas
                )
                self.assertIn(
                    f"modelos/calibradores/umbrales_{modo}_{region}.json", rutas
                )

    def test_incluye_holdout_targets_y_lock(self):
        rutas = {
            p.relative_to(ROOT).as_posix()
            for p in congelar.required_artifacts()
        }
        self.assertIn("datos/processed/holdout_confirmacion/celdas_holdout.parquet", rutas)
        self.assertIn("datos/processed/holdout_confirmacion/celdas_buffer.parquet", rutas)
        self.assertIn("datos/processed/targets_poblacionales/train_ignicion.parquet", rutas)
        self.assertIn("datos/processed/targets_poblacionales/val_ignicion.parquet", rutas)
        self.assertIn("datos/processed/targets_poblacionales/test_ignicion.parquet", rutas)
        self.assertIn("modelos/calibradores/prevalencias_poblacionales_d2.json", rutas)
        self.assertIn("requirements-lock.txt", rutas)

    def test_no_hay_rutas_duplicadas(self):
        rutas = congelar.required_artifacts()
        self.assertEqual(len(rutas), len({p.resolve() for p in rutas}))
