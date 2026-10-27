"""
Script 05b: applies the D2 labels (primary ignition and propagation) to the
existing partitions. Does not resample; only adds target_ignicion and
target_propagacion. Saves a prior copy in datos/processed/legacy_v1/.
"""

from pathlib import Path
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl

from src.constantes import DIR_PROCESSED
from src.targets import definir_targets_d2


def main():
    path_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    if not path_focos.exists():
        raise FileNotFoundError(path_focos)
    focos = pl.read_parquet(path_focos, columns=["cell_id", "acq_date"])

    legacy = DIR_PROCESSED / "legacy_v1"
    legacy.mkdir(parents=True, exist_ok=True)

    for nombre in ["train_dataset.parquet", "val_dataset.parquet", "test_dataset.parquet"]:
        ruta = DIR_PROCESSED / nombre
        if not ruta.exists():
            raise FileNotFoundError(ruta)

        respaldo = legacy / nombre
        if not respaldo.exists():
            shutil.copy2(ruta, respaldo)
            print(f" ✓ Respaldo: {respaldo}")

        df = pl.read_parquet(ruta)
        out = definir_targets_d2(df, focos)
        out.write_parquet(ruta, compression="zstd")

        ign = out["target_ignicion"].value_counts().sort("target_ignicion", nulls_last=True).to_dicts()
        prop = out["target_propagacion"].value_counts().sort("target_propagacion", nulls_last=True).to_dicts()
        print(f"\n{nombre}: {out.height:,} filas")
        print(f"  target_ignicion: {ign}")
        print(f"  target_propagacion: {prop}")


if __name__ == "__main__":
    main()
