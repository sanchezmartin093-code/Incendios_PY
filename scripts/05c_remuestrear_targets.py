"""
Script 05c: 1:10 resampling per D2 target over the labeled partitions.
Generates separate files for ignition and propagation.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl

from src.constantes import DIR_PROCESSED
from src.remuestreo import remuestrear_por_target


def main():
    salida = DIR_PROCESSED / "targets"
    salida.mkdir(parents=True, exist_ok=True)

    for particion in ["train", "val", "test"]:
        ruta = DIR_PROCESSED / f"{particion}_dataset.parquet"
        if not ruta.exists():
            raise FileNotFoundError(ruta)
        df = pl.read_parquet(ruta)

        for target, nombre in [
            ("target_ignicion", "ignicion"),
            ("target_propagacion", "propagacion"),
        ]:
            out = remuestrear_por_target(df, target)
            destino = salida / f"{particion}_{nombre}.parquet"
            out.write_parquet(destino, compression="zstd")
            n_pos = int((out[target] == 1).sum())
            n_neg = int((out[target] == 0).sum())
            print(
                f"{destino.name}: {out.height:,} filas | "
                f"pos={n_pos:,} neg={n_neg:,} ratio=1:{n_neg / max(n_pos, 1):.2f}"
            )


if __name__ == "__main__":
    main()
