"""
Script 02e: calculates PSE and input variables for the INPE Fire Risk.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from src.constantes import DIR_PROCESSED
from src.inpe import pse_desde_lluvia_diaria


def main():
    clima = pl.read_parquet(
        DIR_PROCESSED / "clima_era5_diario.parquet",
        columns=["nodo_clima_id", "fecha", "precipitacion_mm", "temp_max_c", "vpd_max_kpa"],
    ).sort(["nodo_clima_id", "fecha"])
    salida = []
    for (nodo,), sub in clima.group_by("nodo_clima_id", maintain_order=True):
        p = sub["precipitacion_mm"].to_numpy()
        tmax = sub["temp_max_c"].to_numpy()
        vpd = sub["vpd_max_kpa"].to_numpy()
        es = 0.61078 * np.exp((17.27 * tmax) / (tmax + 237.3))
        rh = np.clip(100.0 * (es - vpd) / es, 1.0, 100.0)
        pse = pse_desde_lluvia_diaria(p)
        for f, pv, rv, tv in zip(sub["fecha"].to_list(), pse, rh, tmax):
            salida.append({
                "nodo_clima_id": int(nodo),
                "fecha": f,
                "pse": float(pv),
                "rh_min_pct": float(rv),
                "temp_max_c": float(tv),
            })
    out = pl.DataFrame(salida)
    ruta = DIR_PROCESSED / "inpe_diario.parquet"
    out.write_parquet(ruta, compression="zstd")
    print(out.describe())
    print(f"✓ Guardado: {ruta}")


if __name__ == "__main__":
    main()
