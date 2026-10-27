"""
Script 19: D2 population prevalences by target, region and year.
Uses the counts from the D2 population sampling plan.
"""

from pathlib import Path
import sys
import json
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from src.constantes import DIR_PROCESSED, DIR_MODELOS


def main():
    ruta_conteos = DIR_PROCESSED / "conteos_poblacionales_d2.parquet"
    if not ruta_conteos.exists():
        raise FileNotFoundError(ruta_conteos)
    df = pl.read_parquet(ruta_conteos).with_columns(
        pl.col("fecha").str.slice(0, 4).cast(pl.Int32).alias("anio")
    )

    salida = {"ignicion": {}, "propagacion": {}}
    for task in ["ignicion", "propagacion"]:
        for region in ["Chaco", "Oriental"]:
            sub = df.filter((pl.col("task") == task) & (pl.col("region") == region))
            n_pos = int(sub["n_pos"].sum())
            n_neg = int(sub["n_neg"].sum())
            por_anio = {}
            for anio, s in sub.group_by("anio"):
                p = int(s["n_pos"].sum())
                n = int(s["n_neg"].sum())
                por_anio[str(int(anio[0]))] = {
                    "n_pos": p,
                    "n_neg": n,
                    "prevalencia": p / (p + n) if (p + n) else 0.0,
                }
            salida[task][region] = {
                "global": n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "por_anio": por_anio,
            }

    out = {
        "descripcion": "Prevalencia poblacional D2 por target, región y año.",
        "prevalencia_muestreo": 1.0 / 11.0,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": salida,
    }
    ruta = DIR_MODELOS / "calibradores" / "prevalencias_poblacionales_d2.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2)[:2500])
    print(f"✓ Guardado: {ruta}")


if __name__ == "__main__":
    main()
