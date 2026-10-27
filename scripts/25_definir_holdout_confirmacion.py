"""Define the independent spatial confirmation holdout without using labels."""

from pathlib import Path
import hashlib
import json
import sys

import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_PROCESSED
from src.holdout import (
    BUFFER_M,
    FRACCION_CONFIRMACION,
    SAL_CONFIRMACION,
    design_confirmation_holdout,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    grid = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    blocks, holdout, buffer = design_confirmation_holdout(grid)
    out = DIR_PROCESSED / "holdout_confirmacion"
    out.mkdir(parents=True, exist_ok=True)
    blocks.write_parquet(out / "bloques.parquet", compression="zstd")
    holdout.write_parquet(out / "celdas_holdout.parquet", compression="zstd")
    buffer.write_parquet(out / "celdas_buffer.parquet", compression="zstd")

    meta = {
        "descripcion": "Holdout espacial de confirmacion, independiente de script 20",
        "sal": SAL_CONFIRMACION,
        "fraccion_confirmacion": FRACCION_CONFIRMACION,
        "buffer_m": BUFFER_M,
        "bloques_totales": blocks.height,
        "bloques_desarrollo": int(blocks["desarrollo"].sum()),
        "bloques_confirmacion": int(blocks["confirmacion"].sum()),
        "celdas_holdout": holdout.height,
        "celdas_buffer": buffer.height,
        "archivos": {
            "bloques": {"name": "bloques.parquet", "sha256": _sha256(out / "bloques.parquet")},
            "celdas_holdout": {"name": "celdas_holdout.parquet", "sha256": _sha256(out / "celdas_holdout.parquet")},
            "celdas_buffer": {"name": "celdas_buffer.parquet", "sha256": _sha256(out / "celdas_buffer.parquet")},
        },
        "uso": "Solo Fase 5; no usar para seleccion de modelo ni metricas de desarrollo.",
    }
    (out / "holdout_confirmacion.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
