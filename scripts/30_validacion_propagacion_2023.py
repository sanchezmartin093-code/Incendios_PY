"""Script 30: 2023 validation of the tactical propagation layer (no selection)."""

from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_MODELOS
from src.metricas import (
    calcular_brier_score, calcular_brier_skill_score,
    calcular_expected_calibration_error,
)
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy


def main():
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    val = pl.read_parquet(dir_t / "val_propagacion.parquet")
    if val["fecha"].max() >= "2024":
        raise ValueError("val_propagacion contiene fechas >= 2024")
    filas = []
    for region in ["Chaco", "Oriental"]:
        df = val.filter(pl.col("region") == region)
        b, c = cargar_modelo_y_calibrador(
            DIR_MODELOS / f"lightgbm_propagacion_{region.lower()}.booster",
            DIR_MODELOS / "calibradores" / f"calibrador_propagacion_{region.lower()}.joblib",
        )
        X, y, _ = preparar_matrices_xy(df, region, modo="propagacion")
        p = predecir_probabilidad(b, c, X)
        meta = json.loads((DIR_MODELOS / "calibradores" / f"umbrales_propagacion_{region.lower()}.json").read_text())
        cutoff = float(meta["c_f05"])
        yb = (p >= cutoff).astype(int)
        tp = int(((y == 1) & (yb == 1)).sum()); fp = int(((y == 0) & (yb == 1)).sum())
        fn = int(((y == 1) & (yb == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f05 = (1 + 0.25) * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) else 0.0
        filas.append({
            "region": region, "task": "propagacion", "anio": 2023,
            "n": int(len(y)), "n_pos": int(y.sum()), "prevalencia": round(float(y.mean()), 6),
            "cutoff_f05": round(cutoff, 6),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(calcular_brier_score(y, p), 4),
            "bss": round(calcular_brier_skill_score(y, p), 4),
            "ece": round(calcular_expected_calibration_error(y, p), 4),
            "precision": round(prec, 4), "recall": round(rec, 4), "f05": round(f05, 4),
            "fp": fp, "fn": fn,
        })
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tabla10_validacion_2023_propagacion.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
