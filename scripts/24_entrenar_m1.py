"""
Script 24: entrena y serializa el modelo biofísico puro M1 para el fallback de alerta.
"""

from pathlib import Path
import sys, json

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_MODELOS
from src.modelo import (
    FEATURES_M1_CHACO, FEATURES_M1_ORIENTAL,
    calibrar_probabilidades_isotonica, entrenar_modelo_lightgbm,
    guardar_modelo_y_calibrador, predecir_probabilidad,
)
from src.metricas import calcular_umbral_optimo_f_beta


def main():
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    tr = pl.read_parquet(dir_t / "train_ignicion.parquet")
    va = pl.read_parquet(dir_t / "val_ignicion.parquet")
    dir_cal = DIR_MODELOS / "calibradores"
    dir_cal.mkdir(parents=True, exist_ok=True)
    for region, feats in [("Chaco", FEATURES_M1_CHACO), ("Oriental", FEATURES_M1_ORIENTAL)]:
        trr = tr.filter(pl.col("region") == region)
        vaa = va.filter(pl.col("region") == region)
        Xtr = trr.select(feats).to_numpy(); ytr = trr["target_ignicion"].to_numpy().astype(float)
        Xva = vaa.select(feats).to_numpy(); yva = vaa["target_ignicion"].to_numpy().astype(float)
        spw = 1.0 if region == "Chaco" else 2.0
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, feats,
            scale_pos_weight=spw, num_boost_round=45 if region == "Chaco" else 60,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            early_stopping_rounds=None, semilla=42,
        )
        c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
        p = predecir_probabilidad(b, c, Xva)
        cutoff_f05, _ = calcular_umbral_optimo_f_beta(yva, p, beta=0.5)
        cutoff_y, _ = calcular_umbral_optimo_f_beta(yva, p, beta=0.5)
        nombre = region.lower()
        guardar_modelo_y_calibrador(
            b, c, DIR_MODELOS / f"lightgbm_m1_{nombre}.booster",
            dir_cal / f"calibrador_m1_{nombre}.joblib",
        )
        meta = {"region": region, "modo": "m1_biofisico", "c_f05": round(float(cutoff_f05), 4),
                "roc_auc_val": round(float(roc_auc_score(yva, p)), 4),
                "pr_auc_val": round(float(average_precision_score(yva, p)), 4)}
        (dir_cal / f"umbrales_m1_{nombre}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(region, meta)


if __name__ == "__main__":
    main()
