"""Coupled t-1 persistence baseline on the combined fire sample.

This baseline is deliberately coupled: it is trained on a single binary target
(fire at t) using the same case-control ignition + propagation samples and adds
the three causal t-1 fire features. It is compared with the decoupled task
architecture (M3 for ignition and the propagation model for propagation) on
the same combined test rows. It is a comparator for the leakage discussion, not
an operational model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
)
from src.modelo import (
    FEATURES_HIBRIDO_CHACO,
    FEATURES_HIBRIDO_ORIENTAL,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)

T1 = ["dist_foco_t1_km", "focos_25km_t1", "frp_25km_t1"]


def _features(region: str) -> list[str]:
    base = FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL
    return list(dict.fromkeys(base + T1))


def _load(path: Path, task: str, region: str) -> pl.DataFrame:
    d = pl.read_parquet(path)
    d = d.filter(pl.col("region") == region).with_columns(pl.lit(task).alias("_task"))
    return d


def _metrics(y, p):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float(np.mean(y)),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
    }


def main():
    base = DIR_PROCESSED / "targets_poblacionales"
    rows = []
    for region in ["Chaco", "Oriental"]:
        print(f"--- {region} ---")
        tr_i = _load(base / "train_ignicion.parquet", "ignicion", region)
        tr_p = _load(base / "train_propagacion.parquet", "propagacion", region)
        va_i = _load(base / "val_ignicion.parquet", "ignicion", region)
        va_p = _load(base / "val_propagacion.parquet", "propagacion", region)
        te_i = _load(base / "test_ignicion.parquet", "ignicion", region)
        te_p = _load(base / "test_propagacion.parquet", "propagacion", region)
        tr = pl.concat([tr_i, tr_p], how="vertical_relaxed")
        va = pl.concat([va_i, va_p], how="vertical_relaxed")
        te = pl.concat([te_i, te_p], how="vertical_relaxed")
        feats = _features(region)
        Xtr = tr.select(feats).cast(pl.Float32).to_numpy()
        ytr = tr["fuego"].to_numpy().astype(np.float32)
        Xva = va.select(feats).cast(pl.Float32).to_numpy()
        yva = va["fuego"].to_numpy().astype(np.float32)
        Xte = te.select(feats).cast(pl.Float32).to_numpy()
        yte = te["fuego"].to_numpy().astype(int)
        print('rows train/val/test', len(tr), len(va), len(te), 'features', len(feats))
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, feats,
            scale_pos_weight=1.0 if region == "Chaco" else 2.0,
            num_boost_round=45 if region == "Chaco" else 60,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            early_stopping_rounds=None, semilla=42,
        )
        cal = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
        p_coupled = predecir_probabilidad(b, cal, Xte)

        # Decoupled comparator: M3 for ignition rows, propagation model for propagation rows.
        ign = pl.read_parquet(DIR_PROCESSED / "evaluacion_test_ignicion.parquet").filter(pl.col("region") == region)
        prop = pl.read_parquet(DIR_PROCESSED / "evaluacion_test_propagacion.parquet").filter(pl.col("region") == region)
        p_dec = np.concatenate([ign["probabilidad_hibrido"].to_numpy(), prop["probabilidad_propagacion"].to_numpy()])
        # Row order in concat is ignition then propagation.
        m_c = _metrics(yte, p_coupled)
        m_d = _metrics(yte, p_dec)
        rows.append({"region": region, "comparator": "coupled_t1", **m_c})
        rows.append({"region": region, "comparator": "decoupled_task_models", **m_d})
        print('coupled', m_c)
        print('decoupled', m_d)

    out = pd.DataFrame(rows)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIR_TABLAS / "tablaS19_coupled_baseline.csv", index=False)
    (DIR_TABLAS / "tablaS19_coupled_baseline.md").write_text(
        "# Tabla Suplementaria S19. Baseline acoplado t-1 vs arquitectura desacoplada\n\n"
        "Mismo conjunto de test combinado (ignición + propagación, 1:10 por tarea). "
        "El baseline acoplado se entrena con un único target de fuego e incorpora las tres "
        "variables de fuego t-1; la arquitectura desacoplada usa M3 en filas de ignición y "
        "el modelo de propagación en filas de propagación. No es un modelo operativo.\n\n"
        + out.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
