"""
Script 27: Development spatial-block validation evaluated on 2023 only.

The block mask is the one already used by script 20 and is derived from the
grid only. The confirmation holdout is excluded by construction. This script
never reads test_*.parquet or the confirmation holdout.

Outputs:
- tablas/tabla08b_validacion_espacial_2023.csv
- tablas/tabla08b_comparaciones_pareadas_2023.csv
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.espaciotemporal import agregar_features_antropicas, enriquecer_dataset_espaciotemporal
from src.holdout import BLOQUE_M, BUFFER_M, development_holdout_blocks
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    FEATURES_HIBRIDO_CHACO,
    FEATURES_HIBRIDO_ORIENTAL,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
    FEATURES_M1_CHACO,
    FEATURES_M1_ORIENTAL,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)


def _block_id(x, y):
    return (np.floor(np.asarray(x) / BLOQUE_M).astype(np.int64)
            * 1_000_003
            + np.floor(np.asarray(y) / BLOQUE_M).astype(np.int64))


def _buffer_cells(holdout_cells, train_cells):
    ref = train_cells.filter(pl.col("cell_id").is_in(list(holdout_cells)))
    if ref.is_empty():
        return set()
    tree = cKDTree(ref.select(["x_utm", "y_utm"]).to_numpy())
    dist, _ = tree.query(train_cells.select(["x_utm", "y_utm"]).to_numpy(), k=1, workers=-1)
    ids = train_cells["cell_id"].to_numpy()
    return set(ids[dist < BUFFER_M].tolist())


def _metricas(y, p, cutoff, evaluacion, region, modelo, blocks):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    y_bin = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (y_bin == 1)).sum())
    fp = int(((y == 0) & (y_bin == 1)).sum())
    fn = int(((y == 1) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1 + 0.25) * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) else 0.0
    return {
        "evaluacion": evaluacion,
        "region": region,
        "modelo": modelo,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalencia": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float(y.mean()),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "cutoff_f05": round(float(cutoff), 6),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "fp": fp,
        "fn": fn,
        "n_bloques": int(len(np.unique(blocks))),
    }


def _delta_bootstrap(y, p1, p0, blocks, n_boot=1000, seed=42):
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1).astype(float)
    p0 = np.asarray(p0).astype(float)
    blocks = np.asarray(blocks)
    unicos, inv = np.unique(blocks, return_inverse=True)
    indices = [np.where(inv == i)[0] for i in range(len(unicos))]
    rng = np.random.default_rng(seed)
    d_auc, d_pr = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unicos), size=len(unicos))
        idx = np.concatenate([indices[j] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
        d_pr.append(average_precision_score(y[idx], p1[idx]) - average_precision_score(y[idx], p0[idx]))
    obs_auc = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    obs_pr = float(average_precision_score(y, p1) - average_precision_score(y, p0))
    if not d_auc:
        return obs_auc, np.nan, np.nan, obs_pr, np.nan, np.nan
    return (obs_auc, float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5)),
            obs_pr, float(np.percentile(d_pr, 2.5)), float(np.percentile(d_pr, 97.5)))


def main():
    print("=" * 72)
    print("VALIDACION ESPACIAL DE DESARROLLO 2023 (50 km / buffer 20 km)")
    print("=" * 72)
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    grid = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    train = pl.read_parquet(dir_t / "train_ignicion.parquet")
    val = pl.read_parquet(dir_t / "val_ignicion.parquet")
    focos = pl.read_parquet(DIR_PROCESSED / "focos_viirs_diarios.parquet")

    if val["fecha"].max() >= "2024":
        raise ValueError("val_ignicion contiene fechas >= 2024")

    holdout_blocks = development_holdout_blocks(grid)
    grid = grid.with_columns(
        pl.Series("block_id", _block_id(grid["x_utm"].to_numpy(), grid["y_utm"].to_numpy()))
    )
    holdout_cells = set(grid.filter(pl.col("block_id").is_in(list(holdout_blocks)))["cell_id"].to_list())
    train_cells = train.select(["cell_id", "x_utm", "y_utm"]).unique("cell_id")
    buffer_cells = _buffer_cells(holdout_cells, train_cells)
    print(f"Bloques desarrollo: {len(holdout_blocks)}; celdas holdout: {len(holdout_cells):,}; buffer train: {len(buffer_cells):,}")

    excluir = list(holdout_cells)
    train_e = agregar_features_antropicas(enriquecer_dataset_espaciotemporal(
        train, focos, usar_priors_causales=True, excluir_cell_ids=excluir))
    val_e = agregar_features_antropicas(enriquecer_dataset_espaciotemporal(
        val, focos, usar_priors_causales=True, excluir_cell_ids=excluir))

    train_e = train_e.with_columns(pl.Series("block_id", _block_id(train_e["x_utm"].to_numpy(), train_e["y_utm"].to_numpy())))
    val_e = val_e.with_columns(pl.Series("block_id", _block_id(val_e["x_utm"].to_numpy(), val_e["y_utm"].to_numpy())))

    train_esp = train_e.filter(~pl.col("cell_id").is_in(list(holdout_cells | buffer_cells)))
    val_esp = val_e.filter(~pl.col("cell_id").is_in(excluir))
    val_holdout = val_e.filter(pl.col("cell_id").is_in(excluir))
    val_heldin = val_e.filter(~pl.col("cell_id").is_in(excluir))

    filas, deltas = [], []
    for region in ["Chaco", "Oriental"]:
        print(f"\n--- {region.upper()} ---")
        tr = train_esp.filter(pl.col("region") == region)
        va = val_esp.filter(pl.col("region") == region)
        vh = val_holdout.filter(pl.col("region") == region)
        vi = val_heldin.filter(pl.col("region") == region)
        b_col = "pct_bosque_seco" if region == "Chaco" else "pct_bosque_humedo"
        modelos = [
            ("M1_biofisico", FEATURES_M1_CHACO if region == "Chaco" else FEATURES_M1_ORIENTAL),
            ("M2_base", FEATURES_IGNICION_CHACO if region == "Chaco" else FEATURES_IGNICION_ORIENTAL),
            ("M3_hibrido", FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL),
        ]
        y_tr = tr["target_ignicion"].to_numpy().astype(np.float32)
        y_va = va["target_ignicion"].to_numpy().astype(np.float32)
        y_h = vh["target_ignicion"].to_numpy().astype(int)
        y_i = vi["target_ignicion"].to_numpy().astype(int)
        print(f"train={len(tr):,} val={len(va):,} holdout={len(vh):,} heldin={len(vi):,}")

        preds_h, preds_i = {}, {}
        for nombre, feats in modelos:
            b = entrenar_modelo_lightgbm(
                tr.select(feats).to_numpy(), y_tr, va.select(feats).to_numpy(), y_va, feats,
                scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                num_boost_round=45 if region == "Chaco" else 60,
                num_leaves=23 if region == "Chaco" else 31,
                min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None, semilla=42,
            )
            cal = calibrar_probabilidades_isotonica(b, va.select(feats).to_numpy(), y_va, metodo="platt_monotono")
            p_va = predecir_probabilidad(b, cal, va.select(feats).to_numpy())
            cutoff, _ = calcular_umbral_optimo_f_beta(y_va, p_va, beta=0.5)
            p_h = predecir_probabilidad(b, cal, vh.select(feats).to_numpy())
            p_i = predecir_probabilidad(b, cal, vi.select(feats).to_numpy())
            preds_h[nombre], preds_i[nombre] = p_h, p_i
            for ev, y_ev, p_ev, df_ev in [("holdout_espacial_2023", y_h, p_h, vh), ("heldin_referencia_2023", y_i, p_i, vi)]:
                m = _metricas(y_ev, p_ev, cutoff, ev, region, nombre, df_ev["block_id"].to_numpy())
                m.update({"n_train": len(tr), "n_val": len(va)})
                filas.append(m)
            print(f"{nombre}: holdout ROC={filas[-2]['roc_auc']:.4f} PR={filas[-2]['pr_auc']:.4f}")

        for a, b in [("M2_base", "M3_hibrido"), ("M3_hibrido", "M1_biofisico"), ("M2_base", "M1_biofisico")]:
            d = _delta_bootstrap(y_h, preds_h[a], preds_h[b], vh["block_id"].to_numpy())
            deltas.append({
                "region": region, "comparacion": f"{a}_minus_{b}",
                "delta_roc_auc": round(d[0], 4), "delta_roc_ci95_low": round(d[1], 4),
                "delta_roc_ci95_high": round(d[2], 4), "delta_pr_auc": round(d[3], 4),
                "delta_pr_ci95_low": round(d[4], 4), "delta_pr_ci95_high": round(d[5], 4),
                "n_bootstrap": 1000, "agrupacion": "bloque_espacial",
            })

    df = pd.DataFrame(filas)
    cols = ["region", "modelo", "evaluacion", "n", "n_pos", "prevalencia", "roc_auc", "pr_auc",
            "brier", "bss", "ece", "cutoff_f05", "precision", "recall", "f05", "fp", "fn",
            "n_bloques", "n_train", "n_val"]
    df[cols].to_csv(DIR_TABLAS / "tabla08b_validacion_espacial_2023.csv", index=False)
    pd.DataFrame(deltas).to_csv(DIR_TABLAS / "tabla08b_comparaciones_pareadas_2023.csv", index=False)
    print("\nHoldout:")
    print(df[df["evaluacion"] == "holdout_espacial_2023"][["region", "modelo", "roc_auc", "pr_auc"]].to_string(index=False))
    print("\nDeltas:")
    print(pd.DataFrame(deltas).to_string(index=False))


if __name__ == "__main__":
    main()
