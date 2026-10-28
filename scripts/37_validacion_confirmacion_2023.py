"""Independent confirmation-holdout evaluation on 2023.

The confirmation blocks are defined from the grid only. Models are retrained
without confirmation cells and their 20-km buffer; the validation held-in cells
are used for calibration and cutoff selection, and the untouched confirmation
cells are evaluated once.
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
from src.espaciotemporal import agregar_features_antropicas, enriquecer_dataset_espaciotemporal
from src.holdout import design_confirmation_holdout
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


def _metricas(y, p, cutoff, evaluation):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) else 0.0
    return {
        "evaluacion": evaluation,
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
    }


def main():
    grid = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    train = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    val = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    focos = pl.read_parquet(DIR_PROCESSED / "focos_viirs_diarios.parquet")

    _, holdout, buffer = design_confirmation_holdout(grid)
    holdout_cells = set(holdout["cell_id"].to_list())
    buffer_cells = set(buffer["cell_id"].to_list())
    print(f"Confirmación: {len(holdout_cells):,} celdas; buffer: {len(buffer_cells):,}")

    train_e = agregar_features_antropicas(enriquecer_dataset_espaciotemporal(
        train, focos, usar_priors_causales=True, excluir_cell_ids=list(holdout_cells)
    ))
    val_e = agregar_features_antropicas(enriquecer_dataset_espaciotemporal(
        val, focos, usar_priors_causales=True, excluir_cell_ids=list(holdout_cells)
    ))

    train_esp = train_e.filter(~pl.col("cell_id").is_in(list(holdout_cells | buffer_cells)))
    val_heldin = val_e.filter(~pl.col("cell_id").is_in(list(holdout_cells)))
    val_confirm = val_e.filter(pl.col("cell_id").is_in(list(holdout_cells)))

    rows = []
    for region in ["Chaco", "Oriental"]:
        print(f"--- {region} ---")
        tr = train_esp.filter(pl.col("region") == region)
        vi = val_heldin.filter(pl.col("region") == region)
        vc = val_confirm.filter(pl.col("region") == region)
        feats_models = [
            ("M1_biofisico", FEATURES_M1_CHACO if region == "Chaco" else FEATURES_M1_ORIENTAL),
            ("M2_base", FEATURES_IGNICION_CHACO if region == "Chaco" else FEATURES_IGNICION_ORIENTAL),
            ("M3_hibrido", FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL),
        ]
        y_tr = tr["target_ignicion"].to_numpy().astype(np.float32)
        y_i = vi["target_ignicion"].to_numpy().astype(int)
        y_c = vc["target_ignicion"].to_numpy().astype(int)
        preds_i, preds_c = {}, {}
        for name, feats in feats_models:
            b = entrenar_modelo_lightgbm(
                tr.select(feats).to_numpy(), y_tr,
                vi.select(feats).to_numpy(), y_i.astype(np.float32), feats,
                scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                num_boost_round=45 if region == "Chaco" else 60,
                num_leaves=23 if region == "Chaco" else 31,
                min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None, semilla=42,
            )
            cal = calibrar_probabilidades_isotonica(b, vi.select(feats).to_numpy(), y_i, metodo="platt_monotono")
            p_i = predecir_probabilidad(b, cal, vi.select(feats).to_numpy())
            cutoff, _ = calcular_umbral_optimo_f_beta(y_i, p_i, beta=0.5)
            p_c = predecir_probabilidad(b, cal, vc.select(feats).to_numpy())
            preds_i[name], preds_c[name] = p_i, p_c
            for ev, y, p in [("heldin_2023", y_i, p_i), ("confirmacion_2023", y_c, p_c)]:
                m = _metricas(y, p, cutoff, ev)
                m.update({"region": region, "modelo": name, "n_train": len(tr), "n_val": len(vi)})
                rows.append(m)
            print(f"{name}: confirm ROC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f}")

        # Operational gated policy: cutoff selected on held-in validation.
        p_i_g = np.where(vi["freq_hist_1km"].to_numpy() <= 0, preds_i["M2_base"], preds_i["M3_hibrido"])
        cutoff_g, _ = calcular_umbral_optimo_f_beta(y_i, p_i_g, beta=0.5)
        p_c_g = np.where(vc["freq_hist_1km"].to_numpy() <= 0, preds_c["M2_base"], preds_c["M3_hibrido"])
        for ev, y, p in [("heldin_2023", y_i, p_i_g), ("confirmacion_2023", y_c, p_c_g)]:
            m = _metricas(y, p, cutoff_g, ev)
            m.update({"region": region, "modelo": "M3_M2_gated", "n_train": len(tr), "n_val": len(vi)})
            rows.append(m)
        print(f"M3/M2 gated: confirm ROC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    cols = ["region", "modelo", "evaluacion", "n", "n_pos", "prevalencia", "roc_auc", "pr_auc",
            "brier", "bss", "ece", "cutoff_f05", "precision", "recall", "f05", "fp", "fn", "n_train", "n_val"]
    out = out[cols]
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIR_TABLAS / "tablaS18_confirmacion_espacial_2023.csv", index=False)
    (DIR_TABLAS / "tablaS18_confirmacion_espacial_2023.md").write_text(
        "# Tabla Suplementaria S18. Validación espacial de confirmación 2023\n\n"
        "Modelos reentrenados excluyendo las celdas de confirmación y su buffer de 20 km. "
        "El cutoff se selecciona en celdas held-in de 2023 y se aplica a la confirmación.\n\n"
        + out.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
