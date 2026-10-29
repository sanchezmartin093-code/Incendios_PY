"""G0-Formosa: frozen transport (G0-Z), conditional recalibration (G0-C)
and regional retraining (G0-R).
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    fbeta_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_MODELOS, DIR_PROCESSED
from src.metricas import (
    bootstrap_diferencia_roc, calcular_bootstrap_ci_roc,
    calcular_expected_calibration_error, calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    CalibradorPlattMonotono, calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm, preparar_matrices_xy, predecir_probabilidad,
)

OUT = DIR_PROCESSED / "formosa_g0"
MOD = DIR_MODELOS
AUD = ROOT / "auditoria" / "G0"
AUD.mkdir(parents=True, exist_ok=True)

BOOSTERS = {
    "M2": ("ignicion", MOD / "lightgbm_ignicion_chaco.booster", MOD / "calibradores" / "calibrador_ignicion_chaco.joblib", MOD / "calibradores" / "umbrales_ignicion_chaco.json"),
    "M3": ("hibrido", MOD / "lightgbm_hibrido_chaco.booster", MOD / "calibradores" / "calibrador_hibrido_chaco.joblib", MOD / "calibradores" / "umbrales_hibrido_chaco.json"),
}


def _metrics(y, p, cut):
    yb = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(calcular_expected_calibration_error(y, p)),
        "cutoff": float(cut), "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f05": float(fbeta_score(y, yb, beta=0.5, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "prevalence": float(np.mean(y)),
    }


def _eval_name(name, mode, model_kind, val, test):
    import lightgbm as lgb
    Xv, yv, feats = preparar_matrices_xy(val, "Chaco", modo=mode)
    Xt, yt, _ = preparar_matrices_xy(test, "Chaco", modo=mode)
    b = lgb.Booster(model_file=str(BOOSTERS[name][1]))
    if model_kind == "Z":
        cal = joblib.load(BOOSTERS[name][2])
        cut = json.loads(BOOSTERS[name][3].read_text())["c_f05"]
    elif model_kind == "C":
        cal = calibrar_probabilidades_isotonica(b, Xv, yv, metodo="platt_monotono")
        pv = predecir_probabilidad(b, cal, Xv)
        cut, _ = calcular_umbral_optimo_f_beta(yv, pv, beta=0.5)
    else:
        raise ValueError(model_kind)
    pv = predecir_probabilidad(b, cal, Xv)
    pt = predecir_probabilidad(b, cal, Xt)
    return pv, pt, yv, yt, cut, feats, b, cal


def main() -> None:
    tr = pl.read_parquet(OUT / "formosa_train_ignicion.parquet")
    va = pl.read_parquet(OUT / "formosa_val_ignicion.parquet")
    te = pl.read_parquet(OUT / "formosa_test_ignicion.parquet")
    excl = pl.read_parquet(OUT / "py_cells_excluir_buffer_formosa.parquet")["cell_id"].to_list()
    py = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    py = py.filter(~pl.col("cell_id").is_in(excl)).filter(pl.col("target_ignicion").is_not_null())
    print(f"Formosa train/val/test: {tr.height}/{va.height}/{te.height}; PY train sin buffer: {py.height}")

    all_rows = []
    pred_rows = []
    models = {}
    for arm in ("Z", "C"):
        for name in ("M2", "M3"):
            pv, pt, yv, yt, cut, feats, b, cal = _eval_name(name, BOOSTERS[name][0], arm, va, te)
            models[(arm, name)] = (b, cal, cut, feats)
            for sub, year in ((pt, "2024"), (pt, "2025"), (pt, "pooled")):
                mask = np.ones(len(yt), bool) if year == "pooled" else (te["anio"].to_numpy().astype(str) == year)
                m = _metrics(yt[mask], sub[mask], cut)
                all_rows.append({"arm": f"G0-{arm}", "model": name, "year": year, **m})
            pred_rows.append(pl.DataFrame({
                "arm": f"G0-{arm}", "model": name, "cell_id": te["cell_id"],
                "fecha": te["fecha"], "y": yt, "p": pt,
            }))

    # G0-R: train pooled Paraguay + Formosa 2018-2022; validate Formosa 2023.
    common = [c for c in tr.columns if c in py.columns]
    tr_all = pl.concat([py.select(common), tr.select(common)], how="diagonal_relaxed")
    print(f"G0-R train pooled: {tr_all.height}")
    for name in ("M2", "M3"):
        mode = BOOSTERS[name][0]
        Xtr, ytr, feats = preparar_matrices_xy(tr_all, "Chaco", modo=mode)
        Xv, yv, _ = preparar_matrices_xy(va, "Chaco", modo=mode)
        Xt, yt, _ = preparar_matrices_xy(te, "Chaco", modo=mode)
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xv, yv, feats, scale_pos_weight=1.0, learning_rate=0.03,
            num_leaves=23, min_child_samples=20, num_boost_round=45,
            early_stopping_rounds=None, semilla=42,
        )
        cal = calibrar_probabilidades_isotonica(b, Xv, yv, metodo="platt_monotono")
        pv = predecir_probabilidad(b, cal, Xv); pt = predecir_probabilidad(b, cal, Xt)
        cut, _ = calcular_umbral_optimo_f_beta(yv, pv, beta=0.5)
        models[("R", name)] = (b, cal, cut, feats)
        for year in ("2024", "2025", "pooled"):
            mask = np.ones(len(yt), bool) if year == "pooled" else (te["anio"].to_numpy().astype(str) == year)
            all_rows.append({"arm": "G0-R", "model": name, "year": year, **_metrics(yt[mask], pt[mask], cut)})
        pred_rows.append(pl.DataFrame({"arm": "G0-R", "model": name, "cell_id": te["cell_id"], "fecha": te["fecha"], "y": yt, "p": pt}))

    # Paired M3-M2 tests per arm/year.
    specs = []
    for arm in ("G0-Z", "G0-C", "G0-R"):
        for year in ("2024", "2025", "pooled"):
            a = pl.concat([p for p in pred_rows if p["arm"][0] == arm and p["model"][0] == "M2"], how="vertical")
            b = pl.concat([p for p in pred_rows if p["arm"][0] == arm and p["model"][0] == "M3"], how="vertical")
            if year != "pooled":
                a = a.filter(pl.col("fecha").str.slice(0, 4) == year)
                b = b.filter(pl.col("fecha").str.slice(0, 4) == year)
            y = a["y"].to_numpy().astype(int)
            d, p = bootstrap_diferencia_roc(y, b["p"].to_numpy(), a["p"].to_numpy(), n_bootstrap=1000, block_ids=a["fecha"].to_numpy())
            specs.append({"arm": arm, "year": year, "delta_roc_m3_m2": round(float(d), 4), "p": round(float(p), 4), "n": len(y), "pos": int(y.sum())})

    pred = pl.concat(pred_rows, how="vertical")
    pred.write_parquet(AUD / "predicciones_g0.parquet", compression="zstd")
    metrics = pl.DataFrame(all_rows).with_columns([pl.col(c).round(4) for c in ["roc_auc", "pr_auc", "brier", "ece", "cutoff", "precision", "recall", "f05", "prevalence"]])
    metrics.write_csv(AUD / "metricas_g0.csv")
    pl.DataFrame(specs).write_csv(AUD / "delta_roc_g0.csv")
    meta = {"rows": len(all_rows), "deltas": specs, "train_pooled": tr_all.height, "excluded_py_cells": len(excl)}
    (AUD / "resultados_g0.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics)
    print(pl.DataFrame(specs))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
