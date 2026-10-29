"""Script 55 (B.3): GBT con indices FWI/INPE + contexto local M3.

Entrena un LightGBM con las features de M3 mas el FWI completo y el riesgo
INPE local. Sirve como referencia intermedia: si iguala a M3, los indices
no aportan informacion adicional una vez que existe contexto local.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    bootstrap_diferencia_roc,
    calcular_bootstrap_ci_roc,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    cargar_modelo_y_calibrador,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

COBERTURA = [
    "pct_pastura", "pct_sabana_pastizal", "pct_bosque_seco",
    "pct_bosque_humedo", "pct_humedal_palmar",
]
PESOS = np.array([6.0, 3.0, 1.72, 1.72, 1.5])


def _enriquecer(df: pl.DataFrame) -> pl.DataFrame:
    grilla = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet",
        columns=["cell_id", "nodo_clima_id"],
    )
    fwi = pl.read_parquet(
        DIR_PROCESSED / "fwi_completo_diario.parquet",
        columns=["nodo_clima_id", "fecha", "fwi"],
    )
    inpe = pl.read_parquet(DIR_PROCESSED / "inpe_diario.parquet")
    x = (
        df.join(grilla, on="cell_id", how="left")
        .join(fwi, on=["nodo_clima_id", "fecha"], how="left")
        .join(inpe, on=["nodo_clima_id", "fecha"], how="left")
    )
    f = x.select(COBERTURA).to_numpy().astype(float)
    total = f.sum(axis=1)
    a = np.divide(f @ PESOS, total, out=np.zeros(len(x)), where=total > 0)
    pse = x["pse"].to_numpy().astype(float)
    madurez = np.clip(a * pse / 120.0, 0.0, 2.0)
    rb = 0.8 * (1.0 + np.sin(madurez * (np.pi / 2.0) - (np.pi / 2.0))) / 2.0
    fur = -0.006 * x["rh_min_pct"].to_numpy().astype(float) + 1.3
    ft = 0.02 * x["temp_max_c"].to_numpy().astype(float) + 0.4
    flat = 1.0 + np.abs(x["lat"].to_numpy().astype(float)) * 0.003
    riesgo = np.clip(rb * fur * ft * flat, 0.0, 1.0)
    return x.with_columns([
        pl.Series("fwi", x["fwi"].to_numpy()),
        pl.Series("inpe", riesgo),
    ])


def _metricas(y, p, cut):
    y_bin = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_bin).ravel()
    return {
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f05": float(fbeta_score(y, y_bin, beta=0.5, zero_division=0)),
        "fp": int(fp),
    }


def main():
    dir_t = DIR_PROCESSED / "targets_poblacionales"
    tr = _enriquecer(pl.read_parquet(dir_t / "train_ignicion.parquet"))
    va = _enriquecer(pl.read_parquet(dir_t / "val_ignicion.parquet"))
    te = _enriquecer(pl.read_parquet(dir_t / "test_ignicion.parquet"))
    filas = []
    for region in ("Chaco", "Oriental"):
        feats = (preparar_matrices_xy(te, region, modo="hibrido")[2] + ["fwi", "inpe"])
        Xtr, ytr, _ = preparar_matrices_xy(tr, region, modo="hibrido")
        Xva, yva, _ = preparar_matrices_xy(va, region, modo="hibrido")
        Xte, yte, _ = preparar_matrices_xy(te, region, modo="hibrido")
        Xtr = np.column_stack([Xtr, tr.filter(pl.col("region") == region)["fwi"].to_numpy(), tr.filter(pl.col("region") == region)["inpe"].to_numpy()])
        Xva = np.column_stack([Xva, va.filter(pl.col("region") == region)["fwi"].to_numpy(), va.filter(pl.col("region") == region)["inpe"].to_numpy()])
        Xte = np.column_stack([Xte, te.filter(pl.col("region") == region)["fwi"].to_numpy(), te.filter(pl.col("region") == region)["inpe"].to_numpy()])

        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, feats,
            scale_pos_weight=1.0 if region == "Chaco" else 2.0,
            learning_rate=0.03,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            num_boost_round=45 if region == "Chaco" else 60,
            early_stopping_rounds=None, semilla=42,
        )
        c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
        p = predecir_probabilidad(b, c, Xte)
        cut, _ = calcular_umbral_optimo_f_beta(yva, predecir_probabilidad(b, c, Xva), beta=0.5)
        m = _metricas(yte, p, cut)
        te_r = te.filter(pl.col("region") == region)
        lo, hi = calcular_bootstrap_ci_roc(yte, p, n_bootstrap=500, block_ids=te_r["fecha"].to_numpy())

        # M3 de referencia
        b3, c3 = cargar_modelo_y_calibrador(
            DIR_MODELOS / f"lightgbm_hibrido_{region.lower()}.booster",
            DIR_MODELOS / "calibradores" / f"calibrador_hibrido_{region.lower()}.joblib",
        )
        Xm3, _, _ = preparar_matrices_xy(te, region, modo="hibrido")
        pm3 = predecir_probabilidad(b3, c3, Xm3)
        delta, pval = bootstrap_diferencia_roc(
            yte, p, pm3, n_bootstrap=1000, block_ids=te_r["fecha"].to_numpy()
        )
        bs = float(brier_score_loss(yte, p))
        bs_clim = float(brier_score_loss(yte, np.full(len(yte), ytr.mean())))
        filas.append({
            "region": region,
            "task": "ignicion",
            "modelo": "M3 + FWI/INPE (contexto local)",
            "roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "roc_ci_95": f"[{lo:.3f} - {hi:.3f}]",
            "pr_auc": round(float(average_precision_score(yte, p)), 4),
            "brier_score": round(bs, 4),
            "brier_skill_score": round(1.0 - bs / (bs_clim + 1e-9), 4),
            "ece": round(float(calcular_expected_calibration_error(yte, p)), 4),
            "cutoff_f05": round(float(cut), 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f05_score": round(m["f05"], 4),
            "falsos_positivos": m["fp"],
            "delta_roc_vs_m3": round(float(delta), 4),
            "p_value": round(float(pval), 4),
        })
        print(region, filas[-1], flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS11b_contexto_local.csv", index=False)
    _md(df)
    print(df)


def _md(df):
    lineas = [
        "| Region | Model | ROC-AUC [95 % CI] | PR-AUC | Brier | BSS | ECE | Cutoff | Precision | Recall | F0.5 | FP | ΔROC vs M3 (p) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.modelo} | {r.roc_auc:.4f} {r.roc_ci_95} | {r.pr_auc:.4f} | "
            f"{r.brier_score:.4f} | {r.brier_skill_score:.4f} | {r.ece:.4f} | "
            f"{r.cutoff_f05:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f05_score:.4f} | "
            f"{r.falsos_positivos:,} | {r.delta_roc_vs_m3:+.4f} ({r.p_value:.3f}) |"
        )
    (DIR_TABLAS / "tablaS11b_contexto_local.md").write_text(
        "# Table S11 (continued). FWI/INPE plus local context\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
