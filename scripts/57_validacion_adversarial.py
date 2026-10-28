"""Script 57 (R.7): validacion adversarial train vs test.

Entrena un LightGBM para separar filas de entrenamiento (2018-2022) de test
(2024-2025) usando las features de M3. Un AUC alto indica cambio de
distribucion; las importancias por ganancia identifican las variables que
mas contribuyen al shift.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.modelo import FEATURES_HIBRIDO_CHACO, FEATURES_HIBRIDO_ORIENTAL


def _datos(region):
    tr = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet").filter(pl.col("region") == region)
    te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet").filter(pl.col("region") == region)
    feats = FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL
    X = np.vstack([tr.select(feats).to_numpy(), te.select(feats).to_numpy()])
    z = np.r_[np.zeros(tr.height, dtype=int), np.ones(te.height, dtype=int)]
    return X, z, feats


def main():
    filas = []
    importancias = []
    for region in ("Chaco", "Oriental"):
        X, z, feats = _datos(region)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        gains = []
        for fold, (i_tr, i_va) in enumerate(skf.split(X, z), start=1):
            ds = lgb.Dataset(X[i_tr], label=z[i_tr], feature_name=feats, free_raw_data=False)
            dv = lgb.Dataset(X[i_va], label=z[i_va], reference=ds, feature_name=feats, free_raw_data=False)
            params = {
                "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                "min_child_samples": 50, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1,
                "lambda_l1": 0.2, "lambda_l2": 1.0, "num_threads": 4,
                "verbose": -1, "seed": 42,
            }
            b = lgb.train(params, ds, num_boost_round=100, valid_sets=[dv],
                          callbacks=[lgb.log_evaluation(period=0)])
            auc = float(roc_auc_score(z[i_va], b.predict(X[i_va])))
            aucs.append(auc)
            gains.append(b.feature_importance(importance_type="gain"))
            filas.append({"region": region, "fold": fold, "auc": round(auc, 4)})
            print(f"{region} fold {fold}: AUC {auc:.4f}", flush=True)
        g = np.mean(gains, axis=0)
        orden = np.argsort(-g)
        for rank, i in enumerate(orden[:10], start=1):
            importancias.append({
                "region": region, "rank": rank, "feature": feats[i],
                "gain": round(float(g[i]), 2),
                "gain_pct": round(100.0 * float(g[i] / g.sum()), 2),
            })
        print(f"{region}: adversarial AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")

    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS29_validacion_adversarial.csv", index=False)
    imp = pd.DataFrame(importancias)
    imp.to_csv(DIR_TABLAS / "tablaS29_validacion_adversarial_features.csv", index=False)
    _md(df, imp)


def _md(df, imp):
    resumen = df.groupby("region")["auc"].agg(["mean", "std"]).reset_index()
    lineas = [
        "| Region | Adversarial AUC (mean ± SD) |",
        "|---|---:|",
    ]
    for r in resumen.itertuples(index=False):
        lineas.append(f"| {r.region} | {r.mean:.4f} ± {r.std:.4f} |")
    lineas += ["", "Top features by gain:", "", "| Region | Rank | Feature | Gain (%) |", "|---|---:|---|---:|"]
    for r in imp.itertuples(index=False):
        lineas.append(f"| {r.region} | {r.rank} | `{r.feature}` | {r.gain_pct:.2f} |")
    (DIR_TABLAS / "tablaS29_validacion_adversarial.md").write_text(
        "# Table S29. Adversarial validation\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
