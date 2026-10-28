"""Script 66 (R.9): evaluacion con muestra sin gate.

Usa la muestra `targets_extra/sin_gate` (1:10, sin filtro biofisico) y la
compara con el baseline gateado. La muestra fue generada por scripts/64.
"""

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.modelo import entrenar_modelo_lightgbm, preparar_matrices_xy


def _fwi(df):
    grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"])
    fwi = pl.read_parquet(DIR_PROCESSED / "fwi_completo_diario.parquet", columns=["nodo_clima_id", "fecha", "fwi"])
    return df.join(grilla, on="cell_id", how="left").join(fwi, on=["nodo_clima_id", "fecha"], how="left")


def _eval(region, tr, va, te):
    out = {}
    for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
        Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
        Xva, yva, _ = preparar_matrices_xy(va, region, modo=modo)
        Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, feats,
            scale_pos_weight=1.0 if region == "Chaco" else 2.0,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            num_boost_round=45 if region == "Chaco" else 60,
            early_stopping_rounds=None, semilla=42,
        )
        out[nombre] = float(roc_auc_score(yte, b.predict(Xte)))
    y = te["target_ignicion"].to_numpy().astype(int)
    out["FWI"] = float(roc_auc_score(y, te["fwi"].to_numpy())) if te["fwi"].null_count() == 0 else float("nan")
    return out


def main():
    base = Path(DIR_PROCESSED) / "targets_extra" / "sin_gate"
    tr = _fwi(pl.read_parquet(base / "train_ignicion.parquet"))
    va = _fwi(pl.read_parquet(base / "val_ignicion.parquet"))
    te = _fwi(pl.read_parquet(base / "test_ignicion.parquet"))
    base_te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
    rows = []
    for region in ("Chaco", "Oriental"):
        b0 = base_te.filter(pl.col("region") == region)
        n0, p0 = b0.height, int(b0["target_ignicion"].sum())
        tr_r = tr.filter(pl.col("region") == region)
        va_r = va.filter(pl.col("region") == region)
        te_r = te.filter(pl.col("region") == region)
        m = _eval(region, tr_r, va_r, te_r)
        rows.append({
            "region": region, "variant": "sin gate", "retention_rows": round(te_r.height / n0, 3),
            "retention_pos": round(int(te_r["target_ignicion"].sum()) / p0, 3),
            "n_train": tr_r.height, "n_test": te_r.height,
            "m3_roc": round(m["M3"], 4), "m2_roc": round(m["M2"], 4),
            "fwi_roc": round(m["FWI"], 4), "delta_m3_m2": round(m["M3"] - m["M2"], 4),
        })
        print(region, rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DIR_TABLAS / "tablaS31_sin_gate.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
