"""G0-Formosa sensitivity: H-B (class 6 as wetland) and H-C (mask class 6)."""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_MODELOS, DIR_PROCESSED
from src.espaciotemporal import calcular_indice_ecotono
from src.metricas import bootstrap_diferencia_roc, calcular_umbral_optimo_f_beta
from src.modelo import preparar_matrices_xy, predecir_probabilidad

OUT = DIR_PROCESSED / "formosa_g0"
AUD = ROOT / "auditoria" / "G0"
MOD = DIR_MODELOS / "calibradores"


def _hb(df):
    shift = pl.col("pct_clase6").clip(0, 100)
    df = df.with_columns([
        (pl.col("pct_bosque_seco") - shift).clip(0, 100).alias("pct_bosque_seco"),
        (pl.col("pct_humedal_palmar") + shift).clip(0, 100).alias("pct_humedal_palmar"),
    ])
    return df.with_columns(calcular_indice_ecotono(df))


def _metrics(y, p, cut):
    yb = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f05": fbeta_score(y, yb, beta=0.5, zero_division=0),
    }


def _run(df, label, out_rows):
    va = df.filter(pl.col("anio") == 2023)
    te = df.filter(pl.col("anio").is_in([2024, 2025]))
    import lightgbm as lgb
    for name, mode, bpath, cpath, upath in (
        ("M2", "ignicion", DIR_MODELOS / "lightgbm_ignicion_chaco.booster", MOD / "calibrador_ignicion_chaco.joblib", MOD / "umbrales_ignicion_chaco.json"),
        ("M3", "hibrido", DIR_MODELOS / "lightgbm_hibrido_chaco.booster", MOD / "calibrador_hibrido_chaco.joblib", MOD / "umbrales_hibrido_chaco.json"),
    ):
        Xv, yv, _ = preparar_matrices_xy(va, "Chaco", modo=mode)
        Xt, yt, _ = preparar_matrices_xy(te, "Chaco", modo=mode)
        b = lgb.Booster(model_file=str(bpath))
        from src.modelo import calibrar_probabilidades_isotonica
        cal = calibrar_probabilidades_isotonica(b, Xv, yv, metodo="platt_monotono")
        pv = predecir_probabilidad(b, cal, Xv); pt = predecir_probabilidad(b, cal, Xt)
        cut, _ = calcular_umbral_optimo_f_beta(yv, pv, beta=0.5)
        for year in ("2024", "2025", "pooled"):
            mask = np.ones(len(yt), bool) if year == "pooled" else (te["anio"].to_numpy().astype(str) == year)
            out_rows.append({"variant": label, "model": name, "year": year, **_metrics(yt[mask], pt[mask], cut), "cutoff": cut, "n": int(mask.sum()), "pos": int(yt[mask].sum())})


def main() -> None:
    df = pl.read_parquet(OUT / "formosa_dataset.parquet")
    rows = []
    _run(df, "H-A", rows)
    _run(_hb(df), "H-B", rows)
    _run(df.filter(pl.col("pct_clase6") < 50.0), "H-C", rows)
    out = pl.DataFrame(rows)
    out.write_csv(AUD / "sensibilidad_armonizacion_g0.csv")
    print(out.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
