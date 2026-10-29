"""G0-Formosa: 20/25/30 km t-1 buffer sensitivity for G0-C."""

from pathlib import Path
import sys

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_MODELOS, DIR_PROCESSED
from src.metricas import bootstrap_diferencia_roc, calcular_umbral_optimo_f_beta
from src.modelo import calibrar_probabilidades_isotonica, preparar_matrices_xy, predecir_probabilidad

OUT = DIR_PROCESSED / "formosa_g0"
AUD = ROOT / "auditoria" / "G0"
MOD = DIR_MODELOS / "calibradores"


def _metrics(y, p, cut):
    yb = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f05": fbeta_score(y, yb, beta=0.5, zero_division=0),
    }


def main() -> None:
    rows = []
    import lightgbm as lgb
    for tag, buffer_km in (("", 25), ("_b20", 20), ("_b30", 30)):
        va = pl.read_parquet(OUT / f"formosa_val_ignicion{tag}.parquet")
        te = pl.read_parquet(OUT / f"formosa_test_ignicion{tag}.parquet")
        for name, mode, bpath in (
            ("M2", "ignicion", DIR_MODELOS / "lightgbm_ignicion_chaco.booster"),
            ("M3", "hibrido", DIR_MODELOS / "lightgbm_hibrido_chaco.booster"),
        ):
            Xv, yv, _ = preparar_matrices_xy(va, "Chaco", modo=mode)
            Xt, yt, _ = preparar_matrices_xy(te, "Chaco", modo=mode)
            b = lgb.Booster(model_file=str(bpath))
            cal = calibrar_probabilidades_isotonica(b, Xv, yv, metodo="platt_monotono")
            pv = predecir_probabilidad(b, cal, Xv); pt = predecir_probabilidad(b, cal, Xt)
            cut, _ = calcular_umbral_optimo_f_beta(yv, pv, beta=0.5)
            for year in ("2024", "2025", "pooled"):
                mask = np.ones(len(yt), bool) if year == "pooled" else (te["anio"].to_numpy().astype(str) == year)
                rows.append({"buffer_km": buffer_km, "model": name, "year": year, **_metrics(yt[mask], pt[mask], cut), "cutoff": cut, "n": int(mask.sum()), "pos": int(yt[mask].sum())})
    out = pl.DataFrame(rows)
    out.write_csv(AUD / "sensibilidad_buffer_g0.csv")
    print(out.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
