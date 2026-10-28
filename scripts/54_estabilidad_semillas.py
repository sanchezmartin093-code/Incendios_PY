"""Script 54 (R.1): estabilidad de M2/M3 a semillas y capacidad.

5 semillas x 2 capacidades por region. La capacidad alta incrementa
num_leaves y num_boost_round en 20 %. Se reportan media, desviacion,
rango y retencion de signo de Delta(M3-M2).
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.modelo import (
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

SEMILLAS = (42, 7, 13, 21, 33)
CONFIGS = {
    "actual": {"factor": 1.0},
    "alta": {"factor": 1.2},
}
DATA = {}


def _cargar():
    tr = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    va = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
    for region in ("Chaco", "Oriental"):
        for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
            Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
            Xva, yva, _ = preparar_matrices_xy(va, region, modo=modo)
            Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
            DATA[(region, nombre)] = (Xtr, ytr, Xva, yva, Xte, yte, feats)


def _params(region, factor, semilla):
    leaves = int(round((23 if region == "Chaco" else 31) * factor))
    rounds = int(round((45 if region == "Chaco" else 60) * factor))
    return dict(
        scale_pos_weight=1.0 if region == "Chaco" else 2.0,
        learning_rate=0.03,
        num_leaves=leaves,
        min_child_samples=20 if region == "Chaco" else 10,
        num_boost_round=rounds,
        early_stopping_rounds=None,
        semilla=semilla,
    )


def _run(region, config, semilla):
    factor = CONFIGS[config]["factor"]
    out = []
    probs = {}
    for nombre in ("M2", "M3"):
        Xtr, ytr, Xva, yva, Xte, yte, feats = DATA[(region, nombre)]
        b = entrenar_modelo_lightgbm(Xtr, ytr, Xva, yva, feats, **_params(region, factor, semilla))
        c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
        p = predecir_probabilidad(b, c, Xte)
        probs[nombre] = p
        out.append({
            "region": region, "config": config, "semilla": semilla, "model": nombre,
            "roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "pr_auc": round(float(average_precision_score(yte, p)), 4),
        })
    delta = float(roc_auc_score(yte, probs["M3"]) - roc_auc_score(yte, probs["M2"]))
    out[-1]["delta_roc_m3_m2"] = round(delta, 4)
    out[0]["delta_roc_m3_m2"] = np.nan
    return out


def main():
    _cargar()
    tareas = [(r, c, s) for r in ("Chaco", "Oriental") for c in CONFIGS for s in SEMILLAS]
    filas = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futuros = {ex.submit(_run, *t): t for t in tareas}
        for fut in as_completed(futuros):
            filas.extend(fut.result())
            print("ok", futuros[fut], flush=True)

    df = pd.DataFrame(filas).sort_values(["region", "config", "semilla", "model"])
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR_TABLAS / "tablaS25_estabilidad_semillas.csv", index=False)
    resumen = _resumen(df)
    resumen.to_csv(DIR_TABLAS / "tablaS25_estabilidad_semillas_resumen.csv", index=False)
    _md(resumen)
    print(resumen.to_string(index=False))


def _resumen(df):
    filas = []
    for region in ("Chaco", "Oriental"):
        for config in CONFIGS:
            base = df[(df.region == region) & (df.config == config)]
            d = base[base.model == "M3"].merge(
                base[base.model == "M2"], on=["region", "config", "semilla"], suffixes=("_m3", "_m2")
            )
            delta = d["roc_auc_m3"] - d["roc_auc_m2"]
            n = len(delta)
            media = float(delta.mean())
            sd = float(delta.std(ddof=1)) if n > 1 else 0.0
            ic = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            filas.append({
                "region": region, "config": config, "n_corridas": n,
                "m3_roc_media": round(float(base[base.model == "M3"]["roc_auc"].mean()), 4),
                "m3_roc_sd": round(float(base[base.model == "M3"]["roc_auc"].std(ddof=1)), 4),
                "m2_roc_media": round(float(base[base.model == "M2"]["roc_auc"].mean()), 4),
                "m2_roc_sd": round(float(base[base.model == "M2"]["roc_auc"].std(ddof=1)), 4),
                "delta_media": round(media, 4),
                "delta_sd": round(sd, 4),
                "delta_ic95": round(ic, 4),
                "delta_min": round(float(delta.min()), 4),
                "delta_max": round(float(delta.max()), 4),
                "signo_positivo_pct": round(100.0 * float((delta > 0).mean()), 1),
            })
    return pd.DataFrame(filas)


def _md(df):
    lineas = [
        "| Region | Capacity | Runs | M3 ROC (mean ± SD) | M2 ROC (mean ± SD) | ΔROC mean ± SD | 95 % CI | Range | Δ > 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.config} | {r.n_corridas} | {r.m3_roc_media:.4f} ± {r.m3_roc_sd:.4f} | "
            f"{r.m2_roc_media:.4f} ± {r.m2_roc_sd:.4f} | {r.delta_media:+.4f} ± {r.delta_sd:.4f} | "
            f"±{r.delta_ic95:.4f} | {r.delta_min:+.4f} to {r.delta_max:+.4f} | {r.signo_positivo_pct:.0f} % |"
        )
    (DIR_TABLAS / "tablaS25_estabilidad_semillas.md").write_text(
        "# Table S25. Seed and capacity stability\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
