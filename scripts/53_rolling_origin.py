"""Script 53 (R.2): backtest rolling-origin con cuatro folds.

Folds causales: train hasta Y-2, validacion Y-1, test Y, para
Y = 2022, 2023, 2024 y 2025. Se reentrenan M1, M2 y M3 por region.
El cutoff y la calibracion se ajustan solo en el ano de validacion.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.metricas import bootstrap_diferencia_roc
from src.modelo import (
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

FOLDS = {
    2022: (list(range(2018, 2021)), 2021, 2022),
    2023: (list(range(2018, 2022)), 2022, 2023),
    2024: (list(range(2018, 2023)), 2023, 2024),
    2025: (list(range(2018, 2024)), 2024, 2025),
}
MODOS = (("M1", "m1"), ("M2", "ignicion"), ("M3", "hibrido"))


def _cargar() -> pl.DataFrame:
    frames = [
        pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / f"{s}_ignicion.parquet")
        for s in ("train", "val", "test")
    ]
    d = pl.concat(frames)
    return d.with_columns(
        pl.col("fecha").str.slice(0, 4).cast(pl.Int32).alias("anio")
    )


def _params(region):
    return dict(
        scale_pos_weight=1.0 if region == "Chaco" else 2.0,
        learning_rate=0.03,
        num_leaves=23 if region == "Chaco" else 31,
        min_child_samples=20 if region == "Chaco" else 10,
        num_boost_round=45 if region == "Chaco" else 60,
        early_stopping_rounds=None,
        semilla=42,
    )


def _run_fold(region: str, anio_test: int):
    train_years, val_year, test_year = FOLDS[anio_test]
    d = _cargar().filter(pl.col("region") == region)
    tr = d.filter(pl.col("anio").is_in(train_years))
    va = d.filter(pl.col("anio") == val_year)
    te = d.filter(pl.col("anio") == test_year)
    if min(tr.height, va.height, te.height) == 0:
        raise ValueError(f"Fold vacio {region}/{test_year}")

    out = []
    probs = {}
    for nombre, modo in MODOS:
        Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
        Xva, yva, _ = preparar_matrices_xy(va, region, modo=modo)
        Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
        b = entrenar_modelo_lightgbm(Xtr, ytr, Xva, yva, feats, **_params(region))
        c = calibrar_probabilidades_isotonica(b, Xva, yva, metodo="platt_monotono")
        p = predecir_probabilidad(b, c, Xte)
        probs[nombre] = p
        out.append({
            "region": region,
            "fold": f"test_{test_year}",
            "test_year": test_year,
            "val_year": val_year,
            "model": nombre,
            "roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "pr_auc": round(float(average_precision_score(yte, p)), 4),
            "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]),
            "n_test": int(Xte.shape[0]),
            "n_pos_test": int(yte.sum()),
            "delta_roc_m3_m2": np.nan,
            "p_value": np.nan,
        })

    delta, pval = bootstrap_diferencia_roc(
        yte, probs["M3"], probs["M2"], n_bootstrap=1000,
        block_ids=te["fecha"].to_numpy(),
    )
    out[-1]["delta_roc_m3_m2"] = round(float(delta), 4)
    out[-1]["p_value"] = round(float(pval), 4)
    return out


def main():
    tareas = [(r, y) for y in FOLDS for r in ("Chaco", "Oriental")]
    filas = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futuros = {ex.submit(_run_fold, r, y): (r, y) for r, y in tareas}
        for fut in as_completed(futuros):
            r, y = futuros[fut]
            filas.extend(fut.result())
            print(f"ok {r} {y}", flush=True)

    df = pd.DataFrame(filas).sort_values(["region", "test_year", "model"])
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_TABLAS / "tablaS26_rolling_origin.csv"
    df.to_csv(ruta, index=False)
    _md(df)
    print(df[["region", "test_year", "model", "roc_auc", "pr_auc", "delta_roc_m3_m2", "p_value"]])


def _md(df):
    lineas = [
        "| Region | Fold | Model | ROC-AUC | PR-AUC | n train | n val | n test | Pos test | ΔROC M3−M2 (p) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        delta = "—" if not np.isfinite(r.delta_roc_m3_m2) else f"{r.delta_roc_m3_m2:+.4f} ({r.p_value:.3f})"
        lineas.append(
            f"| {r.region} | {r.fold} | {r.model} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | "
            f"{r.n_train:,} | {r.n_val:,} | {r.n_test:,} | {r.n_pos_test:,} | {delta} |"
        )
    (DIR_TABLAS / "tablaS26_rolling_origin.md").write_text(
        "# Table S26. Rolling-origin backtest\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
