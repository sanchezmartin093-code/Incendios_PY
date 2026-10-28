"""Script 52 (B.4): sensibilidad del INPE a la constante de flamabilidad A.

Version primaria: A local, ponderada por fracciones de cobertura (MapBiomas).
Sensibilidad: A fija = 3.0, coeficiente INPE de sabana abierta, aplicado a todas
las celdas. El cutoff se selecciona en validacion 2023 con F0.5 y se congela.

Salida: Tabla S22.
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
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.inpe import constante_a_desde_fracciones
from src.metricas import (
    bootstrap_diferencia_roc,
    calcular_bootstrap_ci_roc,
    calcular_umbral_optimo_f_beta,
)

A_FIJA = 3.0
COLS_COBERTURA = [
    "pct_pastura",
    "pct_sabana_pastizal",
    "pct_bosque_seco",
    "pct_bosque_humedo",
    "pct_humedal_palmar",
]


def _join(df: pl.DataFrame) -> pl.DataFrame:
    grilla = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet",
        columns=["cell_id", "nodo_clima_id"],
    )
    inpe = pl.read_parquet(DIR_PROCESSED / "inpe_diario.parquet")
    return df.join(grilla, on="cell_id", how="left").join(
        inpe, on=["nodo_clima_id", "fecha"], how="left"
    )


def _a_local(x: pl.DataFrame) -> np.ndarray:
    return np.array([
        constante_a_desde_fracciones(*fila)
        for fila in x.select(COLS_COBERTURA).iter_rows()
    ])


def _inpe(x: pl.DataFrame, a: np.ndarray) -> np.ndarray:
    rb = 0.8 * (
        1.0 + np.sin((a * x["pse"].to_numpy() / 120.0) * (np.pi / 2.0) - (np.pi / 2.0))
    ) / 2.0
    fur = -0.006 * x["rh_min_pct"].to_numpy() + 1.3
    ft = 0.02 * x["temp_max_c"].to_numpy() + 0.4
    flat = 1.0 + np.abs(x["lat"].to_numpy()) * 0.003
    return np.clip(rb * fur * ft * flat, 0.0, 1.0)


def _metricas(y, p, cut):
    y_bin = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_bin).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision),
        "recall": float(recall),
        "f05": float(fbeta_score(y, y_bin, beta=0.5, zero_division=0)),
        "fp": int(fp),
    }


def main():
    filas = []
    for region in ("Chaco", "Oriental"):
        val = _join(
            pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
            .filter(pl.col("region") == region)
        )
        test = _join(
            pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
            .filter(pl.col("region") == region)
        )
        if test["pse"].null_count() or test["rh_min_pct"].null_count():
            raise ValueError(f"Datos INPE incompletos en {region}")

        a_local_val = _a_local(val)
        a_local_test = _a_local(test)
        p_local_val = _inpe(val, a_local_val)
        p_local = _inpe(test, a_local_test)
        p_fija_val = _inpe(val, np.full(val.height, A_FIJA))
        p_fija = _inpe(test, np.full(test.height, A_FIJA))

        y_val = val["target_ignicion"].to_numpy().astype(int)
        y = test["target_ignicion"].to_numpy().astype(int)
        bloques = test["fecha"].to_numpy()

        variantes = (
            ("Local (land cover)", "cell", p_local_val, p_local),
            (f"Fixed A = {A_FIJA:.1f}", f"{A_FIJA:.1f}", p_fija_val, p_fija),
        )
        for nombre, a_txt, pv, pt in variantes:
            cut, _ = calcular_umbral_optimo_f_beta(y_val, pv, beta=0.5)
            m = _metricas(y, pt, cut)
            lo, hi = calcular_bootstrap_ci_roc(y, pt, n_bootstrap=500, block_ids=bloques)
            fila = {
                "region": region,
                "version": nombre,
                "A": a_txt,
                "roc_auc": round(m["roc_auc"], 4),
                "roc_ci_95": f"[{lo:.3f} - {hi:.3f}]",
                "pr_auc": round(m["pr_auc"], 4),
                "cutoff_f05": round(float(cut), 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f05_score": round(m["f05"], 4),
                "falsos_positivos": m["fp"],
                "delta_roc_fija_local": np.nan,
                "p_value": np.nan,
            }
            if nombre.startswith("Fixed"):
                delta, pval = bootstrap_diferencia_roc(
                    y, pt, p_local, n_bootstrap=1000, block_ids=bloques
                )
                fila["delta_roc_fija_local"] = round(float(delta), 4)
                fila["p_value"] = round(float(pval), 4)
            filas.append(fila)

        print(
            f"{region}: local A mean {a_local_test.mean():.3f} "
            f"(median {np.median(a_local_test):.3f}); "
            f"ROC local {filas[-2]['roc_auc']:.4f}, fija {filas[-1]['roc_auc']:.4f}"
        )

    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS22_sensibilidad_inpe_a.csv", index=False)
    _md(df)
    print(df[["region", "version", "roc_auc", "delta_roc_fija_local", "p_value"]])


def _md(df: pd.DataFrame):
    lineas = [
        "| Region | INPE version | A | ROC-AUC [95 % CI] | PR-AUC | Cutoff | Precision | Recall | F0.5 | FP | ΔROC vs local (p) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        if np.isfinite(r.delta_roc_fija_local):
            delta = f"{r.delta_roc_fija_local:+.4f} ({r.p_value:.3f})"
        else:
            delta = "—"
        lineas.append(
            f"| {r.region} | {r.version} | {r.A} | {r.roc_auc:.4f} {r.roc_ci_95} | "
            f"{r.pr_auc:.4f} | {r.cutoff_f05:.4f} | {r.precision:.4f} | "
            f"{r.recall:.4f} | {r.f05_score:.4f} | {r.falsos_positivos:,} | {delta} |"
        )
    bloque = "\n".join(lineas)
    (DIR_TABLAS / "tablaS22_sensibilidad_inpe_a.md").write_text(
        "# Table S22. INPE sensitivity to A\n\n" + bloque + "\n",
        encoding="utf-8",
    )
    for path in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario_es.qmd",
    ):
        if path.exists():
            _splice(path, bloque, "tablaS22")


def _splice(path, bloque, tag):
    txt = path.read_text(encoding="utf-8").split("\n")
    ini = [i for i, l in enumerate(txt) if f"<!-- {tag}:inicio -->" in l]
    fin = [i for i, l in enumerate(txt) if f"<!-- {tag}:fin -->" in l]
    if len(ini) != 1 or len(fin) != 1 or ini[0] >= fin[0]:
        raise ValueError(f"Marcadores {tag} invalidos en {path}")
    path.write_text(
        "\n".join(txt[: ini[0] + 1] + ["", bloque, ""] + txt[fin[0]:]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
