"""Script 45 (R.3): metricas del test sin solapamiento train/test.

Recomputa las metricas del test retrospectivo 2024-2025 sobre cuatro
subconjuntos que neutralizan el solapamiento temporal de celdas:

- full:             test completo (control de consistencia con la Tabla 2)
- sin_celda_vista:  celdas que no aparecen en train (responde al 90.65 %)
- sin_fuego_train:  celdas sin ninguna deteccion de fuego en train
- sin_historia:     celdas con freq_hist_1km == 0 (dominio del fallback M2)
- solo_2025:        solo el anio 2025 (fuel lag 2)

Los cortes son los congelados en validacion 2023; no se reoptimizan.
BSS usa como referencia la prevalencia del test completo de cada region,
de modo que la fila `full` reproduce la Tabla 2 y los subconjuntos son
comparables entre si.

Salidas:
- entregables/tablas/tablaS27_metricas_sin_solapamiento.csv
- entregables/tablas/tablaS27_metricas_sin_solapamiento.md
- insercion de la tabla entre los marcadores <!-- tablaS27:inicio/fin -->
  de los dos suplementos.
"""

from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)

from src.constantes import DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import (
    calcular_brier_score,
    calcular_expected_calibration_error,
)

SUBSETS = [
    ("full", "Full"),
    ("sin_celda_vista", "Unseen cell"),
    ("sin_fuego_train", "No train fire"),
    ("sin_historia", "No history"),
    ("solo_2025", "2025 only"),
]

REFERENCIA_ROC = {
    ("ignicion", "Chaco", "M3"): 0.8021,
    ("ignicion", "Chaco", "M2"): 0.7991,
    ("ignicion", "Chaco", "Gated"): 0.8035,
    ("ignicion", "Oriental", "M3"): 0.7534,
    ("ignicion", "Oriental", "M2"): 0.7406,
    ("ignicion", "Oriental", "Gated"): 0.7502,
    ("propagacion", "Chaco", "Propagation"): 0.9039,
    ("propagacion", "Oriental", "Propagation"): 0.7859,
}


def _bootstrap_ci_roc_pr(y, p, blocks, n_bootstrap=500, alpha=0.05, semilla=42):
    """IC percentil agrupado por fecha para ROC-AUC y PR-AUC."""
    rng = np.random.default_rng(semilla)
    bloques = np.asarray(blocks)
    unicos = np.unique(bloques)
    mapa = {b: np.where(bloques == b)[0] for b in unicos}
    rocs, prs = [], []
    for _ in range(n_bootstrap):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([mapa[b] for b in elegidos])
        if len(np.unique(y[idx])) < 2:
            continue
        rocs.append(roc_auc_score(y[idx], p[idx]))
        prs.append(average_precision_score(y[idx], p[idx]))
    if not rocs:
        r = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))
        return r, r, pr, pr
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    return (
        float(np.percentile(rocs, lo)),
        float(np.percentile(rocs, hi)),
        float(np.percentile(prs, lo)),
        float(np.percentile(prs, hi)),
    )


def _metricas(y, p, cutoff, blocks, prevalencia_ref):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    y_hat = (p >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = fbeta_score(y, y_hat, beta=0.5, zero_division=0)
    brier = calcular_brier_score(y, p)
    brier_ref = calcular_brier_score(y, np.full_like(p, prevalencia_ref))
    bss = 1.0 - brier / brier_ref if brier_ref > 0 else 0.0
    roc_lo, roc_hi, pr_lo, pr_hi = _bootstrap_ci_roc_pr(y, p, blocks)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "roc_ci_low": round(roc_lo, 4),
        "roc_ci_high": round(roc_hi, 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "pr_ci_low": round(pr_lo, 4),
        "pr_ci_high": round(pr_hi, 4),
        "brier": round(float(brier), 4),
        "bss": round(float(bss), 4),
        "ece": round(float(calcular_expected_calibration_error(y, p)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f05": round(float(f05), 4),
        "fp": int(fp),
    }


def _subset(df, subset, celdas_train, celdas_fuego_train):
    if subset == "full":
        return df
    if subset == "sin_celda_vista":
        return df.filter(
            ~pl.col("cell_id").is_in(pl.Series(list(celdas_train)).implode())
        )
    if subset == "sin_fuego_train":
        return df.filter(
            ~pl.col("cell_id").is_in(pl.Series(list(celdas_fuego_train)).implode())
        )
    if subset == "sin_historia":
        return df.filter(pl.col("freq_hist_1km") <= 0)
    if subset == "solo_2025":
        return df.filter(pl.col("fecha").str.slice(0, 4) == "2025")
    raise ValueError(subset)


def _celdas_train(task):
    train = pl.read_parquet(
        DIR_PROCESSED / "targets_poblacionales" / f"train_{task}.parquet",
        columns=["cell_id", f"target_{task}"],
    )
    todas = set(train["cell_id"].unique().to_list())
    con_fuego = set(
        train.filter(pl.col(f"target_{task}") == 1)["cell_id"].unique().to_list()
    )
    return todas, con_fuego


def _corte(modo, region):
    ruta = DIR_MODELOS / "calibradores" / f"umbrales_{modo}_{region.lower()}.json"
    return float(json.loads(ruta.read_text(encoding="utf-8"))["c_f05"])


def _cortes_gated():
    df = pl.read_csv(DIR_TABLAS / "tabla13_politica_gated.csv")
    return {r["region"]: float(r["cutoff_f05"]) for r in df.iter_rows(named=True)}


def _md(df):
    cols = [
        ("Region", "region", None),
        ("Subset", "subset_label", None),
        ("Model", "model", None),
        ("n", "n", "int"),
        ("n+", "n_pos", "int"),
        ("Prev. (%)", "prevalence", "pct"),
        ("ROC-AUC", "roc_auc", "f4"),
        ("ROC 95% CI", "roc_ci", "ci"),
        ("PR-AUC", "pr_auc", "f4"),
        ("PR 95% CI", "pr_ci", "ci"),
        ("Brier", "brier", "f4"),
        ("BSS", "bss", "f4"),
        ("Prec.", "precision", "f4"),
        ("Rec.", "recall", "f4"),
        ("F0.5", "f05", "f4"),
        ("FP", "fp", "int"),
        ("ΔROC", "delta_roc_vs_full", "delta"),
        ("ΔF0.5", "delta_f05_vs_full", "delta"),
    ]

    def fmt(row, key, kind):
        if kind is None:
            return str(row[key])
        if kind == "int":
            return f"{int(row[key])}"
        if kind == "pct":
            return f"{100.0 * row[key]:.2f}"
        if kind == "f4":
            return f"{row[key]:.4f}"
        if kind == "delta":
            if row["subset"] == "full":
                return "—"
            return f"{row[key]:+.4f}"
        return str(row[key])

    lineas = []
    for task, titulo in (("ignicion", "Ignition"), ("propagacion", "Propagation")):
        lineas.append(f"**{titulo}**")
        lineas.append("")
        lineas.append("| " + " | ".join(c[0] for c in cols) + " |")
        lineas.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in df.filter(pl.col("task") == task).iter_rows(named=True):
            vals = []
            for _, key, kind in cols:
                if kind in (None, "int", "pct", "f4", "delta"):
                    vals.append(fmt(row, key, kind))
                elif key == "roc_ci":
                    vals.append(f"[{row['roc_ci_low']:.3f} - {row['roc_ci_high']:.3f}]")
                elif key == "pr_ci":
                    vals.append(f"[{row['pr_ci_low']:.3f} - {row['pr_ci_high']:.3f}]")
            lineas.append("| " + " | ".join(vals) + " |")
        lineas.append("")
    return "\n".join(lineas).rstrip()


def _splice(path, bloque):
    txt = path.read_text(encoding="utf-8").split("\n")
    inicio = [i for i, l in enumerate(txt) if "<!-- tablaS27:inicio -->" in l]
    fin = [i for i, l in enumerate(txt) if "<!-- tablaS27:fin -->" in l]
    if len(inicio) != 1 or len(fin) != 1 or inicio[0] >= fin[0]:
        raise ValueError(f"Marcadores S27 invalidos en {path}")
    nuevo = txt[: inicio[0] + 1] + ["", bloque, ""] + txt[fin[0]:]
    path.write_text("\n".join(nuevo), encoding="utf-8")


def main():
    cortes = _cortes_gated()
    filas = []

    for task in ("ignicion", "propagacion"):
        celdas_train, celdas_fuego = _celdas_train(task)
        test = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet")
        target = f"target_{task}"

        if task == "ignicion":
            modelos = [
                ("M3", "probabilidad_hibrido", "hibrido"),
                ("M2", "probabilidad_ignicion", "ignicion"),
                ("Gated", None, None),
            ]
        else:
            modelos = [
                ("Propagation", "probabilidad_propagacion", "propagacion")
            ]

        for region in ("Chaco", "Oriental"):
            df_reg = test.filter(pl.col("region") == region)
            prevalencia_full = float(df_reg[target].mean())

            for subset, etiqueta in SUBSETS:
                sub = _subset(df_reg, subset, celdas_train, celdas_fuego)
                if sub.height == 0:
                    raise ValueError(f"Subconjunto vacio: {task}/{region}/{subset}")
                if task == "propagacion" and subset == "sin_celda_vista":
                    print(
                        f"Omitido {task}/{region}/{subset}: "
                        f"n={sub.height} pos={int(sub[target].sum())}"
                    )
                    continue

                y = sub[target].to_numpy()
                bloques = sub["fecha"].to_numpy()

                for nombre, col, modo in modelos:
                    if nombre == "Gated":
                        p = np.where(
                            sub["freq_hist_1km"].to_numpy() <= 0,
                            sub["probabilidad_ignicion"].to_numpy(),
                            sub["probabilidad_hibrido"].to_numpy(),
                        )
                        corte = cortes[region]
                    else:
                        p = sub[col].to_numpy()
                        corte = _corte(modo, region)
                    met = _metricas(y, p, corte, bloques, prevalencia_full)
                    filas.append(
                        {
                            "task": task,
                            "region": region,
                            "subset": subset,
                            "subset_label": etiqueta,
                            "model": nombre,
                            "cutoff_f05": round(float(corte), 6),
                            **met,
                        }
                    )

    base = {
        (r["task"], r["region"], r["model"]): r
        for r in filas
        if r["subset"] == "full"
    }
    for r in filas:
        b = base[(r["task"], r["region"], r["model"])]
        r["delta_roc_vs_full"] = round(r["roc_auc"] - b["roc_auc"], 4)
        r["delta_f05_vs_full"] = round(r["f05"] - b["f05"], 4)

    for r in filas:
        if r["subset"] == "full":
            esperado = REFERENCIA_ROC[(r["task"], r["region"], r["model"])]
            if abs(r["roc_auc"] - esperado) > 0.001:
                raise AssertionError(
                    f"Full no reproduce Tabla 2: {r['task']}/{r['region']}/"
                    f"{r['model']} {r['roc_auc']} != {esperado}"
                )

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS27_metricas_sin_solapamiento.csv")
    md = _md(df)
    (DIR_TABLAS / "tablaS27_metricas_sin_solapamiento.md").write_text(
        "# Table S27. Train/test overlap sensitivity\n\n" + md + "\n",
        encoding="utf-8",
    )

    for q in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario_es.qmd",
    ):
        if q.exists():
            _splice(q, md)

    print(df.select(
        ["task", "region", "subset", "model", "n", "n_pos", "roc_auc", "roc_ci_low", "roc_ci_high", "delta_roc_vs_full"]
    ))
    print(f"\n{len(df)} filas -> {DIR_TABLAS/'tablaS27_metricas_sin_solapamiento.csv'}")


if __name__ == "__main__":
    main()
