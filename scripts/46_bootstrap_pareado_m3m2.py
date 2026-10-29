"""Script 46: bootstrap pareado M3 - M2 por subconjunto de solapamiento (R.3).

Para cada region y subconjunto de la Tabla S27, contrasta M3 contra M2 con
un bootstrap pareado agrupado por fecha (B = 1000) y reporta la diferencia
observada, su IC del 95 % y el p bilateral para ROC-AUC y PR-AUC.

Salidas:
- entregables/fuentes/tablas/tablaS27b_bootstrap_pareado_m3m2.csv
- bloque insertado entre <!-- tablaS27b:inicio/fin --> en los suplementos.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS

SUBSETS = [
    ("full", "Full"),
    ("sin_celda_vista", "Unseen cell"),
    ("sin_fuego_train", "No train fire"),
    ("sin_historia", "No history"),
    ("solo_2025", "2025 only"),
]
B = 1000


def _celdas(train):
    todas = set(train["cell_id"].unique().to_list())
    con_fuego = set(
        train.filter(pl.col("target_ignicion") == 1)["cell_id"].unique().to_list()
    )
    return todas, con_fuego


def _subset(df, subset, celdas_train, celdas_fuego):
    if subset == "full":
        return df
    if subset == "sin_celda_vista":
        return df.filter(
            ~pl.col("cell_id").is_in(pl.Series(list(celdas_train)).implode())
        )
    if subset == "sin_fuego_train":
        return df.filter(
            ~pl.col("cell_id").is_in(pl.Series(list(celdas_fuego)).implode())
        )
    if subset == "sin_historia":
        return df.filter(pl.col("freq_hist_1km") <= 0)
    if subset == "solo_2025":
        return df.filter(pl.col("fecha").str.slice(0, 4) == "2025")
    raise ValueError(subset)


def _pareado(y, p1, p0, bloques, n_boot=B, semilla=42):
    rng = np.random.default_rng(semilla)
    bloques = np.asarray(bloques)
    unicos = np.unique(bloques)
    mapa = {b: np.where(bloques == b)[0] for b in unicos}
    d_roc, d_pr = [], []
    for _ in range(n_boot):
        elegidos = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([mapa[b] for b in elegidos])
        if len(np.unique(y[idx])) < 2:
            continue
        d_roc.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
        d_pr.append(
            average_precision_score(y[idx], p1[idx])
            - average_precision_score(y[idx], p0[idx])
        )

    obs_roc = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    obs_pr = float(average_precision_score(y, p1) - average_precision_score(y, p0))
    if not d_roc:
        return obs_roc, np.nan, np.nan, np.nan, obs_pr, np.nan, np.nan, np.nan

    d_roc = np.asarray(d_roc)
    d_pr = np.asarray(d_pr)
    ci_roc = np.percentile(d_roc, [2.5, 97.5])
    ci_pr = np.percentile(d_pr, [2.5, 97.5])
    p_roc = 2.0 * min(float((d_roc <= 0).mean()), float((d_roc >= 0).mean()))
    p_pr = 2.0 * min(float((d_pr <= 0).mean()), float((d_pr >= 0).mean()))
    return (
        obs_roc,
        float(ci_roc[0]),
        float(ci_roc[1]),
        min(1.0, p_roc),
        obs_pr,
        float(ci_pr[0]),
        float(ci_pr[1]),
        min(1.0, p_pr),
    )


def _md(df):
    lineas = [
        "| Region | Subset | n | n+ | ΔROC (M3 − M2) | 95% CI | p | ΔPR-AUC | 95% CI | p |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for r in df.iter_rows(named=True):
        lineas.append(
            "| {region} | {subset_label} | {n} | {n_pos} | {dr:+.4f} | "
            "[{rlo:+.4f}, {rhi:+.4f}] | {pr} | {dp:+.4f} | "
            "[{plo:+.4f}, {phi:+.4f}] | {pp} |".format(
                region=r["region"],
                subset_label=r["subset_label"],
                n=r["n"],
                n_pos=r["n_pos"],
                dr=r["delta_roc"],
                rlo=r["roc_ci_low"],
                rhi=r["roc_ci_high"],
                pr="≤0.001" if r["p_roc"] <= 0.001 else f"{r['p_roc']:.3f}",
                dp=r["delta_pr"],
                plo=r["pr_ci_low"],
                phi=r["pr_ci_high"],
                pp="≤0.001" if r["p_pr"] <= 0.001 else f"{r['p_pr']:.3f}",
            )
        )
    return "\n".join(lineas)


def _splice(path, bloque):
    txt = path.read_text(encoding="utf-8").split("\n")
    ini = [i for i, l in enumerate(txt) if "<!-- tablaS27b:inicio -->" in l]
    fin = [i for i, l in enumerate(txt) if "<!-- tablaS27b:fin -->" in l]
    if len(ini) != 1 or len(fin) != 1 or ini[0] >= fin[0]:
        raise ValueError(f"Marcadores S27b invalidos en {path}")
    path.write_text(
        "\n".join(txt[: ini[0] + 1] + ["", bloque, ""] + txt[fin[0]:]),
        encoding="utf-8",
    )


def main():
    train = pl.read_parquet(
        DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet",
        columns=["cell_id", "target_ignicion"],
    )
    celdas_train, celdas_fuego = _celdas(train)
    test = pl.read_parquet(DIR_PROCESSED / "evaluacion_test_ignicion.parquet")

    filas = []
    for region in ("Chaco", "Oriental"):
        df_reg = test.filter(pl.col("region") == region)
        for subset, etiqueta in SUBSETS:
            sub = _subset(df_reg, subset, celdas_train, celdas_fuego)
            y = sub["target_ignicion"].to_numpy().astype(int)
            p3 = sub["probabilidad_hibrido"].to_numpy()
            p2 = sub["probabilidad_ignicion"].to_numpy()
            dr, rlo, rhi, pr, dp, plo, phi, pp = _pareado(
                y, p3, p2, sub["fecha"].to_numpy()
            )
            filas.append(
                {
                    "region": region,
                    "subset": subset,
                    "subset_label": etiqueta,
                    "n": int(len(y)),
                    "n_pos": int(y.sum()),
                    "delta_roc": round(dr, 4),
                    "roc_ci_low": round(rlo, 4),
                    "roc_ci_high": round(rhi, 4),
                    "p_roc": round(pr, 4),
                    "delta_pr": round(dp, 4),
                    "pr_ci_low": round(plo, 4),
                    "pr_ci_high": round(phi, 4),
                    "p_pr": round(pp, 4),
                    "n_bootstrap": B,
                }
            )
            print(
                f"{region:8s} {subset:16s} dROC={dr:+.4f} "
                f"[{rlo:+.4f},{rhi:+.4f}] p={pr:.3f} | "
                f"dPR={dp:+.4f} p={pp:.3f}"
            )

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS27b_bootstrap_pareado_m3m2.csv")
    md = _md(df)
    (DIR_TABLAS / "tablaS27b_bootstrap_pareado_m3m2.md").write_text(
        "# Table S27b. Paired M3 - M2 bootstrap by subset\n\n" + md + "\n",
        encoding="utf-8",
    )
    for q in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario_en.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario.qmd",
    ):
        if q.exists():
            _splice(q, md)
    print(f"{len(df)} filas -> tablaS27b_bootstrap_pareado_m3m2.csv")


if __name__ == "__main__":
    main()
