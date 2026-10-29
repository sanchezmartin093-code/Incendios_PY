"""Script 51 (C.3): metricas de carga operativa.

Alertas/dia, precision y recall poblacionales en el corte congelado, recall y
precision con presupuesto de K alertas/dia (K = 10, 50, 100), alertas por
acierto, lift sobre la prevalencia y alpha que maximiza V(alpha).

Salida: bloque "Table S23 (continued)" en los suplementos.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.metricas import corregir_prior_probabilidades

ALPHAS = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
                   0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
KS = (10, 50, 100)


def _prep(task, region):
    d = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(
        pl.col("region") == region
    )
    c = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet").filter(
        (pl.col("task") == task) & (pl.col("region") == region)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(2024, 2025)
    ).select(["fecha", "n_pos", "n_neg"])
    m = d.group_by(["fecha", f"target_{task}"]).len().rename({"len": "n_muestra"})
    b = (
        d.with_row_index("_i")
        .join(m, on=["fecha", f"target_{task}"], how="left")
        .join(c, on="fecha", how="left")
        .sort("_i")
    )
    if b.height != d.height or b["n_pos"].null_count():
        raise ValueError("Pesos desalineados")
    y = b[f"target_{task}"].to_numpy().astype(int)
    w = np.where(
        y == 1,
        b["n_pos"].to_numpy() / b["n_muestra"].to_numpy(),
        b["n_neg"].to_numpy() / b["n_muestra"].to_numpy(),
    )
    return b, y, w


def _operating(p, y, w, cut, nd):
    h = p >= cut
    tp = float((w * ((y == 1) & h)).sum())
    fp = float((w * ((y == 0) & h)).sum())
    fn = float((w * ((y == 1) & (~h))).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "alertas_dia": float((w * h).sum()) / nd,
        "precision": prec,
        "recall": rec,
    }


def _topk(p, y, w, k, nd):
    orden = np.argsort(-p)
    ws, ys = w[orden], y[orden]
    ca = np.cumsum(ws)
    j = int(np.searchsorted(ca, k * nd))
    j = min(j, len(ca) - 1)
    tp = float((ws[: j + 1] * ys[: j + 1]).sum())
    fp = float((ws[: j + 1] * (1 - ys[: j + 1])).sum())
    total_pos = float((w * y).sum())
    return (
        tp / (tp + fp) if (tp + fp) else 0.0,
        tp / total_pos if total_pos else 0.0,
    )


def _valor_alpha(p, y, w, a):
    h = p >= a
    tp = float((w * ((y == 1) & h)).sum())
    fp = float((w * ((y == 0) & h)).sum())
    fn = float((w * ((y == 1) & (~h))).sum())
    n = float(w.sum())
    pi = float((w * y).sum()) / n
    tp, fp, fn = tp / n, fp / n, fn / n
    den = min(a, pi) - a * pi
    num = min(a, pi) - (a * (tp + fp) + fn)
    return max(num / den if abs(den) > 1e-12 else 0.0, -1.0)


def _alpha_optimo(p, y, w):
    vals = np.array([_valor_alpha(p, y, w, a) for a in ALPHAS])
    i = int(np.argmax(vals))
    return float(ALPHAS[i]), float(vals[i])


def main():
    tab6 = pl.read_csv(DIR_TABLAS / "tabla07_evaluacion_poblacional.csv")
    tab13 = pl.read_csv(DIR_TABLAS / "tabla13_politica_gated.csv")
    cost = pl.read_csv(DIR_TABLAS / "tabla11_cost_loss.csv")
    filas = []

    for region in ("Chaco", "Oriental"):
        # Ignition
        b, y, w = _prep("ignicion", region)
        nd = int(b["fecha"].n_unique())
        pi_s = float(y.mean())
        prev = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("modelo") == "M3_ignicion_hibrida")
            )["prevalencia_poblacion_test"][0]
        )
        prev2 = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("modelo") == "M2_ignicion_base")
            )["prevalencia_poblacion_test"][0]
        )
        p3 = corregir_prior_probabilidades(b["probabilidad_hibrido"].to_numpy(), pi_s, prev)
        p2 = corregir_prior_probabilidades(b["probabilidad_calibrada"].to_numpy(), pi_s, prev2)
        gate = b["freq_hist_1km"].to_numpy() <= 0
        pg = np.where(gate, p2, p3)
        cut3 = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("modelo") == "M3_ignicion_hibrida")
            )["cutoff_f05_poblacional"][0]
        )
        cut2 = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("modelo") == "M2_ignicion_base")
            )["cutoff_f05_poblacional"][0]
        )
        cg = float(tab13.filter(pl.col("region") == region)["cutoff_f05"][0])
        cutg = float(corregir_prior_probabilidades(np.array([cg]), pi_s, prev)[0])

        modelos = (("M3", p3, cut3, "M3_ignicion_hibrida"), ("M2", p2, cut2, "M2_ignicion_base"),
                   ("Gated", pg, cutg, None))
        for nombre, p, cut, mod in modelos:
            op = _operating(p, y, w, cut, nd)
            prec, rec = op["precision"], op["recall"]
            row = {
                "region": region,
                "task": "ignicion",
                "model": nombre,
                "alerts_day": round(op["alertas_dia"], 1),
                "precision_pct": round(100 * prec, 3),
                "recall_pct": round(100 * rec, 2),
                "alerts_per_hit": round(1 / prec, 1) if prec else np.nan,
                "lift": round(prec / prev, 1) if prev else np.nan,
            }
            for k in KS:
                pk, rk = _topk(p, y, w, k, nd)
                row[f"precision_top{k}_pct"] = round(100 * pk, 2)
                row[f"recall_top{k}_pct"] = round(100 * rk, 3)
            if mod:
                r = cost.filter(
                    (pl.col("region") == region) & (pl.col("modelo") == mod)
                ).sort("valor", descending=True).head(1).to_dicts()[0]
                row["alpha_optimo"] = round(float(r["alpha"]), 4)
                row["v_optimo"] = round(float(r["valor"]), 4)
            else:
                a, v = _alpha_optimo(p, y, w)
                row["alpha_optimo"] = round(a, 4)
                row["v_optimo"] = round(v, 4)
            filas.append(row)

        # Propagation
        b, y, w = _prep("propagacion", region)
        nd = int(b["fecha"].n_unique())
        pi_s = float(y.mean())
        prevp = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("task") == "propagacion")
            )["prevalencia_poblacion_test"][0]
        )
        pp = corregir_prior_probabilidades(b["probabilidad_propagacion"].to_numpy(), pi_s, prevp)
        cutp = float(
            tab6.filter(
                (pl.col("region") == region) & (pl.col("task") == "propagacion")
            )["cutoff_f05_poblacional"][0]
        )
        op = _operating(pp, y, w, cutp, nd)
        prec, rec = op["precision"], op["recall"]
        row = {
            "region": region, "task": "propagacion", "model": "Propagation",
            "alerts_day": round(op["alertas_dia"], 1),
            "precision_pct": round(100 * prec, 3), "recall_pct": round(100 * rec, 2),
            "alerts_per_hit": round(1 / prec, 1) if prec else np.nan,
            "lift": round(prec / prevp, 1) if prevp else np.nan,
        }
        for k in KS:
            pk, rk = _topk(pp, y, w, k, nd)
            row[f"precision_top{k}_pct"] = round(100 * pk, 2)
            row[f"recall_top{k}_pct"] = round(100 * rk, 3)
        r = cost.filter(
            (pl.col("region") == region) & (pl.col("modelo") == "Propagacion_tactica")
        ).sort("valor", descending=True).head(1).to_dicts()[0]
        row["alpha_optimo"] = round(float(r["alpha"]), 4)
        row["v_optimo"] = round(float(r["valor"]), 4)
        filas.append(row)

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS23b_carga_operativa.csv")
    _tabla(df)
    print(df.select(["region", "model", "alerts_day", "precision_pct", "alerts_per_hit", "lift", "alpha_optimo"]))


def _tabla(df):
    lineas = [
        "| Region | Model | Alerts/day | Prec. (op.) % | Rec. (op.) % | Alerts/hit | Lift | R@10 | R@50 | R@100 | P@10 | P@50 | P@100 | α* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.iter_rows(named=True):
        lineas.append(
            f"| {r['region']} | {r['model']} | {r['alerts_day']:.1f} | "
            f"{r['precision_pct']:.3f} | {r['recall_pct']:.2f} | "
            f"{r['alerts_per_hit']:.1f} | {r['lift']:.1f} | "
            f"{r['recall_top10_pct']:.3f} | {r['recall_top50_pct']:.3f} | {r['recall_top100_pct']:.3f} | "
            f"{r['precision_top10_pct']:.2f} | {r['precision_top50_pct']:.2f} | {r['precision_top100_pct']:.2f} | "
            f"{r['alpha_optimo']:.4f} |"
        )
    bloque = "\n".join(lineas)
    (DIR_TABLAS / "tablaS23b_carga_operativa.md").write_text(
        "# Table S23 (continued). Operational load\n\n" + bloque + "\n", encoding="utf-8"
    )
    for path in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario_en.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario.qmd",
    ):
        if path.exists():
            _splice(path, bloque, "tablaS23b")


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
