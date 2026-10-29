"""Script 47 (R.11): incertidumbre poblacional y curvas por presupuesto de alertas.

1. IC de Wilson (sobre conteos poblacionales) y bootstrap agrupado por fecha
   (B = 1000) para precision y recall poblacionales de la Tabla S6.
2. Curvas de precision poblacional frente a alertas/dia para ignicion
   (M3, M2, gated) y propagacion, y Figura S8.

Salidas:
- entregables/fuentes/tablas/tablaS06b_ic_poblacional.csv
- entregables/fuentes/tablas/tablaS08_presupuesto_alertas.csv
- entregables/fuentes/figuras/supp_fig08_presupuesto_alertas.png|.pdf
- Tabla S6 regenerada entre <!-- tablaS6:inicio/fin -->.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constantes import DIR_FIGURAS, DIR_PROCESSED, DIR_TABLAS
from src.metricas import corregir_prior_probabilidades

N_BOOT = 1000
BUDGETS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
COLORES = {"M3": "#1f4e79", "M2": "#d95f02", "Gated": "#6a3d9a", "Propagation": "#1b7837"}
ETIQUETAS = {"M3": "M3", "M2": "M2", "Gated": "M3/M2 gated", "Propagation": "Propagation"}


def _wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2.0 * n)) / den
    half = z * np.sqrt(ph * (1.0 - ph) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _prep(conteos, task, region):
    preds = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(
        pl.col("region") == region
    )
    target = f"target_{task}"
    counts = conteos.filter(
        (pl.col("task") == task) & (pl.col("region") == region)
    ).select(["fecha", "n_pos", "n_neg"])
    muestra = preds.group_by(["fecha", "target"]).len().rename({"len": "n_muestra"})
    base = (
        preds.with_row_index("_i")
        .join(muestra, on=["fecha", "target"], how="left")
        .join(counts, on="fecha", how="left")
        .sort("_i")
    )
    if base.height != preds.height:
        raise ValueError("La alineacion de pesos cambio el numero de filas")
    if base["n_pos"].null_count() or base["n_muestra"].null_count():
        raise ValueError("Pesos con nulos")
    y = base[target].to_numpy().astype(int)
    w = np.where(
        y == 1,
        base["n_pos"].to_numpy() / base["n_muestra"].to_numpy(),
        base["n_neg"].to_numpy() / base["n_muestra"].to_numpy(),
    )
    return base, y, w


def _poblacional(base, col, pi_s, prev):
    return corregir_prior_probabilidades(base[col].to_numpy(), pi_s, prev)


def _bootstrap(y, p, w, cutoff, fechas, n_boot=N_BOOT, semilla=42):
    rng = np.random.default_rng(semilla)
    fechas = np.asarray(fechas)
    unicos = np.unique(fechas)
    mapa = {b: np.where(fechas == b)[0] for b in unicos}
    ps, rs = [], []
    for _ in range(n_boot):
        pick = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([mapa[b] for b in pick])
        yy, pp, ww = y[idx], p[idx], w[idx]
        h = pp >= cutoff
        tp = float(ww[(yy == 1) & h].sum())
        fp = float(ww[(yy == 0) & h].sum())
        fn = float(ww[(yy == 1) & (~h)].sum())
        if tp + fp > 0:
            ps.append(tp / (tp + fp))
        if tp + fn > 0:
            rs.append(tp / (tp + fn))
    if not ps or not rs:
        return (np.nan, np.nan), (np.nan, np.nan)
    return (
        (float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))),
        (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))),
    )


def _intervalos(conteos, tab):
    filas = []
    for r in tab.iter_rows(named=True):
        task, region, modelo = r["task"], r["region"], r["modelo"]
        col = {
            "M3_ignicion_hibrida": "probabilidad_hibrido",
            "M2_ignicion_base": "probabilidad_ignicion",
            "M2_propagacion_tactica": "probabilidad_propagacion",
        }[modelo]
        base, y, w = _prep(conteos, task, region)
        p = _poblacional(base, col, r["prevalencia_muestra_test"], r["prevalencia_poblacion_test"])
        cut = float(r["cutoff_f05_poblacional"])
        h = p >= cut
        tp = float(w[(y == 1) & h].sum())
        fp = float(w[(y == 0) & h].sum())
        fn = float(w[(y == 1) & (~h)].sum())
        n_alert = tp + fp
        n_pos = tp + fn

        prec = float(r["precision_poblacional_pct"]) / 100.0
        rec = float(r["recall_poblacional_pct"]) / 100.0
        w_prec = _wilson(prec * n_alert, n_alert)
        w_rec = _wilson(rec * n_pos, n_pos)
        b_prec, b_rec = _bootstrap(y, p, w, cut, base["fecha"].to_numpy())

        filas.append(
            {
                "region": region,
                "task": task,
                "modelo": modelo,
                "precision_pct": round(100 * prec, 3),
                "precision_wilson_low_pct": round(100 * w_prec[0], 3),
                "precision_wilson_high_pct": round(100 * w_prec[1], 3),
                "precision_boot_low_pct": round(100 * b_prec[0], 3),
                "precision_boot_high_pct": round(100 * b_prec[1], 3),
                "recall_pct": round(100 * rec, 3),
                "recall_wilson_low_pct": round(100 * w_rec[0], 2),
                "recall_wilson_high_pct": round(100 * w_rec[1], 2),
                "recall_boot_low_pct": round(100 * b_rec[0], 2),
                "recall_boot_high_pct": round(100 * b_rec[1], 2),
                "n_alerted_pop": int(round(n_alert)),
                "n_pos_pop": int(round(n_pos)),
                "n_bootstrap": N_BOOT,
            }
        )
        print(
            f"{region:8s} {task:11s} {modelo[:6]} "
            f"prec {100*prec:.3f} [{100*w_prec[0]:.3f}-{100*w_prec[1]:.3f} | "
            f"{100*b_prec[0]:.3f}-{100*b_prec[1]:.3f}] "
            f"rec {100*rec:.2f} [{100*w_rec[0]:.2f}-{100*w_rec[1]:.2f} | "
            f"{100*b_rec[0]:.2f}-{100*b_rec[1]:.2f}]"
        )
    return pl.DataFrame(filas)


def _tabla_s6(tab, ic):
    tab = tab.with_row_index("_orden")
    df = tab.join(
        ic.select(
            [
                "region",
                "task",
                "modelo",
                "precision_wilson_low_pct",
                "precision_wilson_high_pct",
                "precision_boot_low_pct",
                "precision_boot_high_pct",
                "recall_wilson_low_pct",
                "recall_wilson_high_pct",
                "recall_boot_low_pct",
                "recall_boot_high_pct",
            ]
        ),
        on=["region", "task", "modelo"],
        how="left",
    ).sort("_orden")
    lineas = [
        "| Region | Task | Model | Pop. prevalence | Sample prevalence | Pop. cutoff | "
        "ROC-AUC | PR-AUC | Precision (%) [W; B] | Recall (%) [W; B] | F0.5 | "
        "Brier | BSS | ECE | Pop. N | Pop. positives |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.iter_rows(named=True):
        prec = (
            f"{r['precision_poblacional_pct']:.3f} "
            f"[{r['precision_wilson_low_pct']:.3f}–{r['precision_wilson_high_pct']:.3f}; "
            f"{r['precision_boot_low_pct']:.3f}–{r['precision_boot_high_pct']:.3f}]"
        )
        rec = (
            f"{r['recall_poblacional_pct']:.3f} "
            f"[{r['recall_wilson_low_pct']:.2f}–{r['recall_wilson_high_pct']:.2f}; "
            f"{r['recall_boot_low_pct']:.2f}–{r['recall_boot_high_pct']:.2f}]"
        )
        lineas.append(
            f"| {r['region']} | {r['task']} | {r['modelo']} | "
            f"{r['prevalencia_poblacion_test']:.6f} | {r['prevalencia_muestra_test']:.6f} | "
            f"{r['cutoff_f05_poblacional']:.6f} | {r['roc_auc_poblacional']:.4f} | "
            f"{r['pr_auc_poblacional']:.4f} | {prec} | {rec} | "
            f"{r['f05_poblacional']:.4f} | {r['brier_poblacional']:.6f} | "
            f"{r['bss_poblacional']:.4f} | {r['ece_poblacional']:.4f} | "
            f"{r['n_poblacion_test']} | {r['n_pos_poblacion_test']} |"
        )
    return "\n".join(lineas)


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


def _modelos_curva(base, task, tab, region, cut_gated):
    pi_s = 0.09090909090909091
    if task == "ignicion":
        prev = float(
            tab.filter(
                (pl.col("region") == region)
                & (pl.col("task") == "ignicion")
                & (pl.col("modelo") == "M3_ignicion_hibrida")
            )["prevalencia_poblacion_test"][0]
        )
        p3 = _poblacional(base, "probabilidad_hibrido", pi_s, prev)
        p2 = _poblacional(base, "probabilidad_ignicion", pi_s, prev)
        gate = base["freq_hist_1km"].to_numpy() <= 0
        pg = np.where(gate, p2, p3)
        return (
            prev,
            {
                "M3": (p3, float(tab.filter((pl.col("region") == region) & (pl.col("modelo") == "M3_ignicion_hibrida"))["cutoff_f05_poblacional"][0])),
                "M2": (p2, float(tab.filter((pl.col("region") == region) & (pl.col("modelo") == "M2_ignicion_base"))["cutoff_f05_poblacional"][0])),
                "Gated": (pg, cut_gated[region]),
            },
        )
    prev = float(
        tab.filter(
            (pl.col("region") == region)
            & (pl.col("task") == "propagacion")
            & (pl.col("modelo") == "M2_propagacion_tactica")
        )["prevalencia_poblacion_test"][0]
    )
    pp = _poblacional(base, "probabilidad_propagacion", pi_s, prev)
    return (
        prev,
        {
            "Propagation": (
                pp,
                float(tab.filter((pl.col("region") == region) & (pl.col("modelo") == "M2_propagacion_tactica"))["cutoff_f05_poblacional"][0]),
            )
        },
    )


def _curvas(conteos, tab):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7})
    filas = []
    cut_gated = {
        r["region"]: float(r["cutoff_f05"])
        for r in pl.read_csv(DIR_TABLAS / "tabla13_politica_gated.csv").iter_rows(named=True)
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), dpi=600)

    for fila, task in enumerate(["ignicion", "propagacion"]):
        for col, region in enumerate(["Chaco", "Oriental"]):
            base, y, w = _prep(conteos, task, region)
            prev, modelos = _modelos_curva(base, task, tab, region, cut_gated)
            n_dias = base["fecha"].n_unique()
            ax = axes[fila, col]

            for nombre, (p, corte) in modelos.items():
                orden = np.argsort(-p)
                ys, ps_sorted, ws = y[orden], p[orden], w[orden]
                ca = np.cumsum(ws)
                ct = np.cumsum(ws * ys)
                total_pos = ct[-1] if len(ct) else 0.0

                xs, ys_plot = [], []
                for k in BUDGETS:
                    j = int(np.searchsorted(ca, k * n_dias))
                    j = min(j, len(ca) - 1)
                    if ca[j] <= 0:
                        continue
                    prec = 100.0 * ct[j] / ca[j]
                    rec = 100.0 * ct[j] / total_pos if total_pos else np.nan
                    xs.append(k)
                    ys_plot.append(prec)
                    filas.append(
                        {
                            "task": task,
                            "region": region,
                            "model": nombre,
                            "kind": "budget",
                            "alerts_per_day": k,
                            "precision_pct": round(prec, 4),
                            "recall_pct": round(rec, 4),
                            "threshold": round(float(ps_sorted[j]), 8),
                        }
                    )
                ax.plot(xs, ys_plot, "-o", ms=2.5, lw=1.2, color=COLORES[nombre], label=ETIQUETAS[nombre])

                if corte is not None:
                    if nombre == "Gated":
                        corte = float(
                            corregir_prior_probabilidades(
                                np.array([corte]), 0.09090909090909091, prev
                            )[0]
                        )
                    h = p >= corte
                    a_dia = float(w[h].sum()) / n_dias
                    tp = float(w[(y == 1) & h].sum())
                    fp = float(w[(y == 0) & h].sum())
                    prec_op = 100.0 * tp / (tp + fp) if tp + fp else 0.0
                    ax.scatter([a_dia], [prec_op], s=22, marker="D", color=COLORES[nombre],
                               edgecolor="white", linewidth=0.4, zorder=5)
                    fn = float(w[(y == 1) & (~h)].sum())
                    rec_op = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
                    filas.append(
                        {
                            "task": task,
                            "region": region,
                            "model": nombre,
                            "kind": "operating",
                            "alerts_per_day": round(a_dia, 2),
                            "precision_pct": round(prec_op, 4),
                            "recall_pct": round(rec_op, 4),
                            "threshold": round(float(corte), 8),
                        }
                    )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(4, 15000)
            ax.set_ylim(0.012, 60)
            ax.axhline(100.0 * prev, color="0.35", ls="--", lw=0.8, zorder=1)
            ax.set_xlabel("Alerts per day")
            ax.set_ylabel("Population precision (%)")
            ax.set_title(f"{region} — {'Ignition' if task == 'ignicion' else 'Propagation'}", fontsize=7.5)
            ax.grid(alpha=0.25, which="both", lw=0.4)
            ax.legend(frameon=False, fontsize=6, loc="lower left")

    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / "supp_fig08_presupuesto_alertas.png", dpi=600, bbox_inches="tight")
    fig.savefig(DIR_FIGURAS / "supp_fig08_presupuesto_alertas.pdf", bbox_inches="tight")
    plt.close(fig)
    return pl.DataFrame(filas)


def main():
    conteos = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet")
    tab = pl.read_csv(DIR_TABLAS / "tabla07_evaluacion_poblacional.csv")

    ic = _intervalos(conteos, tab)
    ic.write_csv(DIR_TABLAS / "tablaS06b_ic_poblacional.csv")

    tabla = _tabla_s6(tab, ic)
    for q in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario_en.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario.qmd",
    ):
        if q.exists():
            _splice(q, tabla, "tablaS6")

    cur = _curvas(conteos, tab)
    cur.write_csv(DIR_TABLAS / "tablaS08_presupuesto_alertas.csv")
    print(f"{len(ic)} filas IC -> tablaS06b_ic_poblacional.csv")
    print(f"{len(cur)} filas curvas -> tablaS08_presupuesto_alertas.csv")


if __name__ == "__main__":
    main()
