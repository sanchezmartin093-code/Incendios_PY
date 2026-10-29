"""Script 48 (C.1): intervalos bootstrap de V(alpha) agrupados por fecha.

Remuestrea fechas (no filas), B = 1000
CUTOFFS = {
    ("FWI", "Chaco"): 3.1285, ("FWI", "Oriental"): 20.4720,
    ("INPE", "Chaco"): 0.0364, ("INPE", "Oriental"): 0.0596,
}, recalcula V(alpha) para cada modelo,
region y alpha, y reporta el IC 95 % percentil. Regenera la Tabla S14 y la
Figura S3 con bandas.

Salidas:
- entregables/tablas/tabla11_cost_loss.csv (valor_lo, valor_hi)
- entregables/figuras/supp_fig03_cost_loss.png|.pdf
- Tabla S14 insertada entre <!-- tablaS14:inicio/fin -->.
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
from src.cost_loss import valor_cost_loss
from src.inpe import constante_a_desde_fracciones
from src.metricas import corregir_prior_probabilidades

MODELOS = [
    ("M3_ignicion_hibrida", "ignicion", "probabilidad_hibrido"),
    ("M2_ignicion_base", "ignicion", "probabilidad_calibrada"),
    ("Propagacion_tactica", "propagacion", "probabilidad_propagacion"),
]
ALPHAS = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
                   0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
B = 1000
CUTOFFS = {
    ("FWI", "Chaco"): 3.1285, ("FWI", "Oriental"): 20.4720,
    ("INPE", "Chaco"): 0.0364, ("INPE", "Oriental"): 0.0596,
}


def _prevalencia(c, region, task):
    s = c.filter(
        (pl.col("region") == region) & (pl.col("task") == task)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(2024, 2025)
    )
    p = int(s["n_pos"].sum())
    n = int(s["n_neg"].sum())
    return p / (p + n)


def _datos(region, task, col):
    preds = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(
        pl.col("region") == region
    )
    c = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet").filter(
        (pl.col("task") == task) & (pl.col("region") == region)
    ).select(["fecha", "n_pos", "n_neg"])
    target = f"target_{task}"
    muestra = preds.group_by(["fecha", "target"]).len().rename({"len": "n_muestra"})
    base = (
        preds.with_row_index("_i")
        .join(muestra, on=["fecha", "target"], how="left")
        .join(c, on="fecha", how="left")
        .sort("_i")
    )
    if base.height != preds.height or base["n_pos"].null_count():
        raise ValueError(f"Pesos desalineados en {region}/{task}")
    y = base[target].to_numpy().astype(int)
    w = np.where(
        y == 1,
        base["n_pos"].to_numpy() / base["n_muestra"].to_numpy(),
        base["n_neg"].to_numpy() / base["n_muestra"].to_numpy(),
    )
    pi_s = float(y.mean())
    prev = _prevalencia(
        pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet"),
        region, task,
    )
    p = corregir_prior_probabilidades(base[col].to_numpy(), pi_s, prev)
    if not np.isfinite(w).all() or not np.isfinite(p).all():
        raise ValueError(f"Pesos o probabilidades no finitos en {region}/{task}")
    return base, y, p, w


def _indices(base):
    g = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"]
    )
    f = pl.read_parquet(
        DIR_PROCESSED / "fwi_completo_diario.parquet",
        columns=["nodo_clima_id", "fecha", "fwi"],
    )
    i = pl.read_parquet(
        DIR_PROCESSED / "inpe_diario.parquet",
        columns=["nodo_clima_id", "fecha", "pse", "rh_min_pct"],
    )
    out = (
        base.join(g, on="cell_id", how="left")
        .join(f, on=["nodo_clima_id", "fecha"], how="left")
        .join(i, on=["nodo_clima_id", "fecha"], how="left")
        .sort("_i")
    )
    if out.height != base.height or out["fwi"].null_count() or out["pse"].null_count():
        raise ValueError("Join de indices incompleto")
    a = np.array([
        constante_a_desde_fracciones(p, s, bs, bh, palm)
        for p, s, bs, bh, palm in zip(
            out["pct_pastura"].to_numpy(), out["pct_sabana_pastizal"].to_numpy(),
            out["pct_bosque_seco"].to_numpy(), out["pct_bosque_humedo"].to_numpy(),
            out["pct_humedal_palmar"].to_numpy(),
        )
    ])
    madurez = np.clip(a * out["pse"].to_numpy() / 120.0, 0.0, 2.0)
    rb = 0.8 * (1 + np.sin(madurez * (np.pi / 2) - (np.pi / 2))) / 2
    fur = -0.006 * out["rh_min_pct"].to_numpy() + 1.3
    ft = 0.02 * out["temp_max_c"].to_numpy() + 0.4
    flat = 1 + np.abs(out["lat"].to_numpy()) * 0.003
    return out.with_columns(pl.Series("p_inpe_calc", np.clip(rb * fur * ft * flat, 0, 1)))


def _por_fecha_mask(base, y, w, mask):
    fechas = base["fecha"].to_numpy()
    unicos = np.unique(fechas)
    n_d = len(unicos)
    w_sum = np.zeros(n_d)
    pos_w = np.zeros(n_d)
    tp = np.zeros((n_d, len(ALPHAS)))
    fp = np.zeros((n_d, len(ALPHAS)))
    fn = np.zeros((n_d, len(ALPHAS)))
    for d, f in enumerate(unicos):
        ii = np.where(fechas == f)[0]
        yd, wd, md = y[ii], w[ii], mask[ii]
        w_sum[d] = wd.sum()
        pos_w[d] = wd[yd == 1].sum()
        tp[d] = (wd[:, None] * ((yd == 1)[:, None] & md)).sum(0)
        fp[d] = (wd[:, None] * ((yd == 0)[:, None] & md)).sum(0)
        fn[d] = (wd[:, None] * ((yd == 1)[:, None] & ~md)).sum(0)
    return w_sum, pos_w, tp, fp, fn


def _por_fecha(base, y, p, w):
    fechas = base["fecha"].to_numpy()
    unicos = np.unique(fechas)
    idx = [np.where(fechas == d)[0] for d in unicos]
    n_d, n_a = len(unicos), len(ALPHAS)
    w_sum = np.zeros(n_d)
    pos_w = np.zeros(n_d)
    tp = np.zeros((n_d, n_a))
    fp = np.zeros((n_d, n_a))
    fn = np.zeros((n_d, n_a))
    for d, ii in enumerate(idx):
        yd, pd, wd = y[ii], p[ii], w[ii]
        w_sum[d] = wd.sum()
        pos_w[d] = wd[yd == 1].sum()
        h = pd[:, None] >= ALPHAS[None, :]
        tp[d] = (wd[:, None] * ((yd == 1)[:, None] & h)).sum(0)
        fp[d] = (wd[:, None] * ((yd == 0)[:, None] & h)).sum(0)
        fn[d] = (wd[:, None] * ((yd == 1)[:, None] & ~h)).sum(0)
    return w_sum, pos_w, tp, fp, fn


def _valor(tp, fp, fn, w_sum, pos_w):
    tp = tp / w_sum
    fp = fp / w_sum
    fn = fn / w_sum
    pi = pos_w / w_sum
    den = np.minimum(ALPHAS, pi) - ALPHAS * pi
    num = np.minimum(ALPHAS, pi) - (ALPHAS * (tp + fp) + fn)
    v = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    return np.maximum(v, -1.0), tp, fp, fn, pi


def main():
    filas = []
    for region in ("Chaco", "Oriental"):
        for nombre, task, col in MODELOS:
            base, y, p, w = _datos(region, task, col)
            w_sum_d, pos_w_d, tp_d, fp_d, fn_d = _por_fecha(base, y, p, w)

            v_pt, tp, fp, fn, pi = _valor(
                tp_d.sum(0), fp_d.sum(0), fn_d.sum(0), w_sum_d.sum(), pos_w_d.sum()
            )
            for i, a in enumerate(ALPHAS):
                ref = valor_cost_loss(y, p, w, float(a))
                if abs(ref["valor"] - v_pt[i]) > 1e-9:
                    raise AssertionError(f"V no coincide con valor_cost_loss: {region}/{nombre}/{a}")

            rng = np.random.default_rng(42)
            n_d = len(w_sum_d)
            V = np.zeros((B, len(ALPHAS)))
            for b in range(B):
                m = rng.multinomial(n_d, np.full(n_d, 1.0 / n_d)).astype(float)
                v, *_ = _valor(
                    np.sum(m[:, None] * tp_d, axis=0),
                    np.sum(m[:, None] * fp_d, axis=0),
                    np.sum(m[:, None] * fn_d, axis=0),
                    float(np.sum(m * w_sum_d)),
                    float(np.sum(m * pos_w_d)),
                )
                V[b] = v
            lo = np.percentile(V, 2.5, axis=0)
            hi = np.percentile(V, 97.5, axis=0)

            for i, a in enumerate(ALPHAS):
                filas.append({
                    "region": region,
                    "task": task,
                    "modelo": nombre,
                    "alpha": float(a),
                    "valor": float(v_pt[i]),
                    "valor_lo": float(lo[i]),
                    "valor_hi": float(hi[i]),
                    "tp": float(tp[i]),
                    "fp": float(fp[i]),
                    "fn": float(fn[i]),
                    "prevalencia": float(pi),
                    "threshold": float(a),
                    "rule": "p_ge_alpha",
                })
            print(f"{region:8s} {nombre:24s} listo")

        base_idx, _, _, w_idx = _datos(region, "ignicion", "probabilidad_calibrada")
        base_idx = _indices(base_idx)
        y_idx = base_idx["target_ignicion"].to_numpy().astype(int)
        for idx_name, col_idx in (("FWI", "fwi"), ("INPE", "p_inpe_calc")):
            corte = CUTOFFS[(idx_name, region)]
            mask = np.repeat(
                (base_idx[col_idx].to_numpy() >= corte)[:, None], len(ALPHAS), axis=1
            )
            w_sum_d, pos_w_d, tp_d, fp_d, fn_d = _por_fecha_mask(base_idx, y_idx, w_idx, mask)
            v_pt, tp, fp, fn, pi = _valor(
                tp_d.sum(0), fp_d.sum(0), fn_d.sum(0), w_sum_d.sum(), pos_w_d.sum()
            )
            rng = np.random.default_rng(42)
            n_d = len(w_sum_d)
            V = np.zeros((B, len(ALPHAS)))
            for b in range(B):
                m = rng.multinomial(n_d, np.full(n_d, 1.0 / n_d)).astype(float)
                v, *_ = _valor(
                    np.sum(m[:, None] * tp_d, axis=0),
                    np.sum(m[:, None] * fp_d, axis=0),
                    np.sum(m[:, None] * fn_d, axis=0),
                    float(np.sum(m * w_sum_d)),
                    float(np.sum(m * pos_w_d)),
                )
                V[b] = v
            lo = np.percentile(V, 2.5, axis=0)
            hi = np.percentile(V, 97.5, axis=0)
            for i, a in enumerate(ALPHAS):
                filas.append({
                    "region": region,
                    "task": "ignicion",
                    "modelo": f"{idx_name}_frozen",
                    "alpha": float(a),
                    "valor": float(v_pt[i]),
                    "valor_lo": float(lo[i]),
                    "valor_hi": float(hi[i]),
                    "tp": float(tp[i]),
                    "fp": float(fp[i]),
                    "fn": float(fn[i]),
                    "prevalencia": float(pi),
                    "threshold": float(corte),
                    "rule": "frozen_F0.5",
                })
            print(f"{region:8s} {idx_name:24s} congelado listo")

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tabla11_cost_loss.csv")

    _tabla_s14(df)
    _figura(df)
    print(f"{len(df)} filas -> tabla11_cost_loss.csv")


def _tabla_s14(df):
    lbl = {"M3_ignicion_hibrida": "M3", "M2_ignicion_base": "M2",
           "Propagacion_tactica": "Propagation", "FWI_frozen": "FWI", "INPE_frozen": "INPE"}
    task_lbl = {"ignicion": "Ign.", "propagacion": "Prop."}
    lineas = [
        "| Region | Task | Model | α | V(α) | V low | V high | TP rate | FP rate | FN rate | Prevalence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.iter_rows(named=True):
        lineas.append(
            f"| {r['region']} | {task_lbl[r['task']]} | {lbl[r['modelo']]} | "
            f"{r['alpha']:.4f} | {r['valor']:.6f} | {r['valor_lo']:.6f} | {r['valor_hi']:.6f} | "
            f"{r['tp']:.6f} | {r['fp']:.6f} | {r['fn']:.6f} | {r['prevalencia']:.6f} |"
        )
    bloque = "\n".join(lineas)
    (DIR_TABLAS / "tabla11_cost_loss.md").write_text(
        "# Tabla 11. Analisis cost-loss poblacional (IC bootstrap por fecha)\n\n"
        + bloque + "\n",
        encoding="utf-8",
    )
    for path in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario_es.qmd",
    ):
        if path.exists():
            _splice(path, bloque, "tablaS14")


def _figura(df):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    colors = {"Chaco": "#c0392b", "Oriental": "#1f7a4d"}
    labels = {"M3_ignicion_hibrida": "M3 hybrid", "M2_ignicion_base": "M2 baseline",
              "Propagacion_tactica": "Tactical propagation"}
    styles = {"M3_ignicion_hibrida": "-", "M2_ignicion_base": "--", "Propagacion_tactica": "-"}

    fig, (ax_prop, ax_ign) = plt.subplots(
        1, 2, figsize=(13, 5), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.08]}
    )

    for reg in ("Chaco", "Oriental"):
        sub = df.filter(
            (pl.col("region") == reg) & (pl.col("task") == "propagacion")
        ).sort("alpha")
        a = sub["alpha"].to_numpy()
        ax_prop.plot(a, sub["valor"].to_numpy(), color=colors[reg], lw=2.0,
                     label=f"{reg} tactical propagation")
        ax_prop.fill_between(a, sub["valor_lo"].to_numpy(), sub["valor_hi"].to_numpy(),
                             color=colors[reg], alpha=0.18, lw=0)
    ax_prop.axhline(0, color="grey", lw=0.8)
    ax_prop.set_title("Propagation: value over the full range", fontweight="bold")
    ax_prop.set_xlabel("α = C/L")
    ax_prop.set_ylabel("Normalized value V(α)")
    ax_prop.set_xlim(0, 0.5)
    ax_prop.legend(fontsize=8, loc="upper right", frameon=True)

    for reg in ("Chaco", "Oriental"):
        for nombre in ("M3_ignicion_hibrida", "M2_ignicion_base"):
            sub = df.filter(
                (pl.col("region") == reg) & (pl.col("task") == "ignicion")
                & (pl.col("modelo") == nombre) & (pl.col("alpha") <= 0.006)
            ).sort("alpha")
            a = sub["alpha"].to_numpy()
            ax_ign.plot(a, sub["valor"].to_numpy(), color=colors[reg],
                        linestyle=styles[nombre], lw=2.0,
                        label=f"{reg} {labels[nombre]}")
            ax_ign.fill_between(a, sub["valor_lo"].to_numpy(), sub["valor_hi"].to_numpy(),
                                color=colors[reg], alpha=0.10 if nombre == "M2_ignicion_base" else 0.18,
                                lw=0)
    ax_ign.axhline(0, color="grey", lw=0.8)
    ax_ign.set_title("Ignition: value concentrates at low α", fontweight="bold")
    ax_ign.set_xlabel("α = C/L")
    ax_ign.set_ylabel("Normalized value V(α)")
    ax_ign.set_xlim(0, 0.006)
    ax_ign.set_xticks([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    ax_ign.legend(fontsize=7.5, loc="upper right", frameon=True)

    fig.suptitle("Cost-loss decision value under D2 and prevalence correction",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    for nombre in ("supp_fig03_cost_loss", "fig08_cost_loss"):
        fig.savefig(DIR_FIGURAS / f"{nombre}.png", dpi=300, bbox_inches="tight")
        fig.savefig(DIR_FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)


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
