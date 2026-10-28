"""Script 49 (C.5): bootstrap pareado de Delta V(alpha) = V(M3) - V(referencia).

Referencias: M2 y FWI. Para M3/M2 se usa la regla calibrada p >= alpha. FWI no
es una probabilidad: su umbral se elige en la validacion 2023 maximizando
V(alpha) y se congela para el test; esto le da al indice su mejor punto de
operacion posible y hace la comparacion conservadora para M3.

Remuestrea fechas (B = 1000) y calcula V(M3), V(M2) y V(FWI) en cada replica
para obtener Delta V, IC 95 % y p bilateral.

Salidas:
- entregables/tablas/tablaS24_delta_valor.csv|.md
- Tabla S24 insertada entre <!-- tablaS24:inicio/fin -->.
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
B = 1000
TASK = "ignicion"


def _conteos(region, ini=None, fin=None):
    c = pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet").filter(
        (pl.col("task") == TASK) & (pl.col("region") == region)
    )
    if ini is not None:
        c = c.filter(pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(ini, fin))
    return c


def _fwi_join(df):
    g = pl.read_parquet(
        DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"]
    )
    f = pl.read_parquet(
        DIR_PROCESSED / "fwi_completo_diario.parquet",
        columns=["nodo_clima_id", "fecha", "fwi"],
    )
    out = df.join(g, on="cell_id", how="left").join(
        f, on=["nodo_clima_id", "fecha"], how="left"
    )
    if out.height != df.height:
        raise ValueError("El join de FWI duplico filas")
    if out["fwi"].null_count():
        raise ValueError("FWI con nulos tras el join")
    return out


def _pesos(df, conteos, target):
    muestra = df.group_by(["fecha", target]).len().rename({"len": "n_muestra"})
    base = (
        df.with_row_index("_i")
        .join(muestra, on=["fecha", target], how="left")
        .join(conteos.select(["fecha", "n_pos", "n_neg"]), on="fecha", how="left")
        .sort("_i")
    )
    if base.height != df.height or base["n_pos"].null_count():
        raise ValueError("Pesos desalineados")
    y = base[target].to_numpy().astype(int)
    w = np.where(
        y == 1,
        base["n_pos"].to_numpy() / base["n_muestra"].to_numpy(),
        base["n_neg"].to_numpy() / base["n_muestra"].to_numpy(),
    )
    return base, y, w


def _valor(tp, fp, fn, w_sum, pos_w):
    tp, fp, fn = tp / w_sum, fp / w_sum, fn / w_sum
    pi = pos_w / w_sum
    den = np.minimum(ALPHAS, pi) - ALPHAS * pi
    num = np.minimum(ALPHAS, pi) - (ALPHAS * (tp + fp) + fn)
    v = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    return np.maximum(v, -1.0)


def _tau_validacion(y, fwi, w):
    """Umbral FWI por alpha que maximiza V(alpha) en validacion 2023."""
    orden = np.argsort(-fwi)
    fs = fwi[orden]
    ws = w[orden]
    ys = y[orden]
    ca = np.concatenate([[0.0], np.cumsum(ws)])
    ctp = np.concatenate([[0.0], np.cumsum(ws * ys)])
    cfp = np.concatenate([[0.0], np.cumsum(ws * (1 - ys))])
    total_pos = ctp[-1]
    wsum = ca[-1]
    j = np.unique(np.round(np.linspace(0, len(fwi), 801)).astype(int))
    j = np.clip(j, 0, len(fwi))
    tp = ctp[j]
    fp = cfp[j]
    fn = total_pos - tp
    tp_n, fp_n, fn_n = tp / wsum, fp / wsum, fn / wsum
    pi = total_pos / wsum
    den = np.minimum(ALPHAS[:, None], pi) - ALPHAS[:, None] * pi
    num = np.minimum(ALPHAS[:, None], pi) - (
        ALPHAS[:, None] * (tp_n[None, :] + fp_n[None, :]) + fn_n[None, :]
    )
    v = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    v = np.maximum(v, -1.0)
    best = np.argmax(v, axis=1)
    tau = np.where(j[best] == 0, np.inf, fs[np.maximum(j[best] - 1, 0)])
    return tau.astype(float)


def _por_fecha(base, y, w, masks):
    fechas = base["fecha"].to_numpy()
    unicos = np.unique(fechas)
    n_d = len(unicos)
    out = {}
    for nombre, mask in masks.items():
        tp = np.zeros((n_d, len(ALPHAS)))
        fp = np.zeros((n_d, len(ALPHAS)))
        fn = np.zeros((n_d, len(ALPHAS)))
        for d, f in enumerate(unicos):
            ii = np.where(fechas == f)[0]
            yd, wd, md = y[ii], w[ii], mask[ii]
            tp[d] = (wd[:, None] * ((yd == 1)[:, None] & md)).sum(0)
            fp[d] = (wd[:, None] * ((yd == 0)[:, None] & md)).sum(0)
            fn[d] = (wd[:, None] * ((yd == 1)[:, None] & ~md)).sum(0)
        out[nombre] = (tp, fp, fn)
    return unicos, out


def main():
    filas = []
    for region in ("Chaco", "Oriental"):
        test = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{TASK}.parquet").filter(
            pl.col("region") == region
        )
        c_test = _conteos(region, 2024, 2025)
        base, y, w = _pesos(test, c_test, f"target_{TASK}")
        base = _fwi_join(base).sort("_i")

        prev = float(c_test["n_pos"].sum()) / float(
            (c_test["n_pos"].sum() + c_test["n_neg"].sum())
        )
        pi_s = float(y.mean())
        p3 = corregir_prior_probabilidades(base["probabilidad_hibrido"].to_numpy(), pi_s, prev)
        p2 = corregir_prior_probabilidades(base["probabilidad_calibrada"].to_numpy(), pi_s, prev)
        fwi = base["fwi"].to_numpy()

        val = pl.read_parquet(
            DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet"
        ).filter(pl.col("region") == region)
        c_val = _conteos(region, 2023, 2023)
        val, yv, wv = _pesos(val, c_val, f"target_{TASK}")
        val = _fwi_join(val).sort("_i")
        tau = _tau_validacion(yv, val["fwi"].to_numpy(), wv)

        masks = {
            "M3": p3[:, None] >= ALPHAS[None, :],
            "M2": p2[:, None] >= ALPHAS[None, :],
            "FWI": fwi[:, None] >= tau[None, :],
        }
        unicos, sumas = _por_fecha(base, y, w, masks)
        w_sum_d = np.array([w[(base["fecha"].to_numpy() == f)].sum() for f in unicos])
        pos_w_d = np.array([
            w[(base["fecha"].to_numpy() == f) & (y == 1)].sum() for f in unicos
        ])

        v_pt = {}
        for nombre in ("M3", "M2", "FWI"):
            tp, fp, fn = sumas[nombre]
            v_pt[nombre] = _valor(
                tp.sum(0), fp.sum(0), fn.sum(0), w_sum_d.sum(), pos_w_d.sum()
            )

        ref_tab = pl.read_csv(DIR_TABLAS / "tabla11_cost_loss.csv")
        for nombre, modelo in (("M3", "M3_ignicion_hibrida"), ("M2", "M2_ignicion_base")):
            for i, a in enumerate(ALPHAS):
                r = ref_tab.filter(
                    (pl.col("region") == region) & (pl.col("task") == "ignicion")
                    & (pl.col("modelo") == modelo) & (pl.col("alpha") == float(a))
                )
                if r.height and abs(float(r["valor"][0]) - v_pt[nombre][i]) > 1e-6:
                    raise AssertionError(
                        f"V no coincide con Tabla S14: {region}/{nombre}/{a}"
                    )

        rng = np.random.default_rng(42)
        n_d = len(unicos)
        V = {n: np.zeros((B, len(ALPHAS))) for n in ("M3", "M2", "FWI")}
        for b in range(B):
            m = rng.multinomial(n_d, np.full(n_d, 1.0 / n_d)).astype(float)
            ws = float(np.sum(m * w_sum_d))
            pw = float(np.sum(m * pos_w_d))
            for nombre in ("M3", "M2", "FWI"):
                tp, fp, fn = sumas[nombre]
                V[nombre][b] = _valor(
                    np.sum(m[:, None] * tp, axis=0),
                    np.sum(m[:, None] * fp, axis=0),
                    np.sum(m[:, None] * fn, axis=0),
                    ws, pw,
                )

        for ref in ("M2", "FWI"):
            d_pt = v_pt["M3"] - v_pt[ref]
            d_boot = V["M3"] - V[ref]
            lo = np.percentile(d_boot, 2.5, axis=0)
            hi = np.percentile(d_boot, 97.5, axis=0)
            p = np.minimum(
                1.0,
                2.0 * np.minimum((d_boot <= 0).mean(0), (d_boot >= 0).mean(0)),
            )
            excl = (lo > 0) | (hi < 0)
            for i, a in enumerate(ALPHAS):
                filas.append({
                    "region": region,
                    "comparison": f"M3 - {ref}",
                    "alpha": float(a),
                    "v_m3": float(v_pt["M3"][i]),
                    "v_ref": float(v_pt[ref][i]),
                    "delta_v": float(d_pt[i]),
                    "ci_low": float(lo[i]),
                    "ci_high": float(hi[i]),
                    "p_value": float(p[i]),
                    "excludes_zero": bool(excl[i]),
                    "tau_ref": float(tau[i]) if ref == "FWI" else np.nan,
                    "n_bootstrap": B,
                })
            print(
                f"{region:8s} M3-{ref:3s} excluye 0 en alpha: "
                f"{[round(float(a),4) for a in ALPHAS[excl]].__str__()}"
            )

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS24_delta_valor.csv")
    _tabla_s24(df)
    print(f"{len(df)} filas -> tablaS24_delta_valor.csv")


def _tabla_s24(df):
    lineas = [
        "| Region | Comparison | α | V(M3) | V(ref) | ΔV | 95% CI | p | CI excludes 0 |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in df.iter_rows(named=True):
        p = "≤0.001" if r["p_value"] <= 0.001 else f"{r['p_value']:.3f}"
        lineas.append(
            f"| {r['region']} | {r['comparison']} | {r['alpha']:.4f} | "
            f"{r['v_m3']:.4f} | {r['v_ref']:.4f} | {r['delta_v']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {p} | "
            f"{'yes' if r['excludes_zero'] else 'no'} |"
        )
    bloque = "\n".join(lineas)
    (DIR_TABLAS / "tablaS24_delta_valor.md").write_text(
        "# Table S24. Paired Delta V(alpha) bootstrap\n\n" + bloque + "\n",
        encoding="utf-8",
    )
    for path in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario_es.qmd",
    ):
        if path.exists():
            _splice(path, bloque, "tablaS24")


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
