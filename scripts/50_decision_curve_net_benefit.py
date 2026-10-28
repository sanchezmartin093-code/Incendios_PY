"""Script 50 (C.2): curvas de beneficio neto (decision curve analysis) en la muestra caso-control.

NB(p_t) = TP/N - (FP/N) * p_t/(1 - p_t), expresado por 1000 celdas-dia
de la muestra caso-control. Modelos calibrados: alerta si p >= p_t. FWI e INPE
no son probabilidades: para cada p_t se elige en validacion 2023 el umbral del
indice que maximiza NB y se congela para el test.

Salidas:
- entregables/tablas/tablaS23_net_benefit.csv
- entregables/figuras/supp_fig04_net_benefit.png|.pdf
- Tabla S23 compacta insertada entre <!-- tablaS23:inicio/fin -->.
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

from src.constantes import DIR_FIGURAS, DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.inpe import constante_a_desde_fracciones
from src.modelo import cargar_modelo_y_calibrador, predecir_probabilidad, preparar_matrices_xy

PT_IGN = np.linspace(0.0, 0.30, 61)
PT_PROP = np.linspace(0.0, 0.90, 91)
SEL_IGN = [0.05, 0.10, 0.20]
SEL_PROP = [0.30, 0.50, 0.70]
COLORES = {
    "M1": "#7f7f7f", "M2": "#d95f02", "M3": "#1f4e79", "Gated": "#6a3d9a",
    "FWI": "#c0392b", "INPE": "#8c564b", "Propagation": "#1b7837",
    "No alert": "#000000", "Alert all": "#999999", "Climatology": "#666666",
}


def _conteos(region, task, ini, fin):
    return pl.read_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet").filter(
        (pl.col("task") == task) & (pl.col("region") == region)
        & pl.col("fecha").str.slice(0, 4).cast(pl.Int32).is_between(ini, fin)
    )


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


def _indices(df):
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
    if "_i" not in df.columns:
        df = df.with_row_index("_i")
    out = (
        df.join(g, on="cell_id", how="left")
        .join(f, on=["nodo_clima_id", "fecha"], how="left")
        .join(i, on=["nodo_clima_id", "fecha"], how="left")
        .sort("_i")
    )
    if out.height != df.height or out["fwi"].null_count() or out["pse"].null_count():
        raise ValueError("Join de indices incompleto")
    a = np.array([
        constante_a_desde_fracciones(p, s, bs, bh, palm)
        for p, s, bs, bh, palm in zip(
            out["pct_pastura"].to_numpy(), out["pct_sabana_pastizal"].to_numpy(),
            out["pct_bosque_seco"].to_numpy(), out["pct_bosque_humedo"].to_numpy(),
            out["pct_humedal_palmar"].to_numpy(),
        )
    ])
    rb = 0.8 * (1 + np.sin((a * out["pse"].to_numpy() / 120.0) * (np.pi / 2) - (np.pi / 2))) / 2
    fur = -0.006 * out["rh_min_pct"].to_numpy() + 1.3
    ft = 0.02 * out["temp_max_c"].to_numpy() + 0.4
    flat = 1 + np.abs(out["lat"].to_numpy()) * 0.003
    out = out.with_columns(pl.Series("p_inpe_calc", np.clip(rb * fur * ft * flat, 0, 1)))
    return out


def _nb(y, mask, pt):
    tp = float(((y == 1) & mask).sum())
    fp = float(((y == 0) & mask).sum())
    n = float(len(y))
    return 1000.0 * (tp - fp * pt / (1.0 - pt)) / n


def _curva(y, score, pts):
    return np.array([_nb(y, score >= pt, pt) for pt in pts])


def _tau_indice(yv, sv, pts):
    orden = np.argsort(-sv)
    ss = sv[orden]
    ys = yv[orden]
    ctp = np.concatenate([[0.0], np.cumsum(ys)])
    cfp = np.concatenate([[0.0], np.cumsum(1 - ys)])
    n = float(len(yv))
    j = np.unique(np.round(np.linspace(0, len(sv), 801)).astype(int))
    j = np.clip(j, 0, len(sv))
    tp = ctp[j][None, :]
    fp = cfp[j][None, :]
    p = pts[:, None]
    nb = 1000.0 * (tp - fp * p / (1.0 - p)) / n
    best = np.argmax(nb, axis=1)
    return np.where(j[best] == 0, np.inf, ss[np.maximum(j[best] - 1, 0)])


def _refs(y, pts, pi_s):
    n = float(len(y))
    tp = float((y == 1).sum())
    fp = float((y == 0).sum())
    all_nb = np.array([1000.0 * (tp - fp * pt / (1.0 - pt)) / n for pt in pts])
    clim = np.where(pts <= pi_s, all_nb, 0.0)
    return all_nb, clim


def main():
    filas = []
    plt.rcParams.update({"font.family": "Arial", "font.size": 7})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=300)

    for fila, task in enumerate(("ignicion", "propagacion")):
        pts = PT_IGN if task == "ignicion" else PT_PROP
        for col, region in enumerate(("Chaco", "Oriental")):
            test = pl.read_parquet(DIR_PROCESSED / f"evaluacion_test_{task}.parquet").filter(
                pl.col("region") == region
            )
            c_test = _conteos(region, task, 2024, 2025)
            base, y, _ = _pesos(test, c_test, f"target_{task}")
            pi_s = float(y.mean())

            modelos = {}
            if task == "ignicion":
                base = _indices(base)
                y = base["target_ignicion"].to_numpy().astype(int)
                p2 = base["probabilidad_calibrada"].to_numpy()
                p3 = base["probabilidad_hibrido"].to_numpy()
                p_m1 = _m1(base, region)
                gate = base["freq_hist_1km"].to_numpy() <= 0
                modelos = {
                    "M3": p3, "M2": p2, "M1": p_m1,
                    "Gated": np.where(gate, p2, p3),
                }

                val = pl.read_parquet(
                    DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet"
                ).filter(pl.col("region") == region)
                c_val = _conteos(region, "ignicion", 2023, 2023)
                val, yv, _ = _pesos(val, c_val, "target_ignicion")
                val = _indices(val)
                yv = val["target_ignicion"].to_numpy().astype(int)
                for nombre, colv in (("FWI", "fwi"), ("INPE", "p_inpe_calc")):
                    tau = _tau_indice(yv, val[colv].to_numpy(), pts)
                    modelos[nombre] = (base[colv].to_numpy(), tau)
            else:
                modelos = {"Propagation": base["probabilidad_propagacion"].to_numpy()}

            for nombre, val_modelo in modelos.items():
                if isinstance(val_modelo, tuple):
                    score, tau = val_modelo
                    nb = np.array([
                        _nb(y, score >= tau[i], pt) for i, pt in enumerate(pts)
                    ])
                else:
                    nb = _curva(y, val_modelo, pts)
                for pt, v in zip(pts, nb):
                    filas.append({
                        "region": region, "task": task, "model": nombre,
                        "pt": round(float(pt), 6), "nb_per_1000": float(v),
                    })

            all_nb, clim = _refs(y, pts, pi_s)
            for nombre, vals in (("No alert", np.zeros_like(pts)), ("Alert all", all_nb), ("Climatology", clim)):
                for pt, v in zip(pts, vals):
                    filas.append({
                        "region": region, "task": task, "model": nombre,
                        "pt": round(float(pt), 6), "nb_per_1000": float(v),
                    })

            ax = axes[fila, col]
            ax.plot(pts, np.zeros_like(pts), color=COLORES["No alert"], lw=1.0, ls="-")
            ax.plot(pts, all_nb, color=COLORES["Alert all"], lw=1.0, ls=":")
            ax.plot(pts, clim, color=COLORES["Climatology"], lw=1.0, ls="-.")
            for nombre, val_modelo in modelos.items():
                score = val_modelo[0] if isinstance(val_modelo, tuple) else val_modelo
                if isinstance(val_modelo, tuple):
                    nb = np.array([_nb(y, score >= val_modelo[1][i], pt) for i, pt in enumerate(pts)])
                else:
                    nb = _curva(y, score, pts)
                ax.plot(pts, nb, color=COLORES[nombre], lw=1.3, label=nombre,
                        ls="--" if nombre == "Gated" else ("-." if nombre == "INPE" else (":" if nombre == "FWI" else "-")))
            ax.set_xlim(0, pts[-1])
            ax.set_ylim(-20, 110)
            ax.set_xlabel("Threshold probability $p_t$")
            ax.set_ylabel("Net benefit per 1 000 sampled cell-days")
            ax.set_title(f"{region} — {'Ignition' if task == 'ignicion' else 'Propagation'}", fontsize=7.5)
            ax.grid(alpha=0.25, lw=0.4)

    handles = [
        plt.Line2D([], [], color=COLORES[n], lw=1.3, label=n)
        for n in ("M1", "M2", "M3", "Gated", "FWI", "INPE", "Propagation", "No alert", "Alert all", "Climatology")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for nombre in ("supp_fig04_net_benefit",):
        fig.savefig(DIR_FIGURAS / f"{nombre}.png", dpi=300, bbox_inches="tight")
        fig.savefig(DIR_FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)

    df = pl.DataFrame(filas)
    df.write_csv(DIR_TABLAS / "tablaS23_net_benefit.csv")
    _tabla_s23(df)
    print(f"{len(df)} filas -> tablaS23_net_benefit.csv")


def _m1(base, region):
    b, c = cargar_modelo_y_calibrador(
        DIR_MODELOS / f"lightgbm_m1_{region.lower()}.booster",
        DIR_MODELOS / "calibradores" / f"calibrador_m1_{region.lower()}.joblib",
    )
    X, _, _ = preparar_matrices_xy(base, region, modo="m1")
    if X.shape[0] != base.height:
        raise ValueError("M1 no alineado")
    return predecir_probabilidad(b, c, X)


def _tabla_s23(df):
    lineas = [
        "| Region | Task | Model | $p_t$ | NB per 1 000 |",
        "|---|---|---|---:|---:|",
    ]
    for region in ("Chaco", "Oriental"):
        for task, sels, modelos in (
            ("ignicion", SEL_IGN, ("M1", "M2", "M3", "Gated", "FWI", "INPE")),
            ("propagacion", SEL_PROP, ("Propagation",)),
        ):
            for pt in sels:
                for m in modelos:
                    r = df.filter(
                        (pl.col("region") == region) & (pl.col("task") == task)
                        & (pl.col("model") == m)
                        & ((pl.col("pt") - float(pt)).abs() < 1e-9)
                    )
                    if r.height:
                        lineas.append(
                            f"| {region} | {task} | {m} | {pt:.2f} | {float(r['nb_per_1000'][0]):.3f} |"
                        )
    bloque = "\n".join(lineas)
    (DIR_TABLAS / "tablaS23_net_benefit.md").write_text(
        "# Table S23. Net benefit at selected thresholds\n\n" + bloque + "\n",
        encoding="utf-8",
    )
    for path in (
        ROOT_DIR / "manuscrito" / "en" / "material_suplementario.qmd",
        ROOT_DIR / "manuscrito" / "es" / "material_suplementario_es.qmd",
    ):
        if path.exists():
            _splice(path, bloque, "tablaS23")


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
