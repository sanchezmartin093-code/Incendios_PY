"""Script 68: validacion espacial externa leave-one-department-out (Boqueron).

- Excluye Boqueron + buffer de 25 km de train y validacion.
- Recalcula priors causales excluyendo esas celdas.
- Entrena M2/M3, calibra y elige cutoff F0.5 en 2023 sin Boqueron.
- Evalua 2024-2025 solo en Boqueron.
- Compara FWI e INPE con cutoffs seleccionados en 2023 sin Boqueron.
"""

from pathlib import Path
import os
import sys
from datetime import datetime, timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.constantes import DIR_FIGURAS, DIR_PROCESSED, DIR_TABLAS
from src.espaciotemporal import agregar_features_antropicas, enriquecer_dataset_espaciotemporal
from src.metricas import (
    bootstrap_diferencia_roc,
    calcular_bootstrap_ci_roc,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

DEPARTAMENTO = "BOQUERON"
BUFFER_KM = float(os.environ.get("BUFFER_KM", "25"))
COBERTURA = ["pct_pastura", "pct_sabana_pastizal", "pct_bosque_seco", "pct_bosque_humedo", "pct_humedal_palmar"]
PESOS = np.array([6.0, 3.0, 1.72, 1.72, 1.5])


def _population_counts():
    """Cuenta positivos/negativos poblacionales de Boqueron por fecha D2."""
    g = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    boq = g.filter(pl.col("departamento") == DEPARTAMENTO)
    ids = boq["cell_id"].to_numpy(); coords = np.column_stack([boq["x_utm"].to_numpy(), boq["y_utm"].to_numpy()])
    nodos = boq["nodo_clima_id"].to_numpy(); idx = {int(c): i for i, c in enumerate(ids)}
    tree = cKDTree(np.column_stack([g["x_utm"].to_numpy(), g["y_utm"].to_numpy()]))
    tree_boq = cKDTree(coords)
    all_ids = g["cell_id"].to_numpy()
    all_ids_set = set(all_ids.tolist())
    coord_map = {int(c): (float(x), float(y)) for c, x, y in zip(all_ids.tolist(), g["x_utm"].to_list(), g["y_utm"].to_list())}
    clim = pl.read_parquet(DIR_PROCESSED / "clima_era5_diario.parquet", columns=["nodo_clima_id", "fecha", "vpd_max_kpa", "precipitacion_48h_mm"])
    comb = pl.read_parquet(DIR_PROCESSED / "combustible_mapbiomas_1km_anual.parquet", columns=["cell_id", "anio_fila", "es_combustible"])
    comb = comb.filter(pl.col("cell_id").is_in(ids.tolist()))
    fuel = {}
    for anio in sorted(comb["anio_fila"].unique().to_list()):
        sub = comb.filter(pl.col("anio_fila") == anio)
        mp = dict(zip(sub["cell_id"].to_list(), sub["es_combustible"].to_list()))
        fuel[int(anio)] = np.array([bool(mp.get(int(c), False)) for c in ids])
    focos = pl.read_parquet(DIR_PROCESSED / "focos_viirs_diarios.parquet", columns=["cell_id", "acq_date"])
    focos = focos.with_columns(pl.col("acq_date").cast(pl.String).str.slice(0, 10).alias("fecha"))
    fire_by_date = {f[0]: sub["cell_id"].to_list() for f, sub in focos.group_by("fecha", maintain_order=True)}
    clim_by_date = {f[0]: sub for f, sub in clim.group_by("fecha", maintain_order=True)}
    dates = sorted(clim_by_date)
    ultimo_day = {}
    rows = []
    for f in dates:
        if int(f[:4]) not in fuel:
            continue
        dt = datetime.strptime(f, "%Y-%m-%d")
        c = clim_by_date[f]
        vpd = np.full(int(max(nodos.max(), c["nodo_clima_id"].max())) + 1, np.nan)
        rain = np.full_like(vpd, np.nan)
        vpd[c["nodo_clima_id"].to_numpy()] = c["vpd_max_kpa"].to_numpy(); rain[c["nodo_clima_id"].to_numpy()] = c["precipitacion_48h_mm"].to_numpy()
        gate = fuel[int(f[:4])] & (vpd[nodos] >= 0.8) & (rain[nodos] < 10.0)
        fire_cells = [int(x) for x in fire_by_date.get(f, [])]
        fire = np.zeros(len(ids), dtype=bool)
        for cid in fire_cells:
            j = idx.get(cid)
            if j is not None: fire[j] = True
        ayer = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        buffer = np.zeros(len(ids), dtype=bool)
        if fire_by_date.get(ayer):
            pts = np.array([coord_map[int(cid)] for cid in fire_by_date[ayer] if int(cid) in all_ids_set])
            if len(pts):
                for arr in tree_boq.query_ball_point(pts, r=25000.0):
                    buffer[arr] = True
        if f[:4] in ("2024", "2025"):
            recent = np.array([(int(cid) not in ultimo_day) or (dt - ultimo_day[int(cid)]).days > 10 for cid in ids], dtype=bool)
            pos = fire & gate & ~buffer & recent
            neg = gate & ~fire & ~buffer
            rows.append({"fecha": f, "n_pos": int(pos.sum()), "n_neg": int(neg.sum())})
        for cid in fire_cells:
            if cid in idx:
                ultimo_day[int(cid)] = dt
    return pl.DataFrame(rows)


def _buffer_cells():
    g = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    b = g.filter(pl.col("departamento") == DEPARTAMENTO)
    tree = cKDTree(np.column_stack([g["x_utm"].to_numpy(), g["y_utm"].to_numpy()]))
    idx = tree.query_ball_point(np.column_stack([b["x_utm"].to_numpy(), b["y_utm"].to_numpy()]), r=BUFFER_KM * 1000)
    cells = set()
    ids = g["cell_id"].to_numpy()
    for arr in idx:
        cells.update(ids[arr].tolist())
    return b["cell_id"].to_list(), sorted(cells)


def _enriquecer(df, focos, excluir=None):
    df = enriquecer_dataset_espaciotemporal(df, focos, usar_priors_causales=True, excluir_cell_ids=excluir)
    return agregar_features_antropicas(df)


def _join_indices(df):
    g = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"])
    fwi = pl.read_parquet(DIR_PROCESSED / "fwi_completo_diario.parquet", columns=["nodo_clima_id", "fecha", "fwi"])
    inpe = pl.read_parquet(DIR_PROCESSED / "inpe_diario.parquet")
    x = df.join(g, on="cell_id", how="left").join(fwi, on=["nodo_clima_id", "fecha"], how="left").join(inpe, on=["nodo_clima_id", "fecha"], how="left")
    f = x.select(COBERTURA).to_numpy().astype(float)
    total = f.sum(axis=1)
    a = np.divide(f @ PESOS, total, out=np.zeros(len(x)), where=total > 0)
    madurez = np.clip(a * x["pse"].to_numpy() / 120.0, 0.0, 2.0)
    rb = 0.8 * (1.0 + np.sin(madurez * (np.pi / 2.0) - (np.pi / 2.0))) / 2.0
    fur = -0.006 * x["rh_min_pct"].to_numpy() + 1.3
    ft = 0.02 * x["temp_max_c"].to_numpy() + 0.4
    flat = 1.0 + np.abs(x["lat"].to_numpy()) * 0.003
    return x.with_columns(pl.Series("inpe", np.clip(rb * fur * ft * flat, 0.0, 1.0)))


def _metricas(y, p, cut):
    yb = (p >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yb).ravel()
    return {
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f05": float(fbeta_score(y, yb, beta=0.5, zero_division=0)),
        "fp": int(fp), "tp": int(tp), "fn": int(fn),
    }


def main():
    tr = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet")
    va = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet")
    te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet")
    boq_cells, buffer_cells = _buffer_cells()
    tr = tr.filter(~pl.col("cell_id").is_in(buffer_cells))
    va = va.filter(~pl.col("cell_id").is_in(buffer_cells))
    te = te.filter(pl.col("departamento") == DEPARTAMENTO)
    print("train/val/test", tr.height, va.height, te.height, "pos", int(tr["target_ignicion"].sum()), int(va["target_ignicion"].sum()), int(te["target_ignicion"].sum()))

    focos = pl.read_parquet(DIR_PROCESSED / "focos_viirs_diarios.parquet")
    print("recalculando priors train...", flush=True)
    tr = _enriquecer(tr, focos, excluir=buffer_cells)
    print("recalculando priors val...", flush=True)
    va = _enriquecer(va, focos, excluir=buffer_cells)
    print("recalculando priors test...", flush=True)
    te = _enriquecer(te, focos, excluir=None)

    # indices para FWI/INPE
    va_i = _join_indices(va)
    te_i = _join_indices(te)
    yva = va_i["target_ignicion"].to_numpy().astype(int)
    yte = te_i["target_ignicion"].to_numpy().astype(int)

    rows = []
    probs = {}
    for name, mode in (("M2", "ignicion"), ("M3", "hibrido")):
        Xtr, ytr, feats = preparar_matrices_xy(tr, "Chaco", modo=mode)
        Xva, yva_m, _ = preparar_matrices_xy(va_i, "Chaco", modo=mode)
        Xte, yte_m, _ = preparar_matrices_xy(te_i, "Chaco", modo=mode)
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva_m, feats,
            scale_pos_weight=1.0, learning_rate=0.03, num_leaves=23,
            min_child_samples=20, num_boost_round=45,
            early_stopping_rounds=None, semilla=42,
        )
        c = calibrar_probabilidades_isotonica(b, Xva, yva_m, metodo="platt_monotono")
        pva = predecir_probabilidad(b, c, Xva)
        pte = predecir_probabilidad(b, c, Xte)
        cut, _ = calcular_umbral_optimo_f_beta(yva_m, pva, beta=0.5)
        m = _metricas(yte_m, pte, cut)
        probs[name] = pte
        lo, hi = calcular_bootstrap_ci_roc(yte_m, pte, n_bootstrap=500, block_ids=te_i["fecha"].to_numpy())
        rows.append({
            "model": name, "roc_auc": round(float(roc_auc_score(yte_m, pte)), 4),
            "ci": f"[{lo:.3f}–{hi:.3f}]", "pr_auc": round(float(average_precision_score(yte_m, pte)), 4),
            "brier": round(float(brier_score_loss(yte_m, pte)), 4),
            "ece": round(float(calcular_expected_calibration_error(yte_m, pte)), 4),
            "cutoff": round(float(cut), 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        })

    # FWI / INPE
    for name, col in (("FWI", "fwi"), ("INPE", "inpe")):
        pva = va_i[col].to_numpy().astype(float)
        pte = te_i[col].to_numpy().astype(float)
        cut, _ = calcular_umbral_optimo_f_beta(yva, pva, beta=0.5)
        m = _metricas(yte, pte, cut)
        probs[name] = pte
        lo, hi = calcular_bootstrap_ci_roc(yte, pte, n_bootstrap=500, block_ids=te_i["fecha"].to_numpy())
        rows.append({
            "model": name, "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
            "ci": f"[{lo:.3f}–{hi:.3f}]", "pr_auc": round(float(average_precision_score(yte, pte)), 4),
            "brier": np.nan, "ece": np.nan,
            "cutoff": round(float(cut), 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
        })

    delta, pval = bootstrap_diferencia_roc(yte, probs["M3"], probs["M2"], n_bootstrap=1000, block_ids=te_i["fecha"].to_numpy())

    # Pesos poblacionales departamentales
    counts = _population_counts()
    muestra = te_i.group_by(["fecha", "target_ignicion"]).len().rename({"len": "n_muestra"})
    b = te_i.with_row_index("_i").join(muestra, on=["fecha", "target_ignicion"]).join(counts, on="fecha").sort("_i")
    y_pob = b["target_ignicion"].to_numpy().astype(int)
    w_pob = np.where(y_pob == 1, b["n_pos"].to_numpy() / b["n_muestra"].to_numpy(), b["n_neg"].to_numpy() / b["n_muestra"].to_numpy())
    def _pob(p, cut):
        h = p >= cut
        tp = float((w_pob * ((y_pob == 1) & h)).sum()); fp = float((w_pob * ((y_pob == 0) & h)).sum()); fn = float((w_pob * ((y_pob == 1) & (~h))).sum())
        return (tp / (tp + fp) if (tp + fp) else 0.0, tp / (tp + fn) if (tp + fn) else 0.0)
    p_map = {"M2": probs["M2"], "M3": probs["M3"], "FWI": probs["FWI"], "INPE": probs["INPE"]}
    for r in rows:
        pp, rr = _pob(p_map[r["model"]], r["cutoff"])
        r["precision_pob"] = round(float(pp), 6); r["recall_pob"] = round(float(rr), 6)

    df = pd.DataFrame(rows)
    df["delta_vs_m2"] = np.nan
    df.loc[df.model == "M3", "delta_vs_m2"] = round(float(delta), 4)
    df["p_vs_m2"] = np.nan
    df.loc[df.model == "M3", "p_vs_m2"] = round(float(pval), 4)
    df.to_csv(DIR_TABLAS / (f"tablaS33_buffer_{BUFFER_KM:g}.csv" if BUFFER_KM != 25 else "tablaS33_validacion_externa_boqueron.csv"), index=False)
    _md(df)
    _fig(yte, probs)
    print(df)


def _md(df):
    lines = [
        "| Model | ROC-AUC [95 % CI] | PR-AUC | Brier | ECE | Cutoff | Precision | Recall | F0.5 | FP | ΔROC vs M2 (p) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        d = "—" if not np.isfinite(r.delta_vs_m2) else f"{r.delta_vs_m2:+.4f} ({r.p_vs_m2:.3f})"
        b = "—" if not np.isfinite(r.brier) else f"{r.brier:.4f}"
        e = "—" if not np.isfinite(r.ece) else f"{r.ece:.4f}"
        lines.append(f"| {r.model} | {r.roc_auc:.4f} {r.ci} | {r.pr_auc:.4f} | {b} | {e} | {r.cutoff:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f05:.4f} | {r.fp:,} | {d} |")
    (DIR_TABLAS / (f"tablaS33_buffer_{BUFFER_KM:g}.md" if BUFFER_KM != 25 else "tablaS33_validacion_externa_boqueron.md")).write_text(
        "# Table S33. External validation in Boquerón\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


def _fig(y, probs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("Boquerón — ROC")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Boquerón — PR")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout()
    ruta = DIR_FIGURAS / "supp_fig09_external_boqueron"
    fig.savefig(ruta.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
