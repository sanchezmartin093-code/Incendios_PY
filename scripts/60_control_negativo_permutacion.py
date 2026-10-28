"""Script 60 (R.10): control negativo y permutation importance.

- Control negativo: M3 con etiquetas de entrenamiento permutadas; el AUC de
  test debe caer a ~0.5.
- Permutation importance: caida de ROC-AUC al permutar cada variable y cada
  familia de variables en el test.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from src.constantes import DIR_FIGURAS, DIR_MODELOS, DIR_PROCESSED, DIR_TABLAS
from src.modelo import (
    cargar_modelo_y_calibrador,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
    preparar_matrices_xy,
)

HISTORICAS = {"dist_foco_hist_km", "densidad_hist_25km", "freq_hist_1km"}

FAMILIAS = {
    "weather": ["temp_max_c", "temp_min_c", "rango_termico_diario", "vpd_max_kpa",
                "deficit_hidrico_acumulado", "precipitacion_mm", "precipitacion_48h_mm",
                "dias_secos_consecutivos", "viento_max_kmh", "potencial_viento_cuadratico",
                "interaccion_viento_vpd", "ffmc", "isi", "swvl1_m3m3",
                "fdd_heladas_60d", "deficit_hidro_60d", "riesgo_estero_seco",
                "combustible_curado_helada"],
    "fuel": ["pct_pastura_ganadera", "pct_agricultura_anual", "pct_plantacion_forestal",
             "pct_sabana_pastizal", "pct_humedal_palmar", "pct_bosque_seco",
             "pct_bosque_humedo", "indice_ecotono", "borde_bosque_agro_pct",
             "pct_desmonte_t1", "pct_desmonte_t2", "presion_desmonte_vpd"],
    "anthropogenic": ["dist_corredor_antropico_km", "densidad_hist_5km", "densidad_hist_10km",
                      "temporada_quema_ganadera", "presion_quema_pastura",
                      "focalizacion_antropica_local", "interaccion_acceso_vpd"],
    "historical": ["dist_foco_hist_km", "densidad_hist_25km", "freq_hist_1km"],
    "calendar": ["dia_semana", "es_fin_de_semana", "mes", "dia_del_anio", "doy_sin", "doy_cos"],
}


def _negativo(region, drop_hist=False):
    tr = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "train_ignicion.parquet").filter(pl.col("region") == region)
    va = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "val_ignicion.parquet").filter(pl.col("region") == region)
    te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet").filter(pl.col("region") == region)
    Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo="hibrido")
    Xva, yva, _ = preparar_matrices_xy(va, region, modo="hibrido")
    Xte, yte, _ = preparar_matrices_xy(te, region, modo="hibrido")
    if drop_hist:
        keep = [i for i, f in enumerate(feats) if f not in HISTORICAS]
        Xtr, Xva, Xte = Xtr[:, keep], Xva[:, keep], Xte[:, keep]
        feats = [feats[i] for i in keep]
    rng = np.random.default_rng(123)
    y_perm = rng.permutation(ytr)
    b = entrenar_modelo_lightgbm(
        Xtr, y_perm, Xva, yva, feats,
        scale_pos_weight=1.0 if region == "Chaco" else 2.0,
        num_leaves=23 if region == "Chaco" else 31,
        min_child_samples=20 if region == "Chaco" else 10,
        num_boost_round=45 if region == "Chaco" else 60,
        early_stopping_rounds=None, semilla=42,
    )
    p = b.predict(Xte)
    return float(roc_auc_score(yte, p)), yte, p, feats


def _permutacion(region):
    te = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / "test_ignicion.parquet").filter(pl.col("region") == region)
    X, y, feats = preparar_matrices_xy(te, region, modo="hibrido")
    b, c = cargar_modelo_y_calibrador(
        DIR_MODELOS / f"lightgbm_hibrido_{region.lower()}.booster",
        DIR_MODELOS / "calibradores" / f"calibrador_hibrido_{region.lower()}.joblib",
    )
    base = float(roc_auc_score(y, predecir_probabilidad(b, c, X)))
    rng = np.random.default_rng(42)
    filas = []
    for j, f in enumerate(feats):
        drops = []
        for _ in range(3):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - float(roc_auc_score(y, predecir_probabilidad(b, c, Xp))))
        filas.append({"region": region, "feature": f, "delta_roc": float(np.mean(drops))})
    fam = []
    for familia, cols in FAMILIAS.items():
        idx = [feats.index(c) for c in cols if c in feats]
        if not idx:
            continue
        drops = []
        for _ in range(3):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])
            Xp[:, idx] = Xp[perm][:, idx]
            drops.append(base - float(roc_auc_score(y, predecir_probabilidad(b, c, Xp))))
        fam.append({"region": region, "family": familia, "n_features": len(idx),
                    "delta_roc": float(np.mean(drops))})
    return base, filas, fam


def main():
    neg = []
    perm = []
    familias = []
    for region in ("Chaco", "Oriental"):
        for drop, label in ((False, "all features"), (True, "without historical")):
            auc, _, _, _ = _negativo(region, drop_hist=drop)
            neg.append({"region": region, "setting": label, "negative_control_auc": round(auc, 4)})
        base, pf, ff = _permutacion(region)
        perm.extend(pf)
        familias.extend(ff)
        print(region, "negative", round(auc, 4), "base", round(base, 4), flush=True)

    df_neg = pd.DataFrame(neg)
    df_perm = pd.DataFrame(perm).sort_values(["region", "delta_roc"], ascending=[True, False])
    df_fam = pd.DataFrame(familias)
    df_neg.to_csv(DIR_TABLAS / "tablaS32_control_negativo.csv", index=False)
    df_perm.to_csv(DIR_TABLAS / "tablaS32_permutacion.csv", index=False)
    df_fam.to_csv(DIR_TABLAS / "tablaS32_familias.csv", index=False)
    _md(df_neg, df_perm, df_fam)
    _fig(df_perm)


def _md(neg, perm, fam):
    lineas = ["# Table S32. Negative control and permutation importance", "",
              "| Region | Setting | Negative-control AUC |", "|---|---|---:|"]
    for r in neg.itertuples(index=False):
        lineas.append(f"| {r.region} | {r.setting} | {r.negative_control_auc:.4f} |")
    lineas += ["", "| Region | Family | Features | ΔROC |", "|---|---|---:|---:|"]
    for r in fam.itertuples(index=False):
        lineas.append(f"| {r.region} | {r.family} | {r.n_features} | {r.delta_roc:.4f} |")
    lineas += ["", "| Region | Feature | ΔROC |", "|---|---|---:|"]
    for r in perm.groupby("region").head(10).itertuples(index=False):
        lineas.append(f"| {r.region} | `{r.feature}` | {r.delta_roc:.4f} |")
    (DIR_TABLAS / "tablaS32_control_negativo.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _fig(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    for ax, region in zip(axes, ("Chaco", "Oriental")):
        d = df[df.region == region].head(15).sort_values("delta_roc")
        ax.barh(d.feature, d.delta_roc, color="#d62728")
        ax.set_title(region)
        ax.set_xlabel("ΔROC after permutation")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    ruta = DIR_FIGURAS / "supp_fig07_permutation_importance"
    fig.savefig(ruta.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
