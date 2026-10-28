"""Script 62 (R.8): variantes de calidad de etiquetas VIIRS.

Variantes disponibles con el archivo crudo: confianza alta (h) y FRP >= 10 MW.
La variante de nubosidad < 40 % no se puede construir porque el archive FIRMS
no contiene una mascara de nubes; se documenta como limitacion.

Las variantes se aplican a las filas positivas del muestreo 1:10 congelado.
Los negativos se mantienen. Es una sensibilidad de etiquetas, no una
reconstruccion completa del operador D2.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import DIR_PROCESSED, DIR_RAW, DIR_TABLAS
from src.firms import agregar_focos_diarios, procesar_focos_firms_a_grilla
from src.metricas import calcular_umbral_optimo_f_beta
from src.modelo import entrenar_modelo_lightgbm, preparar_matrices_xy

FRP_MIN = 10.0


def _focos_variantes():
    grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "departamento", "region"])
    raw = pd.read_parquet(DIR_RAW / "firms" / "viirs_historico_2018_2025_crudo.parquet")
    det = procesar_focos_firms_a_grilla(raw, grilla, excluir_baja_confianza=True)
    det = det.with_columns(pl.col("acq_date").cast(pl.String).str.slice(0, 10).alias("fecha"))
    variantes = {}
    for nombre, cond in (
        ("baseline", pl.lit(True)),
        ("high_confidence", pl.col("confidence").cast(pl.String).str.to_lowercase() == "h"),
        (f"frp_ge_{FRP_MIN:g}", pl.col("frp") >= FRP_MIN),
    ):
        sub = det.filter(cond)
        agg = agregar_focos_diarios(sub).with_columns(pl.col("acq_date").cast(pl.String).str.slice(0, 10).alias("fecha"))
        variantes[nombre] = agg.select(["cell_id", "fecha"]).unique().with_columns(pl.lit(True).alias("_tiene"))
    return variantes


def _prep(split, region, variante, focos, nubes=None):
    d = pl.read_parquet(DIR_PROCESSED / "targets_poblacionales" / f"{split}_ignicion.parquet").filter(pl.col("region") == region)
    if variante == "cloud_lt_40":
        grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet", columns=["cell_id", "nodo_clima_id"])
        d = d.join(grilla, on="cell_id", how="left").join(nubes, on=["nodo_clima_id", "fecha"], how="left")
        d = d.filter((pl.col("target_ignicion") == 0) | (pl.col("tcc") < 0.40))
    elif variante != "baseline":
        d = d.join(focos, on=["cell_id", "fecha"], how="left", coalesce=True)
        # Descartar solo positivos que no cumplen la variante; los negativos quedan.
        d = d.filter((pl.col("target_ignicion") == 0) | pl.col("_tiene")).drop("_tiene")
    return d


def _eval(region, tr, va, te):
    filas = []
    for nombre, modo in (("M2", "ignicion"), ("M3", "hibrido")):
        Xtr, ytr, feats = preparar_matrices_xy(tr, region, modo=modo)
        Xva, yva, _ = preparar_matrices_xy(va, region, modo=modo)
        Xte, yte, _ = preparar_matrices_xy(te, region, modo=modo)
        b = entrenar_modelo_lightgbm(
            Xtr, ytr, Xva, yva, feats,
            scale_pos_weight=1.0 if region == "Chaco" else 2.0,
            num_leaves=23 if region == "Chaco" else 31,
            min_child_samples=20 if region == "Chaco" else 10,
            num_boost_round=45 if region == "Chaco" else 60,
            early_stopping_rounds=None, semilla=42,
        )
        pva = b.predict(Xva); pte = b.predict(Xte)
        cut, _ = calcular_umbral_optimo_f_beta(yva, pva, beta=0.5)
        yb = (pte >= cut).astype(int)
        tp = int(((yb == 1) & (yte == 1)).sum()); fp = int(((yb == 1) & (yte == 0)).sum())
        fn = int(((yb == 0) & (yte == 1)).sum())
        filas.append({
            "model": nombre, "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
            "pr_auc": round(float(average_precision_score(yte, pte)), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        })
    return filas


def main():
    focos = _focos_variantes()
    filas = []
    for region in ("Chaco", "Oriental"):
        base_tr = _prep("train", region, "baseline", focos["baseline"])
        base_va = _prep("val", region, "baseline", focos["baseline"])
        base_te = _prep("test", region, "baseline", focos["baseline"])
        p0 = int(base_te["target_ignicion"].sum())
        ruta_nubes = DIR_PROCESSED / "nubes_era5_diario.parquet"
        nubes = pl.read_parquet(ruta_nubes).unique(subset=["nodo_clima_id", "fecha"]) if ruta_nubes.exists() else None
        variantes = list(focos.keys()) + (["cloud_lt_40"] if nubes is not None else [])
        for var in variantes:
            foco = focos.get(var, focos["baseline"])
            tr = _prep("train", region, var, foco, nubes=nubes)
            va = _prep("val", region, var, foco, nubes=nubes)
            te = _prep("test", region, var, foco, nubes=nubes)
            if int((tr["target_ignicion"] == 1).sum()) < 100:
                continue
            met = _eval(region, tr, va, te)
            for m in met:
                filas.append({
                    "region": region, "variant": var, "model": m["model"],
                    "n_train": tr.height, "n_test": te.height,
                    "pos_train": int((tr["target_ignicion"] == 1).sum()),
                    "pos_test": int((te["target_ignicion"] == 1).sum()),
                    "pos_retained": round(int((te["target_ignicion"] == 1).sum()) / p0, 3),
                    **{k: v for k, v in m.items() if k != "model"},
                })
            print(region, var, "done", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(DIR_TABLAS / "tablaS30_variantes_viirs.csv", index=False)
    _md(df)
    print(df)


def _md(df):
    lineas = [
        "| Region | Variant | Model | Pos train | Pos test | Pos retained | ROC-AUC | PR-AUC | Precision | Recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples(index=False):
        lineas.append(
            f"| {r.region} | {r.variant} | {r.model} | {r.pos_train:,} | {r.pos_test:,} | "
            f"{r.pos_retained:.3f} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.precision:.4f} | {r.recall:.4f} |"
        )
    (DIR_TABLAS / "tablaS30_variantes_viirs.md").write_text(
        "# Table S30. VIIRS label-quality variants\n\n" + "\n".join(lineas) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
