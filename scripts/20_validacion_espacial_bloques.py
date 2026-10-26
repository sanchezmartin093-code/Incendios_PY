"""
Script 20: Validación espacial por bloques con buffer de 20 km.

Diseño:
- Se dividen las celdas en bloques de 50 x 50 km.
- ~20 % de los bloques se retienen (holdout) de forma determinística.
- Se excluyen del entrenamiento las celdas de holdout y las celdas de train a
  menos de 20 km de cualquier celda de holdout (buffer espacial).
- Los priors históricos se recalculan excluyendo los fuegos de las celdas de
  holdout, para que éstas no se informen a sí mismas.
- El modelo M3 se entrena con 2018–2022 (bloques retenidos), se calibra en
  2023 (bloques retenidos) y se evalúa en 2024–2025 sobre:
    * bloques holdout (generalización espacial + temporal)
    * bloques no holdout (referencia temporal)

Exporta `tablas/tabla08_validacion_espacial_bloques.csv/.md`.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score

from src.constantes import DIR_PROCESSED, DIR_TABLAS
from src.espaciotemporal import (
    agregar_features_antropicas,
    enriquecer_dataset_espaciotemporal,
)
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_bootstrap_ci_roc,
    calcular_expected_calibration_error,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    FEATURES_HIBRIDO_CHACO,
    FEATURES_HIBRIDO_ORIENTAL,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)

BLOQUE_M = 50000.0
BUFFER_M = 20000.0
PROPORCION_HOLDOUT = 5  # 1 de cada 5 bloques

FEATURES_M1_BASE = [
    "temp_max_c", "temp_min_c", "rango_termico_diario", "vpd_max_kpa",
    "deficit_hidrico_acumulado", "precipitacion_mm", "precipitacion_48h_mm",
    "dias_secos_consecutivos", "viento_max_kmh", "potencial_viento_cuadratico",
    "interaccion_viento_vpd", "ffmc", "isi", "pct_pastura",
    "pct_sabana_pastizal", "pct_humedal_palmar",
]


def _bloque_id(df: pl.DataFrame) -> pl.Series:
    bx = (pl.col("x_utm") // BLOQUE_M).cast(pl.Int64)
    by = (pl.col("y_utm") // BLOQUE_M).cast(pl.Int64)
    return df.select(
        (bx * 1_000_003 + by).alias("block_id")
    )["block_id"]


def _mascara_holdout_global(*dfs: pl.DataFrame) -> tuple[set[int], set[int], float]:
    celdas = pl.concat([
        d.select(["cell_id", "x_utm", "y_utm"]) for d in dfs
    ]).unique("cell_id")
    block = _bloque_id(celdas)
    celdas = celdas.with_columns(block.alias("block_id"))

    # Hash determinístico por bloque (no por celda)
    holdout_blocks = (
        celdas.select("block_id").unique()
        .with_columns(
            (
                (pl.col("block_id") * 7_389_560_93).cast(pl.Int64)
                ^ (pl.col("block_id") * 1_934_966_3).cast(pl.Int64)
            ).alias("_hash")
        )
        .filter((pl.col("_hash") % PROPORCION_HOLDOUT) == 0)
        ["block_id"].to_list()
    )
    mask_holdout = celdas["block_id"].is_in(holdout_blocks)
    holdout_cells = set(celdas.filter(mask_holdout)["cell_id"].to_list())
    heldin_cells = set(celdas.filter(~mask_holdout)["cell_id"].to_list())
    frac = len(holdout_cells) / len(celdas)
    return holdout_cells, heldin_cells, frac


def _celdas_buffer(holdout_cells: set[int], train_cells: pl.DataFrame) -> set[int]:
    """Celdas de train a <20 km de una celda holdout."""
    ref = train_cells.filter(pl.col("cell_id").is_in(list(holdout_cells)))
    if ref.is_empty():
        return set()
    coords_ref = ref.select(["x_utm", "y_utm"]).to_numpy()
    tree = cKDTree(coords_ref)
    coords_train = train_cells.select(["x_utm", "y_utm"]).to_numpy()
    dists, _ = tree.query(coords_train, k=1, workers=-1)
    ids = train_cells["cell_id"].to_numpy()
    return set(ids[dists < BUFFER_M].tolist())


def _metricas(y, p, cutoff, nombre):
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float(np.mean(y))
    ci_l, ci_h = calcular_bootstrap_ci_roc(y, p, n_bootstrap=300) if len(np.unique(y)) > 1 else (0.5, 0.5)
    y_bin = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (y_bin == 1)).sum())
    fp = int(((y == 0) & (y_bin == 1)).sum())
    fn = int(((y == 1) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1 + 0.25) * prec * rec / (0.25 * prec + rec) if (0.25 * prec + rec) else 0.0
    return {
        "evaluacion": nombre,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalencia": round(float(np.mean(y)), 6),
        "roc_auc": round(auc, 4),
        "roc_ci_95": f"[{ci_l:.3f} - {ci_h:.3f}]",
        "pr_auc": round(pr, 4),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "ece": round(calcular_expected_calibration_error(y, p), 4),
        "cutoff_f05": round(float(cutoff), 5),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "fp": fp,
        "fn": fn,
    }


def main():
    print("=" * 72)
    print("VALIDACIÓN ESPACIAL POR BLOQUES DE 50 km CON BUFFER DE 20 km")
    print("=" * 72)

    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    p_train = dir_targets / "train_ignicion.parquet"
    p_val = dir_targets / "val_ignicion.parquet"
    p_test = dir_targets / "test_ignicion.parquet"
    p_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    for p in [p_train, p_val, p_test, p_focos]:
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}")

    train = pl.read_parquet(p_train)
    val = pl.read_parquet(p_val)
    test = pl.read_parquet(p_test)
    focos = pl.read_parquet(p_focos)

    holdout_cells, heldin_cells, frac = _mascara_holdout_global(train, val, test)
    print(f"Bloques holdout: {frac*100:.1f}% de las celdas ({len(holdout_cells):,} celdas)")

    train_cells = train.select(["cell_id", "x_utm", "y_utm"]).unique("cell_id")
    buffer_cells = _celdas_buffer(holdout_cells, train_cells)
    print(f"Celdas de train excluidas por buffer 20 km: {len(buffer_cells):,}")

    # Recalcular priors excluyendo fuegos de celdas holdout.
    print("Recalculando priors causales excluyendo celdas holdout...")
    excluir = list(holdout_cells)
    train_e = agregar_features_antropicas(
        enriquecer_dataset_espaciotemporal(
            train, focos, usar_priors_causales=True, excluir_cell_ids=excluir
        )
    )
    val_e = agregar_features_antropicas(
        enriquecer_dataset_espaciotemporal(
            val, focos, usar_priors_causales=True, excluir_cell_ids=excluir
        )
    )
    test_e = agregar_features_antropicas(
        enriquecer_dataset_espaciotemporal(
            test, focos, usar_priors_causales=True, excluir_cell_ids=excluir
        )
    )

    # Particiones espaciales
    train_esp = train_e.filter(
        ~pl.col("cell_id").is_in(list(holdout_cells | buffer_cells))
    )
    val_esp = val_e.filter(~pl.col("cell_id").is_in(list(holdout_cells)))
    test_holdout = test_e.filter(pl.col("cell_id").is_in(list(holdout_cells)))
    test_heldin = test_e.filter(~pl.col("cell_id").is_in(list(holdout_cells)))

    filas = []
    for region in ["Chaco", "Oriental"]:
        print(f"\n--- {region.upper()} ---")
        tr = train_esp.filter(pl.col("region") == region)
        va = val_esp.filter(pl.col("region") == region)
        te_h = test_holdout.filter(pl.col("region") == region)
        te_i = test_heldin.filter(pl.col("region") == region)
        b_col = "pct_bosque_seco" if region == "Chaco" else "pct_bosque_humedo"
        if region == "Chaco":
            feats_m2 = FEATURES_IGNICION_CHACO
            feats_m3 = FEATURES_HIBRIDO_CHACO
        else:
            feats_m2 = FEATURES_IGNICION_ORIENTAL
            feats_m3 = FEATURES_HIBRIDO_ORIENTAL
        modelos = [
            ("M1_biofisico", FEATURES_M1_BASE + [b_col]),
            ("M2_base", feats_m2),
            ("M3_hibrido", feats_m3),
        ]

        y_tr = tr["target_ignicion"].to_numpy().astype(np.float32)
        y_va = va["target_ignicion"].to_numpy().astype(np.float32)
        y_th = te_h["target_ignicion"].to_numpy().astype(int)
        y_ti = te_i["target_ignicion"].to_numpy().astype(int)

        print(
            f"train={len(tr):,} val={len(va):,} "
            f"test_holdout={len(te_h):,} test_heldin={len(te_i):,}"
        )

        spw = 1.0 if region == "Chaco" else 2.0
        leaves = 23 if region == "Chaco" else 31
        rounds = 45 if region == "Chaco" else 60

        for nombre_modelo, feats in modelos:
            X_tr = tr.select(feats).to_numpy()
            X_va = va.select(feats).to_numpy()
            X_th = te_h.select(feats).to_numpy()
            X_ti = te_i.select(feats).to_numpy()

            booster = entrenar_modelo_lightgbm(
                X_tr, y_tr, X_va, y_va, feats,
                scale_pos_weight=spw, num_boost_round=rounds,
                num_leaves=leaves, min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None,
            )
            cal = calibrar_probabilidades_isotonica(booster, X_va, y_va, metodo="platt_monotono")
            p_va = predecir_probabilidad(booster, cal, X_va)
            cutoff, _ = calcular_umbral_optimo_f_beta(y_va, p_va, beta=0.5)

            p_th = predecir_probabilidad(booster, cal, X_th)
            p_ti = predecir_probabilidad(booster, cal, X_ti)

            for nombre_eval, y_eval, p_eval in [
                ("holdout_espacial", y_th, p_th),
                ("heldin_referencia", y_ti, p_ti),
            ]:
                m = _metricas(y_eval, p_eval, cutoff, nombre_eval)
                m["region"] = region
                m["modelo"] = nombre_modelo
                m["n_train"] = int(len(tr))
                m["n_val"] = int(len(va))
                filas.append(m)

    df = pd.DataFrame(filas)
    # Reordenar columnas
    cols = ["region", "modelo", "evaluacion", "n", "n_pos", "prevalencia", "roc_auc", "roc_ci_95",
            "pr_auc", "brier", "bss", "ece", "cutoff_f05", "precision", "recall",
            "f05", "fp", "fn", "n_train", "n_val"]
    df = df[cols]
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    p_csv = DIR_TABLAS / "tabla08_validacion_espacial_bloques.csv"
    p_md = DIR_TABLAS / "tabla08_validacion_espacial_bloques.md"
    df.to_csv(p_csv, index=False)
    p_md.write_text(
        "# Tabla 8. Validación espacial por bloques de 50 km con buffer de 20 km\n\n"
        "El modelo M3 se entrena excluyendo los bloques holdout y un buffer de "
        "20 km; los priors también excluyen los fuegos de las celdas holdout. "
        "`heldin_referencia` usa las mismas fechas 2024–2025 en bloques no "
        "retenidos. La diferencia entre ambas filas estima la pérdida por "
        "generalización espacial.\n\n"
        + df.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n✓ Guardado en {p_csv.name} y {p_md.name}")
    print(df.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
