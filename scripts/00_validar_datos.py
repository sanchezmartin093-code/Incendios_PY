"""
Script 00b: Validación de integridad y calidad de los datos procesados.

No entrena ni modifica nada. Genera un informe de invariantes que la suite de
tests unitarios no cubre: prevalencia real, fuga de priors, rangos físicos de
clima, cobertura de la grilla, tipos de FIRMS y estado del manifiesto.
"""

from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from src.constantes import DIR_PROCESSED, DIR_TABLAS, DIR_MODELOS

FILAS = []


def add(seccion: str, chequeo: str, valor, estado: str, nota: str = ""):
    FILAS.append({
        "seccion": seccion,
        "chequeo": chequeo,
        "valor": valor,
        "estado": estado,
        "nota": nota,
    })


def main():
    con = duckdb.connect()

    # 1. Grilla
    p_grilla = DIR_PROCESSED / "grilla_nacional_1km.parquet"
    if p_grilla.exists():
        df = pl.read_parquet(p_grilla)
        add("grilla", "filas", df.height, "OK" if df.height == 400006 else "FAIL")
        add("grilla", "cell_id_unicos", df["cell_id"].n_unique(), "OK" if df["cell_id"].n_unique() == df.height else "FAIL")
        regiones = df.group_by("region").len().sort("region").to_dicts()
        for r in regiones:
            add("grilla", f"celdas_{r['region']}", r["len"], "OK")
    else:
        add("grilla", "archivo", "ausente", "FAIL")

    # 2. Clima
    p_clima = DIR_PROCESSED / "clima_era5_diario.parquet"
    if p_clima.exists():
        clima = pl.read_parquet(p_clima)
        add("clima", "filas", clima.height, "OK")
        add("clima", "nodos", clima["nodo_clima_id"].n_unique(), "OK" if clima["nodo_clima_id"].n_unique() == 640 else "WARN")
        add("clima", "fecha_min", clima["fecha"].min(), "OK")
        add("clima", "fecha_max", clima["fecha"].max(), "OK")
        n_prec_extrema = int((clima["precipitacion_mm"] > 300.0).sum())
        add("clima", "precipitacion_mayor_300mm", n_prec_extrema, "WARN" if n_prec_extrema > 0 else "OK",
            "valores posibles pero improbables; revisar desacumulación")
        add("clima", "precipitacion_max_mm", round(float(clima["precipitacion_mm"].max()), 3),
            "WARN" if float(clima["precipitacion_mm"].max()) > 500 else "OK",
            "pendiente ejecutar scripts/02c_corregir_clima_openmeteo.py cuando haya cuota externa")
        add("clima", "et0_distinto_de_cero", int((clima["et0_mm"] != 0).sum()), "WARN")
        diff_rafaga = float((clima["rafaga_max_kmh"] - 1.3 * clima["viento_max_kmh"]).abs().max())
        add("clima", "rafaga_es_proxy_1.3x", round(diff_rafaga, 6), "WARN" if diff_rafaga < 1e-3 else "OK")
    else:
        add("clima", "archivo", "ausente", "FAIL")

    # 3. Focos FIRMS
    p_focos = DIR_PROCESSED / "focos_viirs_diarios.parquet"
    if p_focos.exists():
        focos = pl.read_parquet(p_focos)
        add("firms", "celda_dias_positivos", focos.height, "OK")
        add("firms", "fecha_min", focos["acq_date"].min(), "OK")
        add("firms", "fecha_max", focos["acq_date"].max(), "WARN",
            "puede exceder la ventana de clima")
    p_raw = ROOT_DIR / "datos" / "raw" / "firms" / "viirs_historico_2018_2025_crudo.parquet"
    if p_raw.exists():
        raw = con.sql(f"SELECT COUNT(*) n, MAX(scan) max_scan, SUM(CASE WHEN type<>0 THEN 1 ELSE 0 END) no_vegetal FROM '{p_raw}'").df()
        add("firms_raw", "detecciones", int(raw["n"][0]), "OK")
        add("firms_raw", "scan_max_km", float(raw["max_scan"][0]), "WARN",
            "si max_scan<=1.5 el filtro Bow-Tie no descarta registros")
        add("firms_raw", "detecciones_type_no_0", int(raw["no_vegetal"][0]), "WARN",
            "no se filtró type; revisar fuentes industriales/offshore")
    else:
        add("firms_raw", "archivo", "ausente", "WARN")

    # 4. Dataset unificado: prevalencia real
    p_ds = DIR_PROCESSED / "dataset_unificado_candidatos.parquet"
    if p_ds.exists():
        stats = con.sql(f"""
            SELECT COUNT(*) filas, SUM(fuego) fuegos, AVG(fuego) prevalencia,
                   COUNT(DISTINCT cell_id) celdas
            FROM '{p_ds}'
        """).df()
        add("dataset", "filas", int(stats["filas"][0]), "OK")
        add("dataset", "fuegos", int(stats["fuegos"][0]), "OK")
        add("dataset", "prevalencia", round(float(stats["prevalencia"][0]), 6), "OK",
            "prevalencia real de celdas-día candidatas")
        add("dataset", "celdas", int(stats["celdas"][0]), "OK")
    else:
        add("dataset", "archivo", "ausente", "FAIL")

    # 5. Particiones y fuga de priors
    dir_targets = DIR_PROCESSED / "targets_poblacionales"
    for nombre in ["train_ignicion", "val_ignicion", "test_ignicion"]:
        p = dir_targets / f"{nombre}.parquet"
        if not p.exists():
            add("particiones", nombre, "ausente", "FAIL")
            continue
        cols = [
            "region", "target_ignicion", "fecha", "cell_id",
            "dist_foco_hist_km", "freq_hist_1km", "dist_corredor_antropico_km",
        ]
        df = pl.read_parquet(p, columns=cols)
        prev = float(df["target_ignicion"].mean())
        add("particiones", f"{nombre}_filas", df.height, "OK")
        add("particiones", f"{nombre}_prevalencia", round(prev, 6),
            "OK" if 0.05 < prev < 0.15 else "WARN",
            "case-control esperada ~9%")
        for reg in ["Chaco", "Oriental"]:
            pos = df.filter((pl.col("region") == reg) & (pl.col("target_ignicion") == 1))
            if pos.height:
                pct_cero = 100.0 * float((pos["dist_foco_hist_km"] == 0).mean())
                add("priors", f"{nombre}_{reg}_pos_dist_hist_0_pct", round(pct_cero, 2),
                    "FAIL" if pct_cero > 99.9 else "OK",
                    "100% indicaría fuga del propio evento")

    # 6. Fuga espacial train/test
    p_tr = dir_targets / "train_ignicion.parquet"
    p_te = dir_targets / "test_ignicion.parquet"
    if p_tr.exists() and p_te.exists():
        tr = pl.read_parquet(p_tr, columns=["cell_id"])
        te = pl.read_parquet(p_te, columns=["cell_id"])
        tr_cells = set(tr["cell_id"].unique().to_list())
        te_cells = set(te["cell_id"].unique().to_list())
        overlap = len(tr_cells & te_cells) / max(1, len(te_cells))
        add("fuga_espacial", "pct_celdas_test_presentes_en_train", round(100 * overlap, 2),
            "WARN",
            "el test es temporal; un holdout espacial requiere bloques/regiones")

    # 7. Modelos
    for f in ["lightgbm_hibrido_chaco.booster", "lightgbm_hibrido_oriental.booster",
              "lightgbm_propagacion_chaco.booster", "lightgbm_propagacion_oriental.booster"]:
        ruta = DIR_MODELOS / f
        add("modelos", f, "presente" if ruta.exists() else "ausente",
            "OK" if ruta.exists() else "WARN")

    # 8. Manifiesto
    p_man = ROOT_DIR / "fuentes" / "MANIFIESTO_DATOS_v2.csv"
    if p_man.exists():
        man = pd.read_csv(p_man)
        if "sha256_hash" in man.columns:
            pendientes = int((man["sha256_hash"].astype(str).str.upper() == "PENDIENTE").sum())
            add("manifiesto", "hashes_pendientes", pendientes, "WARN" if pendientes else "OK")
            add("manifiesto", "filas", len(man), "OK")
    else:
        add("manifiesto", "archivo", "ausente", "FAIL")

    df_out = pd.DataFrame(FILAS)
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    p_csv = DIR_TABLAS / "tabla00_validacion_datos.csv"
    p_md = DIR_TABLAS / "tabla00_validacion_datos.md"
    df_out.to_csv(p_csv, index=False)
    p_md.write_text(
        "# Validación de integridad de datos\n\n"
        + df_out.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("VALIDACIÓN DE INTEGRIDAD")
    print("=" * 70)
    print(df_out.to_string(index=False))
    print(f"\n✓ Guardado en {p_csv.name} y {p_md.name}")
    fallos = int((df_out["estado"] == "FAIL").sum())
    print(f"Chequeos FAIL: {fallos}")
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
