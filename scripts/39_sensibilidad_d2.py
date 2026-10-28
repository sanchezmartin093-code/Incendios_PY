"""D2 sensitivity: retrain ignition models for radius/window grids."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constantes import (
    DIR_PROCESSED,
    DIR_TABLAS,
    FECHA_INICIO_TRAIN_SIN_WARMUP,
    RATIO_AUSENCIAS_PRESENCIAS,
    UMBRAL_LLUVIA_48H_MM,
    UMBRAL_VPD_MINIMO_KPA,
)
from src.espaciotemporal import (
    agregar_features_antropicas,
    enriquecer_dataset_espaciotemporal,
)
from src.metricas import (
    calcular_brier_score,
    calcular_brier_skill_score,
    calcular_umbral_optimo_f_beta,
)
from src.modelo import (
    FEATURES_HIBRIDO_CHACO,
    FEATURES_HIBRIDO_ORIENTAL,
    FEATURES_IGNICION_CHACO,
    FEATURES_IGNICION_ORIENTAL,
    FEATURES_M1_CHACO,
    FEATURES_M1_ORIENTAL,
    calibrar_probabilidades_isotonica,
    entrenar_modelo_lightgbm,
    predecir_probabilidad,
)

RADIOS = (15, 25, 35)
VENTANAS = (7, 10, 14)
BASE = DIR_PROCESSED / "sensibilidad_d2"


def _cargar_bases():
    grilla = pl.read_parquet(DIR_PROCESSED / "grilla_nacional_1km.parquet")
    combustible = pl.read_parquet(
        DIR_PROCESSED / "combustible_mapbiomas_1km_anual.parquet",
        columns=["cell_id", "anio_fila", "es_combustible"],
    )
    clima = pl.read_parquet(
        DIR_PROCESSED / "clima_era5_diario.parquet",
        columns=["nodo_clima_id", "fecha", "vpd_max_kpa", "precipitacion_48h_mm"],
    )
    focos = pl.read_parquet(
        DIR_PROCESSED / "focos_viirs_diarios.parquet",
        columns=["cell_id", "acq_date", "sum_frp_dia"],
    )
    return grilla, combustible, clima, focos


def _mapas_bool_anuales(grilla: pl.DataFrame, combustible: pl.DataFrame) -> dict[int, np.ndarray]:
    celdas = grilla["cell_id"].to_list()
    mapas: dict[int, np.ndarray] = {}
    for anio in sorted(combustible["anio_fila"].unique().to_list()):
        sub = combustible.filter(pl.col("anio_fila") == anio)
        mapa = dict(zip(sub["cell_id"].to_list(), sub["es_combustible"].to_list()))
        mapas[int(anio)] = np.array([bool(mapa.get(c, False)) for c in celdas], dtype=bool)
    return mapas


def _planificar(grilla, combustible, clima, focos, radio_km: int, ventana_dias: int):
    cell_ids = grilla["cell_id"].to_numpy()
    x = grilla["x_utm"].to_numpy()
    y = grilla["y_utm"].to_numpy()
    regiones = np.array(grilla["region"].to_list())
    nodos = grilla["nodo_clima_id"].to_numpy()
    idx_cell = {int(c): i for i, c in enumerate(cell_ids)}
    fuel_por_anio = _mapas_bool_anuales(grilla, combustible)
    tree = cKDTree(np.column_stack([x, y]))
    radio_m = float(radio_km) * 1000.0
    max_nodo = int(max(nodos.max(), clima["nodo_clima_id"].max()))
    rng = np.random.default_rng(42)

    focos_por_fecha = {
        f[0]: sub["cell_id"].to_list()
        for f, sub in focos.group_by("acq_date", maintain_order=True)
    }
    fechas = sorted(clima["fecha"].unique().to_list())
    clima_por_fecha = {
        f[0]: sub for f, sub in clima.group_by("fecha", maintain_order=True)
    }

    ultimo_fuego: dict[int, str] = {}
    plan = {"cell_id": [], "fecha": [], "region": [], "target": []}

    for n_fecha, fecha in enumerate(fechas, start=1):
        anio_fila = int(fecha[:4])
        if anio_fila not in fuel_por_anio:
            raise ValueError(f"No hay máscara de combustible para el año {anio_fila}")
        fuel = fuel_por_anio[anio_fila]
        sub_clima = clima_por_fecha[fecha]
        vpd_nodo = np.full(max_nodo + 1, np.nan, dtype=float)
        lluvia_nodo = np.full(max_nodo + 1, np.nan, dtype=float)
        vpd_nodo[sub_clima["nodo_clima_id"].to_numpy()] = sub_clima["vpd_max_kpa"].to_numpy()
        lluvia_nodo[sub_clima["nodo_clima_id"].to_numpy()] = sub_clima["precipitacion_48h_mm"].to_numpy()
        gating = (
            fuel
            & (vpd_nodo[nodos] >= UMBRAL_VPD_MINIMO_KPA)
            & (lluvia_nodo[nodos] < UMBRAL_LLUVIA_48H_MM)
        )

        fuegos_hoy = focos_por_fecha.get(fecha, [])
        fire_mask = np.zeros(len(cell_ids), dtype=bool)
        fire_idx = []
        for cid in fuegos_hoy:
            i = idx_cell.get(int(cid))
            if i is not None:
                fire_mask[i] = True
                fire_idx.append(i)

        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        ayer = (fecha_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        fuegos_ayer = focos_por_fecha.get(ayer, [])
        buffer = np.zeros(len(cell_ids), dtype=bool)
        if fuegos_ayer:
            coords_ayer = np.array([[x[idx_cell[int(c)]], y[idx_cell[int(c)]]] for c in fuegos_ayer if int(c) in idx_cell])
            if len(coords_ayer):
                vecinos = tree.query_ball_point(coords_ayer, r=radio_m, workers=-1)
                for arr in vecinos:
                    buffer[arr] = True

        candidato = gating & ~fire_mask
        pos_ign = np.zeros(len(cell_ids), dtype=bool)
        pos_prop = np.zeros(len(cell_ids), dtype=bool)
        for i in fire_idx:
            if not gating[i]:
                continue
            if buffer[i]:
                pos_prop[i] = True
                continue
            cid = int(cell_ids[i])
            anterior = ultimo_fuego.get(cid)
            if anterior is None or (fecha_dt - datetime.strptime(anterior, "%Y-%m-%d")).days > ventana_dias:
                pos_ign[i] = True

        neg_ign = candidato & ~buffer
        neg_prop = candidato & buffer

        for reg in ["Chaco", "Oriental"]:
            mask_reg = regiones == reg
            for task, pos_mask, neg_mask in [
                ("ignicion", pos_ign & mask_reg, neg_ign & mask_reg),
                ("propagacion", pos_prop & mask_reg, neg_prop & mask_reg),
            ]:
                idx_pos = np.where(pos_mask)[0]
                idx_neg = np.where(neg_mask)[0]
                if len(idx_pos) == 0 or len(idx_neg) == 0:
                    continue
                n_neg = min(len(idx_neg), RATIO_AUSENCIAS_PRESENCIAS * len(idx_pos))
                if len(idx_neg) >= RATIO_AUSENCIAS_PRESENCIAS * len(idx_pos):
                    idx_pos_sel = idx_pos
                else:
                    n_pos_sel = n_neg // RATIO_AUSENCIAS_PRESENCIAS
                    if n_pos_sel == 0:
                        continue
                    idx_pos_sel = rng.choice(idx_pos, size=n_pos_sel, replace=False)
                idx_neg_sel = rng.choice(idx_neg, size=n_neg, replace=False)
                if task != "ignicion":
                    continue
                for i in idx_pos_sel:
                    plan["cell_id"].append(int(cell_ids[i]))
                    plan["fecha"].append(fecha)
                    plan["region"].append(reg)
                    plan["target"].append(1)
                for i in idx_neg_sel:
                    plan["cell_id"].append(int(cell_ids[i]))
                    plan["fecha"].append(fecha)
                    plan["region"].append(reg)
                    plan["target"].append(0)

        for cid in fuegos_hoy:
            ultimo_fuego[int(cid)] = fecha

        if n_fecha % 366 == 0:
            print(f"  plan {radio_km}/{ventana_dias}: {fecha} ({n_fecha}/{len(fechas)})", flush=True)

    return pl.DataFrame(plan).sort(["fecha", "region", "cell_id"])


def _extraer_features(radio_km: int, ventana_dias: int, plan: pl.DataFrame, focos: pl.DataFrame) -> dict[str, pl.DataFrame]:
    out_dir = BASE / f"r{radio_km}_w{ventana_dias}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan_ignicion.parquet"
    plan.write_parquet(plan_path, compression="zstd")

    con = duckdb.connect()
    con.execute(f"CREATE TEMP TABLE plan AS SELECT * FROM '{plan_path}'")
    dataset = con.sql(f"""
        SELECT d.*, p.target AS target_ignicion
        FROM read_parquet('{DIR_PROCESSED / "dataset_unificado_candidatos" / "dataset_*.parquet"}') d
        INNER JOIN plan p
          ON d.cell_id = p.cell_id AND d.fecha = p.fecha
    """).pl()
    con.close()
    dataset = dataset.sort(["fecha", "cell_id"])
    print(f"  features r{radio_km}/w{ventana_dias}: {dataset.height:,} filas", flush=True)

    dataset = enriquecer_dataset_espaciotemporal(dataset, focos, usar_priors_causales=True)
    dataset = agregar_features_antropicas(dataset)
    dataset = dataset.with_columns(pl.col("target_ignicion").cast(pl.Int8))

    anio = dataset["fecha"].str.slice(0, 4).cast(pl.Int32)
    particiones = {
        "train": anio.is_in([2018, 2019, 2020, 2021, 2022])
        & (pl.col("fecha") >= FECHA_INICIO_TRAIN_SIN_WARMUP),
        "val": anio == 2023,
        "test": anio.is_in([2024, 2025]),
    }
    salida = {}
    for nombre, mask in particiones.items():
        sub = dataset.filter(mask).sort(["fecha", "cell_id"])
        ruta = out_dir / f"{nombre}_ignicion.parquet"
        sub.write_parquet(ruta, compression="zstd")
        salida[nombre] = sub
        print(
            f"  {nombre}: {sub.height:,} | pos={int(sub['target_ignicion'].sum()):,} "
            f"neg={int((sub['target_ignicion'] == 0).sum()):,}",
            flush=True,
        )
    return salida


def _metricas(y, p, cutoff, evaluacion):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= cutoff).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25 * prec + rec) else 0.0
    return {
        "evaluacion": evaluacion,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalencia": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else float(y.mean()),
        "brier": round(calcular_brier_score(y, p), 4),
        "bss": round(calcular_brier_skill_score(y, p), 4),
        "cutoff_f05": round(float(cutoff), 6),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
    }


def _entrenar(radio_km: int, ventana_dias: int, data: dict[str, pl.DataFrame]) -> list[dict]:
    filas: list[dict] = []
    for region in ["Chaco", "Oriental"]:
        tr = data["train"].filter(pl.col("region") == region)
        va = data["val"].filter(pl.col("region") == region)
        te = data["test"].filter(pl.col("region") == region)
        if tr.is_empty() or va.is_empty() or te.is_empty():
            print(f"  AVISO: partición vacía {region} r{radio_km}/w{ventana_dias}")
            continue
        y_tr = tr["target_ignicion"].to_numpy().astype(np.float32)
        y_va = va["target_ignicion"].to_numpy().astype(int)
        y_te = te["target_ignicion"].to_numpy().astype(int)
        modelos = [
            ("M1_biofisico", FEATURES_M1_CHACO if region == "Chaco" else FEATURES_M1_ORIENTAL),
            ("M2_base", FEATURES_IGNICION_CHACO if region == "Chaco" else FEATURES_IGNICION_ORIENTAL),
            ("M3_hibrido", FEATURES_HIBRIDO_CHACO if region == "Chaco" else FEATURES_HIBRIDO_ORIENTAL),
        ]
        preds_va: dict[str, np.ndarray] = {}
        preds_te: dict[str, np.ndarray] = {}
        for nombre, feats in modelos:
            booster = entrenar_modelo_lightgbm(
                tr.select(feats).to_numpy(), y_tr,
                va.select(feats).to_numpy(), y_va.astype(np.float32), feats,
                scale_pos_weight=1.0 if region == "Chaco" else 2.0,
                num_boost_round=45 if region == "Chaco" else 60,
                num_leaves=23 if region == "Chaco" else 31,
                min_child_samples=20 if region == "Chaco" else 10,
                early_stopping_rounds=None, semilla=42,
            )
            cal = calibrar_probabilidades_isotonica(booster, va.select(feats).to_numpy(), y_va, metodo="platt_monotono")
            p_va = predecir_probabilidad(booster, cal, va.select(feats).to_numpy())
            p_te = predecir_probabilidad(booster, cal, te.select(feats).to_numpy())
            cutoff, _ = calcular_umbral_optimo_f_beta(y_va, p_va, beta=0.5)
            preds_va[nombre], preds_te[nombre] = p_va, p_te
            for ev, y, p in [("validacion_2023", y_va, p_va), ("test_2024_2025", y_te, p_te)]:
                m = _metricas(y, p, cutoff, ev)
                m.update({
                    "radio_km": radio_km, "ventana_dias": ventana_dias,
                    "region": region, "modelo": nombre,
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                })
                filas.append(m)
            print(
                f"  {region} {nombre} r{radio_km}/w{ventana_dias}: "
                f"val ROC={filas[-2]['roc_auc']:.4f} PR={filas[-2]['pr_auc']:.4f}; "
                f"test ROC={filas[-1]['roc_auc']:.4f} PR={filas[-1]['pr_auc']:.4f}",
                flush=True,
            )

        p_va_g = np.where(va["freq_hist_1km"].to_numpy() <= 0, preds_va["M2_base"], preds_va["M3_hibrido"])
        p_te_g = np.where(te["freq_hist_1km"].to_numpy() <= 0, preds_te["M2_base"], preds_te["M3_hibrido"])
        cutoff_g, _ = calcular_umbral_optimo_f_beta(y_va, p_va_g, beta=0.5)
        for ev, y, p in [("validacion_2023", y_va, p_va_g), ("test_2024_2025", y_te, p_te_g)]:
            m = _metricas(y, p, cutoff_g, ev)
            m.update({
                "radio_km": radio_km, "ventana_dias": ventana_dias,
                "region": region, "modelo": "M3_M2_gated",
                "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            })
            filas.append(m)
        print(
            f"  {region} gated r{radio_km}/w{ventana_dias}: "
            f"val ROC={filas[-2]['roc_auc']:.4f} PR={filas[-2]['pr_auc']:.4f}; "
            f"test ROC={filas[-1]['roc_auc']:.4f} PR={filas[-1]['pr_auc']:.4f}",
            flush=True,
        )
    return filas


def _ejecutar(radios: list[int], ventanas: list[int], reuse: bool):
    BASE.mkdir(parents=True, exist_ok=True)
    grid, combustible, clima, focos = _cargar_bases()
    filas: list[dict] = []
    for radio in radios:
        for ventana in ventanas:
            out_dir = BASE / f"r{radio}_w{ventana}"
            data: dict[str, pl.DataFrame] = {}
            ok = reuse and all((out_dir / f"{p}_ignicion.parquet").exists() for p in ("train", "val", "test"))
            if ok:
                print(f"Reutilizando features r{radio}/w{ventana}", flush=True)
                for p in ("train", "val", "test"):
                    data[p] = pl.read_parquet(out_dir / f"{p}_ignicion.parquet")
            else:
                print(f"Planificando r{radio}/w{ventana}", flush=True)
                plan = _planificar(grid, combustible, clima, focos, radio, ventana)
                print(
                    f"Plan r{radio}/w{ventana}: {plan.height:,} filas; "
                    f"pos={int(plan['target'].sum()):,}",
                    flush=True,
                )
                data = _extraer_features(radio, ventana, plan, focos)
            filas.extend(_entrenar(radio, ventana, data))
            DIR_TABLAS.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(filas).to_csv(DIR_TABLAS / "tablaS20_sensibilidad_d2.csv", index=False)
    out = pd.DataFrame(filas)
    cols = [
        "radio_km", "ventana_dias", "region", "modelo", "evaluacion", "n", "n_pos",
        "prevalencia", "roc_auc", "pr_auc", "brier", "bss", "cutoff_f05",
        "precision", "recall", "f05", "n_train", "n_val", "n_test",
    ]
    out = out[cols]
    out.to_csv(DIR_TABLAS / "tablaS20_sensibilidad_d2.csv", index=False)
    resumen = out[out["modelo"] == "M3_hibrido"].copy()
    compact_rows = []
    for (radio, ventana, region), sub in resumen.groupby(["radio_km", "ventana_dias", "region"]):
        val = sub[sub["evaluacion"] == "validacion_2023"].iloc[0]
        test = sub[sub["evaluacion"] == "test_2024_2025"].iloc[0]
        compact_rows.append({
            "radio_km": radio, "ventana_dias": ventana, "region": region,
            "n_val": int(val["n"]), "pos_val": int(val["n_pos"]),
            "roc_val": val["roc_auc"], "pr_val": val["pr_auc"],
            "n_test": int(test["n"]), "pos_test": int(test["n_pos"]),
            "roc_test": test["roc_auc"], "pr_test": test["pr_auc"],
        })
    compact = pd.DataFrame(compact_rows).sort_values(["radio_km", "ventana_dias", "region"])
    compact.to_csv(DIR_TABLAS / "tablaS20_sensibilidad_d2_m3.csv", index=False)
    (DIR_TABLAS / "tablaS20_sensibilidad_d2_m3.md").write_text(
        "# Tabla Suplementaria S20. Sensibilidad D2 con reentrenamiento (M3)\n\n"
        "Cada fila regenera el muestreo 1:10, reentrena M3 y calibra en 2023. "
        "El radio modifica solo la definición del target D2. La tabla completa con M1, M2 y la política gated está en `tablaS20_sensibilidad_d2.csv`.\n\n"
        + compact.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    (DIR_TABLAS / "tablaS20_sensibilidad_d2.md").write_text(
        "# Tabla Suplementaria S20. Sensibilidad del operador D2 con reentrenamiento\n\n"
        "Cada combinación de radio (15/25/35 km) y ventana (7/10/14 d) regenera "
        "el muestreo de ignición 1:10, reentrena M1, M2 y M3, calibra en 2023 y "
        "evalúa 2023 y 2024–2025. El radio solo modifica la definición del target "
        "D2 y el dominio de propagación; las features del modelo no se alteran.\n\n"
        "## M3 híbrido\n\n" + resumen.to_markdown(index=False) +
        "\n\n## Todos los modelos\n\n" + out.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radios", nargs="+", type=int, default=list(RADIOS))
    ap.add_argument("--ventanas", nargs="+", type=int, default=list(VENTANAS))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    _ejecutar(args.radios, args.ventanas, args.reuse)


if __name__ == "__main__":
    main()
