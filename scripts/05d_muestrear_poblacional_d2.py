"""
Script 05d: population sampling D2 from dataset_unificado v2.

Generates a sampling plan by date/region:
- Ignition: positives outside the t-1 buffer and without recent fire in the cell;
  gated negatives outside the t-1 buffer.
- Propagation: positives inside the t-1 buffer; gated negatives inside the
  t-1 buffer.

Then retrieves the features from the v2 dataset, enriches them and writes
train/val/test per target with a 1:10 ratio.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import duckdb
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from src.constantes import (
    DIR_PROCESSED,
    DIAS_IGNICION_SIN_FUEGO,
    RADIO_PROPAGACION_KM,
    RATIO_AUSENCIAS_PRESENCIAS,
    UMBRAL_LLUVIA_48H_MM,
    UMBRAL_VPD_MINIMO_KPA,
)
from src.espaciotemporal import (
    agregar_features_antropicas,
    enriquecer_dataset_espaciotemporal,
)


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


def _mapas_bool_anuales(
    grilla: pl.DataFrame, combustible: pl.DataFrame
) -> dict[int, np.ndarray]:
    """Returns one boolean fuel mask per modelling year (anio_fila)."""
    celdas = grilla["cell_id"].to_list()
    mapas: dict[int, np.ndarray] = {}
    for anio in sorted(combustible["anio_fila"].unique().to_list()):
        sub = combustible.filter(pl.col("anio_fila") == anio)
        mapa = dict(zip(sub["cell_id"].to_list(), sub["es_combustible"].to_list()))
        mapas[int(anio)] = np.array(
            [bool(mapa.get(c, False)) for c in celdas], dtype=bool
        )
    return mapas


def _planificar(grilla, combustible, clima, focos):
    cell_ids = grilla["cell_id"].to_numpy()
    x = grilla["x_utm"].to_numpy()
    y = grilla["y_utm"].to_numpy()
    regiones = np.array(grilla["region"].to_list())
    nodos = grilla["nodo_clima_id"].to_numpy()
    idx_cell = {int(c): i for i, c in enumerate(cell_ids)}
    fuel_por_anio = _mapas_bool_anuales(grilla, combustible)
    tree = cKDTree(np.column_stack([x, y]))
    radio_m = RADIO_PROPAGACION_KM * 1000.0
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

    ultimo_fuego = {}
    plan = {"cell_id": [], "fecha": [], "task": [], "target": [], "region": []}
    conteos = {"fecha": [], "region": [], "task": [], "n_pos": [], "n_neg": []}

    for fecha in fechas:
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
            coords_ayer = np.array([
                [x[idx_cell[int(c)]], y[idx_cell[int(c)]]]
                for c in fuegos_ayer if int(c) in idx_cell
            ])
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
            else:
                cid = int(cell_ids[i])
                anterior = ultimo_fuego.get(cid)
                if anterior is None or (fecha_dt - datetime.strptime(anterior, "%Y-%m-%d")).days > DIAS_IGNICION_SIN_FUEGO:
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
                if len(idx_pos) == 0:
                    continue
                conteos["fecha"].append(fecha)
                conteos["region"].append(reg)
                conteos["task"].append(task)
                conteos["n_pos"].append(int(len(idx_pos)))
                conteos["n_neg"].append(int(len(idx_neg)))
                if len(idx_neg) == 0:
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

                for i in idx_pos_sel:
                    plan["cell_id"].append(int(cell_ids[i]))
                    plan["fecha"].append(fecha)
                    plan["task"].append(task)
                    plan["target"].append(1)
                    plan["region"].append(reg)
                for i in idx_neg_sel:
                    plan["cell_id"].append(int(cell_ids[i]))
                    plan["fecha"].append(fecha)
                    plan["task"].append(task)
                    plan["target"].append(0)
                    plan["region"].append(reg)

        for cid in fuegos_hoy:
            ultimo_fuego[int(cid)] = fecha

    return pl.DataFrame(plan), pl.DataFrame(conteos)


def main():
    print("Cargando bases...")
    grilla, combustible, clima, focos = _cargar_bases()
    print("Planificando muestreo D2...")
    plan, conteos = _planificar(grilla, combustible, clima, focos)
    print(f"Plan: {plan.height:,} filas")
    print(plan.group_by(["task", "target"]).len().sort(["task", "target"]).to_dicts())

    ruta_plan = DIR_PROCESSED / "plan_muestreo_d2.parquet"
    plan.write_parquet(ruta_plan, compression="zstd")
    conteos.write_parquet(DIR_PROCESSED / "conteos_poblacionales_d2.parquet", compression="zstd")

    print("Recuperando features desde dataset v2...")
    con = duckdb.connect()
    con.execute(f"CREATE TEMP TABLE plan AS SELECT * FROM '{ruta_plan}'")
    dataset = con.sql(f"""
        SELECT d.*, p.task, p.target
        FROM '{DIR_PROCESSED / "dataset_unificado_candidatos.parquet"}' d
        INNER JOIN plan p
          ON d.cell_id = p.cell_id AND d.fecha = p.fecha
    """).pl()
    con.close()
    print(f"Dataset recuperado: {dataset.height:,} filas")

    print("Enriqueciendo features espaciotemporales...")
    dataset = enriquecer_dataset_espaciotemporal(dataset, focos, usar_priors_causales=True)
    dataset = agregar_features_antropicas(dataset)

    dataset = dataset.with_columns([
        pl.when(pl.col("task") == "ignicion")
        .then(pl.col("target"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("target_ignicion"),
        pl.when(pl.col("task") == "propagacion")
        .then(pl.col("target"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("target_propagacion"),
    ])

    salida = DIR_PROCESSED / "targets_poblacionales"
    salida.mkdir(parents=True, exist_ok=True)
    anio = dataset["fecha"].str.slice(0, 4).cast(pl.Int32)
    particiones = {
        "train": anio.is_in([2018, 2019, 2020, 2021, 2022]),
        "val": anio == 2023,
        "test": anio.is_in([2024, 2025]),
    }

    for nombre, mask in particiones.items():
        sub = dataset.filter(mask)
        for task in ["ignicion", "propagacion"]:
            out = sub.filter(pl.col("task") == task)
            ruta = salida / f"{nombre}_{task}.parquet"
            out.write_parquet(ruta, compression="zstd")
            n_pos = int((out[f"target_{task}"] == 1).sum())
            n_neg = int((out[f"target_{task}"] == 0).sum())
            print(
                f"{ruta.name}: {out.height:,} | pos={n_pos:,} "
                f"neg={n_neg:,} ratio=1:{n_neg / max(n_pos, 1):.2f}"
            )


if __name__ == "__main__":
    main()
