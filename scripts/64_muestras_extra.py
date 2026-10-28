"""Script 64: muestras adicionales para R.9 (sin gate) y R.4 (1:20).

Genera dos planes de ignicion desde el archivo candidato completo:
- ``sin_gate``: sin filtro biofisico, ratio 1:10.
- ``gate_1_20``: gate basal (VPD >= 0.8, lluvia 48 h < 10 mm), ratio 1:20
  con negativos independientes (no duplicados).

Luego recupera features y enriquece con la misma logica de scripts/05d.
Salida: datos/processed/targets_extra/{variant}/{split}_ignicion.parquet.
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
    DIR_PROCESSED, DIAS_IGNICION_SIN_FUEGO, RADIO_PROPAGACION_KM,
    UMBRAL_LLUVIA_48H_MM, UMBRAL_VPD_MINIMO_KPA, FECHA_INICIO_TRAIN_SIN_WARMUP,
)
from src.espaciotemporal import agregar_features_antropicas, enriquecer_dataset_espaciotemporal


VARIANTES = {
    "sin_gate": {"gate": False, "ratio": 10},
    "gate_1_20": {"gate": True, "ratio": 20},
}


def _cargar():
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


def _mapas_bool_anuales(grilla, combustible):
    celdas = grilla["cell_id"].to_list()
    mapas = {}
    for anio in sorted(combustible["anio_fila"].unique().to_list()):
        sub = combustible.filter(pl.col("anio_fila") == anio)
        mapa = dict(zip(sub["cell_id"].to_list(), sub["es_combustible"].to_list()))
        mapas[int(anio)] = np.array([bool(mapa.get(c, False)) for c in celdas], dtype=bool)
    return mapas


def _planificar(grilla, combustible, clima, focos):
    cell_ids = grilla["cell_id"].to_numpy()
    x = grilla["x_utm"].to_numpy(); y = grilla["y_utm"].to_numpy()
    regiones = np.array(grilla["region"].to_list())
    nodos = grilla["nodo_clima_id"].to_numpy()
    idx_cell = {int(c): i for i, c in enumerate(cell_ids)}
    fuel_por_anio = _mapas_bool_anuales(grilla, combustible)
    tree = cKDTree(np.column_stack([x, y]))
    radio_m = RADIO_PROPAGACION_KM * 1000.0
    max_nodo = int(max(nodos.max(), clima["nodo_clima_id"].max()))
    rng = np.random.default_rng(42)

    focos_por_fecha = {f[0]: sub["cell_id"].to_list() for f, sub in focos.group_by("acq_date", maintain_order=True)}
    fechas = sorted(clima["fecha"].unique().to_list())
    clima_por_fecha = {f[0]: sub for f, sub in clima.group_by("fecha", maintain_order=True)}

    ultimo_fuego = {}
    plan = {"cell_id": [], "fecha": [], "variant": [], "target": [], "region": []}
    for fecha in fechas:
        anio = int(fecha[:4])
        fuel = fuel_por_anio[anio]
        sub_clima = clima_por_fecha[fecha]
        vpd_nodo = np.full(max_nodo + 1, np.nan); lluvia_nodo = np.full(max_nodo + 1, np.nan)
        vpd_nodo[sub_clima["nodo_clima_id"].to_numpy()] = sub_clima["vpd_max_kpa"].to_numpy()
        lluvia_nodo[sub_clima["nodo_clima_id"].to_numpy()] = sub_clima["precipitacion_48h_mm"].to_numpy()
        gate = fuel & (vpd_nodo[nodos] >= UMBRAL_VPD_MINIMO_KPA) & (lluvia_nodo[nodos] < UMBRAL_LLUVIA_48H_MM)

        fuegos_hoy = focos_por_fecha.get(fecha, [])
        fire_mask = np.zeros(len(cell_ids), dtype=bool)
        fire_idx = []
        for cid in fuegos_hoy:
            i = idx_cell.get(int(cid))
            if i is not None:
                fire_mask[i] = True; fire_idx.append(i)

        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        ayer = (fecha_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        buffer = np.zeros(len(cell_ids), dtype=bool)
        if focos_por_fecha.get(ayer):
            coords = np.array([[x[idx_cell[int(c)]], y[idx_cell[int(c)]]] for c in focos_por_fecha[ayer] if int(c) in idx_cell])
            if len(coords):
                for arr in tree.query_ball_point(coords, r=radio_m, workers=-1):
                    buffer[arr] = True

        for variant, cfg in VARIANTES.items():
            base = gate if cfg["gate"] else np.ones(len(cell_ids), dtype=bool)
            candidato = base & ~fire_mask
            pos = np.zeros(len(cell_ids), dtype=bool)
            for i in fire_idx:
                if not base[i]:
                    continue
                cid = int(cell_ids[i]); anterior = ultimo_fuego.get(cid)
                if (not buffer[i]) and (anterior is None or (fecha_dt - datetime.strptime(anterior, "%Y-%m-%d")).days > DIAS_IGNICION_SIN_FUEGO):
                    pos[i] = True
            neg = candidato & ~buffer
            for reg in ("Chaco", "Oriental"):
                mask = regiones == reg
                idx_pos = np.where(pos & mask)[0]; idx_neg = np.where(neg & mask)[0]
                if len(idx_pos) == 0 or len(idx_neg) == 0:
                    continue
                ratio = cfg["ratio"]
                if len(idx_neg) >= ratio * len(idx_pos):
                    idx_pos_sel = idx_pos
                else:
                    n_pos = len(idx_neg) // ratio
                    if n_pos == 0:
                        continue
                    idx_pos_sel = rng.choice(idx_pos, size=n_pos, replace=False)
                idx_neg_sel = rng.choice(idx_neg, size=min(len(idx_neg), ratio * len(idx_pos_sel)), replace=False)
                for i in idx_pos_sel:
                    plan["cell_id"].append(int(cell_ids[i])); plan["fecha"].append(fecha)
                    plan["variant"].append(variant); plan["target"].append(1); plan["region"].append(reg)
                for i in idx_neg_sel:
                    plan["cell_id"].append(int(cell_ids[i])); plan["fecha"].append(fecha)
                    plan["variant"].append(variant); plan["target"].append(0); plan["region"].append(reg)

        for cid in fuegos_hoy:
            ultimo_fuego[int(cid)] = fecha

    return pl.DataFrame(plan)


def main():
    print("Cargando bases...")
    grilla, combustible, clima, focos = _cargar()
    print("Planificando sin gate y 1:20...")
    plan = _planificar(grilla, combustible, clima, focos)
    print(plan.group_by(["variant", "target"]).len().sort(["variant", "target"]))
    ruta_plan = DIR_PROCESSED / "plan_muestreo_extra.parquet"
    plan.write_parquet(ruta_plan, compression="zstd")

    print("Recuperando features...")
    con = duckdb.connect()
    con.execute(f"CREATE TEMP TABLE plan AS SELECT * FROM '{ruta_plan}'")
    dataset = con.sql(f"""
        SELECT d.*, p.variant, p.target
        FROM read_parquet('{DIR_PROCESSED / "dataset_unificado_candidatos" / "dataset_*.parquet"}') d
        INNER JOIN plan p ON d.cell_id = p.cell_id AND d.fecha = p.fecha
    """).pl()
    con.close()
    dataset = dataset.sort(["variant", "fecha", "cell_id"])
    print(f"Dataset recuperado: {dataset.height:,}")

    print("Enriqueciendo...")
    dataset = enriquecer_dataset_espaciotemporal(dataset, focos, usar_priors_causales=True)
    dataset = agregar_features_antropicas(dataset)
    dataset = dataset.with_columns(pl.col("target").cast(pl.Int8).alias("target_ignicion"))

    for variant in VARIANTES:
        sub = dataset.filter(pl.col("variant") == variant)
        out_dir = DIR_PROCESSED / "targets_extra" / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        anio = sub["fecha"].str.slice(0, 4).cast(pl.Int32)
        mascaras = {
            "train": anio.is_in([2018, 2019, 2020, 2021, 2022]) & (pl.col("fecha") >= FECHA_INICIO_TRAIN_SIN_WARMUP),
            "val": anio == 2023,
            "test": anio.is_in([2024, 2025]),
        }
        for split, mask in mascaras.items():
            out = sub.filter(mask)
            out.write_parquet(out_dir / f"{split}_ignicion.parquet", compression="zstd")
            print(variant, split, out.height, int(out["target_ignicion"].sum()))


if __name__ == "__main__":
    main()
