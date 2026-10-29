"""G0-Formosa: D2 ignition sampling plan (1:10) and population counts."""

from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constantes import DIR_PROCESSED

OUT = DIR_PROCESSED / "formosa_g0"
RATIO = 10
BUFFER_KM = float(os.environ.get("G0_BUFFER_KM", "25"))
TAG = os.environ.get("G0_TAG", "")
BUFFER_M = BUFFER_KM * 1000.0


def main() -> None:
    grid = pl.read_parquet(OUT / "grilla_formosa_1km.parquet")
    fuel = pl.read_parquet(OUT / "combustible_formosa_anual.parquet")
    clim = pl.read_parquet(OUT / "clima_era5_diario.parquet", columns=["nodo_clima_id", "fecha", "vpd_max_kpa", "precipitacion_48h_mm"])
    focos = pl.read_parquet(OUT / "focos_formosa_diarios.parquet", columns=["cell_id", "acq_date"])

    ids = grid["cell_id"].to_numpy()
    x = grid["x_utm"].to_numpy(); y = grid["y_utm"].to_numpy()
    nodos = grid["nodo_clima_id"].to_numpy()
    idx = {int(c): i for i, c in enumerate(ids)}
    tree = cKDTree(np.column_stack([x, y]))

    fuel_by_year = {}
    for anio in sorted(fuel["anio_fila"].unique().to_list()):
        sub = fuel.filter(pl.col("anio_fila") == anio)
        mp = dict(zip(sub["cell_id"].to_list(), sub["es_combustible"].to_list()))
        fuel_by_year[int(anio)] = np.array([bool(mp.get(int(c), False)) for c in ids])

    clim_by_date = {k[0]: v for k, v in clim.group_by("fecha", maintain_order=True)}
    fires_by_date = {k[0]: v["cell_id"].to_list() for k, v in focos.group_by("acq_date", maintain_order=True)}
    fechas = sorted(clim_by_date)
    max_nodo = int(max(nodos.max(), clim["nodo_clima_id"].max()))

    rng = np.random.default_rng(42)
    ultimo = {}
    plan = {"cell_id": [], "fecha": [], "target": []}
    counts = {"fecha": [], "n_pos": [], "n_neg": [], "n_pos_all": [], "n_neg_all": []}

    for fecha in fechas:
        anio = int(fecha[:4])
        if anio not in fuel_by_year:
            continue
        dt = datetime.strptime(fecha, "%Y-%m-%d")
        c = clim_by_date[fecha]
        vpd = np.full(max_nodo + 1, np.nan); rain = np.full(max_nodo + 1, np.nan)
        vpd[c["nodo_clima_id"].to_numpy()] = c["vpd_max_kpa"].to_numpy()
        rain[c["nodo_clima_id"].to_numpy()] = c["precipitacion_48h_mm"].to_numpy()
        gate = fuel_by_year[anio] & (vpd[nodos] >= 0.8) & (rain[nodos] < 10.0)

        fire = np.zeros(len(ids), dtype=bool)
        fire_idx = []
        for cid in fires_by_date.get(fecha, []):
            i = idx.get(int(cid))
            if i is not None:
                fire[i] = True; fire_idx.append(i)

        ayer = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        buffer = np.zeros(len(ids), dtype=bool)
        if fires_by_date.get(ayer):
            pts = np.array([[x[idx[int(c)]], y[idx[int(c)]]] for c in fires_by_date[ayer] if int(c) in idx])
            if len(pts):
                for arr in tree.query_ball_point(pts, r=BUFFER_M, workers=-1):
                    buffer[arr] = True

        pos = np.zeros(len(ids), dtype=bool)
        for i in fire_idx:
            if not gate[i] or buffer[i]:
                continue
            cid = int(ids[i])
            prev = ultimo.get(cid)
            if prev is None or (dt - datetime.strptime(prev, "%Y-%m-%d")).days > 10:
                pos[i] = True
        neg = gate & ~fire & ~buffer
        n_pos_all = int(pos.sum()); n_neg_all = int(neg.sum())
        counts["fecha"].append(fecha); counts["n_pos_all"].append(n_pos_all); counts["n_neg_all"].append(n_neg_all)

        ip = np.where(pos)[0]; ineg = np.where(neg)[0]
        if len(ip) and len(ineg):
            n_neg = min(len(ineg), RATIO * len(ip))
            ip_sel = ip
            if len(ineg) < RATIO * len(ip):
                n_pos_sel = n_neg // RATIO
                if n_pos_sel == 0:
                    continue
                ip_sel = rng.choice(ip, size=n_pos_sel, replace=False)
            ineg_sel = rng.choice(ineg, size=n_neg, replace=False)
            for i in ip_sel:
                plan["cell_id"].append(int(ids[i])); plan["fecha"].append(fecha); plan["target"].append(1)
            for i in ineg_sel:
                plan["cell_id"].append(int(ids[i])); plan["fecha"].append(fecha); plan["target"].append(0)
            counts["n_pos"].append(len(ip_sel)); counts["n_neg"].append(n_neg)
        else:
            counts["n_pos"].append(0); counts["n_neg"].append(0)

        for cid in fires_by_date.get(fecha, []):
            ultimo[int(cid)] = fecha

    out = pl.DataFrame(plan).with_columns(pl.col("target").cast(pl.Int8))
    out.write_parquet(OUT / f"plan_formosa_d2{TAG}.parquet", compression="zstd")
    pl.DataFrame(counts).write_parquet(OUT / f"conteos_formosa_d2{TAG}.parquet", compression="zstd")
    meta = {
        "plan_rows": out.height,
        "pos": int(out["target"].sum()),
        "neg": int((out["target"] == 0).sum()),
        "dates": [out["fecha"].min(), out["fecha"].max()],
        "buffer_km": BUFFER_KM,
        "por_anio": out.with_columns(pl.col("fecha").str.slice(0, 4).alias("anio")).group_by("anio").agg(pl.len(), pl.col("target").sum()).sort("anio").to_dicts(),
    }
    (OUT / f"plan_meta{TAG}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
