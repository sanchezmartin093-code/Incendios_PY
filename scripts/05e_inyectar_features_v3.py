"""
Script 05e: injects the v3 fuel, desmonte and soil-moisture features into the
canonical D2 partitions.

Canonical files
---------------
``datos/processed/targets_poblacionales/{train,val,test}_{ignicion,propagacion}.parquet``

This is the dataset consumed by scripts 06, 07, 08, 13, 16, 18, 20, 21 and 24.
The legacy ``train_dataset.parquet`` / ``val_dataset.parquet`` /
``test_dataset.parquet`` files are NOT the modelling inputs and usually carry
the old synthetic fuel profile. Do not inject into them.

The script preserves the D2 targets, row count and row order. It validates the
join cardinality, raises on unexpected nulls and writes atomically.
"""

from pathlib import Path
import os
import shutil
import sys

import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constantes import DIR_PROCESSED
from src.espaciotemporal import agregar_features_antropicas

ARCHIVOS = [
    f"{split}_{task}.parquet"
    for split in ("train", "val", "test")
    for task in ("ignicion", "propagacion")
]

COLS_NUEVAS_CLIMA = [
    "swvl1_m3m3",
    "et0_mm",
    "fdd_heladas_60d",
    "deficit_hidro_60d",
]
COLS_NUEVAS_DESMONTE = [
    "pct_desmonte_t1",
    "pct_desmonte_t2",
    "pct_desmonte_2a",
]
COLS_DERIVADAS = [
    "presion_desmonte_vpd",
    "riesgo_estero_seco",
    "combustible_curado_helada",
]
COLS_NUEVAS = COLS_NUEVAS_CLIMA + COLS_NUEVAS_DESMONTE


def _validar_unicas(df: pl.DataFrame, columnas: list[str], nombre: str) -> None:
    if not df.select(columnas).is_unique().all():
        raise ValueError(f"{nombre} contiene claves duplicadas en {columnas}")


def _leer_requerido(path: Path, columnas: list[str], nombre: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {nombre} en {path}. Ejecutá primero el script que lo genera."
        )
    df = pl.read_parquet(path)
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"{nombre} no tiene las columnas requeridas: {faltantes}. "
            "Regenerá ese artefacto antes de ejecutar 05e."
        )
    return df


def main() -> None:
    print("=" * 70)
    print("INYECCIÓN V3 SOBRE PARTICIONES CANÓNICAS D2")
    print("=" * 70)

    grilla = _leer_requerido(
        DIR_PROCESSED / "grilla_nacional_1km.parquet",
        ["cell_id", "nodo_clima_id"],
        "grilla_nacional_1km.parquet",
    ).select(["cell_id", "nodo_clima_id"])
    _validar_unicas(grilla, ["cell_id"], "grilla")

    clima = _leer_requerido(
        DIR_PROCESSED / "clima_era5_diario.parquet",
        ["nodo_clima_id", "fecha"] + COLS_NUEVAS_CLIMA,
        "clima_era5_diario.parquet",
    ).select(["nodo_clima_id", "fecha"] + COLS_NUEVAS_CLIMA)
    _validar_unicas(clima, ["nodo_clima_id", "fecha"], "clima")

    desmonte = _leer_requerido(
        DIR_PROCESSED / "desmonte_hansen_1km.parquet",
        ["cell_id", "anio"] + COLS_NUEVAS_DESMONTE,
        "desmonte_hansen_1km.parquet",
    ).select(["cell_id", "anio"] + COLS_NUEVAS_DESMONTE)
    _validar_unicas(desmonte, ["cell_id", "anio"], "desmonte")

    dir_backup = DIR_PROCESSED / "legacy_v3"
    dir_backup.mkdir(parents=True, exist_ok=True)

    for nombre in ARCHIVOS:
        ruta = DIR_PROCESSED / "targets_poblacionales" / nombre
        if not ruta.exists():
            print(f" [aviso] No existe {ruta.name}; se omite.")
            continue

        df = pl.read_parquet(ruta)
        n0 = df.height
        df = df.drop([c for c in COLS_NUEVAS + COLS_DERIVADAS + ["nodo_clima_id", "anio"] if c in df.columns])
        df = df.with_row_index("_orden")
        df = df.with_columns(
            pl.col("fecha").cast(pl.String).str.slice(0, 4).cast(pl.Int32).alias("anio")
        )

        df = (
            df.join(grilla, on="cell_id", how="left")
            .join(clima, on=["nodo_clima_id", "fecha"], how="left")
            .join(desmonte, on=["cell_id", "anio"], how="left")
        )

        if df.height != n0:
            raise RuntimeError(
                f"{nombre}: la unión cambió el número de filas ({n0} -> {df.height}). "
                "Revisá duplicados en las tablas auxiliares."
            )

        df = df.sort("_orden").drop(["_orden", "nodo_clima_id", "anio"])
        df = agregar_features_antropicas(df)

        nulos = {
            c: int(df[c].null_count())
            for c in COLS_NUEVAS
            if int(df[c].null_count()) > 0
        }
        if nulos:
            raise ValueError(
                f"{nombre}: quedaron valores nulos en las features nuevas: {nulos}. "
                "Revisá cobertura Hansen, clima y combustible."
            )

        if not (dir_backup / nombre).exists():
            shutil.copy2(ruta, dir_backup / nombre)

        ruta_tmp = ruta.with_name(ruta.name + ".tmp")
        df.write_parquet(ruta_tmp, compression="zstd")
        os.replace(ruta_tmp, ruta)

        print(
            f" ✓ {nombre}: {n0:,} filas, {len(df.columns)} columnas "
            f"(+{len(COLS_NUEVAS)} features, +{len(COLS_DERIVADAS)} derivadas)"
        )

    print("=" * 70)
    print("Las particiones canónicas quedaron actualizadas. Reentrená y regenerá")
    print("métricas/figuras; los boosters actuales no contienen estas variables.")
    print("=" * 70)


if __name__ == "__main__":
    main()
