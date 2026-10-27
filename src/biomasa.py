"""
Module biomasa.py: MapBiomas land use reclassification and vegetal fuel typification.
- Mapping of MapBiomas Paraguay C2/C3 codes to functional fire types.
- Percentage proportions per 1 km² cell (pasture, annual agriculture, forest
  plantation, savanna, dry/humid forest, estero/wetland) and forest-agriculture edge.
- Hard biophysical exclusion mask for non-combustibles (water/urban > 80%).
- Modular interface to process massive 30m GeoTIFFs or provide an ecoregional base model.
"""

from pathlib import Path
import numpy as np
import polars as pl
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window

# Official mapping of MapBiomas Cono Sur coverage codes
ORIGEN_PERFIL_SINTETICO = "sintetico_referencia"


def seleccionar_tif_estatico(dir_mapbiomas: Path, anio_mapa: int = 2023) -> Path:
    """Returns the single MapBiomas GeoTIFF used by the static fuel fallback.

    The static profile must never be built from the annual collection: doing so
    would sum/normalise overlapping maps. The annual prior-year product is
    handled separately by ``scripts/03a_procesar_combustible_anual.py``.
    """
    ruta = Path(dir_mapbiomas) / f"mapbiomas_paraguay_collection2_{int(anio_mapa)}.tif"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el mapa MapBiomas estático esperado: {ruta}. "
            "Para el perfil estático usá un único año (por defecto 2023); "
            "para la serie anual usá scripts/03a_procesar_combustible_anual.py."
        )
    return ruta


# MapBiomas Paraguay Collection 2/3 legend (10 classes).
# 3/4/6: natural woodlands (closed, open, flooded); 9: forest plantation;
# 11: flooded grassland/estero; 12: grassland; 15: pasture;
# 18: annual agriculture; 22/25: non-vegetated; 26: water.
# The Paraguayan legend does NOT contain class 21 ("mosaico de usos"), which
# belongs to other MapBiomas country legends.
CODIGOS_MAPBIOMAS = {
    3: "bosque",
    4: "bosque",
    6: "bosque",
    9: "plantacion_forestal",
    11: "humedal_estero",
    12: "sabana_pastizal",
    15: "pastura_ganadera",
    18: "agricultura_anual",
    22: "no_combustible",
    24: "no_combustible",  # legacy alias (not present in the Paraguayan C2 raster)
    25: "no_combustible",  # C3 bare/non-vegetated area
    26: "no_combustible",
    33: "no_combustible",  # legacy alias (C3 remaps 33 -> 26)
}

def categorizar_combustible(codigo: int, region: str) -> str:
    """
    Assigns a MapBiomas numeric code to a functional fuel category
    according to the ecoregion (Chaco vs. Oriental Region).
    """
    if codigo == 15:
        return "pastura_ganadera"
    elif codigo == 18:
        return "agricultura_anual"
    elif codigo == 9:
        return "plantacion_forestal"
    elif codigo in (3, 4, 6):
        # Closed, open or flooded woodland. In the Chaco it is xerophytic;
        # in the Oriental, Atlantic Forest.
        return "bosque_seco" if region == "Chaco" else "bosque_humedo"
    elif codigo == 12:
        return "sabana_pastizal"  # Natural grassland
    elif codigo == 11:
        return "humedal_estero"  # Flooded grassland / estero
    elif codigo in (22, 24, 25, 26, 33):
        return "no_combustible"  # Non-vegetated, urban or water
    else:
        return "otro"

def calcular_metricas_combustible_desde_conteo(conteo_codigos: dict[int, int], region: str) -> dict[str, float | bool]:
    """
    From a dictionary of MapBiomas pixel counts (at 30m) within a 1 km² cell,
    computes the percentage proportions [0.0, 100.0] and the hard exclusion mask.

    Only codes present in the official MapBiomas legend are used in the
    denominator. This is consistent with the raster processing path, where
    class 0 / unobserved pixels are treated as no-data.
    """
    conteo_valido = {
        int(cod): int(cant)
        for cod, cant in conteo_codigos.items()
        if int(cod) in CODIGOS_MAPBIOMAS and int(cant) >= 0
    }
    total_pixeles = sum(conteo_valido.values())
    if total_pixeles == 0:
        return {
            "pct_pastura": 0.0,
            "pct_pastura_ganadera": 0.0,
            "pct_agricultura_anual": 0.0,
            "pct_plantacion_forestal": 0.0,
            "pct_sabana_pastizal": 0.0,
            "pct_bosque_seco": 0.0,
            "pct_bosque_humedo": 0.0,
            "pct_humedal_palmar": 0.0,
            "borde_bosque_agro_pct": 0.0,
            "pct_no_combustible": 0.0,
            "es_combustible": True
        }

    acumulados = {
        "pastura_ganadera": 0,
        "agricultura_anual": 0,
        "plantacion_forestal": 0,
        "sabana_pastizal": 0,
        "bosque_seco": 0,
        "bosque_humedo": 0,
        "humedal_estero": 0,
        "no_combustible": 0,
        "otro": 0,
    }

    for cod, cant in conteo_valido.items():
        cat = categorizar_combustible(cod, region)
        acumulados[cat] += cant

    pct_pastura_gan = float(100.0 * acumulados["pastura_ganadera"] / total_pixeles)
    pct_agro = float(100.0 * acumulados["agricultura_anual"] / total_pixeles)
    pct_plant = float(100.0 * acumulados["plantacion_forestal"] / total_pixeles)
    pct_pastura = pct_pastura_gan + pct_agro + pct_plant
    pct_sabana = float(100.0 * acumulados["sabana_pastizal"] / total_pixeles)
    pct_b_seco = float(100.0 * acumulados["bosque_seco"] / total_pixeles)
    pct_b_humedo = float(100.0 * acumulados["bosque_humedo"] / total_pixeles)
    pct_palmar = float(100.0 * acumulados["humedal_estero"] / total_pixeles)
    pct_no_comb = float(100.0 * acumulados["no_combustible"] / total_pixeles)

    # Stage 1 Mask: If more than 80% of the cell is permanent water or city, it is NOT combustible
    es_combustible = bool(pct_no_comb < 80.0)

    return {
        "pct_pastura": pct_pastura,
        "pct_pastura_ganadera": pct_pastura_gan,
        "pct_agricultura_anual": pct_agro,
        "pct_plantacion_forestal": pct_plant,
        "pct_sabana_pastizal": pct_sabana,
        "pct_bosque_seco": pct_b_seco,
        "pct_bosque_humedo": pct_b_humedo,
        "pct_humedal_palmar": pct_palmar,
        "borde_bosque_agro_pct": 0.0,
        "pct_no_combustible": pct_no_comb,
        "es_combustible": es_combustible
    }

def _mascara_borde_bosque_agro(data: np.ndarray) -> np.ndarray:
    """
    Returns a 30 m mask of forest pixels that touch pasture/agriculture.

    Uses an 8-neighbourhood. The input must include a one-pixel halo around the
    window that will be reprojected, so edges at the window border are not lost.
    """
    es_bosque = np.isin(data, (3, 4, 6))
    es_agro = np.isin(data, (15, 18))
    if not es_bosque.any() or not es_agro.any():
        return np.zeros_like(data, dtype=np.float32)

    h, w = es_agro.shape
    pad = np.pad(es_agro, 1, mode="constant", constant_values=False)
    vecino_agro = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            vecino_agro |= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]

    return (es_bosque & vecino_agro).astype(np.float32)


def procesar_geotiff_mapbiomas_a_grilla(
    df_grilla: pl.DataFrame,
    rutas_tif: list[Path],
    resolucion_m: int = 1000,
    ventana_px: int = 2048,
) -> pl.DataFrame:
    """
    Extracts MapBiomas class fractions (30 m) to the 1 km grid.

    For each code a binary mask is created at the native resolution and
    reprojected by windows to the 1 km grid with ``average``. Window-based
    processing avoids loading the entire raster (Paraguay ~1.1e9 pixels) into
    memory. It is assumed that the GeoTIFFs do not overlap; if they overlap,
    fractions are summed and then normalized to 100.

    The implemented legend is that of MapBiomas Paraguay Collection 2/3:
    3/4/6 woodlands, 9 forest plantation, 11 flooded grassland/estero,
    12 grassland, 15 pasture, 18 annual agriculture, 22/25 non-vegetated,
    26 water. Class 21 (mosaic) is not part of the Paraguayan legend.
    """
    if not rutas_tif:
        raise ValueError("Se requiere al menos un GeoTIFF MapBiomas")

    x = df_grilla["x_utm"].to_numpy()
    y = df_grilla["y_utm"].to_numpy()
    min_x = float(np.min(x)) - resolucion_m / 2.0
    max_y = float(np.max(y)) + resolucion_m / 2.0
    max_x = float(np.max(x)) + resolucion_m / 2.0
    min_y = float(np.min(y)) - resolucion_m / 2.0
    width = int(np.ceil((max_x - min_x) / resolucion_m - 1e-9))
    height = int(np.ceil((max_y - min_y) / resolucion_m - 1e-9))
    dst_transform = from_origin(min_x, max_y, resolucion_m, resolucion_m)

    c_idx = np.round((x - min_x - resolucion_m / 2.0) / resolucion_m).astype(int)
    r_idx = np.round((max_y - resolucion_m / 2.0 - y) / resolucion_m).astype(int)
    if c_idx.min() < 0 or c_idx.max() >= width or r_idx.min() < 0 or r_idx.max() >= height:
        raise ValueError("La grilla no cabe en el raster destino")

    codigos = list(CODIGOS_MAPBIOMAS.keys())
    acumulado = {cod: np.zeros(width * height, dtype=np.float32) for cod in codigos}
    acumulado_borde = np.zeros(width * height, dtype=np.float32)

    for ruta in rutas_tif:
        print(f"  Procesando MapBiomas: {ruta.name}")
        with rasterio.open(ruta) as src:
            destinos = {cod: np.zeros((height, width), dtype=np.float32) for cod in codigos}
            destino_borde = np.zeros((height, width), dtype=np.float32)

            for row in range(0, src.height, ventana_px):
                for col in range(0, src.width, ventana_px):
                    h = min(ventana_px, src.height - row)
                    w = min(ventana_px, src.width - col)
                    window = Window(col, row, w, h)
                    data = src.read(1, window=window)
                    src_transform = src.window_transform(window)

                    for cod in codigos:
                        mascara = (data == cod).astype(np.float32)
                        if mascara.max() == 0:
                            continue
                        reproject(
                            source=mascara,
                            destination=destinos[cod],
                            src_transform=src_transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs="EPSG:32721",
                            resampling=Resampling.average,
                            init_dest_nodata=False,
                        )

                    # One-pixel halo for a correct 8-neighbour edge mask.
                    row0 = max(0, row - 1)
                    col0 = max(0, col - 1)
                    row1 = min(src.height, row + h + 1)
                    col1 = min(src.width, col + w + 1)
                    data_halo = src.read(1, window=Window(col0, row0, col1 - col0, row1 - row0))
                    borde_halo = _mascara_borde_bosque_agro(data_halo)
                    borde = borde_halo[
                        row - row0:row - row0 + h,
                        col - col0:col - col0 + w,
                    ]
                    if borde.max() > 0:
                        reproject(
                            source=borde,
                            destination=destino_borde,
                            src_transform=src_transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs="EPSG:32721",
                            resampling=Resampling.average,
                            init_dest_nodata=False,
                        )

            for cod in codigos:
                acumulado[cod] += destinos[cod].ravel()
            acumulado_borde += destino_borde.ravel()

    # Fractions per code in grid order (0-100)
    frac = {cod: acumulado[cod][r_idx * width + c_idx] * 100.0 for cod in codigos}

    suma = np.zeros(len(df_grilla), dtype=np.float32)
    for cod in codigos:
        suma += frac[cod]
    factor = np.where(suma > 100.0, 100.0 / np.maximum(suma, 1e-9), 1.0).astype(np.float32)

    pct_pastura_gan = (frac[15] * factor).astype(np.float32)
    pct_agricultura = (frac[18] * factor).astype(np.float32)
    pct_plantacion = (frac[9] * factor).astype(np.float32)
    pct_pastura = (pct_pastura_gan + pct_agricultura + pct_plantacion).astype(np.float32)

    regiones = np.array(df_grilla["region"].to_list(), dtype=object)
    es_chaco = regiones == "Chaco"
    bosque = (frac[3] + frac[4] + frac[6]) * factor
    pct_bosque_seco = np.where(es_chaco, bosque, 0.0).astype(np.float32)
    pct_bosque_humedo = np.where(es_chaco, 0.0, bosque).astype(np.float32)
    pct_no_comb = (
        (frac[22] + frac[24] + frac[25] + frac[26] + frac[33]) * factor
    ).astype(np.float32)
    pct_no_comb = np.clip(pct_no_comb, 0.0, 100.0)
    borde_bosque_agro_pct = np.clip(
        acumulado_borde[r_idx * width + c_idx] * 100.0, 0.0, 100.0
    ).astype(np.float32)

    return pl.DataFrame({
        "cell_id": df_grilla["cell_id"],
        "pct_pastura": pct_pastura,
        "pct_pastura_ganadera": pct_pastura_gan,
        "pct_agricultura_anual": pct_agricultura,
        "pct_plantacion_forestal": pct_plantacion,
        "pct_sabana_pastizal": (frac[12] * factor).astype(np.float32),
        "pct_bosque_seco": pct_bosque_seco,
        "pct_bosque_humedo": pct_bosque_humedo,
        "pct_humedal_palmar": (frac[11] * factor).astype(np.float32),
        "borde_bosque_agro_pct": borde_bosque_agro_pct,
        "pct_no_combustible": pct_no_comb,
        "es_combustible": pct_no_comb < 80.0,
        "origen_datos": ["mapbiomas_real"] * len(df_grilla),
    })


def generar_perfil_combustible_base(df_grilla: pl.DataFrame) -> pl.DataFrame:
    """
    Generates the reference biophysical fuel profile for the 400 006 cells of Paraguay
    based on the official ecoregional distribution of MapBiomas Chaco and Atlantic Forest.
    Ensures that the pipeline works immediately and deterministically.
    """
    regiones = df_grilla["region"].to_list()
    deptos = df_grilla["departamento"].to_list()
    lats = df_grilla["lat"].to_numpy()
    lons = df_grilla["lon"].to_numpy()
    cell_ids = df_grilla["cell_id"].to_list()

    n = len(cell_ids)
    pct_pastura = np.zeros(n, dtype=np.float32)
    pct_sabana = np.zeros(n, dtype=np.float32)
    pct_b_seco = np.zeros(n, dtype=np.float32)
    pct_b_humedo = np.zeros(n, dtype=np.float32)
    pct_palmar = np.zeros(n, dtype=np.float32)
    pct_no_comb = np.zeros(n, dtype=np.float32)
    es_comb = np.ones(n, dtype=bool)

    # Fixed seed for mathematical reproducibility
    rng = np.random.default_rng(42)

    for i in range(n):
        reg = regiones[i]
        dep = deptos[i]
        lat = lats[i]
        lon = lons[i]

        if reg == "Chaco":
            # Chaco Boreal: predominance of dry xerophytic forest and expanding pastures
            if dep == "ALTO PARAGUAY":
                # Pantanal and northern humid Chaco
                p_seco = rng.uniform(55.0, 85.0)
                p_palmar = rng.uniform(5.0, 25.0)
                p_past = rng.uniform(5.0, 20.0)
                p_sab = max(0.0, 100.0 - (p_seco + p_palmar + p_past))
                pct_b_seco[i] = p_seco
                pct_palmar[i] = p_palmar
                pct_pastura[i] = p_past
                pct_sabana[i] = p_sab
            elif dep == "BOQUERON":
                # Central dry Chaco
                p_seco = rng.uniform(50.0, 75.0)
                p_past = rng.uniform(15.0, 40.0) # Mennonite colonies and Gatton Panic pastures
                p_sab = max(0.0, 100.0 - (p_seco + p_past))
                pct_b_seco[i] = p_seco
                pct_pastura[i] = p_past
                pct_sabana[i] = p_sab
            else: # PRESIDENTE HAYES
                # Lower Chaco: karanda'y palm groves and floodable grasslands
                p_palmar = rng.uniform(25.0, 50.0)
                p_past = rng.uniform(20.0, 40.0)
                p_seco = max(0.0, 100.0 - (p_palmar + p_past))
                pct_palmar[i] = p_palmar
                pct_pastura[i] = p_past
                pct_b_seco[i] = p_seco

        else: # Oriental Region
            if dep == "ASUNCION":
                # Dense urban core
                pct_no_comb[i] = 85.0
                pct_pastura[i] = 15.0
                es_comb[i] = False
            else:
                # Productive Oriental Region
                p_past = rng.uniform(40.0, 70.0) # Dominant cultivated pastures
                p_humedo = rng.uniform(10.0, 35.0) # BAAPA remnants
                p_sab = max(0.0, 100.0 - (p_past + p_humedo))
                pct_pastura[i] = p_past
                pct_b_humedo[i] = p_humedo
                pct_sabana[i] = p_sab

    return pl.DataFrame({
        "cell_id": cell_ids,
        "pct_pastura": pct_pastura,
        "pct_pastura_ganadera": pct_pastura,
        "pct_agricultura_anual": np.zeros(n, dtype=np.float32),
        "pct_plantacion_forestal": np.zeros(n, dtype=np.float32),
        "pct_sabana_pastizal": pct_sabana,
        "pct_bosque_seco": pct_b_seco,
        "pct_bosque_humedo": pct_b_humedo,
        "pct_humedal_palmar": pct_palmar,
        "borde_bosque_agro_pct": np.zeros(n, dtype=np.float32),
        "pct_no_combustible": pct_no_comb,
        "es_combustible": es_comb,
        "origen_datos": [ORIGEN_PERFIL_SINTETICO] * n,
    })
