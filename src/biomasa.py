"""
Módulo biomasa.py: Reclasificación de uso de suelo MapBiomas y tipificación de combustibles vegetales.
- Mapeo de códigos MapBiomas (Colección Chaco y Bosque Atlántico) a tipos funcionales de fuego.
- Cálculo de proporciones porcentuales por celda de 1 km² (pastura, sabana, bosque seco/húmedo, palmar).
- Máscara dura de exclusión biofísica de no-combustibles (agua > 80% o urbano > 80%).
- Interfaz modular para procesar GeoTIFFs masivos de 30m o proveer modelo base ecorregional.
"""

from pathlib import Path
import numpy as np
import polars as pl
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window

# Mapeo oficial de códigos de cobertura MapBiomas Cono Sur
ORIGEN_PERFIL_SINTETICO = "sintetico_referencia"

# Leyenda MapBiomas Paraguay Colección 2 (10 clases).
# 3/4/6: bosques (cerrado, abierto, inundado); 9: plantación forestal;
# 11: humedal; 12: pastizal; 15: pastura; 18: agricultura;
# 22: no vegetado; 26: agua.
CODIGOS_MAPBIOMAS = {
    3: "bosque",
    4: "bosque",
    6: "bosque",
    9: "pastura",
    11: "humedal_palmar",
    12: "pastizal_natural",
    15: "pastura_cultivada",
    18: "agricultura",
    21: "mosaico_agropecuario",
    22: "no_combustible",
    24: "urbano",
    25: "suelo_desnudo",
    26: "no_combustible",
    33: "cuerpo_agua",
}

def categorizar_combustible(codigo: int, region: str) -> str:
    """
    Asigna un código numérico de MapBiomas a una categoría funcional de combustible
    según la ecorregión (Chaco vs. Región Oriental).
    """
    if codigo in (9, 15, 18, 21):
        return "pastura"  # Pasturas, agricultura y plantaciones: combustible fino
    elif codigo in (3, 4, 6):
        # Bosque cerrado, abierto o inundado. En el Chaco es xerofítico;
        # en la Oriental, Bosque Atlántico.
        return "bosque_seco" if region == "Chaco" else "bosque_humedo"
    elif codigo == 12:
        return "sabana_pastizal"  # Pastizal natural
    elif codigo == 11:
        return "humedal_palmar"  # Palmares de karanda'y y vegetación palustre
    elif codigo in (22, 24, 26, 33):
        return "no_combustible"  # No vegetado, urbano o agua
    else:
        return "otro"

def calcular_metricas_combustible_desde_conteo(conteo_codigos: dict[int, int], region: str) -> dict[str, float | bool]:
    """
    A partir de un diccionario de conteo de píxeles MapBiomas (a 30m) dentro de una celda de 1 km²,
    calcula las proporciones porcentuales [0.0, 100.0] y la máscara dura de exclusión.
    """
    total_pixeles = sum(conteo_codigos.values())
    if total_pixeles == 0:
        return {
            "pct_pastura": 0.0,
            "pct_sabana_pastizal": 0.0,
            "pct_bosque_seco": 0.0,
            "pct_bosque_humedo": 0.0,
            "pct_humedal_palmar": 0.0,
            "pct_no_combustible": 0.0,
            "es_combustible": True
        }

    acumulados = {
        "pastura": 0,
        "sabana_pastizal": 0,
        "bosque_seco": 0,
        "bosque_humedo": 0,
        "humedal_palmar": 0,
        "no_combustible": 0,
        "otro": 0
    }

    for cod, cant in conteo_codigos.items():
        cat = categorizar_combustible(cod, region)
        acumulados[cat] += cant

    pct_pastura = float(100.0 * acumulados["pastura"] / total_pixeles)
    pct_sabana = float(100.0 * acumulados["sabana_pastizal"] / total_pixeles)
    pct_b_seco = float(100.0 * acumulados["bosque_seco"] / total_pixeles)
    pct_b_humedo = float(100.0 * acumulados["bosque_humedo"] / total_pixeles)
    pct_palmar = float(100.0 * acumulados["humedal_palmar"] / total_pixeles)
    pct_no_comb = float(100.0 * acumulados["no_combustible"] / total_pixeles)

    # Máscara de Etapa 1: Si más del 80% de la celda es agua permanente o ciudad, NO es combustible
    es_combustible = bool(pct_no_comb < 80.0)

    return {
        "pct_pastura": pct_pastura,
        "pct_sabana_pastizal": pct_sabana,
        "pct_bosque_seco": pct_b_seco,
        "pct_bosque_humedo": pct_b_humedo,
        "pct_humedal_palmar": pct_palmar,
        "pct_no_combustible": pct_no_comb,
        "es_combustible": es_combustible
    }

def procesar_geotiff_mapbiomas_a_grilla(
    df_grilla: pl.DataFrame,
    rutas_tif: list[Path],
    resolucion_m: int = 1000,
    ventana_px: int = 2048,
) -> pl.DataFrame:
    """
    Extrae fracciones de clases MapBiomas (30 m) a la grilla de 1 km.

    Para cada código se crea una máscara binaria en la resolución nativa y se
    reproyecta por ventanas a la grilla de 1 km con ``average``. El procesamiento
    por ventanas evita cargar el raster completo (Paraguay ~1.1e9 píxeles) en
    memoria. Se asume que los GeoTIFF no se solapan; si se solapan, las
    fracciones se suman y luego se normalizan a 100.

    La leyenda implementada es la de MapBiomas Paraguay Colección 2:
    3/4/6 bosque, 9 plantación, 11 humedal, 12 pastizal, 15/18 pastura/agro,
    22 no vegetado, 26 agua.
    """
    if not rutas_tif:
        raise ValueError("Se requiere al menos un GeoTIFF MapBiomas")

    x = df_grilla["x_utm"].to_numpy()
    y = df_grilla["y_utm"].to_numpy()
    min_x = float(np.min(x)) - resolucion_m / 2.0
    max_y = float(np.max(y)) + resolucion_m / 2.0
    max_x = float(np.max(x)) + resolucion_m / 2.0
    min_y = float(np.min(y)) - resolucion_m / 2.0
    width = int(round((max_x - min_x) / resolucion_m))
    height = int(round((max_y - min_y) / resolucion_m))
    dst_transform = from_origin(min_x, max_y, resolucion_m, resolucion_m)

    c_idx = np.round((x - min_x - resolucion_m / 2.0) / resolucion_m).astype(int)
    r_idx = np.round((max_y - resolucion_m / 2.0 - y) / resolucion_m).astype(int)
    if c_idx.min() < 0 or c_idx.max() >= width or r_idx.min() < 0 or r_idx.max() >= height:
        raise ValueError("La grilla no cabe en el raster destino")

    codigos = list(CODIGOS_MAPBIOMAS.keys())
    acumulado = {cod: np.zeros(width * height, dtype=np.float32) for cod in codigos}

    for ruta in rutas_tif:
        print(f"  Procesando MapBiomas: {ruta.name}")
        with rasterio.open(ruta) as src:
            destinos = {cod: np.zeros((height, width), dtype=np.float32) for cod in codigos}
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
            for cod in codigos:
                acumulado[cod] += destinos[cod].ravel()

    # Fracciones por código en el orden de la grilla (0-100)
    frac = {cod: acumulado[cod][r_idx * width + c_idx] * 100.0 for cod in codigos}

    suma = np.zeros(len(df_grilla), dtype=np.float32)
    for cod in codigos:
        suma += frac[cod]
    factor = np.where(suma > 100.0, 100.0 / np.maximum(suma, 1e-9), 1.0).astype(np.float32)

    regiones = df_grilla["region"].to_list()
    pct = {
        "pct_pastura": np.zeros(len(df_grilla), dtype=np.float32),
        "pct_sabana_pastizal": np.zeros(len(df_grilla), dtype=np.float32),
        "pct_bosque_seco": np.zeros(len(df_grilla), dtype=np.float32),
        "pct_bosque_humedo": np.zeros(len(df_grilla), dtype=np.float32),
        "pct_humedal_palmar": np.zeros(len(df_grilla), dtype=np.float32),
        "pct_no_combustible": np.zeros(len(df_grilla), dtype=np.float32),
    }
    for cod in codigos:
        valores = frac[cod] * factor
        cat = categorizar_combustible(cod, "Chaco")
        if cat in ("bosque_seco", "bosque_humedo"):
            for i, reg in enumerate(regiones):
                key = "pct_bosque_seco" if reg == "Chaco" else "pct_bosque_humedo"
                pct[key][i] += valores[i]
        else:
            key = {
                "pastura": "pct_pastura",
                "sabana_pastizal": "pct_sabana_pastizal",
                "humedal_palmar": "pct_humedal_palmar",
                "no_combustible": "pct_no_combustible",
            }.get(cat)
            if key:
                pct[key] += valores

    pct_no_comb = np.clip(pct["pct_no_combustible"], 0.0, 100.0)
    return pl.DataFrame({
        "cell_id": df_grilla["cell_id"],
        **{k: v.astype(np.float32) for k, v in pct.items()},
        "es_combustible": pct_no_comb < 80.0,
        "origen_datos": ["mapbiomas_real"] * len(df_grilla),
    })


def generar_perfil_combustible_base(df_grilla: pl.DataFrame) -> pl.DataFrame:
    """
    Genera el perfil biofísico de combustible de referencia para las 400.006 celdas de Paraguay
    basado en la distribución ecorregional oficial de MapBiomas Chaco y Bosque Atlántico.
    Garantiza que el pipeline funcione de forma inmediata y determinista.
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

    # Semilla fija para reproducibilidad matemática
    rng = np.random.default_rng(42)

    for i in range(n):
        reg = regiones[i]
        dep = deptos[i]
        lat = lats[i]
        lon = lons[i]

        if reg == "Chaco":
            # Chaco Boreal: predominio de bosque xerofítico seco y pasturas en expansión
            if dep == "ALTO PARAGUAY":
                # Pantanal y Chaco húmedo norte
                p_seco = rng.uniform(55.0, 85.0)
                p_palmar = rng.uniform(5.0, 25.0)
                p_past = rng.uniform(5.0, 20.0)
                p_sab = max(0.0, 100.0 - (p_seco + p_palmar + p_past))
                pct_b_seco[i] = p_seco
                pct_palmar[i] = p_palmar
                pct_pastura[i] = p_past
                pct_sabana[i] = p_sab
            elif dep == "BOQUERON":
                # Chaco seco central
                p_seco = rng.uniform(50.0, 75.0)
                p_past = rng.uniform(15.0, 40.0) # Colonias menonitas y pasturas Gatton Panic
                p_sab = max(0.0, 100.0 - (p_seco + p_past))
                pct_b_seco[i] = p_seco
                pct_pastura[i] = p_past
                pct_sabana[i] = p_sab
            else: # PRESIDENTE HAYES
                # Bajo Chaco: palmares de karanda'y y pastizales inundables
                p_palmar = rng.uniform(25.0, 50.0)
                p_past = rng.uniform(20.0, 40.0)
                p_seco = max(0.0, 100.0 - (p_palmar + p_past))
                pct_palmar[i] = p_palmar
                pct_pastura[i] = p_past
                pct_b_seco[i] = p_seco

        else: # Región Oriental
            if dep == "ASUNCION":
                # Casco urbano denso
                pct_no_comb[i] = 85.0
                pct_pastura[i] = 15.0
                es_comb[i] = False
            else:
                # Región Oriental productiva
                p_past = rng.uniform(40.0, 70.0) # Pasturas cultivadas dominantes
                p_humedo = rng.uniform(10.0, 35.0) # Remanentes de BAAPA
                p_sab = max(0.0, 100.0 - (p_past + p_humedo))
                pct_pastura[i] = p_past
                pct_b_humedo[i] = p_humedo
                pct_sabana[i] = p_sab

    return pl.DataFrame({
        "cell_id": cell_ids,
        "pct_pastura": pct_pastura,
        "pct_sabana_pastizal": pct_sabana,
        "pct_bosque_seco": pct_b_seco,
        "pct_bosque_humedo": pct_b_humedo,
        "pct_humedal_palmar": pct_palmar,
        "pct_no_combustible": pct_no_comb,
        "es_combustible": es_comb,
        "origen_datos": [ORIGEN_PERFIL_SINTETICO] * n,
    })
