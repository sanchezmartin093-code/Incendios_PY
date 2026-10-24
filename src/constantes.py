"""
Constantes geoespaciales, rutas canónicas y parámetros del sistema Incendios_PY.
"""

from pathlib import Path

# Rutas raíz relativas
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_DATOS = ROOT_DIR / "datos"
DIR_RAW = DIR_DATOS / "raw"
DIR_PROCESSED = DIR_DATOS / "processed"
DIR_OUTPUTS = DIR_DATOS / "outputs"
DIR_MODELOS = ROOT_DIR / "modelos"
DIR_FIGURAS = ROOT_DIR / "figuras"
DIR_TABLAS = ROOT_DIR / "tablas"

# Sistemas de Referencia de Coordenadas (CRS)
CRS_GEOGRAFICO = "EPSG:4326"       # WGS84 para ingesta satelital y APIs
CRS_PROYECTADO = "EPSG:32721"      # UTM Zona 21 Sur (oficial para métricas métricas en Paraguay)

# Parámetros de la Grilla Nacional
RESOLUCION_GRILLA_METROS = 1000    # 1 km x 1 km

# Bounding box aproximado de Paraguay continental
BBOX_PARAGUAY = {
    "min_lon": -62.64,
    "max_lon": -54.25,
    "min_lat": -27.60,
    "max_lat": -19.29
}

# Períodos de partición temporal (Anti-Data-Leakage)
YEARS_TRAIN = list(range(2018, 2023))  # 2018 a 2022
YEAR_VAL = 2023                        # Ajuste de hiperparámetros
YEAR_TEST = 2024                       # Testeo ciego out-of-time (Crisis histórica Chaco)

# Rigor GeoAI: Buffer de Zona Muerta Espacial (Dead-Zone Buffer)
BUFFER_ZONA_MUERTA_KM = 20             # Amortiguamiento para anular autocorrelación espacial entre folds

# Control de Calidad Satelital (Filtro Bow-Tie y Ángulo Cenital VIIRS)
MAX_SCAN_ANGLE_DEG = 45.0              # Excluye deformaciones extremas en los bordes de la franja orbital
MAX_COBERTURA_NUBOSA_PCT = 60.0        # Excluye celdas ciegas para evitar falsas ausencias (label noise)

# Proporción de muestreo para pseudo-ausencias
RATIO_AUSENCIAS_PRESENCIAS = 10        # 10 negativas por cada positiva en entrenamiento

# Umbrales Biofísicos para Filtrado de Etapa 1 (Two-Stage Gating)
UMBRAL_VPD_MINIMO_KPA = 0.8            # Por debajo de 0.8 kPa el aire está saturado; riesgo de ignición casi nulo
UMBRAL_LLUVIA_48H_MM = 10.0            # Lluvias acumuladas > 10 mm en 48h extinguen o inhiben la ignición
DIAS_RESET_COMBUSTIBLE = 30            # Período de exclusión de biomasa tras un fuego severo en el píxel
