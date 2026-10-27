"""
Geospatial constants, canonical paths, and parameters for the Incendios_PY system.
"""

from pathlib import Path

# Relative root paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DIR_DATOS = ROOT_DIR / "datos"
DIR_RAW = DIR_DATOS / "raw"
DIR_PROCESSED = DIR_DATOS / "processed"
DIR_OUTPUTS = DIR_DATOS / "outputs"
DIR_MODELOS = ROOT_DIR / "modelos"
DIR_FIGURAS = ROOT_DIR / "figuras"
DIR_TABLAS = ROOT_DIR / "tablas"

# Coordinate Reference Systems (CRS)
CRS_GEOGRAFICO = "EPSG:4326"       # WGS84 for satellite ingestion and APIs
CRS_PROYECTADO = "EPSG:32721"      # UTM Zone 21 South (official for metric measurements in Paraguay)

# National Grid Parameters
RESOLUCION_GRILLA_METROS = 1000    # 1 km x 1 km

# Approximate bounding box of continental Paraguay
BBOX_PARAGUAY = {
    "min_lon": -62.64,
    "max_lon": -54.25,
    "min_lat": -27.60,
    "max_lat": -19.29
}

# Temporal partition periods (Anti-Data-Leakage)
YEARS_TRAIN = list(range(2018, 2023))  # 2018 to 2022 (5 years of synoptic and interannual variability)
YEAR_VAL = 2023                        # Hyperparameter tuning and Platt calibration
YEARS_TEST = [2024, 2025]              # Blind out-of-time testing (historic Chaco crisis and recent season)
YEAR_TEST = 2024                       # Primary retrospective test year (compatibility)

# GeoAI Rigor: Spatial Dead-Zone Buffer
BUFFER_ZONA_MUERTA_KM = 20             # Damping to cancel spatial autocorrelation between folds

# Satellite Quality Control (Bow-Tie Filter and VIIRS Zenith Angle)
MAX_SCAN_ANGLE_DEG = 45.0              # Excludes extreme distortions at the edges of the orbital swath
MAX_COBERTURA_NUBOSA_PCT = 60.0        # Excludes blind cells to avoid false absences (label noise)

# Sampling proportion for pseudo-absences
RATIO_AUSENCIAS_PRESENCIAS = 10        # 10 negatives for every positive in training

# D2 definition of targets
DIAS_IGNICION_SIN_FUEGO = 10            # A primary ignition had no fire in the cell in the last N days
RADIO_PROPAGACION_KM = 25               # Propagation domain: cells within R km of a fire at t-1
WARMUP_DIAS = 60                         # Antecedent window requiring a complete 60-day history
FECHA_INICIO_TRAIN_SIN_WARMUP = "2018-03-02"  # First date with a complete 60-day window

# Biophysical Thresholds for Stage 1 Filtering (Two-Stage Gating)
UMBRAL_VPD_MINIMO_KPA = 0.8            # Below 0.8 kPa the air is saturated; ignition risk is nearly zero
UMBRAL_LLUVIA_48H_MM = 10.0            # Accumulated rainfall > 10 mm in 48h extinguishes or inhibits ignition
DIAS_RESET_COMBUSTIBLE = 30            # Biomass exclusion period after a severe fire in the pixel
