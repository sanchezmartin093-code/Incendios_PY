"""
Módulo modelo.py: Entrenamiento, calibración isotónica y serialización C++ de modelos LightGBM.
Filosofía SOTA:
- Modelado regional estratificado: Modelo Chaco vs. Modelo Región Oriental.
- Algoritmo Histogram-based Gradient Boosting con soporte nativo de valores faltantes y regularización L1/L2.
- Manejo de desbalance mediante penalización de pesos positivos (scale_pos_weight).
- Calibración Isotónica (Platt/Isotonic) para garantizar probabilidades reales P(Fuego) ∈ [0.0, 1.0].
- Serialización interoperable en formato nativo C++ Booster (.booster) y Joblib.
"""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
import joblib

# Lista canónica de variables predictoras (Features)
FEATURES_BASE = [
    # Clima dinámico y desecación física
    "temp_max_c",
    "temp_min_c",
    "vpd_max_kpa",
    "precipitacion_mm",
    "precipitacion_48h_mm",
    "dias_secos_consecutivos",
    # Comportamiento de propagación y viento
    "viento_max_kmh",
    "rafaga_max_kmh",
    "interaccion_viento_vpd",
    "ffmc",
    "isi",
    # Calendario antrópico
    "dia_semana",
    "es_fin_de_semana",
    "mes",
    "dia_del_anio",
    # Coberturas de biomasa
    "pct_pastura",
    "pct_sabana_pastizal",
    "pct_humedal_palmar"
]

FEATURES_CHACO = FEATURES_BASE + ["pct_bosque_seco"]
FEATURES_ORIENTAL = FEATURES_BASE + ["pct_bosque_humedo"]

def preparar_matrices_xy(df: pl.DataFrame, region: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Filtra el dataset por región, extrae las features correspondientes y la etiqueta dependiente Y.
    """
    df_reg = df.filter(pl.col("region") == region)
    features = FEATURES_CHACO if region == "Chaco" else FEATURES_ORIENTAL
    
    # Asegurar que todas las columnas existan
    feats_existentes = [f for f in features if f in df_reg.columns]
    
    X = df_reg.select(feats_existentes).to_numpy()
    y = df_reg["fuego"].to_numpy().astype(np.float32)

    return X, y, feats_existentes

def entrenar_modelo_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    nombres_features: list[str],
    scale_pos_weight: float = 10.0,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    semilla: int = 42
) -> lgb.Booster:
    """
    Entrena un ensamble LightGBM para clasificación binaria optimizado para eventos raros.
    """
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=nombres_features, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=nombres_features, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 10,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": semilla,
        "verbose": -1
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks
    )

    return booster

def calibrar_probabilidades_isotonica(
    booster: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> IsotonicRegression:
    """
    Ajusta una regresión isotónica no paramétrica sobre las predicciones de validación
    para transformar los scores en probabilidades reales bien calibradas P(Y=1) ∈ [0.0, 1.0].
    """
    y_pred_raw = booster.predict(X_val)
    # Isotonic Regression requiere valores acotados
    calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrador.fit(y_pred_raw, y_val)
    return calibrador

def predecir_probabilidad(
    booster: lgb.Booster,
    calibrador: IsotonicRegression | None,
    X: np.ndarray
) -> np.ndarray:
    """
    Genera la inferencia probabilística: score del árbol + calibración isotónica.
    """
    raw_scores = booster.predict(X)
    if calibrador is not None:
        probs = calibrador.predict(raw_scores)
    else:
        probs = raw_scores
    return np.clip(probs, 0.0, 1.0)

def guardar_modelo_y_calibrador(
    booster: lgb.Booster,
    calibrador: IsotonicRegression,
    ruta_booster: Path,
    ruta_calibrador: Path
):
    """
    Serializa el modelo en formato nativo C++ Booster y el calibrador en Joblib.
    """
    ruta_booster.parent.mkdir(parents=True, exist_ok=True)
    ruta_calibrador.parent.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(ruta_booster))
    joblib.dump(calibrador, str(ruta_calibrador))

def cargar_modelo_y_calibrador(
    ruta_booster: Path,
    ruta_calibrador: Path
) -> tuple[lgb.Booster, IsotonicRegression]:
    """
    Carga el booster nativo C++ y el calibrador serializado.
    """
    booster = lgb.Booster(model_file=str(ruta_booster))
    calibrador = joblib.load(str(ruta_calibrador))
    return booster, calibrador
