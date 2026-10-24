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
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
import joblib

class CalibradorPlattMonotono:
    """
    Calibrador de Platt Monótono No-Decreciente (Monotonic Platt Scaling).
    Ajusta una transformación sigmoide P = 1 / (1 + exp(-(w*z + b))) sobre logits z,
    garantizando que w >= 0.1 para que el ranking del ensamble (ROC-AUC) se preserve estrictamente.
    """
    def __init__(self):
        self.w = 1.0
        self.b = 0.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray):
        eps = 1e-6
        p_clip = np.clip(p_raw, eps, 1.0 - eps)
        z = np.log(p_clip / (1.0 - p_clip))

        def brier_obj(params):
            w_val, b_val = params
            p_cal = 1.0 / (1.0 + np.exp(-(w_val * z + b_val)))
            return float(np.mean((y - p_cal) ** 2))

        res = minimize(brier_obj, [1.0, 0.0], bounds=[(0.1, 10.0), (-10.0, 10.0)])
        self.w, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if isinstance(p_raw, np.ndarray) and p_raw.ndim == 2:
            p_raw = p_raw.ravel()
        eps = 1e-6
        p_clip = np.clip(p_raw, eps, 1.0 - eps)
        z = np.log(p_clip / (1.0 - p_clip))
        return 1.0 / (1.0 + np.exp(-(self.w * z + self.b)))

    def predict_proba(self, p_raw: np.ndarray) -> np.ndarray:
        p_1d = self.predict(p_raw)
        return np.column_stack([1.0 - p_1d, p_1d])

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.predict(p_raw)

# Lista canónica de variables predictoras (Features)
FEATURES_BASE = [
    # Clima dinámico y desecación física
    "temp_max_c",
    "temp_min_c",
    "rango_termico_diario",
    "vpd_max_kpa",
    "deficit_hidrico_acumulado",
    "precipitacion_mm",
    "precipitacion_48h_mm",
    "dias_secos_consecutivos",
    # Comportamiento de propagación y viento
    "viento_max_kmh",
    "rafaga_max_kmh",
    "potencial_viento_cuadratico",
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
    scale_pos_weight: float = 2.0,
    learning_rate: float = 0.03,
    num_leaves: int = 15,
    min_child_samples: int = 10,
    num_boost_round: int = 100,
    early_stopping_rounds: int | None = 15,
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
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": 5,
        "min_child_samples": min_child_samples,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 1.0,
        "random_state": semilla,
        "verbose": -1
    }

    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

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
    y_val: np.ndarray,
    metodo: str = "platt_monotono"
):
    """
    Calibra los scores del ensamble en probabilidades reales P(Y=1) ∈ [0.0, 1.0].
    - 'platt_monotono': Calibración logística monotónica (preserva 100% el ROC-AUC sin colapsos).
    - 'isotonica': Regresión isotónica no paramétrica.
    """
    y_pred_raw = booster.predict(X_val)
    if metodo == "platt_monotono":
        calibrador = CalibradorPlattMonotono()
        calibrador.fit(y_pred_raw, y_val)
    else:
        calibrador = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrador.fit(y_pred_raw, y_val)
    return calibrador

def predecir_probabilidad(
    booster: lgb.Booster,
    calibrador,
    X: np.ndarray
) -> np.ndarray:
    """
    Genera la inferencia probabilística: score del árbol + calibración.
    """
    raw_scores = booster.predict(X)
    if calibrador is not None:
        if hasattr(calibrador, "predict_proba"):
            probs = calibrador.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        elif hasattr(calibrador, "transform"):
            probs = calibrador.transform(raw_scores)
        else:
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
