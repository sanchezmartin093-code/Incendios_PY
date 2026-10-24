"""
Módulo metricas.py: Funciones vectorizadas de evaluación de modelos y optimización de umbrales.
Implementación directa con NumPy puro (sin depender de scikit-learn).
Incluye:
- Búsqueda de umbral óptimo mediante Estadística J de Youden (TPR - FPR)
- Búsqueda de umbral óptimo para F-beta (F0.5 para priorizar Precisión)
- Brier Score y Brier Skill Score (BSS)
"""

import numpy as np

def calcular_curva_roc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula FPR, TPR y umbrales ordenados de forma vectorizada.
    """
    orden = np.argsort(y_prob)[::-1]
    y_true_ord = y_true[orden]
    y_prob_ord = y_prob[orden]
    
    tp_acum = np.cumsum(y_true_ord)
    fp_acum = np.cumsum(1.0 - y_true_ord)
    
    total_pos = tp_acum[-1]
    total_neg = fp_acum[-1]
    
    tpr = tp_acum / total_pos if total_pos > 0 else np.zeros_like(tp_acum)
    fpr = fp_acum / total_neg if total_neg > 0 else np.zeros_like(fp_acum)
    
    return fpr, tpr, y_prob_ord

def calcular_umbral_optimo_youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Calcula el umbral de corte que maximiza el estadístico J de Youden:
    J = Sensibilidad + Especificidad - 1 = TPR - FPR
    Retorna: (umbral_optimo, max_j)
    """
    fpr, tpr, thresholds = calcular_curva_roc(y_true, y_prob)
    j_scores = tpr - fpr
    idx_opt = np.argmax(j_scores)
    return float(thresholds[idx_opt]), float(j_scores[idx_opt])

def calcular_umbral_optimo_f_beta(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 0.5) -> tuple[float, float]:
    """
    Calcula el umbral que maximiza el F-beta score.
    Con beta = 0.5, se le da el doble de peso a la Precisión (minimizar falsas alarmas) sobre el Recall.
    Retorna: (umbral_optimo, max_f_beta)
    """
    orden = np.argsort(y_prob)[::-1]
    y_true_ord = y_true[orden]
    y_prob_ord = y_prob[orden]
    
    tp_acum = np.cumsum(y_true_ord)
    fp_acum = np.cumsum(1.0 - y_true_ord)
    total_pos = tp_acum[-1]
    
    precision = np.where((tp_acum + fp_acum) > 0, tp_acum / (tp_acum + fp_acum), 1.0)
    recall = tp_acum / total_pos if total_pos > 0 else np.zeros_like(tp_acum)
    
    numerador = (1.0 + beta ** 2) * (precision * recall)
    denominador = ((beta ** 2) * precision) + recall
    
    f_beta = np.where(denominador > 0, numerador / denominador, 0.0)
    idx_opt = np.argmax(f_beta)
    return float(y_prob_ord[idx_opt]), float(f_beta[idx_opt])

def calcular_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula el Brier Score: promedio de (y_prob - y_true)^2"""
    return float(np.mean((y_prob - y_true) ** 2))

def calcular_brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcula el Brier Skill Score (BSS) respecto a la tasa climatológica media de fuego.
    BSS = 1 - (BS_modelo / BS_climatologia)
    BSS > 0 indica que el modelo predice significativamente mejor que la frecuencia media.
    """
    bs_modelo = calcular_brier_score(y_true, y_prob)
    prob_base = np.mean(y_true)
    y_base = np.full_like(y_true, prob_base, dtype=float)
    bs_base = calcular_brier_score(y_true, y_base)
    
    if bs_base == 0:
        return 0.0
    return float(1.0 - (bs_modelo / bs_base))
