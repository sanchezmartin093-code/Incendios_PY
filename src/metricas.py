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
    
    with np.errstate(divide="ignore", invalid="ignore"):
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

def calcular_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calcula el Expected Calibration Error (ECE) ponderado por el tamaño de cada decil.
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Donde acc(B_m) es la prevalencia observada y conf(B_m) la probabilidad media predicha.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        
        n_bin = np.sum(mask)
        if n_bin > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (n_bin / n) * np.abs(acc - conf)

    return float(ece)

def calcular_bootstrap_ci_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    semilla: int = 42
) -> tuple[float, float]:
    """
    Calcula el Intervalo de Confianza al (1 - alpha)% (por defecto 95%) para el ROC-AUC
    mediante muestreo no paramétrico con reemplazo.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) > 1:
            boot_scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    if not boot_scores:
        val = float(roc_auc_score(y_true, y_prob))
        return val, val

    inf = float(np.percentile(boot_scores, 100.0 * (alpha / 2.0)))
    sup = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha / 2.0)))
    return inf, sup
