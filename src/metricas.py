"""
Módulo metricas.py: Funciones vectorizadas de evaluación de modelos y optimización de umbrales.
Implementación directa con NumPy puro (sin depender de scikit-learn).
Incluye:
- Búsqueda de umbral óptimo mediante Estadística J de Youden (TPR - FPR)
- Búsqueda de umbral óptimo para F-beta (F0.5 para priorizar Precisión)
- Brier Score y Brier Skill Score (BSS)
"""

import numpy as np

def _puntos_roc_agrupados(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula puntos ROC agrupando puntuaciones empatadas.

    Con empates, el orden interno del argsort es arbitrario. Si se acumulan
    TP/FP registro a registro se generan puntos ROC inválidos y umbrales que
    no reproducen la métrica al clasificar con ``>=``. Este helper acumula al
    final de cada grupo de puntuaciones idénticas.
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])

    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ord = y_true[orden]
    p_ord = y_prob[orden]

    # Último índice de cada grupo de puntuaciones idénticas
    cambios = np.where(np.diff(p_ord) != 0)[0]
    fines = np.r_[cambios, n - 1]

    tp_acum = np.cumsum(y_ord)[fines]
    fp_acum = np.cumsum(1.0 - y_ord)[fines]
    thresholds = p_ord[fines]

    total_pos = float(tp_acum[-1])
    total_neg = float(fp_acum[-1])
    tpr = tp_acum / total_pos if total_pos > 0 else np.zeros_like(tp_acum)
    fpr = fp_acum / total_neg if total_neg > 0 else np.zeros_like(fp_acum)

    return fpr, tpr, thresholds


def calcular_curva_roc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula FPR, TPR y umbrales agrupados por puntuación (tie-aware).
    """
    return _puntos_roc_agrupados(y_true, y_prob)

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
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    if n == 0:
        return 1.0, 0.0

    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ord = y_true[orden]
    p_ord = y_prob[orden]

    # Agrupar empates: el umbral se aplica con >=, por lo que todos los
    # registros con la misma puntuación entran juntos.
    cambios = np.where(np.diff(p_ord) != 0)[0]
    fines = np.r_[cambios, n - 1]

    tp_acum = np.cumsum(y_ord)[fines]
    fp_acum = np.cumsum(1.0 - y_ord)[fines]
    thresholds = p_ord[fines]
    total_pos = float(tp_acum[-1])

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp_acum + fp_acum) > 0, tp_acum / (tp_acum + fp_acum), 1.0)
        recall = tp_acum / total_pos if total_pos > 0 else np.zeros_like(tp_acum)

        numerador = (1.0 + beta ** 2) * (precision * recall)
        denominador = ((beta ** 2) * precision) + recall
        f_beta = np.where(denominador > 0, numerador / denominador, 0.0)

    idx_opt = np.argmax(f_beta)
    return float(thresholds[idx_opt]), float(f_beta[idx_opt])

def corregir_prior_probabilidades(
    p_muestreo: np.ndarray,
    prevalencia_muestreo: float,
    prevalencia_poblacion: float,
) -> np.ndarray:
    """
    Corrige probabilidades calibradas sobre una muestra case-control para
    estimar la probabilidad poblacional.

    Si el modelo fue calibrado con una prevalencia artificial ``pi_s`` y la
    prevalencia real es ``pi_p``, el ajuste correcto es un desplazamiento del
    logit:

        logit(p_pob) = logit(p_muestreo) + logit(pi_p) - logit(pi_s)

    Es la corrección estándar de intercepto para muestreo caso-control.
    """
    pi_s = float(prevalencia_muestreo)
    pi_p = float(prevalencia_poblacion)
    if not (0.0 < pi_s < 1.0) or not (0.0 < pi_p < 1.0):
        raise ValueError("Las prevalencias deben estar estrictamente en (0, 1)")

    p = np.clip(np.asarray(p_muestreo, dtype=float), 1e-9, 1.0 - 1e-9)
    odds = p / (1.0 - p)
    factor = (pi_p / (1.0 - pi_p)) / (pi_s / (1.0 - pi_s))
    odds_corr = odds * factor
    return odds_corr / (1.0 + odds_corr)


def calcular_metricas_poblacionales(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    cutoff: float,
    beta: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Métricas ponderadas por el inverso de la probabilidad de muestreo.

    ``weights`` debe ser 1 para las presencias y ``N_neg_poblacion /
    N_neg_muestra`` para las ausencias. Así la muestra case-control se
    reexpande a la población de celdas-día.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if len(y) != len(p) or len(y) != len(w):
        raise ValueError("y_true, y_prob y weights deben tener la misma longitud")
    if np.any(w < 0):
        raise ValueError("weights no puede contener valores negativos")
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        raise ValueError("La suma de weights debe ser positiva")
    w_norm = w / w_sum

    pred = (p >= cutoff).astype(int)
    tp = float(np.sum(w_norm * ((y == 1) & (pred == 1))))
    fp = float(np.sum(w_norm * ((y == 0) & (pred == 1))))
    fn = float(np.sum(w_norm * ((y == 1) & (pred == 0))))
    tn = float(np.sum(w_norm * ((y == 0) & (pred == 0))))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fbeta = (
        (1.0 + beta ** 2) * precision * recall
        / ((beta ** 2) * precision + recall)
        if ((beta ** 2) * precision + recall) > 0 else 0.0
    )

    brier = float(np.sum(w_norm * (p - y) ** 2))
    prev_pob = float(np.sum(w_norm * y))
    brier_base = float(np.sum(w_norm * (prev_pob - y) ** 2))
    bss = 1.0 - brier / brier_base if brier_base > 0 else 0.0

    # ECE ponderado por peso poblacional
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (p >= low) & (p <= high) if i == n_bins - 1 else (p >= low) & (p < high)
        w_bin = float(np.sum(w_norm[mask]))
        if w_bin > 0:
            acc = float(np.sum(w_norm[mask] * y[mask]) / w_bin)
            conf = float(np.sum(w_norm[mask] * p[mask]) / w_bin)
            ece += w_bin * abs(acc - conf)

    auc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else 0.5
    pr_auc = (
        float(average_precision_score(y, p, sample_weight=w))
        if len(np.unique(y)) > 1 else prev_pob
    )

    return {
        "roc_auc_poblacional": auc,
        "pr_auc_poblacional": pr_auc,
        "precision_poblacional": precision,
        "recall_poblacional": recall,
        "especificidad_poblacional": specificity,
        "f05_poblacional": fbeta,
        "brier_poblacional": brier,
        "bss_poblacional": bss,
        "ece_poblacional": ece,
        "prevalencia_poblacional": prev_pob,
        "tp_poblacional": tp,
        "fp_poblacional": fp,
        "fn_poblacional": fn,
        "tn_poblacional": tn,
    }


def bootstrap_diferencia_roc(
    y_true: np.ndarray,
    p_modelo: np.ndarray,
    p_referencia: np.ndarray,
    n_bootstrap: int = 1000,
    semilla: int = 42,
    block_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Test pareado de diferencia de ROC-AUC mediante bootstrap.

    Remuestrea filas (o bloques completos si ``block_ids`` se provee) con
    reemplazo, calcula AUC(modelo) - AUC(referencia) y devuelve (delta
    observado, p-valor bilateral). El p-valor es una aproximación bootstrap,
    no un test exacto.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).reshape(-1)
    p1 = np.asarray(p_modelo).reshape(-1)
    p0 = np.asarray(p_referencia).reshape(-1)
    if not (len(y) == len(p1) == len(p0)):
        raise ValueError("y_true, p_modelo y p_referencia deben tener la misma longitud")

    delta = float(roc_auc_score(y, p1) - roc_auc_score(y, p0))
    rng = np.random.default_rng(semilla)
    n = len(y)
    deltas = []
    if block_ids is not None:
        bloques = np.asarray(block_ids)
        unicos = np.unique(bloques)
        mapa = {b: np.where(bloques == b)[0] for b in unicos}
    for _ in range(n_bootstrap):
        if block_ids is None:
            idx = rng.choice(n, size=n, replace=True)
        else:
            elegidos = rng.choice(unicos, size=len(unicos), replace=True)
            idx = np.concatenate([mapa[b] for b in elegidos])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p0[idx]))
    if not deltas:
        return delta, 1.0
    deltas = np.asarray(deltas)
    # p bilateral: proporción de deltas al otro lado de cero
    p_inf = float(np.mean(deltas <= 0))
    p_sup = float(np.mean(deltas >= 0))
    p_val = 2.0 * min(p_inf, p_sup)
    return delta, float(min(max(p_val, 1.0 / len(deltas)), 1.0))


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
    semilla: int = 42,
    block_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Calcula el Intervalo de Confianza al (1 - alpha)% (por defecto 95%) para el ROC-AUC
    mediante bootstrap no paramétrico. Si ``block_ids`` se provee, remuestrea
    bloques completos (p. ej., fechas) en lugar de filas individuales.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    boot_scores = []
    if block_ids is not None:
        bloques = np.asarray(block_ids)
        unicos = np.unique(bloques)
        mapa = {b: np.where(bloques == b)[0] for b in unicos}

    for _ in range(n_bootstrap):
        if block_ids is None:
            idx = rng.choice(n, size=n, replace=True)
        else:
            elegidos = rng.choice(unicos, size=len(unicos), replace=True)
            idx = np.concatenate([mapa[b] for b in elegidos])
        if len(np.unique(y_true[idx])) > 1:
            boot_scores.append(roc_auc_score(y_true[idx], y_prob[idx]))

    if not boot_scores:
        val = float(roc_auc_score(y_true, y_prob))
        return val, val

    inf = float(np.percentile(boot_scores, 100.0 * (alpha / 2.0)))
    sup = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha / 2.0)))
    return inf, sup

def calcular_cuantil_conforme(
    y_val: np.ndarray,
    p_val: np.ndarray,
    alpha: float = 0.10
) -> float:
    """
    Calcula el cuantil de no-conformidad q_hat sobre el conjunto de calibración/validación
    para Split Conformal Prediction con garantía de cobertura marginal 1 - alpha (ej. 90%).
    
    Score de no-conformidad: s_i = 1 - p_val[i] si y_val[i] == 1 else p_val[i].
    q_hat = cuantil ceiling((n + 1)(1 - alpha)) / n de s_1, ..., s_n.
    """
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)
    n = len(y_val)
    if n == 0:
        return 1.0
    
    scores = np.where(y_val == 1, 1.0 - p_val, p_val)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q_hat = float(np.sort(scores)[k - 1])
    return q_hat

def predecir_conjuntos_conformes(
    p_test: np.ndarray,
    q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera conjuntos de predicción conformalmente válidos para nuevas probabilidades p_test.
    Retorna: (incluye_cero, incluye_uno) boolean arrays donde C(x) = {y: incluye_y[i]}.
    Garantía marginal: P(Y_test in C(X_test)) >= 1 - alpha.
    """
    p_test = np.asarray(p_test, dtype=float)
    incluye_cero = p_test <= q_hat
    incluye_uno = (1.0 - p_test) <= q_hat
    return incluye_cero, incluye_uno

def evaluar_cobertura_conforme(
    y_true: np.ndarray,
    p_test: np.ndarray,
    q_hat: float
) -> float:
    """
    Evalúa la proporción empírica de etiquetas verdaderas cubiertas por los conjuntos conformes.
    """
    y_true = np.asarray(y_true, dtype=int)
    c0, c1 = predecir_conjuntos_conformes(p_test, q_hat)
    cubierto = np.where(y_true == 1, c1, c0)
    return float(np.mean(cubierto))

