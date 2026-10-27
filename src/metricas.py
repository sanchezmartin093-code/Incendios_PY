"""
metricas.py module: Vectorized model evaluation functions and threshold optimization.
Direct implementation with pure NumPy (without depending on scikit-learn).
Includes:
- Optimal threshold search via Youden's J statistic (TPR - FPR)
- Optimal threshold search for F-beta (F0.5 to prioritize Precision)
- Brier Score and Brier Skill Score (BSS)
"""

import numpy as np

def _puntos_roc_agrupados(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes ROC points by grouping tied scores.

    With ties, the internal order of argsort is arbitrary. If TP/FP are
    accumulated record by record, invalid ROC points and thresholds are
    generated that do not reproduce the metric when classifying with ``>=``.
    This helper accumulates at the end of each group of identical scores.
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    if n == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0])

    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ord = y_true[orden]
    p_ord = y_prob[orden]

    # Last index of each group of identical scores
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
    Computes FPR, TPR and thresholds grouped by score (tie-aware).
    """
    return _puntos_roc_agrupados(y_true, y_prob)

def calcular_umbral_optimo_youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Computes the cutoff threshold that maximizes Youden's J statistic:
    J = Sensitivity + Specificity - 1 = TPR - FPR
    Returns: (optimal_threshold, max_j)
    """
    fpr, tpr, thresholds = calcular_curva_roc(y_true, y_prob)
    j_scores = tpr - fpr
    idx_opt = np.argmax(j_scores)
    return float(thresholds[idx_opt]), float(j_scores[idx_opt])

def calcular_umbral_optimo_f_beta(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 0.5) -> tuple[float, float]:
    """
    Computes the threshold that maximizes the F-beta score.
    With beta = 0.5, Precision (minimizing false alarms) is given twice the weight of Recall.
    Returns: (optimal_threshold, max_f_beta)
    """
    y_true = np.asarray(y_true).astype(float).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    if n == 0:
        return 1.0, 0.0

    orden = np.argsort(y_prob, kind="mergesort")[::-1]
    y_ord = y_true[orden]
    p_ord = y_prob[orden]

    # Group ties: the threshold is applied with >=, so all records
    # with the same score enter together.
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
    Corrects probabilities calibrated on a case-control sample to
    estimate the population probability.

    If the model was calibrated with an artificial prevalence ``pi_s`` and the
    real prevalence is ``pi_p``, the correct adjustment is a shift of the
    logit:

        logit(p_pop) = logit(p_sample) + logit(pi_p) - logit(pi_s)

    This is the standard intercept correction for case-control sampling.
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
    Metrics weighted by the inverse sampling probability.

    ``weights`` must be 1 for presences and ``N_neg_population /
    N_neg_sample`` for absences. This re-expands the case-control sample
    to the population of cell-days.
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

    # ECE weighted by population weight
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
    Paired test for ROC-AUC difference via bootstrap.

    Resamples rows (or complete blocks if ``block_ids`` is provided) with
    replacement, computes AUC(model) - AUC(reference) and returns (observed
    delta, two-sided p-value). The p-value is a bootstrap approximation,
    not an exact test.
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
    # two-sided p: proportion of deltas on the other side of zero
    p_inf = float(np.mean(deltas <= 0))
    p_sup = float(np.mean(deltas >= 0))
    p_val = 2.0 * min(p_inf, p_sup)
    return delta, float(min(max(p_val, 1.0 / len(deltas)), 1.0))


def calcular_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes the Brier Score: mean of (y_prob - y_true)^2"""
    return float(np.mean((y_prob - y_true) ** 2))

def calcular_brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes the Brier Skill Score (BSS) relative to the mean climatological fire rate.
    BSS = 1 - (BS_model / BS_climatology)
    BSS > 0 indicates that the model predicts significantly better than the mean frequency.
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
    Computes the Expected Calibration Error (ECE) weighted by the size of each decile.
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Where acc(B_m) is the observed prevalence and conf(B_m) the mean predicted probability.
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
    Computes the (1 - alpha)% Confidence Interval (95% by default) for ROC-AUC
    via non-parametric bootstrap. If ``block_ids`` is provided, it resamples
    complete blocks (e.g., dates) instead of individual rows.
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
    Computes the non-conformity quantile q_hat over the calibration/validation set
    for Split Conformal Prediction with marginal coverage guarantee 1 - alpha (e.g., 90%).
    
    Non-conformity score: s_i = 1 - p_val[i] if y_val[i] == 1 else p_val[i].
    q_hat = ceiling((n + 1)(1 - alpha)) / n quantile of s_1, ..., s_n.
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
    Generates conformally valid prediction sets for new probabilities p_test.
    Returns: (includes_zero, includes_one) boolean arrays where C(x) = {y: includes_y[i]}.
    Marginal guarantee: P(Y_test in C(X_test)) >= 1 - alpha.
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
    Evaluates the empirical proportion of true labels covered by the conformal sets.
    """
    y_true = np.asarray(y_true, dtype=int)
    c0, c1 = predecir_conjuntos_conformes(p_test, q_hat)
    cubierto = np.where(y_true == 1, c1, c0)
    return float(np.mean(cubierto))

