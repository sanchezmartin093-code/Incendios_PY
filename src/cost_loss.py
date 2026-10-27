"""
cost_loss.py module: cost-loss analysis for operational decisions.
"""

import numpy as np


def valor_cost_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Relative value V(alpha) using probabilities and population weights."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    pi = float(np.sum(w * y))
    pred = (p >= alpha).astype(int)
    tp = float(np.sum(w * ((y == 1) & (pred == 1))))
    fp = float(np.sum(w * ((y == 0) & (pred == 1))))
    fn = float(np.sum(w * ((y == 1) & (pred == 0))))
    den = min(alpha, pi) - alpha * pi
    num = min(alpha, pi) - (alpha * (tp + fp) + fn)
    valor = num / den if abs(den) > 1e-12 else 0.0
    return {
        "alpha": float(alpha),
        "valor": max(float(valor), -1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "prevalencia": pi,
    }
