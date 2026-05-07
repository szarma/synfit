"""
ZIP (Zero Interaction Potency) synergy — per-slice Hill fits and delta score.

Sign convention matches Bliss/HSA: negative = synergy, positive = antagonism.
"""
import numpy as np
from scipy.optimize import minimize

from .hill import hill_curve


def _fit_slice(
    response_1d: np.ndarray,
    conc_1d: np.ndarray,
    effect_0: float,
    effect_inf: float,
) -> tuple[float, float]:
    """
    Two-parameter Hill fit (log10(c50), hill) with fixed asymptotes.

    Returns (c50, hill) in linear c50 space, or (nan, nan) if the slice cannot be fit.
    """
    if effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")

    y = np.asarray(response_1d, dtype=float).ravel()
    c = np.asarray(conc_1d, dtype=float).ravel()
    if y.shape != c.shape:
        raise ValueError("response_1d and conc_1d must have the same shape")

    pos = c[c > 0]
    if pos.size == 0:
        return (np.nan, np.nan)
    uniq_pos = np.unique(pos)
    if uniq_pos.size < 2:
        return (np.nan, np.nan)

    scale = abs(effect_0 - effect_inf)
    if np.ptp(y) < max(1e-12, 1e-9 * scale):
        return (np.nan, np.nan)

    log_lo = np.log10(uniq_pos.min())
    log_hi = np.log10(uniq_pos.max())
    if log_lo > log_hi:
        log_lo, log_hi = log_hi, log_lo

    span = log_hi - log_lo
    pad = max(0.5 * span, 0.25)
    bounds = [(log_lo - pad, log_hi + pad), (0.1, 4.0)]
    x0 = np.array([0.5 * (log_lo + log_hi), 1.0])

    def objective(x: np.ndarray) -> float:
        log_c50, hill = float(x[0]), float(x[1])
        c50 = 10.0**log_c50
        pred = hill_curve(c, c50=c50, hill=hill, effect_0=effect_0, effect_inf=effect_inf)
        return float(np.sum((y - pred) ** 2))

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    log_c50_fit, hill_fit = float(res.x[0]), float(res.x[1])
    c50_fit = 10.0**log_c50_fit

    if not np.isfinite(c50_fit) or not np.isfinite(hill_fit) or c50_fit <= 0:
        return (np.nan, np.nan)

    return (c50_fit, hill_fit)


def zip_reference(
    mean_matrix: np.ndarray,
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> np.ndarray:
    """
    ZIP predicted combination surface (pre-subtraction).

    For each row of the observed matrix, fit a Hill curve in horizontal concentrations;
    for each column, in vertical concentrations. The ZIP prediction at (i, j) is the
    mean of the two slice predictions. Failed slice fits fall back to the global (c50,
    hill) for that axis.
    """
    if effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")

    m = np.asarray(mean_matrix, dtype=float)
    n_ver, n_hor = m.shape
    ch = np.asarray(conc_hor, dtype=float).ravel()
    cv = np.asarray(conc_ver, dtype=float).ravel()
    if ch.size != n_hor or cv.size != n_ver:
        raise ValueError("mean_matrix shape does not match concentration vectors")

    row_c50 = np.empty(n_ver)
    row_hill = np.empty(n_ver)
    for i in range(n_ver):
        c50, hill = _fit_slice(m[i, :], ch, effect_0, effect_inf)
        if np.isfinite(c50) and np.isfinite(hill):
            row_c50[i], row_hill[i] = c50, hill
        else:
            row_c50[i], row_hill[i] = c50_hor, hill_hor

    col_c50 = np.empty(n_hor)
    col_hill = np.empty(n_hor)
    for j in range(n_hor):
        c50, hill = _fit_slice(m[:, j], cv, effect_0, effect_inf)
        if np.isfinite(c50) and np.isfinite(hill):
            col_c50[j], col_hill[j] = c50, hill
        else:
            col_c50[j], col_hill[j] = c50_ver, hill_ver

    pred_h = np.empty_like(m)
    for i in range(n_ver):
        pred_h[i, :] = hill_curve(
            ch,
            c50=row_c50[i],
            hill=row_hill[i],
            effect_0=effect_0,
            effect_inf=effect_inf,
        )
    pred_v = np.empty_like(m)
    for j in range(n_hor):
        pred_v[:, j] = hill_curve(
            cv,
            c50=col_c50[j],
            hill=col_hill[j],
            effect_0=effect_0,
            effect_inf=effect_inf,
        )

    return 0.5 * (pred_h + pred_v)


def zip_delta(
    mean_matrix: np.ndarray,
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> np.ndarray:
    """
    ZIP delta-score matrix: (observed - ZIP_predicted) / (effect_0 - effect_inf).

    Cells where either concentration is zero are masked as NaN — those cells are
    single-drug measurements, not combinations, so delta ≈ 0 is tautological.
    """
    zip_pred = zip_reference(
        mean_matrix, conc_hor, conc_ver,
        c50_hor, c50_ver, hill_hor, hill_ver,
        effect_0, effect_inf,
    )
    scale = effect_0 - effect_inf
    delta = (np.asarray(mean_matrix, dtype=float) - zip_pred) / scale
    ch = np.asarray(conc_hor, dtype=float)
    cv = np.asarray(conc_ver, dtype=float)
    edge = (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)
    return np.where(edge, np.nan, delta)
