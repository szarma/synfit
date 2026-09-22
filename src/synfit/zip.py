"""
ZIP (Zero Interaction Potency) synergy after Yadav et al. 2015.

Yadav, Wennerberg, Aittokallio, Tang (2015). *Comput Struct Biotechnol J* 13:504–513.
doi:10.1016/j.csbj.2015.09.001

Fraction affected (unit- and direction-neutral):

    f(y) = (effect_0 - y) / (effect_0 - effect_inf)
    y(f) = effect_0 - f * (effect_0 - effect_inf)

Zero-interaction expectation (Bliss on fraction affected):

    f_zip[i, j] = f_h(c_h[j]) + f_v(c_v[i]) - f_h(c_h[j]) * f_v(c_v[i])

Per-slice fits in normalised space with fixed baseline ``f0`` (the other drug's
single-agent fraction affected) and free ``log10 m``, Hill slope λ, and ``Emax``.
The fitted combination surface averages row and column slice predictions;
δ compares that surface to ``f_zip``. synfit sign convention: **negative = synergy**
(``delta = -(f_c - f_zip)`` in fraction-affected terms).
"""
import warnings

import numpy as np
from scipy.optimize import minimize

from .hill import hill_curve


def _response_to_fraction_affected(
    y: np.ndarray, effect_0: float, effect_inf: float
) -> np.ndarray:
    scale = effect_0 - effect_inf
    return (effect_0 - np.asarray(y, dtype=float)) / scale


def _fraction_affected_to_response(
    f: np.ndarray, effect_0: float, effect_inf: float
) -> np.ndarray:
    scale = effect_0 - effect_inf
    return effect_0 - np.asarray(f, dtype=float) * scale


def _single_agent_fraction_affected(
    conc: np.ndarray,
    c50: float,
    hill: float,
    effect_0: float,
    effect_inf: float,
    asymmetry: float | None,
) -> np.ndarray:
    y = hill_curve(
        conc,
        c50=c50,
        hill=hill,
        effect_0=effect_0,
        effect_inf=effect_inf,
        asymmetry=asymmetry,
    )
    return _response_to_fraction_affected(y, effect_0, effect_inf)


def _slice_model(conc: np.ndarray, f0: float, log10_m: float, hill: float, emax: float) -> np.ndarray:
    """g(x) = (f0 + Emax * (x/m)^λ) / (1 + (x/m)^λ)."""
    c = np.asarray(conc, dtype=float)
    m = 10.0 ** log10_m
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(c > 0, (c / m) ** hill, 0.0)
    return (f0 + emax * ratio) / (1.0 + ratio)


def _log10_m_bounds(pos_conc: np.ndarray) -> tuple[float, float]:
    uniq_pos = np.unique(pos_conc[pos_conc > 0])
    if uniq_pos.size < 1:
        return (0.0, 0.0)
    log_lo = np.log10(uniq_pos.min())
    log_hi = np.log10(uniq_pos.max())
    if log_lo > log_hi:
        log_lo, log_hi = log_hi, log_lo
    span = log_hi - log_lo
    pad = max(0.5 * span, 0.25)
    return (log_lo - pad, log_hi + pad)


def _fit_slice_normalized(
    fraction_1d: np.ndarray,
    conc_1d: np.ndarray,
    f0: float,
    log10_m0: float,
    hill0: float,
) -> tuple[float, float, float] | None:
    """
    Three-parameter slice fit (log10 m, λ, Emax) with fixed f0.

    Returns (log10_m, hill, emax) or None if the slice cannot be fit.
    """
    f = np.asarray(fraction_1d, dtype=float).ravel()
    c = np.asarray(conc_1d, dtype=float).ravel()
    if f.shape != c.shape:
        raise ValueError("fraction_1d and conc_1d must have the same shape")

    pos_mask = (c > 0) & np.isfinite(f)
    if np.count_nonzero(pos_mask) < 3:
        return None

    c_pos = c[pos_mask]
    f_pos = f[pos_mask]

    log_lo, log_hi = _log10_m_bounds(c_pos)
    if not np.isfinite(log_lo) or not np.isfinite(log_hi):
        return None

    bounds = [(log_lo, log_hi), (0.1, 10.0), (0.0, 1.5)]
    x0 = np.array([log10_m0, hill0, 1.0], dtype=float)
    x0[0] = float(np.clip(x0[0], bounds[0][0], bounds[0][1]))
    x0[1] = float(np.clip(x0[1], bounds[1][0], bounds[1][1]))

    def objective(x: np.ndarray) -> float:
        pred = _slice_model(c_pos, f0, float(x[0]), float(x[1]), float(x[2]))
        return float(np.sum((f_pos - pred) ** 2))

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return None
    log10_m_fit, hill_fit, emax_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])
    if not all(np.isfinite(v) for v in (log10_m_fit, hill_fit, emax_fit)):
        return None
    return (log10_m_fit, hill_fit, emax_fit)


def _validate_matrix_shape(
    mean_matrix: np.ndarray, conc_hor: np.ndarray, conc_ver: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray]:
    m = np.asarray(mean_matrix, dtype=float)
    n_ver, n_hor = m.shape
    ch = np.asarray(conc_hor, dtype=float).ravel()
    cv = np.asarray(conc_ver, dtype=float).ravel()
    if ch.size != n_hor or cv.size != n_ver:
        raise ValueError("mean_matrix shape does not match concentration vectors")
    return n_ver, n_hor, ch, cv


def _fraction_affected_zip(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float | None,
    asymmetry_ver: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_h = _single_agent_fraction_affected(
        ch, c50_hor, hill_hor, effect_0, effect_inf, asymmetry_hor
    )
    f_v = _single_agent_fraction_affected(
        cv, c50_ver, hill_ver, effect_0, effect_inf, asymmetry_ver
    )
    f_zip = f_v[:, np.newaxis] + f_h[np.newaxis, :] - f_v[:, np.newaxis] * f_h[np.newaxis, :]
    return f_h, f_v, f_zip


def _zip_slice_fits(
    mean_matrix: np.ndarray,
    ch: np.ndarray,
    cv: np.ndarray,
    f_h: np.ndarray,
    f_v: np.ndarray,
    f_zip: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
) -> tuple[np.ndarray, int]:
    m = np.asarray(mean_matrix, dtype=float)
    n_ver, n_hor = m.shape
    f_obs = _response_to_fraction_affected(m, effect_0, effect_inf)

    log10_m_hor0 = np.log10(c50_hor) if c50_hor > 0 else 0.0
    log10_m_ver0 = np.log10(c50_ver) if c50_ver > 0 else 0.0

    n_failed = 0
    f_row = np.empty_like(f_zip)
    for i in range(n_ver):
        f0 = float(f_v[i])
        fit = _fit_slice_normalized(f_obs[i, :], ch, f0, log10_m_hor0, hill_hor)
        if fit is None:
            n_failed += 1
            f_row[i, :] = f_zip[i, :]
        else:
            log10_m, hill, emax = fit
            f_row[i, :] = _slice_model(ch, f0, log10_m, hill, emax)

    f_col = np.empty_like(f_zip)
    for j in range(n_hor):
        f0 = float(f_h[j])
        fit = _fit_slice_normalized(f_obs[:, j], cv, f0, log10_m_ver0, hill_ver)
        if fit is None:
            n_failed += 1
            f_col[:, j] = f_zip[:, j]
        else:
            log10_m, hill, emax = fit
            f_col[:, j] = _slice_model(cv, f0, log10_m, hill, emax)

    f_c = 0.5 * (f_row + f_col)
    return f_c, n_failed


def _warn_slice_failures(n_failed: int) -> None:
    if n_failed > 0:
        warnings.warn(
            f"ZIP: {n_failed} slice fit(s) failed; using zero-interaction expectation "
            "for those slice directions.",
            UserWarning,
            stacklevel=3,
        )


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
    *,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
) -> np.ndarray:
    """
    ZIP zero-interaction expectation in response units.

    ``mean_matrix`` is used only to validate shape against the concentration
    vectors. The reference is the Bliss combination of the supplied single-drug
    Hill parameters in fraction-affected space, mapped back to responses.
    """
    if effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")
    _, _, ch, cv = _validate_matrix_shape(mean_matrix, conc_hor, conc_ver)
    _, _, f_zip = _fraction_affected_zip(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver,
        effect_0, effect_inf, asymmetry_hor, asymmetry_ver,
    )
    return _fraction_affected_to_response(f_zip, effect_0, effect_inf)


def _zip_fitted_fraction_affected(
    mean_matrix: np.ndarray,
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")
    _, _, ch, cv = _validate_matrix_shape(mean_matrix, conc_hor, conc_ver)
    f_h, f_v, f_zip = _fraction_affected_zip(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver,
        effect_0, effect_inf, asymmetry_hor, asymmetry_ver,
    )
    f_c, n_failed = _zip_slice_fits(
        mean_matrix, ch, cv, f_h, f_v, f_zip,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
    )
    return f_zip, f_c, n_failed


def zip_fitted_surface(
    mean_matrix: np.ndarray,
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    *,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
) -> np.ndarray:
    """ZIP fitted combination surface ``y(f_c)`` in response units."""
    _, f_c, n_failed = _zip_fitted_fraction_affected(
        mean_matrix, conc_hor, conc_ver,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor, asymmetry_ver,
    )
    _warn_slice_failures(n_failed)
    return _fraction_affected_to_response(f_c, effect_0, effect_inf)


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
    *,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
) -> np.ndarray:
    """
    ZIP δ-score matrix: ``-(f_c - f_zip)`` (negative = synergy).

    Cells where either concentration is zero are NaN (single-drug edges).
    """
    f_zip, f_c, n_failed = _zip_fitted_fraction_affected(
        mean_matrix, conc_hor, conc_ver,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor, asymmetry_ver,
    )
    _warn_slice_failures(n_failed)
    delta = -(f_c - f_zip)
    ch = np.asarray(conc_hor, dtype=float)
    cv = np.asarray(conc_ver, dtype=float)
    edge = (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)
    return np.where(edge, np.nan, delta)
