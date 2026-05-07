"""
Loewe Additivity and Combination Index (CI) for drug combination analysis.

The Combination Index at each concentration pair (a, b) with observed effect E is:

    CI = a / A(E) + b / B(E)

where A(E) and B(E) are the single-agent doses that produce the same effect E,
obtained by analytically inverting the Hill equation:

    c = c50 * ((effect_0 - effect_inf) / (E - effect_inf) - 1)^(1/hill)

This inversion is only valid when E lies strictly between the two asymptotes
(`effect_0` and `effect_inf`, which map to top/bottom for inhibition). Cells where
the observed response falls outside that band are set to NaN — Loewe CI is undefined
beyond the single-agent effect range.

Reference
---------
Chou, T.-C. & Talalay, P. (1984)
"Quantitative analysis of dose-effect relationships: the combined effects of
multiple drugs or enzyme inhibitors"
Advances in Enzyme Regulation, 22, 27–55.
https://doi.org/10.1016/0065-2571(84)90007-4
"""
from typing import Optional, TypedDict

import numpy as np
from scipy.optimize import brentq

from .hill import hill_curve


class SlopeMismatchWarning(TypedDict):
    severity: str  # "yellow" | "red"
    hill_hor: float
    hill_ver: float
    ratio: float
    message: str


# Slope-mismatch thresholds. Loewe's sham-combination principle is rigorously
# defined only when both drugs share the same Hill slope; unequal slopes can
# produce spurious CI values purely from the shape mismatch (Berenbaum 1989;
# Foucquier & Guedj 2015). We compare slopes multiplicatively (ratio) rather
# than by absolute difference because Hill coefficients are multiplicative
# steepnesses — 1.0 vs 1.1 is noise, 0.3 vs 0.4 is a 33% shape change.
_SLOPE_YELLOW_RATIO = 1.5
_SLOPE_RED_RATIO = 3.0
# When both slopes fall inside this band, Loewe is well-behaved regardless of
# the exact ratio (Hill=1 is the special case where Loewe and Bliss coincide
# most closely), so we suppress the warning.
_SLOPE_NEUTRAL_BAND = (0.9, 1.1)


def slope_mismatch_warning(
    hill_hor: Optional[float],
    hill_ver: Optional[float],
) -> Optional[SlopeMismatchWarning]:
    """
    Return a structured warning if the two Hill slopes disagree enough to
    compromise the Loewe CI interpretation, else ``None``.

    Thresholds: yellow at ratio >= 1.5, red at ratio >= 3.0. Suppressed when
    both slopes lie inside ``_SLOPE_NEUTRAL_BAND`` (Loewe behaves well when
    both slopes are near 1 regardless of the exact ratio).

    ``None`` / non-numeric inputs return ``None`` rather than raising, so API
    callers can pass fit-params dicts without pre-validating every field.
    """
    if hill_hor is None or hill_ver is None:
        return None
    try:
        hill_hor = float(hill_hor)
        hill_ver = float(hill_ver)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(hill_hor) and np.isfinite(hill_ver)):
        return None
    if hill_hor <= 0 or hill_ver <= 0:
        return None
    lo, hi = _SLOPE_NEUTRAL_BAND
    if lo <= hill_hor <= hi and lo <= hill_ver <= hi:
        return None
    ratio = max(hill_hor, hill_ver) / min(hill_hor, hill_ver)
    if ratio < _SLOPE_YELLOW_RATIO:
        return None
    severity = "red" if ratio >= _SLOPE_RED_RATIO else "yellow"
    message = (
        f"Hill slopes disagree (hill_hor = {hill_hor:.2g}, hill_ver = {hill_ver:.2g}, "
        f"ratio {ratio:.2g}x). Loewe CI assumes matched slopes; reported values may "
        "partly reflect the slope mismatch rather than real interaction. You can pin "
        "each Hill slope individually in the matrix analysis setup."
    )
    return {
        "severity": severity,
        "hill_hor": float(hill_hor),
        "hill_ver": float(hill_ver),
        "ratio": float(ratio),
        "message": message,
    }


def _invert_hill(
    y: np.ndarray,
    c50: float,
    hill: float,
    effect_0: float,
    effect_inf: float,
    asymmetry: float = 1.0,
) -> np.ndarray:
    """
    Invert the Hill equation analytically: given response y, return concentration c.

    Supports the 5-parameter (asymmetric) Hill form used in ``hill_curve``:

        y = effect_inf + (effect_0 - effect_inf) / (1 + (c/c50)**hill) ** asymmetry

    Solving for c:

        c = c50 * (((effect_0 - effect_inf) / (y - effect_inf)) ** (1/asymmetry) - 1) ** (1/hill)

    With ``asymmetry = 1`` this collapses to the 4-parameter inverse. Returns NaN
    for elements where y is outside the open interval between the asymptotes or
    where the inversion would require a non-positive base under a fractional power.
    """
    if asymmetry is None:
        asymmetry = 1.0
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    scale = effect_0 - effect_inf
    if scale == 0 or asymmetry <= 0:
        return out

    # Valid mask: strictly inside the asymptote band (direction-agnostic).
    valid = (y > min(effect_0, effect_inf)) & (y < max(effect_0, effect_inf))
    if not valid.any():
        return out

    # scale/u is strictly > 1 for y inside the band (for both inhibition and
    # activation directions, since scale and u share sign), so raising to
    # 1/asymmetry stays > 1 and the subtraction below is positive.
    u = y[valid] - effect_inf
    inner = (scale / u) ** (1.0 / asymmetry) - 1.0
    # Numerical guard: tiny negatives near the top asymptote -> NaN instead of
    # raising on the fractional power. Build a flat result aligned with y[valid]
    # so the boolean assignment works for arbitrary input shapes.
    result = np.full_like(u, np.nan)
    positive = inner > 0
    result[positive] = c50 * inner[positive] ** (1.0 / hill)
    out[valid] = result
    return out


def loewe_ci(
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    mean_matrix: np.ndarray,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float = 1.0,
    asymmetry_ver: float = 1.0,
) -> np.ndarray:
    """
    Combination Index matrix under Loewe additivity.

    For each cell (i, j) with observed effect E = mean_matrix[i, j]:

        CI[i, j] = conc_ver[i] / B(E) + conc_hor[j] / A(E)

    where A(E) and B(E) are the single-agent doses producing effect E.

    Parameters
    ----------
    conc_hor:    1-D array of horizontal drug concentrations, shape (n_hor,)
    conc_ver:    1-D array of vertical drug concentrations, shape (n_ver,)
    c50_hor:     C50 of the horizontal drug
    c50_ver:     C50 of the vertical drug
    hill_hor:    Hill coefficient of the horizontal drug
    hill_ver:    Hill coefficient of the vertical drug
    mean_matrix: Observed response matrix, shape (n_ver, n_hor)
    effect_0:    Shared low-concentration asymptote
    effect_inf:  Shared high-concentration asymptote

    Returns
    -------
    np.ndarray of shape (n_ver, n_hor).
    NaN where the observed response is outside the single-agent range (effect_inf, effect_0).
    """
    conc_hor = np.asarray(conc_hor, dtype=float)
    conc_ver = np.asarray(conc_ver, dtype=float)
    mean_matrix = np.asarray(mean_matrix, dtype=float)

    a_e = _invert_hill(mean_matrix, c50_hor, hill_hor, effect_0, effect_inf, asymmetry_hor)
    b_e = _invert_hill(mean_matrix, c50_ver, hill_ver, effect_0, effect_inf, asymmetry_ver)

    with np.errstate(divide="ignore", invalid="ignore"):
        ci = conc_hor[np.newaxis, :] / a_e + conc_ver[:, np.newaxis] / b_e

    invalid = np.isnan(a_e) | np.isnan(b_e) | (a_e == 0) | (b_e == 0)
    # Cells where either concentration is zero are single-drug measurements, not
    # combinations — CI = 1 there is a tautology, not interaction signal.
    edge = (conc_hor[np.newaxis, :] == 0) | (conc_ver[:, np.newaxis] == 0)
    return np.where(invalid | edge, np.nan, ci)


def loewe_reference(
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float = 1.0,
    asymmetry_ver: float = 1.0,
) -> np.ndarray:
    """
    Predicted response surface under Loewe additivity (CI = 1 null).

    At each cell (i, j) with concentrations (c_h, c_v), find the effect E* that
    satisfies c_h / A(E*) + c_v / B(E*) = 1, where A(E) and B(E) are the inverse
    single-agent Hill curves. Uses Brent's method on the open interval between
    the two asymptotes. Boundary cells (either concentration = 0) fall back to
    the corresponding single-drug Hill prediction.

    Returns shape (len(conc_ver), len(conc_hor)). NaN where the root find fails.
    """
    ch = np.asarray(conc_hor, dtype=float).ravel()
    cv = np.asarray(conc_ver, dtype=float).ravel()
    n_hor = ch.size
    n_ver = cv.size
    ref = np.full((n_ver, n_hor), np.nan, dtype=float)

    e_lo, e_hi = min(effect_0, effect_inf), max(effect_0, effect_inf)
    scale = e_hi - e_lo
    if scale == 0:
        return ref
    eps = 1e-9 * scale
    lo, hi = e_lo + eps, e_hi - eps

    def a_of(e: float) -> float:
        return _invert_hill(np.array([e]), c50_hor, hill_hor, effect_0, effect_inf, asymmetry_hor)[0]

    def b_of(e: float) -> float:
        return _invert_hill(np.array([e]), c50_ver, hill_ver, effect_0, effect_inf, asymmetry_ver)[0]

    for i in range(n_ver):
        cv_i = cv[i]
        for j in range(n_hor):
            ch_j = ch[j]
            if ch_j == 0 and cv_i == 0:
                ref[i, j] = effect_0
                continue
            if ch_j == 0:
                ref[i, j] = float(hill_curve(
                    np.array([cv_i]), c50=c50_ver, hill=hill_ver,
                    effect_0=effect_0, effect_inf=effect_inf, asymmetry=asymmetry_ver,
                )[0])
                continue
            if cv_i == 0:
                ref[i, j] = float(hill_curve(
                    np.array([ch_j]), c50=c50_hor, hill=hill_hor,
                    effect_0=effect_0, effect_inf=effect_inf, asymmetry=asymmetry_hor,
                )[0])
                continue

            def residual(e: float) -> float:
                a = a_of(e)
                b = b_of(e)
                if not (np.isfinite(a) and np.isfinite(b)) or a == 0 or b == 0:
                    return np.nan
                return ch_j / a + cv_i / b - 1.0

            try:
                r_lo = residual(lo)
                r_hi = residual(hi)
                if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo * r_hi > 0:
                    continue
                ref[i, j] = brentq(residual, lo, hi, xtol=1e-8, rtol=1e-8, maxiter=64)
            except (ValueError, RuntimeError):
                continue

    return ref
