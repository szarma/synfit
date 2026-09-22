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
single-agent fraction affected), fixed asymmetry ``s`` (the slice drug's 5p
shape, 1 for symmetric 4p), and free ``log10 m``, Hill slope λ, and ``Emax``.
Row slices use the horizontal drug's asymmetry; column slices use the vertical
drug's. The fixed asymmetry is a synfit extension: the published method
(Yadav et al. 2015, SynergyFinder) uses symmetric 4-parameter slices, which
this reduces to exactly when s = 1. The fitted combination surface averages successful row/column slice
predictions (a single direction when the other failed; NaN when both failed).
δ compares that surface to ``f_zip``. synfit sign convention: **negative = synergy**
(``delta = -(f_c - f_zip)`` in fraction-affected terms).
"""
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .hill import hill_curve


@dataclass(frozen=True)
class ZipResult:
    reference: np.ndarray  # zero-interaction expectation, response units
    fitted: np.ndarray  # fitted combination surface (NaN where no slice succeeded)
    delta: np.ndarray  # ZIP δ; NaN on zero-dose edges and unscored interior
    failed_rows: tuple[int, ...]  # vertical-dose indices whose horizontal slice failed
    failed_cols: tuple[int, ...]  # horizontal-dose indices whose vertical slice failed
    n_unscored: int  # interior cells with NaN δ because both slice directions failed


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


def _slice_asymmetry(asymmetry: float | None) -> float:
    return 1.0 if asymmetry is None else float(asymmetry)


def _slice_model(
    conc: np.ndarray,
    f0: float,
    log10_m: float,
    hill: float,
    emax: float,
    asymmetry: float | None = None,
) -> np.ndarray:
    """g(x) = f0 + (Emax - f0) * (1 - (1 + (x/m)^λ)^(-s)); s=1 gives the 4p form."""
    c = np.asarray(conc, dtype=float)
    m = 10.0 ** log10_m
    s = _slice_asymmetry(asymmetry)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(c > 0, (c / m) ** hill, 0.0)
        frac = 1.0 - (1.0 + ratio) ** (-s)
    return f0 + (emax - f0) * frac


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
    asymmetry: float | None = None,
) -> tuple[float, float, float] | None:
    """
    Three-parameter slice fit (log10 m, λ, Emax) with fixed f0 and fixed s.

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

    # Emax is capped at full effect (fraction affected 1), as in SynergyFinder
    # (≤ 100 % inhibition); an uncapped slice overshoots on synergistic data.
    bounds = [(log_lo, log_hi), (0.1, 10.0), (-0.5, 1.0)]
    x0 = np.array([log10_m0, hill0, 1.0], dtype=float)
    x0[0] = float(np.clip(x0[0], bounds[0][0], bounds[0][1]))
    x0[1] = float(np.clip(x0[1], bounds[1][0], bounds[1][1]))

    def objective(x: np.ndarray) -> float:
        pred = _slice_model(
            c_pos, f0, float(x[0]), float(x[1]), float(x[2]), asymmetry=asymmetry
        )
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


def _interior_mask(ch: np.ndarray, cv: np.ndarray) -> np.ndarray:
    return (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)


def _zip_slice_fits(
    mean_matrix: np.ndarray,
    ch: np.ndarray,
    cv: np.ndarray,
    f_h: np.ndarray,
    f_v: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float | None,
    asymmetry_ver: float | None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], int]:
    m = np.asarray(mean_matrix, dtype=float)
    n_ver, n_hor = m.shape
    f_obs = _response_to_fraction_affected(m, effect_0, effect_inf)

    log10_m_hor0 = np.log10(c50_hor) if c50_hor > 0 else 0.0
    log10_m_ver0 = np.log10(c50_ver) if c50_ver > 0 else 0.0

    failed_rows: list[int] = []
    failed_cols: list[int] = []
    row_ok = np.zeros(n_ver, dtype=bool)
    col_ok = np.zeros(n_hor, dtype=bool)
    row_f = np.full((n_ver, n_hor), np.nan)
    col_f = np.full((n_ver, n_hor), np.nan)

    for i in range(n_ver):
        if cv[i] <= 0:
            continue
        f0 = float(f_v[i])
        fit = _fit_slice_normalized(
            f_obs[i, :], ch, f0, log10_m_hor0, hill_hor, asymmetry=asymmetry_hor
        )
        if fit is None:
            failed_rows.append(i)
        else:
            log10_m, hill, emax = fit
            row_ok[i] = True
            row_f[i, :] = _slice_model(
                ch, f0, log10_m, hill, emax, asymmetry=asymmetry_hor
            )

    for j in range(n_hor):
        if ch[j] <= 0:
            continue
        f0 = float(f_h[j])
        fit = _fit_slice_normalized(
            f_obs[:, j], cv, f0, log10_m_ver0, hill_ver, asymmetry=asymmetry_ver
        )
        if fit is None:
            failed_cols.append(j)
        else:
            log10_m, hill, emax = fit
            col_ok[j] = True
            col_f[:, j] = _slice_model(
                cv, f0, log10_m, hill, emax, asymmetry=asymmetry_ver
            )

    has_row = row_ok[:, np.newaxis]
    has_col = col_ok[np.newaxis, :]
    both = has_row & has_col
    one_row = has_row & ~has_col
    one_col = ~has_row & has_col

    f_c = np.full((n_ver, n_hor), np.nan)
    f_c = np.where(both, 0.5 * (row_f + col_f), f_c)
    f_c = np.where(one_row, row_f, f_c)
    f_c = np.where(one_col, col_f, f_c)

    interior = _interior_mask(ch, cv)
    n_unscored = int(np.sum(interior & ~has_row & ~has_col))

    return f_c, tuple(failed_rows), tuple(failed_cols), n_unscored


def _warn_slice_failures(
    failed_rows: tuple[int, ...],
    failed_cols: tuple[int, ...],
    n_unscored: int,
) -> None:
    if not failed_rows and not failed_cols:
        return
    parts: list[str] = []
    if failed_rows:
        parts.append(f"rows {list(failed_rows)}")
    if failed_cols:
        parts.append(f"columns {list(failed_cols)}")
    warnings.warn(
        f"ZIP: slice fit(s) failed for {' and '.join(parts)}; "
        f"{n_unscored} interior cell(s) unscored.",
        UserWarning,
        stacklevel=3,
    )


def zip_scores(
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
) -> ZipResult:
    """Full ZIP result: reference, fitted surface, δ, and slice-failure metadata."""
    if effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")
    _, _, ch, cv = _validate_matrix_shape(mean_matrix, conc_hor, conc_ver)
    f_h, f_v, f_zip = _fraction_affected_zip(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver,
        effect_0, effect_inf, asymmetry_hor, asymmetry_ver,
    )
    f_c, failed_rows, failed_cols, n_unscored = _zip_slice_fits(
        mean_matrix, ch, cv, f_h, f_v,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor, asymmetry_ver,
    )
    _warn_slice_failures(failed_rows, failed_cols, n_unscored)

    delta = -(f_c - f_zip)
    edge = (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)
    delta = np.where(edge, np.nan, delta)

    return ZipResult(
        reference=_fraction_affected_to_response(f_zip, effect_0, effect_inf),
        fitted=_fraction_affected_to_response(f_c, effect_0, effect_inf),
        delta=delta,
        failed_rows=failed_rows,
        failed_cols=failed_cols,
        n_unscored=n_unscored,
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
    return zip_scores(
        mean_matrix, conc_hor, conc_ver,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor=asymmetry_hor, asymmetry_ver=asymmetry_ver,
    ).fitted


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
    Interior cells are NaN when both the row and column slice fits failed.
    """
    return zip_scores(
        mean_matrix, conc_hor, conc_ver,
        c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf,
        asymmetry_hor=asymmetry_hor, asymmetry_ver=asymmetry_ver,
    ).delta
