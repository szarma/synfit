import numpy as np


def hill_curve(conc, c50, hill=1.0, effect_0=1.0, effect_inf=0.0, asymmetry=1.0):
    """
    Hill equation (dose-response curve).

    Parameters:
    - c50: concentration at 50% effect (linear space, not log)
    - hill: Hill coefficient (steepness)
    - effect_0: response at zero drug concentration (low-concentration asymptote)
    - effect_inf: response at saturating drug concentration (high-concentration asymptote)
    - asymmetry: asymmetry factor S. When asymmetry=1, reduces to 4-parameter Hill. None is treated as 1.0.

    Works for both inhibition and activation assays by swapping effect_0 and effect_inf.
    """
    if asymmetry is None:
        asymmetry = 1.0
    return effect_inf + (effect_0 - effect_inf) / (1 + (conc / c50) ** hill) ** asymmetry


def log_wall(x, boundaries, steep=1):
    """Soft linear ramp penalty outside parameter bounds, for use as a prior."""
    xm = np.mean(boundaries)
    dxh = max(boundaries) - xm
    x = np.abs(np.asarray(x, dtype=float) - xm) / dxh
    out = np.zeros_like(x)
    fltr = x > 1
    out[fltr] = steep * (x[fltr] - 1)
    return out


def calculate_concentration_series(
    initial_conc: float,
    fold_dilutions: float,
    length: int,
    has_zero: bool = True,
    ascending: bool = True,
) -> np.ndarray:
    """Calculate a serial dilution concentration series."""
    if initial_conc <= 0:
        raise ValueError("initial_conc must be positive")
    if fold_dilutions <= 1:
        raise ValueError("fold_dilutions must be greater than 1")
    conc = float(fold_dilutions) ** -np.arange(length) * initial_conc
    if has_zero:
        conc[-1] = 0.0
    if ascending:
        conc = conc[::-1]
    return conc
