import numpy as np
from .hill import hill_curve


def _normalise_clip(values: np.ndarray, effect_0: float, effect_inf: float) -> np.ndarray:
    """Map responses to fractional survival in [0, 1] and clip.

    Clipping is required because the Bliss multiplication is a probabilistic
    identity on fractional survivals — values outside [0, 1] (caused by assay
    noise, super-inhibition, or responses above baseline) break the product
    interpretation and can yield out-of-range reference surfaces.
    """
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    norm = (np.asarray(values, dtype=float) - effect_inf) / scale
    return np.clip(norm, 0.0, 1.0)


def bliss_independence(
    conc_hor: np.ndarray,
    conc_ver: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float = 1.0,
    hill_ver: float = 1.0,
    effect_0: float = 1.0,
    effect_inf: float = 0.0,
    asymmetry_hor: float = 1.0,
    asymmetry_ver: float = 1.0,
) -> np.ndarray:
    """
    Predicted combined effect under Bliss independence assumption (model-based).

    Uses fitted Hill-curve responses at ``conc_hor`` / ``conc_ver`` as the
    single-drug marginals. Prefer :func:`bliss_reference` when observed
    single-drug responses are available — it avoids contaminating the
    reference surface with Hill-fit residuals.

    Returns a (len(conc_ver), len(conc_hor)) matrix of expected responses.
    """
    scale = effect_0 - effect_inf
    if scale == 0:
        return np.full((len(conc_ver), len(conc_hor)), effect_0)
    resp_hor = hill_curve(
        conc_hor, c50=c50_hor, hill=hill_hor,
        effect_0=effect_0, effect_inf=effect_inf, asymmetry=asymmetry_hor,
    )
    resp_ver = hill_curve(
        conc_ver, c50=c50_ver, hill=hill_ver,
        effect_0=effect_0, effect_inf=effect_inf, asymmetry=asymmetry_ver,
    )
    norm_hor = _normalise_clip(resp_hor, effect_0, effect_inf)
    norm_ver = _normalise_clip(resp_ver, effect_0, effect_inf)
    combined = np.outer(norm_ver, norm_hor)
    return effect_inf + scale * combined


def bliss_reference(
    resp_horizontal: np.ndarray,
    resp_vertical: np.ndarray,
    effect_0: float,
    effect_inf: float,
) -> np.ndarray:
    """
    Predicted combination response under Bliss independence, from observed
    single-drug marginals.

    ``resp_horizontal``: 1-D array of shape (n_hor,) — horizontal drug alone at each conc
    ``resp_vertical``:   1-D array of shape (n_ver,) — vertical drug alone at each conc

    Uses the observed marginals directly (typically the matrix edges:
    ``mean_matrix[0, :]`` and ``mean_matrix[:, 0]``) so the reference agrees
    with the data at single-drug rows/columns by construction, matching the
    convention used by :func:`hsa_reference`.

    Returns shape (len(resp_vertical), len(resp_horizontal)).
    """
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    norm_hor = _normalise_clip(resp_horizontal, effect_0, effect_inf)
    norm_ver = _normalise_clip(resp_vertical, effect_0, effect_inf)
    combined = np.outer(norm_ver, norm_hor)
    return effect_inf + scale * combined


def bliss_deviation(
    data: np.ndarray,
    bliss: np.ndarray,
    effect_0: float,
    effect_inf: float,
    conc_hor: np.ndarray | None = None,
    conc_ver: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalised deviation of observed data from Bliss independence prediction.

    Negative = synergy (observed response lower than predicted, i.e. more inhibition).
    Positive = antagonism.

    If ``conc_hor`` and ``conc_ver`` are supplied, cells where either concentration
    is zero are masked as NaN — those cells are single-drug measurements, not
    combinations, so the deviation is a tautological zero rather than interaction signal.
    """
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    dev = (np.asarray(data, dtype=float) - effect_inf) / scale - (np.asarray(bliss, dtype=float) - effect_inf) / scale
    if conc_hor is not None and conc_ver is not None:
        ch = np.asarray(conc_hor, dtype=float)
        cv = np.asarray(conc_ver, dtype=float)
        edge = (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)
        dev = np.where(edge, np.nan, dev)
    return dev


def hsa_reference(
    resp_horizontal: np.ndarray,
    resp_vertical: np.ndarray,
    *,
    effect_0: float | None = None,
    effect_inf: float | None = None,
) -> np.ndarray:
    """
    Predicted combination response under Highest Single Agent.

    hsa[i, j] is the response of whichever single agent has the stronger effect
    at the matched concentrations, i.e. the response closer to ``effect_inf``:
    the lower one for inhibition (``effect_inf < effect_0``), the higher one for
    activation (``effect_inf > effect_0``). Without asymptotes the inhibition
    convention (the lower response) is assumed. Both drugs must act in the same
    direction; HSA is undefined for an inhibitor combined with an activator.
    Returns shape (len(resp_vertical), len(resp_horizontal)).
    """
    rh = np.asarray(resp_horizontal, dtype=float)
    rv = np.asarray(resp_vertical, dtype=float)
    if (effect_0 is None) != (effect_inf is None):
        raise ValueError("pass both effect_0 and effect_inf, or neither")
    if effect_0 is not None and effect_0 == effect_inf:
        raise ValueError("effect_0 and effect_inf must differ")
    stronger = np.maximum if (effect_0 is not None and effect_inf > effect_0) else np.minimum
    return stronger(rv[:, None], rh[None, :])


def hsa_deviation(
    mean_matrix: np.ndarray,
    resp_horizontal: np.ndarray,
    resp_vertical: np.ndarray,
    effect_0: float,
    effect_inf: float,
    conc_hor: np.ndarray | None = None,
    conc_ver: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalised deviation of observed data from HSA (Highest Single Agent) prediction.

    Same sign convention as bliss_deviation: negative = synergy, positive = antagonism.

    resp_horizontal: 1-D array of shape (n_hor,) — horizontal drug alone at each conc
    resp_vertical:   1-D array of shape (n_ver,) — vertical drug alone at each conc

    If ``conc_hor`` and ``conc_ver`` are supplied, cells where either concentration
    is zero are masked as NaN — those cells are single-drug measurements, not
    combinations, so the deviation is a tautological zero rather than interaction signal.
    """
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    hsa = hsa_reference(
        resp_horizontal, resp_vertical, effect_0=effect_0, effect_inf=effect_inf
    )
    dev = (np.asarray(mean_matrix, dtype=float) - effect_inf) / scale - (hsa - effect_inf) / scale
    if conc_hor is not None and conc_ver is not None:
        ch = np.asarray(conc_hor, dtype=float)
        cv = np.asarray(conc_ver, dtype=float)
        edge = (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)
        dev = np.where(edge, np.nan, dev)
    return dev
