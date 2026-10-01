"""Map synfit synergy metrics to the GPL-3 ``synergy`` package reference JSON.

Both libraries work in raw response (fraction-affected) space with Hill
``E0`` / ``Emax`` matching synfit's ``effect_0`` / ``effect_inf``. For the
fixtures here that is inhibition with ``effect_0=1``, ``effect_inf=0``.

Bliss reference
    ``synergy.Bliss.E_reference`` is ``E1(d1) * E2(d2)`` on that scale, which
    matches synfit's ``bliss_independence`` (normalise to fractional survival,
    multiply, map back — identical when asymptotes are 0 and 1).

HSA reference
    ``synergy.HSA`` defaults to ``stronger_orientation=np.minimum``, i.e. the
    stronger inhibitory single-agent response. That matches synfit
    ``hsa_reference`` when ``effect_inf < effect_0``. Activation assays
    (``effect_inf > effect_0``) are **not** covered here: synfit switches to
    ``np.maximum`` but the reference package would need
    ``HSA(stronger_orientation=np.maximum)``.

Bliss / HSA deviation
    Package: ``synergy = reference - data`` (positive = synergy).
    synfit: ``deviation = (data - reference) / (effect_0 - effect_inf)``
    (negative = synergy). Hence ``synfit_deviation = -package_synergy / scale``.

Loewe CI
    Both use ``d1/E_inv(E) + d2/E_inv(E)`` with the same four-parameter Hill
    inverse. Sign convention already matches (CI < 1 synergy).

Zero-concentration edges
    synfit masks combination deviations and Loewe CI as NaN when either
    concentration is zero. The package sets Bliss/HSA synergy to 0 and Loewe
    CI to 1 on those cells. Reference JSON tests compare **interior** cells only.

Loewe undefined cells
    When the observed response is outside the open asymptote band, both
    implementations yield NaN. synfit's existing tests cover that edge case;
    fixtures here stay inside the band on interior cells.
"""
from __future__ import annotations

import numpy as np


def package_synergy_to_synfit_deviation(
    package_synergy: np.ndarray,
    effect_0: float,
    effect_inf: float,
) -> np.ndarray:
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    return -np.asarray(package_synergy, dtype=float) / scale


def interior_mask(conc_hor: np.ndarray, conc_ver: np.ndarray) -> np.ndarray:
    ch = np.asarray(conc_hor, dtype=float)
    cv = np.asarray(conc_ver, dtype=float)
    return (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)
