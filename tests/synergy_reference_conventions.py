"""Map synfit synergy metrics to the GPL-3 ``synergy`` package reference JSON.

Both libraries work in raw response space with Hill ``E0`` / ``Emax`` matching
synfit's ``effect_0`` / ``effect_inf``. Standard fixtures use fractional
survival / unaffected response (inhibition with ``effect_0=1``, ``effect_inf=0``);
additional cases exercise offset asymptotes and activation.

Bliss reference (model-based)
    ``synergy.Bliss.E_reference`` is ``E1(d1) * E2(d2)`` on the raw scale. synfit
    ``bliss_independence`` normalises to fractional survival, multiplies, and maps
    back. Compare **normalised** survival ``(y - effect_inf) / (effect_0 - effect_inf)``
    so offset-asymptote fixtures catch omitted scaling.

Bliss reference (observed marginals)
    synfit ``bliss_reference`` uses observed matrix edges (clip, normalise, outer
    product). Independent cases commit perturbed edges and package ``Bliss`` with
    tabulated marginals at the same doses; compare normalised surfaces.

Loewe reference surface (``loewe_reference``)
    JSON field ``loewe_reference`` is ``synergy.Loewe.E_reference`` (SciPy
    ``minimize_scalar`` on squared residual; measured max |package − exact|
    ≈ 1.66e-6 on the standard grid). Field ``loewe_reference_exact`` is mpmath
    (dps=50) ground truth; synfit is checked against that (measured max
    |synfit − exact| ≈ 3.5e-9 on ``loewe_additive``). synfit is also required
    to agree with the package within a measured upper bound (~2e-6).

HSA reference
    ``synergy.HSA`` defaults to ``stronger_orientation=np.minimum`` for inhibition;
    activation fixtures use ``np.maximum``, matching synfit ``hsa_reference`` when
    ``effect_inf > effect_0``. Perturbed edges use tabulated package ``HSA`` at
    committed marginal responses.

Loewe reference (activation)
    On ``activation_synergistic``, package ``Loewe.E_reference`` disagrees with
    synfit (interior max |Δ| ≈ 1.39); Loewe JSON checks skip that case.

Bliss / HSA deviation
    Package: ``synergy = reference - data`` (positive = synergy).
    synfit: ``deviation = (data - reference) / (effect_0 - effect_inf)``
    (negative = synergy). Hence ``synfit_deviation = -package_synergy / scale``.

``slope_mismatch_warning``
    Heuristic UX helper (Hill slope ratio threshold); no external reference.

``zip_scores`` / ZIP helpers
    ``zip_delta`` is validated in ``tests/test_zip.py`` against
    ``tests/data/zip_reference_values.json``; ``zip_reference`` and
    ``zip_fitted_surface`` are checked there against the same fixture when
    ``zip_reference`` / ``zip_fitted`` columns are present.

Loewe CI
    Both use the same four-parameter Hill inverse (closed form). Package ``fit()``
    scores suffice for regression tests.

Zero-concentration edges
    synfit masks combination deviations and Loewe CI as NaN when either
    concentration is zero. Reference JSON tests compare **interior** cells only.
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


def normalized_survival(
    response: np.ndarray,
    effect_0: float,
    effect_inf: float,
) -> np.ndarray:
    scale = effect_0 - effect_inf
    if scale == 0:
        raise ValueError("effect_0 and effect_inf must differ")
    return (np.asarray(response, dtype=float) - effect_inf) / scale


def interior_mask(conc_hor: np.ndarray, conc_ver: np.ndarray) -> np.ndarray:
    ch = np.asarray(conc_hor, dtype=float)
    cv = np.asarray(conc_ver, dtype=float)
    return (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)
