"""Regression tests for scale-relative covariance estimation.

A fit whose response values are large-magnitude (e.g. ELISA / RFU data with
asymptotes of order 1e4) used to return ``param_cov is None`` — the Hessian was
probed with a fixed *absolute* finite-difference step (``_HESS_EPS``), which is
~1e-9 *relative* to a parameter of order 1e4, deep in floating-point
cancellation noise. The estimated curvature came out with the wrong sign, the
inverse Hessian had a negative diagonal, and the whole covariance was discarded
— leaving the UI with "Confidence interval unavailable for this fit (no
covariance matrix)". Probing in scale-relative coordinates fixes it.
"""

import numpy as np
import pandas as pd

from synfit.single import SingleDrugFit
from synfit.data import FitConfig


def _large_magnitude_activation_data(seed: int = 2) -> pd.DataFrame:
    """ELISA/RFU-scale activation curve (asymptote ~1e4) that does not fully
    saturate — top concentration is only ~2x the EC50. Mirrors the bundled
    ``single_drug_activation_calibration`` demo dataset.
    """
    rng = np.random.default_rng(seed)
    c50, hill, effect_0, effect_inf = 20.0, 1.1, 87.0, 12345.0
    conc = np.array([0.0, 0.009765625, 0.0390625, 0.15625, 0.625, 2.5, 10.0, 40.0])
    frac = (conc ** hill) / (conc ** hill + c50 ** hill)  # 0 at conc == 0
    y_clean = effect_0 + (effect_inf - effect_0) * frac
    frames = [
        pd.DataFrame(
            {
                "concentration": conc,
                "y": y_clean * np.exp(rng.normal(0, 0.3, size=conc.shape)),
                "replicate": rep,
            }
        )
        for rep in ("A", "B", "C")
    ]
    return pd.concat(frames, ignore_index=True)


def test_covariance_recovered_for_large_magnitude_asymptote():
    """Large-magnitude fit yields a finite, positive-on-the-diagonal covariance."""
    df = _large_magnitude_activation_data()
    fit = SingleDrugFit(df, FitConfig(direction="activation", noise="lognormal"))
    result = fit.fit()

    assert result.effect_inf > 1_000, "expected a genuinely large-magnitude fit"
    assert result.param_cov is not None, "covariance must be recovered, not None"
    assert np.all(np.diag(result.param_cov) > 0)

    # effect_inf is the weakly-constrained, large-magnitude parameter; it must
    # still receive a real, finite, positive variance.
    einf = result.param_names.index("effect_inf")
    einf_var = result.param_cov[einf, einf]
    assert np.isfinite(einf_var) and einf_var > 0


def test_covariance_back_transform_units_are_consistent():
    """Recovered variances are in original (un-scaled) parameter units.

    The asymptote variance should be enormous compared with the O(1) shape
    parameters — a direct check that the scaled Hessian was correctly mapped
    back by Σ_x = D Σ_z D rather than left in scaled space.
    """
    df = _large_magnitude_activation_data()
    result = SingleDrugFit(
        df, FitConfig(direction="activation", noise="lognormal")
    ).fit()
    assert result.param_cov is not None

    names = result.param_names
    var = {n: result.param_cov[i, i] for i, n in enumerate(names)}
    # hill is O(1); effect_inf is O(1e4). Its variance must dwarf hill's.
    assert var["effect_inf"] > var["hill"]
