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


# --- Scale-equivariance ---------------------------------------------------
#
# The fit is invariant to a constant rescaling of the response. Under
# ``y -> y * s`` with multiplicative (lognormal) noise the negative
# log-likelihood is identical up to an additive constant, so the optimum is
# unchanged except that the asymptotes ride along with the scale:
#
#   c50, hill            : invariant
#   effect_0, effect_inf : multiply by s
#   cov(effect_*)        : multiply by s**2   (back-transform Σ_x = D Σ_z D)
#   cov(log_c50), cov(hill) : invariant
#
# These exercise the scale-relative covariance code across magnitude regimes
# that span the broken (≫1) and the always-worked (≈1) cases.

_TRUE = dict(c50=20.0, hill=1.2, effect_0=10.0, effect_inf=200.0)


def _saturating_activation_data(seed: int = 5) -> pd.DataFrame:
    """Well-determined activation curve: top concentration ~50x EC50, low
    noise, so the fit recovers the true parameters tightly and bounds stay
    inactive (a prerequisite for clean equivariance)."""
    rng = np.random.default_rng(seed)
    conc = np.array([0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    frac = (conc ** _TRUE["hill"]) / (conc ** _TRUE["hill"] + _TRUE["c50"] ** _TRUE["hill"])
    y = _TRUE["effect_0"] + (_TRUE["effect_inf"] - _TRUE["effect_0"]) * frac
    frames = [
        pd.DataFrame(
            {
                "concentration": conc,
                "y": y * np.exp(rng.normal(0, 0.05, size=conc.shape)),
                "replicate": rep,
            }
        )
        for rep in ("A", "B", "C")
    ]
    return pd.concat(frames, ignore_index=True)


def _fit_scaled(base: pd.DataFrame, s: float):
    scaled = base.copy()
    scaled["y"] = base["y"] * s
    return SingleDrugFit(
        scaled, FitConfig(direction="activation", noise="lognormal")
    ).fit()


def test_fit_recovers_known_parameters_at_native_and_scaled_magnitude():
    """The fit recovers the true parameters whether the response is O(100) or
    O(1e5) — i.e. magnitude alone doesn't distort inference."""
    base = _saturating_activation_data()
    for s in (1.0, 1000.0):
        r = _fit_scaled(base, s)
        assert r.param_cov is not None
        np.testing.assert_allclose(r.c50, _TRUE["c50"], rtol=0.2)
        np.testing.assert_allclose(r.hill, _TRUE["hill"], rtol=0.2)
        np.testing.assert_allclose(r.effect_0, _TRUE["effect_0"] * s, rtol=0.3)
        np.testing.assert_allclose(r.effect_inf, _TRUE["effect_inf"] * s, rtol=0.15)


def test_fit_is_scale_equivariant():
    """Rescaling the response by a constant leaves shape parameters unchanged,
    scales the asymptotes by s, and scales their variances by s**2."""
    base = _saturating_activation_data()
    r1 = _fit_scaled(base, 1.0)
    assert r1.param_cov is not None
    names = r1.param_names

    def var(result, name):
        i = result.param_names.index(name)
        return result.param_cov[i, i]

    for s in (1000.0, 0.001):
        rs = _fit_scaled(base, s)
        assert rs.param_cov is not None, f"covariance lost at scale {s}"

        # Shape parameters are invariant.
        np.testing.assert_allclose(rs.c50, r1.c50, rtol=2e-2)
        np.testing.assert_allclose(rs.hill, r1.hill, rtol=2e-2)
        # Asymptotes ride with the scale.
        np.testing.assert_allclose(rs.effect_0, r1.effect_0 * s, rtol=2e-2)
        np.testing.assert_allclose(rs.effect_inf, r1.effect_inf * s, rtol=2e-2)

    # Covariance back-transform: asymptote variance scales by s**2, shape-
    # parameter variances are invariant. Checked for the up-scaled (≫1) case,
    # where the scale-relative step is consistent across both fits.
    r_up = _fit_scaled(base, 1000.0)
    np.testing.assert_allclose(
        var(r_up, "effect_inf"), var(r1, "effect_inf") * 1000.0 ** 2, rtol=0.1
    )
    np.testing.assert_allclose(var(r_up, "log_c50"), var(r1, "log_c50"), rtol=0.1)
    np.testing.assert_allclose(var(r_up, "hill"), var(r1, "hill"), rtol=0.1)
    assert names == r_up.param_names
