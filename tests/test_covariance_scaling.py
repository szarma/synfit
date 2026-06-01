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

from synfit.single import SingleDrugFit, _init_config_from_data
from synfit.data import FitConfig
from synfit.joint_marginal import JointMarginalFit
from synfit.matrix import MatrixFit
from synfit.synthetic import generate_single_drug

from tests.helpers import matrix_from_config


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


def test_covariance_rejects_non_finite_optimum():
    """A non-finite optimum yields no covariance (not a NaN-laden one that
    sneaks past the ``diag < 0`` sign check, since ``NaN < 0`` is False)."""
    df = _large_magnitude_activation_data()
    fitter = SingleDrugFit(df, FitConfig(direction="activation", noise="lognormal"))
    bad = np.array([np.nan, 1.0, 1.0, 1.0])
    assert fitter._estimate_covariance(bad) is None


# --- Shared FitBase path: joint-marginal and matrix fitters ---------------
#
# SingleDrugFit, JointMarginalFit and (now) MatrixFit all route through
# FitBase._run_minimize / _estimate_covariance, so the scale-relative fix must
# hold for the two-drug fitters too — JointMarginalFit is the *live* matrix path
# in the app.

def _two_drug_frames(seed_a: int = 11, seed_b: int = 22):
    def cfg(c50, hill, seed):
        return {
            "seed": seed,
            "hill_params": {"c50": c50, "hill": hill, "effect_0": 1.0, "effect_inf": 0.05},
            "concentration_series": {
                "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
                "length": 8, "has_zero": False,
            },
            "n_replicates": 3,
            "noise_model": "gaussian", "noise_sigma": 0.02, "noise_sigma_log": 0.02,
        }
    return (
        generate_single_drug(cfg(2.0, 1.5, seed_a)),
        generate_single_drug(cfg(10.0, 1.0, seed_b)),
    )


def _scale_frame(df: pd.DataFrame, s: float) -> pd.DataFrame:
    out = df.copy()
    out["y"] = df["y"] * s
    return out


def test_joint_marginal_fit_is_scale_equivariant():
    """The live matrix path (JointMarginalFit) keeps a covariance at large
    magnitude and its shape parameters are scale-invariant."""
    a, b = _two_drug_frames()
    r1 = JointMarginalFit(a, b).fit()
    assert r1.param_cov is not None

    s = 1000.0
    rs = JointMarginalFit(_scale_frame(a, s), _scale_frame(b, s)).fit()
    assert rs.param_cov is not None, "covariance lost at large response magnitude"
    assert np.all(np.diag(rs.param_cov) > 0)

    np.testing.assert_allclose(rs.drug_a.c50, r1.drug_a.c50, rtol=3e-2)
    np.testing.assert_allclose(rs.drug_b.c50, r1.drug_b.c50, rtol=3e-2)
    np.testing.assert_allclose(rs.drug_a.hill, r1.drug_a.hill, rtol=3e-2)
    np.testing.assert_allclose(rs.drug_b.hill, r1.drug_b.hill, rtol=3e-2)
    np.testing.assert_allclose(rs.top, r1.top * s, rtol=3e-2)
    np.testing.assert_allclose(rs.bottom, r1.bottom * s, rtol=3e-2)


def test_custom_x0_is_scale_invariant_with_data_bounds():
    """A hand-crafted x0 that does not match the response magnitude still fits
    scale-invariantly when bounds are derived from the data at each scale."""
    base = _saturating_activation_data()

    def fit_bad_x0(df: pd.DataFrame):
        auto = _init_config_from_data(
            df, direction="activation", noise=FitConfig(noise="lognormal").noise,
        )
        cfg = FitConfig(
            direction="activation",
            noise="lognormal",
            log_c50=0.0,
            hill=1.0,
            effect_0=1.0,
            effect_inf=1.0,
            bounds=auto.bounds,
        )
        return SingleDrugFit(df, cfg).fit()

    r1 = fit_bad_x0(base)
    assert r1.param_cov is not None

    def var(result, name):
        i = result.param_names.index(name)
        return result.param_cov[i, i]

    for s in (1000.0, 0.001):
        rs = fit_bad_x0(_scale_frame(base, s))
        assert rs.param_cov is not None, f"covariance lost at scale {s}"
        np.testing.assert_allclose(rs.c50, r1.c50, rtol=2e-2)
        np.testing.assert_allclose(rs.hill, r1.hill, rtol=2e-2)
        np.testing.assert_allclose(rs.effect_0, r1.effect_0 * s, rtol=2e-2)
        np.testing.assert_allclose(rs.effect_inf, r1.effect_inf * s, rtol=2e-2)

    r_up = fit_bad_x0(_scale_frame(base, 1000.0))
    np.testing.assert_allclose(
        var(r_up, "effect_inf"), var(r1, "effect_inf") * 1000.0 ** 2, rtol=0.1
    )
    np.testing.assert_allclose(var(r_up, "log_c50"), var(r1, "log_c50"), rtol=0.1)
    np.testing.assert_allclose(var(r_up, "hill"), var(r1, "hill"), rtol=0.1)


def _fit_matrix_scaled(reps, conc_h, conc_v, s: float):
    scaled = [r * s for r in reps]
    return MatrixFit(scaled, conc_h, conc_v, error_model="lognormal").fit()


def test_matrix_fit_is_scale_equivariant():
    """The 6-parameter Bliss matrix fit is invariant to response rescaling."""
    reps, conc_h, conc_v = matrix_from_config(seed=42, noise_model="lognormal")
    r1 = _fit_matrix_scaled(reps, conc_h, conc_v, 1.0)
    assert r1.success
    assert r1.param_names == [
        "log_c50_hor", "log_c50_ver", "hill_hor", "hill_ver", "effect_0", "effect_inf",
    ]
    assert r1.param_cov is not None
    assert np.all(np.diag(r1.param_cov)[:4] > 0)

    def var(result, name):
        i = result.param_names.index(name)
        return result.param_cov[i, i]

    for s in (1000.0, 0.01):
        rs = _fit_matrix_scaled(reps, conc_h, conc_v, s)
        assert rs.param_cov is not None, f"covariance lost at scale {s}"

        np.testing.assert_allclose(rs.horizontal.c50, r1.horizontal.c50, rtol=3e-2)
        np.testing.assert_allclose(rs.vertical.c50, r1.vertical.c50, rtol=5e-2)
        np.testing.assert_allclose(rs.horizontal.hill, r1.horizontal.hill, rtol=3e-2)
        np.testing.assert_allclose(rs.vertical.hill, r1.vertical.hill, rtol=3e-2)
        np.testing.assert_allclose(rs.effect_0, r1.effect_0 * s, rtol=3e-2)
        np.testing.assert_allclose(rs.effect_inf, r1.effect_inf * s, rtol=3e-2)

    r_up = _fit_matrix_scaled(reps, conc_h, conc_v, 1000.0)
    np.testing.assert_allclose(
        var(r_up, "effect_inf"), var(r1, "effect_inf") * 1000.0 ** 2, rtol=0.15
    )
    np.testing.assert_allclose(
        var(r_up, "log_c50_hor"), var(r1, "log_c50_hor"), rtol=0.15
    )
    np.testing.assert_allclose(var(r_up, "hill_hor"), var(r1, "hill_hor"), rtol=0.15)

    d = r1.to_dict()
    assert d["param_names"] == r1.param_names
    assert np.asarray(d["param_cov"]).shape == (6, 6)


def test_small_c50_with_wide_log_bounds_keeps_covariance():
    """A legitimately small EC50 must not be over-preconditioned from bounds.

    Regression for the first cut of #14: the bound-magnitude fallback in
    ``parameter_scale`` fired on *every* parameter whose ``|x0|`` was small
    relative to its bounds — including ``log_c50``. With the default seed
    (``log_c50 = 0``) and the wide ``(-5, 5)`` log-decade bounds, ``|0|`` is
    below the threshold, so the optimiser scale for ``log_c50`` was inflated to
    the bound magnitude (5). One optimiser step then spanned ~5 decades of
    EC50, sending the heteroscedastic ``gaussian_linear`` fit to a spurious
    optimum (EC50 ≈ 15 against a true 5) with a non-invertible Hessian, so the
    covariance was discarded. This is exactly the downstream
    ``gaussian_linear`` CI-band fixture that broke; pinning it here keeps the
    failure inside synfit's own suite.

    The fix restricts the fallback to the magnitude-bearing asymptotes, so
    ``log_c50`` keeps its O(1) scale and the fit recovers both the EC50 and a
    finite covariance.
    """
    from synfit.noise import GaussianLinear

    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.02},
        "concentration_series": {
            "initial_conc": 100.0,
            "fold_dilutions": 3.0,
            "length": 8,
            "has_zero": False,
        },
        "n_replicates": 3,
    }
    df = generate_single_drug(
        {**cfg, "noise_model": "gaussian", "noise_var_a": 0.001, "noise_var_b": 0.05},
        rng=np.random.default_rng(42),
    )
    result = SingleDrugFit(
        df, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05))
    ).fit()

    assert result.param_cov is not None, "covariance must survive (was None pre-fix)"
    assert np.all(np.diag(result.param_cov) >= 0)
    # True EC50 is 5.0; the regression landed it near 15. A 30% tolerance both
    # absorbs noise and stays well clear of the spurious optimum.
    np.testing.assert_allclose(result.c50, 5.0, rtol=0.3)
