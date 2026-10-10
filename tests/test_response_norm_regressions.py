"""Regressions for response normalisation reporting, variance, and retry paths."""

import numpy as np
import pandas as pd
import pytest

from synfit.data import FitConfig
from synfit.joint_marginal import fit_joint_marginal_auto
from synfit.noise import GaussianLinear
from synfit.single import SingleDrugFit
from tests.response_scaling_helpers import single_drug_noisy_data
from tests.test_covariance_scaling import _two_drug_frames
from tests.test_variance_anchor import _repro_data


def test_gaussian_constant_log_likelihood_invariant_at_1e_9():
    data = single_drug_noisy_data()
    native = SingleDrugFit(data, FitConfig()).fit()
    s_ext = 1e-9
    scaled = SingleDrugFit(data.assign(y=data.y * s_ext), FitConfig()).fit()
    n = native.n_valid
    expected = native.log_likelihood - n * np.log(s_ext)
    assert scaled.log_likelihood == pytest.approx(expected, abs=1e-3)
    assert scaled.log_likelihood == pytest.approx(515.632, abs=0.05)


def test_fit_joint_marginal_auto_error_model_invariant_under_response_scale():
    data_a, data_b = _two_drug_frames()
    native = fit_joint_marginal_auto(data_a, data_b)
    s = 1e-9
    scaled_a = data_a.assign(y=data_a.y * s)
    scaled_b = data_b.assign(y=data_b.y * s)
    scaled = fit_joint_marginal_auto(scaled_a, scaled_b)
    assert scaled.error_model == native.error_model


def test_unset_variance_initials_normalise_from_their_public_values():
    """With explicit bounds, unset initials keep their public user-unit value
    (``config.var_a``), which must enter the normalised fit divided by s²."""
    from synfit.response_norm import normalized_fit_config

    data = single_drug_noisy_data().assign(y=lambda df: df.y * 1000.0)
    bounds = SingleDrugFit(data, FitConfig(noise=GaussianLinear())).config.bounds
    cfg = SingleDrugFit(
        data, FitConfig(noise=GaussianLinear(a_init=None, b_init=10.0), bounds=bounds),
    ).config
    assert cfg.noise.a_init is None
    s = 512.0
    work = normalized_fit_config(cfg, s)
    assert work.var_a == pytest.approx(cfg.var_a / s**2)
    assert work.var_b == pytest.approx(cfg.var_b / s)


def _excluded_zero_fit(var_a_bounds, *, fit_var_a):
    """Excluding the first zero changes the robust range (1 → 2), so the
    working scale differs from the one the user-unit domain was checked at."""
    data = pd.DataFrame({
        "concentration": [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000.0],
        "y": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "replicate": 0,
    })
    fitter = SingleDrugFit(data, FitConfig(noise=GaussianLinear(a_init=1.5e-12, b_init=0.0)))
    fitter.config.bounds.var_a = var_a_bounds
    if not fit_var_a:
        fitter.config.fitting_parameters = [
            p for p in fitter.config.fitting_parameters if p != "var_a"
        ]
    valids = np.ones(len(data), dtype=bool)
    valids[0] = False
    return fitter.fit(valids=valids)


def test_valid_user_variance_domain_is_not_re_clamped_in_working_units():
    """Bounds valid in user units must not become empty after normalisation."""
    result = _excluded_zero_fit((1e-12, 2e-12), fit_var_a=True)
    assert 1e-12 <= result.variance_params["a"] <= 2e-12


def test_predict_variance_equivariant_below_the_absolute_floor():
    """At s = 1e-9 every variance (~1e-21) is far below the 2.2e-16 floor; the
    reported variance model must still be the fitted one, scaled by s². Interior
    μ only: at the lower asymptote the fitted ``a`` sits on its bound and the
    value is dominated by rounding in μ − anchor."""
    data = single_drug_noisy_data()
    native = SingleDrugFit(data, FitConfig(noise=GaussianLinear())).fit()
    mu = native.effect_inf + np.array([0.25, 0.5, 0.75, 1.0]) * (
        native.effect_0 - native.effect_inf
    )
    s = 1e-9
    scaled = SingleDrugFit(
        data.assign(y=data.y * s), FitConfig(noise=GaussianLinear()),
    ).fit()
    np.testing.assert_allclose(
        scaled.predict_variance(mu * s) / s**2, native.predict_variance(mu), rtol=1e-3,
    )


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e3])
def test_variance_retry_initials_independent_of_response_magnitude(scale):
    from synfit.response_norm import normalized_fit_config, response_scale_from_y

    df = _repro_data()
    if scale != 1.0:
        df = df.assign(y=df.y * scale)
    cfg = FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05))
    fitter = SingleDrugFit(df, cfg)
    s = response_scale_from_y(df.y.values)
    work = normalized_fit_config(fitter.config, s)
    fitter._enter_response_normalization(s, work_config=work)
    x0, bounds = fitter._get_x0_and_bounds()
    x0_retry = fitter._variance_retry_x0(x0, bounds)
    fitter._clear_response_normalization()
    names = list(fitter.config.fitting_parameters)
    idx_a = names.index("var_a")
    idx_b = names.index("var_b")
    assert x0_retry[idx_a] == pytest.approx(1e-3, rel=1e-6)
    assert x0_retry[idx_b] == pytest.approx(0.0, abs=1e-15)


def test_joint_drug_results_share_the_parent_variance_model():
    from synfit.joint_marginal import JointMarginalFit

    data_a, data_b = _two_drug_frames()
    s = 1e-9
    result = JointMarginalFit(
        data_a.assign(y=data_a.y * s), data_b.assign(y=data_b.y * s), noise=GaussianLinear(),
    ).fit()
    mu = 0.5 * (result.top + result.bottom)
    expected = result.predict_variance(mu)
    np.testing.assert_allclose(result.drug_a.predict_variance(mu), expected, rtol=1e-12)
    np.testing.assert_allclose(result.drug_b.predict_variance(mu), expected, rtol=1e-12)


def test_to_dict_carries_the_response_scale():
    """``predict_variance`` depends on the scale, so serialised results keep it."""
    data = single_drug_noisy_data().assign(y=lambda df: df.y * 1e-9)
    result = SingleDrugFit(data, FitConfig(noise=GaussianLinear())).fit()
    assert result.response_scale != 1.0
    assert result.to_dict()["response_scale"] == result.response_scale


def test_every_fitter_reports_the_scale_it_used():
    from synfit.matrix import MatrixFit
    from synfit.single import SingleDrugFitWithError
    from tests.helpers import matrix_from_config

    s = 1e-9
    data = single_drug_noisy_data()
    weighted = SingleDrugFitWithError(
        data.assign(y=data.y * s, y_err=0.1 * s), FitConfig(),
    ).fit()
    assert weighted.response_scale != 1.0
    assert weighted.to_dict()["response_scale"] == weighted.response_scale

    reps, ch, cv = matrix_from_config(seed=42, noise_model="lognormal")
    matrix = MatrixFit([r * s for r in reps], ch, cv, error_model="lognormal").fit()
    assert matrix.horizontal.response_scale == matrix.vertical.response_scale != 1.0
    assert matrix.to_dict()["horizontal"]["response_scale"] == matrix.horizontal.response_scale


def test_unit_response_scale_still_normalises():
    """A fit whose scale is exactly 1 must take the normalised path too: the
    excluded outliers set the construction-time range (~1000), which must not
    leak into the optimiser's variance scaling."""
    from synfit.single import _robust_response_extrema

    base = single_drug_noisy_data()
    lo, hi = _robust_response_extrema(base.y.values)
    base = base.assign(y=(base.y - lo) / (hi - lo))
    outliers = pd.DataFrame({"concentration": [1.0, 2.0, 3.0], "y": 1000.0})
    data = pd.concat([base, outliers], ignore_index=True).reindex(columns=base.columns)
    data["replicate"] = data["replicate"].fillna(0)
    valids = np.r_[np.ones(len(base), bool), np.zeros(len(outliers), bool)]

    fits = {}
    for f in (1.0, 2.0):
        fitter = SingleDrugFit(data.assign(y=data.y * f), FitConfig(noise=GaussianLinear()))
        fits[f] = fitter.fit(valids=valids)
    assert fits[1.0].response_scale == pytest.approx(1.0)
    assert fits[2.0].response_scale == pytest.approx(2.0)
    assert fits[1.0].c50 == pytest.approx(fits[2.0].c50, rel=1e-6)
    assert fits[1.0].hill == pytest.approx(fits[2.0].hill, rel=1e-6)
    assert fits[2.0].log_likelihood == pytest.approx(
        fits[1.0].log_likelihood - fits[1.0].n_valid * np.log(2.0), abs=1e-6
    )
