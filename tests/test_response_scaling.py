"""Response rescaling: lock in equivariance and record known small-magnitude defects."""

import numpy as np
import pytest

from synfit.data import FitConfig
from synfit.single import SingleDrugFit

from tests.response_scaling_helpers import (
    CONC_GRID,
    SCALE_FIT_CASES,
    case_by_id,
    compare_joint_results,
    compare_matrix_results,
    compare_single_results,
    fit_scaled_case,
    hill_predictions,
    native_solution_x,
    result_to_x,
    run_scale_case,
    single_drug_noisy_data,
)


def _noisy_config(noise: str) -> FitConfig:
    return FitConfig(noise="lognormal") if noise == "lognormal" else FitConfig()


def _noisy_data():
    return single_drug_noisy_data()


@pytest.mark.parametrize("noise", ["gaussian", "lognormal"])
@pytest.mark.parametrize("s", [1e-3, 1e-6])
def test_small_response_objective_gap_health(noise, s):
    """Fits on scaled data must succeed with finite outputs; native optimum stays feasible."""
    data = _noisy_data()
    cfg = _noisy_config(noise)
    native = SingleDrugFit(data, cfg).fit()
    scaled_data = data.assign(y=data.y * s)
    fitter = SingleDrugFit(scaled_data, cfg)
    scaled = fitter.fit()

    assert scaled.success
    assert np.isfinite(scaled.c50) and scaled.c50 > 0
    x_native = native_solution_x(fitter, native, s)
    assert np.all(np.isfinite(x_native))
    obj_native = fitter._objective(x_native)
    assert np.isfinite(obj_native)


@pytest.mark.parametrize(
    "noise,s",
    [
        pytest.param(noise, s, id=f"{noise}-s={s:g}")
        for noise in ("gaussian", "lognormal")
        for s in (1e-3, 1e-6)
    ],
)
def test_small_response_objective_at_most_native(noise, s):
    """Attained objective must not exceed the transformed native optimum (strict xfail on known gaps)."""
    data = _noisy_data()
    cfg = _noisy_config(noise)
    native = SingleDrugFit(data, cfg).fit()
    scaled_data = data.assign(y=data.y * s)
    fitter = SingleDrugFit(scaled_data, cfg)
    scaled = fitter.fit()
    x_native = native_solution_x(fitter, native, s)
    obj_fit = fitter._objective(result_to_x(fitter, scaled))
    obj_native = fitter._objective(x_native)
    assert obj_fit <= obj_native + 1e-4


@pytest.mark.parametrize(
    "noise,s,check",
    [
        pytest.param("gaussian", 1e-2, "se_effect_inf", id="gaussian-0.01-se_effect_inf"),
        pytest.param("gaussian", 1e-2, "predict_ci_width", id="gaussian-0.01-predict_ci_width"),
        pytest.param("lognormal", 1e-2, "se_effect_inf", id="lognormal-0.01-se_effect_inf"),
        pytest.param("lognormal", 1e-2, "predict_ci_width", id="lognormal-0.01-predict_ci_width"),
    ],
)
def test_small_response_uncertainty(noise, s, check):
    """Down-scaled lognormal uncertainty should match native (xfail known SE / CI defects)."""
    data = _noisy_data()
    cfg = _noisy_config(noise)
    native = SingleDrugFit(data, cfg).fit()
    assert native.param_cov is not None
    scaled = SingleDrugFit(data.assign(y=data.y * s), cfg).fit()
    assert scaled.param_cov is not None

    se_native = {
        n: float(np.sqrt(native.param_cov[i, i])) for i, n in enumerate(native.param_names)
    }
    se_scaled = {
        n: float(np.sqrt(scaled.param_cov[i, i])) for i, n in enumerate(scaled.param_names)
    }
    if check.startswith("se_"):
        name = check[3:]
        fac = s if name in ("effect_0", "effect_inf") else 1.0
        rel = abs(se_scaled[name] / fac - se_native[name]) / se_native[name]
        assert rel < 0.01
    else:
        lo_n, hi_n = native.predict_ci(CONC_GRID)
        lo_s, hi_s = scaled.predict_ci(CONC_GRID)
        w_native = hi_n - lo_n
        w_scaled = (hi_s - lo_s) / s
        rel = np.abs(w_scaled - w_native) / np.maximum(w_native, 1e-12)
        assert np.all(rel < 0.01)


def test_small_response_recovers_native_at_extreme_downscale():
    """s = 1e-9 on single_drug_noisy should match the native optimum after normalisation."""
    data = _noisy_data()
    cfg = _noisy_config("lognormal")
    native = SingleDrugFit(data, cfg).fit()
    s = 1e-9
    scaled = SingleDrugFit(data.assign(y=data.y * s), cfg).fit()
    assert scaled.success
    np.testing.assert_allclose(scaled.c50, native.c50, rtol=1e-3)
    np.testing.assert_allclose(scaled.log_c50, native.log_c50, rtol=1e-3)


@pytest.mark.parametrize(
    "case_id,s",
    [
        pytest.param(c.case_id, s, id=f"{c.case_id}-s={s:g}")
        for c in SCALE_FIT_CASES
        for s in (1e3, 1e-2)
    ],
)
def test_reported_outputs_scale(case_id, s):
    """Native vs scaled fits agree on reported outputs (covariance only when scaling up)."""
    case = case_by_id(case_id)
    native = run_scale_case(case)
    scaled = fit_scaled_case(case, s)
    check_cov = s >= 1.0

    if case.kind in ("single", "single_with_error"):
        compare_single_results(native, scaled, s, check_covariance=check_cov)
        conc = CONC_GRID
        np.testing.assert_allclose(
            hill_predictions(scaled, conc),
            hill_predictions(native, conc) * s,
            rtol=1e-3,
        )
        if native.variance_params and s != 1.0:
            mu = hill_predictions(native, conc)
            v_native = native.predict_variance(mu)
            v_scaled = scaled.predict_variance(mu * s)
            assert v_native is not None and v_scaled is not None
            np.testing.assert_allclose(v_scaled, v_native * s * s, rtol=0.02)
    elif case.kind == "joint":
        compare_joint_results(native, scaled, s, check_covariance=check_cov)
    elif case.kind == "matrix":
        compare_matrix_results(native, scaled, s, check_covariance=check_cov)
    else:
        pytest.fail(f"unhandled kind {case.kind}")


@pytest.mark.parametrize(
    "case_id,s",
    [
        pytest.param(c.case_id, s, id=f"{c.case_id}-s={s:g}")
        for c in SCALE_FIT_CASES
        if c.kind == "single_with_error"
        for s in (1e-6, 1e-9)
    ],
)
def test_weighted_fits_scale_down(case_id, s):
    """Weighted fits keep their default effect_0 = 1 start, far outside the
    bounds at small scales; it must not set the preconditioning (#61)."""
    case = case_by_id(case_id)
    compare_single_results(run_scale_case(case), fit_scaled_case(case, s), s)
