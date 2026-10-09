"""Anchored heteroscedastic Gaussian variance: domain, scale, shift, retry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult, minimize as real_minimize

from synfit.data import FitBounds, FitConfig
from synfit.hill import calculate_concentration_series, hill_curve
from synfit.joint_marginal import JointMarginalFit, default_joint_marginal_config
from synfit.noise import (
    GaussianConstant,
    GaussianLinear,
    GaussianQuadratic,
    Lognormal,
    from_dict,
    to_dict,
    variance_at,
)
from synfit.single import SingleDrugFit, _robust_response_range
from synfit.synthetic import generate_single_drug


def _repro_data() -> pd.DataFrame:
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
        "noise_model": "gaussian",
        "noise_var_a": 0.001,
        "noise_var_b": 0.05,
    }
    return generate_single_drug(cfg, rng=np.random.default_rng(42))


def _hetero_curve(
    a: float = 5e-4,
    b: float = 0.04,
    c: float = 0.0,
    *,
    seed: int = 11,
) -> pd.DataFrame:
    return generate_single_drug({
        "seed": seed,
        "hill_params": {"c50": 1.0, "hill": 1.5, "effect_0": 0.9, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
            "length": 12, "has_zero": False,
        },
        "n_replicates": 8,
        "noise_model": "gaussian",
        "noise_var_a": a,
        "noise_var_b": b,
        "noise_var_c": c,
    })


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
            "noise_model": "gaussian",
            "noise_sigma": 0.02,
        }
    return generate_single_drug(cfg(2.0, 1.5, seed_a)), generate_single_drug(cfg(10.0, 1.0, seed_b))


def test_reproduction_linear_variance_matches_core_0_3_0():
    """The fixture that stalled at ABNORMAL under the un-anchored polynomial."""
    df = _repro_data()
    fitter = SingleDrugFit(
        df, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05)),
    )
    x0, _bounds = fitter._get_x0_and_bounds()
    names = list(fitter.config.fitting_parameters)
    result = fitter.fit()
    assert result.success
    assert not np.allclose(
        [result.log_c50, result.hill, result.effect_0, result.effect_inf],
        [x0[names.index(n)] for n in ("log_c50", "hill", "effect_0", "effect_inf")],
    )
    assert result.aic == pytest.approx(-24.85, abs=0.15)
    assert result.effect_0 == pytest.approx(1.0279, rel=0.01)
    assert result.effect_inf == pytest.approx(0.0, abs=0.01)
    assert result.hill == pytest.approx(1.3395, rel=0.01)
    assert result.c50 == pytest.approx(4.674, rel=0.01)


def _variance_curve(result, lo, hi, n=25):
    """σ²(μ) on a grid spanning the fitted asymptotes.

    Quadratic a/b/c trade off against each other (weakly identifiable), so
    tests compare the identifiable variance function, not each coefficient.
    """
    mu = np.linspace(lo, hi, n)
    return mu, np.asarray(result.predict_variance(mu), dtype=float)

@pytest.mark.parametrize("variance_model", ["linear", "quadratic"])
def test_heteroscedastic_shift_invariance(variance_model):
    data = _hetero_curve(c=0.03 if variance_model == "quadratic" else 0.0)
    r0 = SingleDrugFit(data, FitConfig(variance_model=variance_model)).fit()
    assert r0.success
    vp0 = r0.variance_params
    assert vp0 is not None

    for offset in (0.5, -0.1):
        shifted = data.copy()
        shifted["y"] = data["y"] + offset
        rs = SingleDrugFit(shifted, FitConfig(variance_model=variance_model)).fit()
        assert rs.success
        np.testing.assert_allclose(rs.hill, r0.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.c50, r0.c50, rtol=1e-3)
        np.testing.assert_allclose(rs.effect_0, r0.effect_0 + offset, atol=1e-3)
        np.testing.assert_allclose(rs.effect_inf, r0.effect_inf + offset, atol=1e-3)
        if variance_model == "linear":
            for key in vp0:
                np.testing.assert_allclose(rs.variance_params[key], vp0[key], rtol=5e-3)
        else:
            lo0, hi0 = sorted((r0.effect_0, r0.effect_inf))
            _, v0 = _variance_curve(r0, lo0, hi0)
            _, vs = _variance_curve(rs, lo0 + offset, hi0 + offset)
            np.testing.assert_allclose(vs, v0, rtol=2e-2)


@pytest.mark.parametrize("variance_model", ["linear", "quadratic"])
def test_heteroscedastic_scale_equivariance(variance_model):
    base = _hetero_curve(c=0.03 if variance_model == "quadratic" else 0.0)
    r1 = SingleDrugFit(base, FitConfig(variance_model=variance_model)).fit()
    assert r1.success
    vp1 = r1.variance_params
    assert vp1 is not None

    for s in (1e-3, 1.0, 1e3):
        scaled = base.copy()
        scaled["y"] = base["y"] * s
        rs = SingleDrugFit(scaled, FitConfig(variance_model=variance_model)).fit()
        assert rs.success
        np.testing.assert_allclose(rs.hill, r1.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.c50, r1.c50, rtol=1e-3)
        np.testing.assert_allclose(rs.effect_0, r1.effect_0 * s, rtol=1e-3)
        np.testing.assert_allclose(rs.effect_inf, r1.effect_inf * s, rtol=1e-3)
        if variance_model == "linear":
            np.testing.assert_allclose(rs.variance_params["a"], vp1["a"] * s * s, rtol=1e-3)
            np.testing.assert_allclose(rs.variance_params["b"], vp1["b"] * s, rtol=1e-3)
        else:
            lo1, hi1 = sorted((r1.effect_0, r1.effect_inf))
            _, v1 = _variance_curve(r1, lo1, hi1)
            _, vs = _variance_curve(rs, lo1 * s, hi1 * s)
            np.testing.assert_allclose(vs, v1 * s * s, rtol=2e-2)


def test_swapped_asymptotes_finite_likelihood_single():
    df = _repro_data()
    fitter = SingleDrugFit(df, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05)))
    x0, _bounds = fitter._get_x0_and_bounds()
    x = np.asarray(x0, dtype=float).copy()
    names = list(fitter.config.fitting_parameters)
    i0, iinf = names.index("effect_0"), names.index("effect_inf")
    x[i0], x[iinf] = x[iinf], x[i0]
    ll = fitter._log_prob_data(x)
    assert np.isfinite(ll)
    a, b, c = fitter._x_to_variance_params(x)
    kwargs = fitter._x_to_kwargs(x)
    y_pred = fitter._curve(df["concentration"].values, kwargs)
    var = variance_at(y_pred, a, b, c, anchor=min(kwargs["effect_0"], kwargs["effect_inf"]))
    assert np.all(var >= a - 1e-15)


def test_swapped_asymptotes_finite_likelihood_joint():
    data_a, data_b = _two_drug_frames()
    fitter = JointMarginalFit(data_a, data_b, noise=GaussianLinear())
    x = np.asarray(fitter._x0, dtype=float).copy()
    i_top = fitter._param_names.index("top")
    i_bot = fitter._param_names.index("bottom")
    x[i_top], x[i_bot] = x[i_bot], x[i_top]
    ll = fitter._log_prob_data(x)
    assert np.isfinite(ll)
    a, b, c = fitter._x_to_variance_params(x)
    p = fitter._unpack(x)
    e0_a, einf_a = fitter._drug_asymptotes(p["top"], p["bottom"], fitter.direction_a)
    from synfit.hill import hill_curve
    y_pred = hill_curve(
        data_a["concentration"].values,
        c50=p["c50_a"], hill=p["hill_a"],
        effect_0=e0_a, effect_inf=einf_a,
    )
    var = variance_at(y_pred, a, b, c, anchor=min(p["top"], p["bottom"]))
    assert np.all(var >= a - 1e-15)


def test_domain_clamp_negative_var_b_bounds_and_init():
    df = _repro_data()
    fitter = SingleDrugFit(
        df,
        FitConfig(
            noise=GaussianLinear(a_init=0.001, b_init=-0.5),
            bounds=FitBounds(
                log_c50=(-5.0, 5.0),
                hill=(0.1, 4.0),
                effect_0=(0.0, 2.0),
                effect_inf=(0.0, 2.0),
                var_b=(-1e4, 1e4),
            ),
        ),
    )
    assert fitter.config.bounds.var_b[0] == 0.0
    assert fitter.config.noise.b_init == 0.0
    result = fitter.fit()
    assert result.success
    assert result.variance_params["b"] >= 0.0


def test_retry_from_default_variance_initials_on_iteration_zero_stall(monkeypatch):
    df = _repro_data()
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        n = calls["n"]
        calls["n"] += 1
        if n == 0:
            x0 = np.asarray(x0, dtype=float)
            return OptimizeResult(
                x=x0, success=False, message="ABNORMAL", nit=0, status=2,
                fun=float(fun(x0)),
            )
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr("synfit.fitting.minimize", fake_minimize)
    result = SingleDrugFit(
        df, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05)),
    ).fit()
    assert calls["n"] == 2
    assert result.success
    assert "retried from default variance initials" in result.message
    assert "optimizer stalled at the start" in result.message


def test_no_retry_when_first_attempt_iterates(monkeypatch):
    df = _repro_data()
    calls = {"n": 0}

    def fake_minimize(fun, x0, **kwargs):
        calls["n"] += 1
        x0 = np.asarray(x0, dtype=float)
        return OptimizeResult(
            x=x0, success=False, message="STOP", nit=3, status=2,
            fun=float(fun(x0)),
        )

    monkeypatch.setattr("synfit.fitting.minimize", fake_minimize)
    result = SingleDrugFit(
        df, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05)),
    ).fit()
    assert calls["n"] == 1
    assert not result.success
    assert "retried from default variance initials" not in result.message


@pytest.mark.parametrize("noise", [GaussianLinear(), GaussianQuadratic()])
def test_joint_marginal_heteroscedastic_converges_with_anchor(noise):
    data_a, data_b = _two_drug_frames()
    result = JointMarginalFit(data_a, data_b, noise=noise).fit()
    assert result.success
    assert result.variance_params is not None
    mu = np.array([result.bottom, 0.5 * (result.top + result.bottom), result.top])
    var = result.predict_variance(mu)
    assert var is not None
    assert np.all(var >= result.variance_params["a"] - 1e-15)


@pytest.mark.parametrize("scale", [1.0, 1e4])
@pytest.mark.parametrize("noise", [GaussianLinear(), GaussianQuadratic()])
def test_default_joint_marginal_config_var_matches_fitter_at_scale(scale, noise):
    data_a, data_b = _two_drug_frames()
    if scale != 1.0:
        data_a = data_a.copy()
        data_b = data_b.copy()
        data_a["y"] = data_a["y"] * scale
        data_b["y"] = data_b["y"] * scale
    result = default_joint_marginal_config(data_a, data_b, noise=noise)
    fitter = JointMarginalFit(data_a, data_b, noise=noise)
    for name, init, (lo, hi) in zip(fitter._param_names, fitter._x0, fitter._bounds):
        if not name.startswith("var_"):
            continue
        assert result[name]["init"] == init
        assert result[name]["lo"] == lo
        assert result[name]["hi"] == hi
    if scale == 1e4:
        assert result["var_a"]["hi"] > 1e4
        assert result["var_a"]["init"] == pytest.approx(
            1e-3 * fitter._response_scale ** 2
        )


def _explicit_curve_bounds(**var_kwargs) -> FitBounds:
    return FitBounds(
        log_c50=(-5.0, 5.0),
        hill=(0.1, 4.0),
        effect_0=(0.0, 2.0),
        effect_inf=(0.0, 2.0),
        **var_kwargs,
    )


def _normalized_positive() -> pd.DataFrame:
    df = generate_single_drug({
        "seed": 7,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 3.0, "length": 8, "has_zero": False,
        },
        "n_replicates": 3,
        "noise_model": "lognormal",
        "noise_sigma_log": 0.05,
    })
    out = df.copy()
    out["y"] = df["y"] / float(df["y"].max())
    return out


def _activation_curve() -> pd.DataFrame:
    return generate_single_drug({
        "seed": 11,
        "hill_params": {"c50": 1.0, "hill": 1.5, "effect_0": 0.05, "effect_inf": 0.9},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
            "length": 12, "has_zero": False,
        },
        "n_replicates": 8,
        "noise_model": "gaussian",
        "noise_var_a": 5e-4,
        "noise_var_b": 0.04,
    })


def _5p_curve() -> pd.DataFrame:
    concs = calculate_concentration_series(
        initial_conc=100.0, fold_dilutions=10 ** (2 / 7), length=12, has_zero=False,
    )
    rng = np.random.default_rng(11)
    rows = []
    effect_0, effect_inf = 0.9, 0.05
    a, b = 5e-4, 0.04
    m = min(effect_0, effect_inf)
    for rep in "ABCDEFGH":
        for conc in concs:
            mu = hill_curve(
                conc, c50=1.0, hill=1.5,
                effect_0=effect_0, effect_inf=effect_inf, asymmetry=2.0,
            )
            d = mu - m
            sigma = np.sqrt(max(a + b * d, 1e-12))
            rows.append({
                "concentration": conc,
                "y": float(mu + rng.normal(0.0, sigma)),
                "replicate": rep,
            })
    return pd.DataFrame(rows)


def test_heteroscedastic_fit_matches_native_after_response_downscale():
    """Internal normalisation keeps heteroscedastic fits equivariant to y → s·y."""
    data = _hetero_curve()
    noise = GaussianLinear()
    native = SingleDrugFit(data, FitConfig(noise=noise)).fit()
    s = 1e-2
    scaled = SingleDrugFit(data.assign(y=data.y * s), FitConfig(noise=noise)).fit()
    assert scaled.c50 == pytest.approx(native.c50, rel=1e-3)
    assert scaled.hill == pytest.approx(native.hill, rel=1e-3)


@pytest.mark.parametrize("variance_model", ["linear", "quadratic"])
def test_joint_marginal_heteroscedastic_shift_invariance(variance_model):
    data_a, data_b = _two_drug_frames()
    noise = GaussianLinear() if variance_model == "linear" else GaussianQuadratic()
    r0 = JointMarginalFit(data_a, data_b, noise=noise).fit()
    assert r0.success
    vp0 = r0.variance_params
    assert vp0 is not None

    for offset in (0.5, -0.1):
        shifted_a, shifted_b = data_a.copy(), data_b.copy()
        shifted_a["y"] = data_a["y"] + offset
        shifted_b["y"] = data_b["y"] + offset
        rs = JointMarginalFit(shifted_a, shifted_b, noise=noise).fit()
        assert rs.success
        np.testing.assert_allclose(rs.drug_a.log_c50, r0.drug_a.log_c50, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_b.log_c50, r0.drug_b.log_c50, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_a.hill, r0.drug_a.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_b.hill, r0.drug_b.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.top, r0.top + offset, atol=1e-3)
        np.testing.assert_allclose(rs.bottom, r0.bottom + offset, atol=1e-3)
        if variance_model == "linear":
            for key in vp0:
                np.testing.assert_allclose(rs.variance_params[key], vp0[key], rtol=5e-3)
        else:
            lo0, hi0 = sorted((r0.top, r0.bottom))
            _, v0 = _variance_curve(r0, lo0, hi0)
            _, vs = _variance_curve(rs, lo0 + offset, hi0 + offset)
            np.testing.assert_allclose(vs, v0, rtol=2e-2)


@pytest.mark.parametrize("variance_model", ["linear", "quadratic"])
def test_joint_marginal_heteroscedastic_scale_equivariance(variance_model):
    data_a, data_b = _two_drug_frames()
    noise = GaussianLinear() if variance_model == "linear" else GaussianQuadratic()
    r1 = JointMarginalFit(data_a, data_b, noise=noise).fit()
    assert r1.success
    vp1 = r1.variance_params
    assert vp1 is not None

    for s in (1e-3, 1e3):
        scaled_a, scaled_b = data_a.copy(), data_b.copy()
        scaled_a["y"] = data_a["y"] * s
        scaled_b["y"] = data_b["y"] * s
        rs = JointMarginalFit(scaled_a, scaled_b, noise=noise).fit()
        assert rs.success
        np.testing.assert_allclose(rs.drug_a.log_c50, r1.drug_a.log_c50, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_b.log_c50, r1.drug_b.log_c50, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_a.hill, r1.drug_a.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.drug_b.hill, r1.drug_b.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.top, r1.top * s, rtol=1e-3)
        np.testing.assert_allclose(rs.bottom, r1.bottom * s, rtol=1e-3)
        if variance_model == "linear":
            np.testing.assert_allclose(rs.variance_params["a"], vp1["a"] * s * s, rtol=5e-3)
            np.testing.assert_allclose(rs.variance_params["b"], vp1["b"] * s, rtol=5e-3)
        else:
            lo1, hi1 = sorted((r1.top, r1.bottom))
            _, v1 = _variance_curve(r1, lo1, hi1)
            _, vs = _variance_curve(rs, lo1 * s, hi1 * s)
            np.testing.assert_allclose(vs, v1 * s * s, rtol=2e-2)


def test_heteroscedastic_shift_invariance_activation():
    data = _activation_curve()
    r0 = SingleDrugFit(data, FitConfig(direction="activation", variance_model="linear")).fit()
    assert r0.success
    vp0 = r0.variance_params
    assert vp0 is not None
    for offset in (0.5, -0.1):
        shifted = data.copy()
        shifted["y"] = data["y"] + offset
        rs = SingleDrugFit(
            shifted, FitConfig(direction="activation", variance_model="linear"),
        ).fit()
        assert rs.success
        np.testing.assert_allclose(rs.hill, r0.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.c50, r0.c50, rtol=1e-3)
        np.testing.assert_allclose(rs.effect_0, r0.effect_0 + offset, atol=1e-3)
        np.testing.assert_allclose(rs.effect_inf, r0.effect_inf + offset, atol=1e-3)
        for key in vp0:
            np.testing.assert_allclose(rs.variance_params[key], vp0[key], rtol=5e-3)


def test_heteroscedastic_shift_invariance_5p():
    data = _5p_curve()
    cfg = FitConfig(
        variance_model="linear",
        fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf", "asymmetry"],
    )
    r0 = SingleDrugFit(data, cfg).fit()
    assert r0.success
    vp0 = r0.variance_params
    assert vp0 is not None
    for offset in (0.5, -0.1):
        shifted = data.copy()
        shifted["y"] = data["y"] + offset
        rs = SingleDrugFit(shifted, FitConfig(
            variance_model="linear",
            fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf", "asymmetry"],
        )).fit()
        assert rs.success
        np.testing.assert_allclose(rs.hill, r0.hill, rtol=1e-3)
        np.testing.assert_allclose(rs.c50, r0.c50, rtol=1e-3)
        np.testing.assert_allclose(rs.effect_0, r0.effect_0 + offset, atol=1e-3)
        np.testing.assert_allclose(rs.effect_inf, r0.effect_inf + offset, atol=1e-3)
        np.testing.assert_allclose(rs.asymmetry, r0.asymmetry, rtol=1e-3)
        for key in vp0:
            np.testing.assert_allclose(rs.variance_params[key], vp0[key], rtol=5e-3)


def test_negative_var_a_bounds_are_clamped_and_fit_is_valid():
    df = _repro_data()
    fitter = SingleDrugFit(
        df,
        FitConfig(
            noise=GaussianLinear(),
            bounds=_explicit_curve_bounds(var_a=(-2.0, 1.0)),
        ),
    )
    assert fitter.config.bounds.var_a[0] > 0
    result = fitter.fit()
    assert result.success
    assert result.variance_params["a"] > 0


def test_all_negative_var_b_bounds_raise_clear_error():
    df = _repro_data()
    with pytest.raises(ValueError, match=r"var_b bounds \(-2.0, -1.0\).*b >= 0"):
        SingleDrugFit(
            df,
            FitConfig(
                noise=GaussianLinear(),
                bounds=_explicit_curve_bounds(var_b=(-2.0, -1.0)),
            ),
        )


def test_joint_negative_var_a_bounds_are_clamped_and_fit_is_valid():
    data_a, data_b = _two_drug_frames()
    fitter = JointMarginalFit(
        data_a, data_b,
        noise=GaussianLinear(),
        param_config={"var_a": {"lo": -2.0, "hi": 1.0, "init": -1.0}},
    )
    i = fitter._param_names.index("var_a")
    assert fitter._bounds[i][0] > 0
    assert fitter._x0[i] > 0
    result = fitter.fit()
    assert result.success
    assert result.variance_params["a"] > 0


def test_joint_all_negative_var_b_bounds_raise_clear_error():
    data_a, data_b = _two_drug_frames()
    with pytest.raises(ValueError, match=r"var_b bounds \(-2.0, -1.0\).*b >= 0"):
        JointMarginalFit(
            data_a, data_b,
            noise=GaussianLinear(),
            param_config={"var_b": {"lo": -2.0, "hi": -1.0}},
        )


def test_explicit_a_init_is_preserved_when_response_scale_is_not_one():
    data = _hetero_curve()
    scaled = data.copy()
    scaled["y"] = data["y"] * 1e3
    fitter = SingleDrugFit(
        scaled, FitConfig(noise=GaussianLinear(a_init=0.001, b_init=0.05)),
    )
    assert fitter.config.noise.a_init == 0.001
    x0, _bounds = fitter._get_x0_and_bounds()
    names = list(fitter.config.fitting_parameters)
    assert x0[names.index("var_a")] == pytest.approx(0.001)


def test_omitted_a_init_becomes_response_scaled_default():
    data = _hetero_curve()
    scaled = data.copy()
    scaled["y"] = data["y"] * 1e3
    fitter = SingleDrugFit(scaled, FitConfig(noise=GaussianLinear()))
    assert fitter.config.noise.a_init == pytest.approx(1e-3 * fitter._response_scale ** 2)


def test_robust_response_range_safe_for_fewer_than_two_values():
    assert _robust_response_range(np.array([])) == 1.0
    assert _robust_response_range(np.array([0.25])) == 1.0
    assert _robust_response_range(np.array([-4.0])) == 4.0


def test_one_row_fit_raises_fit_size_error_not_index_error():
    df = pd.DataFrame({"concentration": [1.0], "y": [0.5], "replicate": ["A"]})
    fitter = SingleDrugFit(
        df,
        FitConfig(noise=GaussianLinear(), bounds=_explicit_curve_bounds()),
    )
    with pytest.raises(ValueError, match="valid data points"):
        fitter.fit()


@pytest.mark.parametrize(
    "spec",
    [
        GaussianLinear(),
        GaussianLinear(a_init=0.001, b_init=0.05),
        GaussianQuadratic(),
        GaussianQuadratic(a_init=0.1, b_init=0.2, c_init=0.3),
    ],
)
def test_gaussian_hetero_spec_serde_round_trip(spec):
    encoded = to_dict(spec)
    if spec.a_init is None:
        assert "a_init" not in encoded
    else:
        assert encoded["a_init"] == spec.a_init
    assert from_dict(encoded) == spec
    assert from_dict({"kind": spec.kind}).a_init is None


@pytest.mark.parametrize("e0, einf", [(1.2, -0.3), (-0.3, 1.2)])
def test_single_drug_likelihood_passes_lower_asymptote_anchor(e0, einf, monkeypatch):
    """The likelihood receives min(effect_0, effect_inf) of the trial point as the anchor.

    Variance-function comparisons cannot detect a wrong-but-fixed anchor for
    quadratic noise (a/b/c can absorb it), so assert the call path directly.
    """
    import synfit.single as single_mod

    seen: list = []
    real = single_mod.noise_log_prob

    def spy(*args, **kwargs):
        seen.append(kwargs.get("variance_anchor"))
        return real(*args, **kwargs)

    monkeypatch.setattr(single_mod, "noise_log_prob", spy)
    fitter = SingleDrugFit(_hetero_curve(c=0.03), FitConfig(variance_model="quadratic"))
    x0, _ = fitter._get_x0_and_bounds()
    names = list(fitter.config.fitting_parameters)
    x = np.array(x0, dtype=float)
    x[names.index("effect_0")] = e0
    x[names.index("effect_inf")] = einf
    fitter._log_prob_data(x)
    assert seen and seen[-1] == pytest.approx(min(e0, einf))


@pytest.mark.parametrize("top, bottom", [(0.9, 0.2), (0.2, 0.9)])
def test_joint_marginal_likelihood_passes_lower_asymptote_anchor(top, bottom, monkeypatch):
    """Both marginal likelihood terms receive min(top, bottom) of the trial point."""
    import synfit.joint_marginal as jm_mod

    seen: list = []
    real = jm_mod.noise_log_prob

    def spy(*args, **kwargs):
        seen.append(kwargs.get("variance_anchor"))
        return real(*args, **kwargs)

    monkeypatch.setattr(jm_mod, "noise_log_prob", spy)
    data_a, data_b = _two_drug_frames()
    fitter = JointMarginalFit(data_a, data_b, noise=GaussianQuadratic())
    x = np.array(fitter._x0, dtype=float)
    x[fitter._param_names.index("top")] = top
    x[fitter._param_names.index("bottom")] = bottom
    seen.clear()
    fitter._log_prob_data(x)
    assert len(seen) == 2
    assert all(a == pytest.approx(min(top, bottom)) for a in seen)
