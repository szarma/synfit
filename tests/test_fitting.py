import copy

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import norm as _norm
from synfit.data import FitBounds, FitConfig, FitResult
from synfit.fitting import FitBase, effective_observation_mask
from synfit.hill import hill_curve
from synfit.noise import GaussianLinear, Lognormal, log_prob as noise_log_prob
from synfit.single import (
    SingleDrugFit,
    SingleDrugFitWithError,
    _robust_response_extrema,
    default_fit_config,
)
from synfit.synthetic import generate_single_drug


def _synthetic_data(c50=1.0, hill=1.5, effect_0=0.9, effect_inf=0.05,
                    noise_model="gaussian", noise_sigma=0.02, noise_sigma_log=0.02,
                    seed=42, length=8):
    """Generate synthetic dose-response data via the canonical generator."""
    config = {
        "seed": seed,
        "hill_params": {"c50": c50, "hill": hill, "effect_0": effect_0, "effect_inf": effect_inf},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10**(2/7), "length": length, "has_zero": False},
        "n_replicates": 3,
        "noise_model": noise_model,
        "noise_sigma": noise_sigma,
        "noise_sigma_log": noise_sigma_log,
    }
    return generate_single_drug(config)


@pytest.mark.parametrize("noise_model,error_model", [
    ("gaussian", "gaussian"),
    ("lognormal", "lognormal"),
])
def test_single_drug_fit_recovers_ic50(noise_model, error_model):
    data = _synthetic_data(c50=2.0, noise_model=noise_model)
    fitter = SingleDrugFit(data, FitConfig(error_model=error_model))
    result = fitter.fit()
    assert result.success
    assert result.c50 == pytest.approx(2.0, rel=0.25)


@pytest.mark.parametrize("noise_model,error_model", [
    ("gaussian", "gaussian"),
    ("lognormal", "lognormal"),
])
def test_single_drug_fit_recovers_hill(noise_model, error_model):
    data = _synthetic_data(hill=2.0, noise_model=noise_model)
    fitter = SingleDrugFit(data, FitConfig(error_model=error_model))
    result = fitter.fit()
    assert result.hill == pytest.approx(2.0, rel=0.6)


def test_single_drug_fit_with_valids():
    data = _synthetic_data(c50=1.0)
    # mark all as valid
    valids = np.ones(len(data), dtype=bool)
    fitter = SingleDrugFit(data)
    result = fitter.fit(valids=valids)
    assert result.n_valid == len(data)
    assert result.n_total == len(data)


def test_single_drug_fit_excluded_points():
    data = _synthetic_data(c50=1.0)
    valids = np.ones(len(data), dtype=bool)
    valids[:3] = False
    fitter = SingleDrugFit(data)
    result = fitter.fit(valids=valids)
    assert result.n_valid == len(data) - 3


def test_single_drug_fit_invalid_columns():
    bad_data = pd.DataFrame({"x": [1, 2], "y": [0.5, 0.3]})
    with pytest.raises(ValueError):
        SingleDrugFit(bad_data)


def test_fitbase_unresolved_bounds_raises_clear_error():
    """A FitBase whose config.bounds is still None at fit time must raise an
    actionable ValueError, not an opaque AttributeError on NoneType. bounds=None
    means "derive from data" — an invariant the data-bearing subclasses resolve;
    this guard keeps it enforced in the base class for any future subclass."""
    from synfit.fitting import FitBase

    base = FitBase(FitConfig())  # bare config → bounds defaults to None
    assert base.config.bounds is None

    with pytest.raises(ValueError, match="bounds is None"):
        base._get_x0_and_bounds()
    with pytest.raises(ValueError, match="derive from data"):
        base._log_prior_prob(np.zeros(len(base.config.fitting_parameters)))


def test_fitbase_concrete_bounds_pass_the_guard():
    """The guard only trips on None — an explicit FitBounds resolves cleanly."""
    from synfit.fitting import FitBase

    base = FitBase(FitConfig(bounds=FitBounds()))
    x0, bounds = base._get_x0_and_bounds()
    assert len(bounds) == len(base.config.fitting_parameters)


def test_single_drug_fit_with_error():
    # length=10 extends the dilution series two steps lower so the curve
    # actually reaches its top plateau (~0.90). With the default 8-point series
    # the response tops out at ~0.61, leaving the top asymptote unobserved — the
    # fit is then free to extrapolate the plateau within the (deliberately
    # generous) upper bound and c50 is not identifiable. This test is about
    # recovering c50, so it needs data that constrains the curve.
    data = _synthetic_data(c50=1.5, length=10)
    # add synthetic error bars
    df = data.groupby("concentration")["y"].agg(["mean", "std"]).reset_index()
    df.columns = ["concentration", "y", "y_err"]
    df["y_err"] = df["y_err"].fillna(0.01)

    fitter = SingleDrugFitWithError(df)
    result = fitter.fit()
    assert result.success
    assert result.c50 == pytest.approx(1.5, rel=0.2)
    # When per-point errors are user-supplied, the fit doesn't estimate σ.
    assert result.sigma is None


@pytest.mark.parametrize("noise_model,error_model,expected_sigma", [
    ("gaussian", "gaussian", 0.05),
    ("lognormal", "lognormal", 0.08),
])
def test_single_drug_fit_populates_sigma(noise_model, error_model, expected_sigma):
    data = _synthetic_data(
        c50=1.0, noise_model=noise_model,
        noise_sigma=expected_sigma, noise_sigma_log=expected_sigma,
    )
    fitter = SingleDrugFit(data, FitConfig(error_model=error_model))
    result = fitter.fit()
    assert result.sigma is not None
    # σ̂ lives in the model's natural likelihood space and should recover the
    # generator's noise level within sampling tolerance.
    assert result.sigma == pytest.approx(expected_sigma, rel=0.4)
    # to_dict() must surface it for DB persistence.
    assert result.to_dict()["sigma"] == result.sigma


def test_to_dict_4p_kappa_equals_ic50():
    """In the symmetric (4p) model κ *is* the half-max; ``kappa``, ``c50``
    and ``ic50``/``ec50`` should all be the same number."""
    data = _synthetic_data(c50=2.0)
    result = SingleDrugFit(data, FitConfig()).fit()
    d = result.to_dict()
    assert d["kappa"] == d["c50"]
    assert d["kappa"] == pytest.approx(d["ic50"], rel=1e-9)
    if "ci" in d and "kappa" in d["ci"]:
        assert d["ci"]["kappa"] == d["ci"]["c50"]
        assert d["ci"]["kappa"] == d["ci"].get("ic50")


def test_to_dict_5p_ic50_is_true_half_max_not_kappa():
    """In the asymmetric (5p) model ``kappa`` is the κ parameter (= c50)
    but the half-max is offset:

        half_max = κ · (2^(1/S) − 1)^(1/h)

    Old code wrote ``ic50 = c50`` even for 5p, mislabelling κ as the
    half-max by tens of percent. Pin the corrected behaviour:
    ``kappa == c50`` (parameter) and ``ic50`` != ``kappa`` whenever S != 1.
    """
    data = _synthetic_data(c50=2.0, hill=1.5)
    config = FitConfig(
        asymmetry=2.0,
        fitting_parameters=("log_c50", "hill", "effect_0", "effect_inf", "asymmetry"),
    )
    result = SingleDrugFit(data, config).fit()
    d = result.to_dict()

    assert d["kappa"] == d["c50"]
    s = d["asymmetry"]
    h = d["hill"]
    expected_half = d["kappa"] * (2.0 ** (1.0 / s) - 1.0) ** (1.0 / h)
    assert d["ic50"] == pytest.approx(expected_half, rel=1e-9)
    # The lie that used to ship: ic50 == c50 for 5p. With S=2 they differ
    # by ~0.41× factor → not within 5%. The test fails loudly if anyone
    # accidentally restores the old behaviour.
    assert abs(d["ic50"] - d["kappa"]) / d["kappa"] > 0.05

    if "ci" in d:
        # Both should appear, and they should be different intervals.
        assert "kappa" in d["ci"]
        assert "ic50" in d["ci"]
        assert d["ci"]["kappa"] != d["ci"]["ic50"]


def test_half_max_ci_propagates_fitted_asymmetry_at_s_equals_one():
    """5p with Ŝ=1 must not reuse the κ CI when asymmetry was estimated."""
    result = FitResult(
        c50=1.0,
        log_c50=0.0,
        hill=1.0,
        effect_0=1.0,
        effect_inf=0.0,
        asymmetry=1.0,
        success=True,
        n_valid=20,
        n_total=20,
        param_names=["log_c50", "hill", "effect_0", "effect_inf", "asymmetry"],
        param_cov=np.diag([0.01, 0.01, 0.01, 0.01, 0.25]),
    )
    assert result.half_max == pytest.approx(result.c50)
    c50_ci = result.param_ci()["c50"]
    half_ci = result._half_max_ci()
    assert half_ci is not None
    c50_width = c50_ci[1] - c50_ci[0]
    half_width = half_ci[1] - half_ci[0]
    assert half_width > c50_width * 1.5


def test_single_drug_fit_with_error_valids_mask():
    data = _synthetic_data(c50=1.0)
    df = data.groupby("concentration")["y"].agg(["mean", "std"]).reset_index()
    df.columns = ["concentration", "y", "y_err"]
    df["y_err"] = df["y_err"].fillna(0.01)

    valids = np.ones(len(df), dtype=bool)
    valids[0] = False
    fitter = SingleDrugFitWithError(df)
    result = fitter.fit(valids=valids)
    assert result.n_valid == len(df) - 1


def test_curve_param_cis_survive_when_effect_inf_pins_to_lower_bound():
    """L-BFGS-B can pin a parameter to its bound; the unconstrained Hessian is
    then non-PD at the optimum. Old behaviour returned None for the entire
    covariance — wiping CIs for *every* parameter, including the well-behaved
    interior ones. Fix: invert only the interior sub-block.

    Repro: synthetic data prefers ``effect_inf ≈ 0.02``, but we set the lower
    bound at ``0.10``, well above. L-BFGS-B pins ``effect_inf`` at 0.10. The
    unconstrained Hessian at that point is non-PD, so the old code returned
    ``None`` for the whole covariance.
    """
    data = _synthetic_data(c50=1.0, effect_inf=0.02, noise_sigma=0.01)
    bounds = FitBounds(effect_inf=(0.10, 1.0))
    config = FitConfig(effect_inf=0.10, bounds=bounds)
    fitter = SingleDrugFit(data, config)
    result = fitter.fit()

    assert result.param_cov is not None, "covariance was wiped by a single bound-active param"
    assert result.param_names is not None

    inf_idx = result.param_names.index("effect_inf")
    assert result.param_cov[inf_idx, inf_idx] == 0.0  # bound-active → zero by design

    # Curve params (log_c50, hill, effect_0) must still have positive variance.
    for name in ("log_c50", "hill", "effect_0"):
        idx = result.param_names.index(name)
        assert result.param_cov[idx, idx] > 0.0, f"{name} CI was erased by the bound-active effect_inf"


def test_predict_ci_jacobian_matches_numerical_finite_differences():
    """Regression for ``predict_ci``'s analytic Jacobian.

    The CI band amplitude is ``√(J · Cov · Jᵀ)``. With a *diagonal* Cov the
    contribution from each parameter sits on ``J²`` and is sign-invariant —
    a sign-flipped Jacobian column would still produce the right band.
    With a non-diagonal Cov the off-diagonals carry signed cross-terms
    ``2·c·J_i·J_j``, so flipping the sign of any column changes the band
    amplitude in a measurable way. This test exercises a non-diagonal
    ``Cov[log_c50, hill]`` against a numerical Jacobian computed by
    ``scipy.optimize.approx_fprime`` on ``hill_curve`` directly.
    """
    log_c50, hill, e0, einf = 0.0, 1.5, 1.0, 0.0
    p0 = np.array([log_c50, hill, e0, einf], dtype=float)

    # Cov with a meaningful off-diagonal between log_c50 and hill — the cross
    # term is what catches a sign-flipped J column.
    cov = np.array([
        [0.04, 0.03, 0.0,  0.0 ],
        [0.03, 0.09, 0.0,  0.0 ],
        [0.0,  0.0,  0.01, 0.0 ],
        [0.0,  0.0,  0.0,  0.01],
    ])

    fit = FitResult(
        c50=10**log_c50, log_c50=log_c50, hill=hill,
        effect_0=e0, effect_inf=einf,
        success=True, n_valid=10, n_total=10,
        param_names=["log_c50", "hill", "effect_0", "effect_inf"],
        param_cov=cov,
    )

    # Concentrations on both sides of c50 — the bug only shows away from c50,
    # because the Jacobian is exactly zero at c=c50.
    concs = np.array([0.05, 0.2, 0.5, 2.0, 5.0, 20.0])
    lo_analytic, hi_analytic = fit.predict_ci(concs, alpha=0.05)

    # Numerical Jacobian column-by-column from hill_curve itself.
    def hill_eval(p, c):
        return float(hill_curve(c, c50=10**p[0], hill=p[1], effect_0=p[2], effect_inf=p[3]))

    J_num = np.zeros((len(concs), 4))
    for i, c in enumerate(concs):
        J_num[i, :] = approx_fprime(p0, lambda p: hill_eval(p, c), epsilon=1e-7)

    var_y_num = np.einsum("ij,jk,ik->i", J_num, cov, J_num)
    std_y_num = np.sqrt(np.maximum(var_y_num, 0.0))
    y_hat = hill_curve(concs, c50=10**log_c50, hill=hill, effect_0=e0, effect_inf=einf)
    z = _norm.ppf(0.975)
    lo_num = y_hat - z * std_y_num
    hi_num = y_hat + z * std_y_num

    np.testing.assert_allclose(lo_analytic, lo_num, atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(hi_analytic, hi_num, atol=2e-3, rtol=1e-3)


# ----------------------------- pinned curve parameters -------------------- #
#
# A FitResult that merely *echoes* the configured value is how this bug hid:
# hill_curve silently substituted its own default, then fit() reported the
# user's number via kwargs.get(..., config.<param>). Every test below asserts
# that the pinned value changed the likelihood / curve, not just the report.

_FREE_CURVE = ["log_c50", "hill", "effect_0", "effect_inf"]

_PIN_PAIRS = (
    ("log_c50", -0.5, 1.0),
    ("hill", 0.5, 3.0),
    ("effect_0", 0.7, 1.3),
    ("effect_inf", 0.0, 0.2),
    ("asymmetry", 0.4, 2.5),
)


def _pinned_config(data, name, value):
    """Data-derived bounds/initials, with ``name`` held at ``value``."""
    base = default_fit_config(data)
    kwargs = dict(
        log_c50=base.log_c50,
        hill=base.hill,
        effect_0=base.effect_0,
        effect_inf=base.effect_inf,
        asymmetry=base.asymmetry,
        bounds=base.bounds,
        fitting_parameters=[p for p in _FREE_CURVE if p != name],
        noise=base.noise,
    )
    kwargs[name] = value
    return FitConfig(**kwargs)


def test_x_to_kwargs_all_free_matches_optimizer_vector():
    """All-free path: kwargs come from ``x``, plus config.asymmetry=1 (the
    hill_curve default), so the evaluated curve is unchanged."""
    cfg = FitConfig(bounds=FitBounds())
    base = FitBase(cfg)
    x = np.array([0.3, 1.7, 0.95, 0.05])
    kwargs = base._x_to_kwargs(x)
    assert kwargs["c50"] == pytest.approx(10 ** 0.3)
    assert kwargs["hill"] == pytest.approx(1.7)
    assert kwargs["effect_0"] == pytest.approx(0.95)
    assert kwargs["effect_inf"] == pytest.approx(0.05)
    assert kwargs["asymmetry"] == pytest.approx(1.0)
    conc = np.array([0.1, 1.0, 10.0])
    y_new = hill_curve(conc, **kwargs)
    y_old = hill_curve(conc, c50=10 ** 0.3, hill=1.7, effect_0=0.95, effect_inf=0.05)
    np.testing.assert_allclose(y_new, y_old)


def test_x_to_kwargs_unlogs_fixed_log_c50():
    cfg = FitConfig(
        log_c50=0.5,
        hill=1.2,
        effect_0=0.9,
        effect_inf=0.1,
        fitting_parameters=["hill", "effect_0", "effect_inf"],
        bounds=FitBounds(),
    )
    kwargs = FitBase(cfg)._x_to_kwargs(np.array([1.2, 0.9, 0.1]))
    assert kwargs["c50"] == pytest.approx(10 ** 0.5)
    assert "log_c50" not in kwargs


@pytest.mark.parametrize("name, v1, v2", _PIN_PAIRS)
def test_pinned_curve_param_moves_the_likelihood(name, v1, v2):
    """Same free-parameter vector, two pinned values → different objective.

    This is the check that would have caught the bug: before the fix, kwargs
    and log-likelihood were identical for hill / effect_0 / effect_inf /
    asymmetry, and log_c50 raised TypeError.
    """
    data = _synthetic_data()
    f1 = SingleDrugFit(data, _pinned_config(data, name, v1))
    f2 = SingleDrugFit(data, _pinned_config(data, name, v2))
    x0, _bounds = f1._get_x0_and_bounds()
    x = np.asarray(x0, dtype=float)
    assert f1.config.fitting_parameters == f2.config.fitting_parameters
    kw1 = f1._x_to_kwargs(x)
    kw2 = f2._x_to_kwargs(x)
    curve_key = "c50" if name == "log_c50" else name
    assert kw1[curve_key] == pytest.approx(10 ** v1 if name == "log_c50" else v1)
    assert kw2[curve_key] == pytest.approx(10 ** v2 if name == "log_c50" else v2)
    assert kw1[curve_key] != pytest.approx(kw2[curve_key])
    ll1 = f1._log_prob_data(x)
    ll2 = f2._log_prob_data(x)
    assert np.isfinite(ll1) and np.isfinite(ll2)
    assert abs(ll1 - ll2) > 1e-6


@pytest.mark.parametrize("name, v1, v2", _PIN_PAIRS)
def test_pinned_curve_param_fit_uses_the_pinned_value(name, v1, v2):
    """End-to-end: two pins produce different curves / MLE free params, and
    the result reports the pin rather than echoing a value the fit ignored."""
    data = _synthetic_data()
    f1 = SingleDrugFit(data, _pinned_config(data, name, v1))
    f2 = SingleDrugFit(data, _pinned_config(data, name, v2))
    r1 = f1.fit()
    r2 = f2.fit()

    reported1 = getattr(r1, name)
    reported2 = getattr(r2, name)
    assert reported1 == pytest.approx(v1)
    assert reported2 == pytest.approx(v2)
    if name == "log_c50":
        assert r1.c50 == pytest.approx(10 ** v1)
        assert r2.c50 == pytest.approx(10 ** v2)

    pred1 = f1.predict(r1)
    pred2 = f2.predict(r2)
    assert not np.allclose(pred1, pred2)
    assert r1.log_likelihood is not None and r2.log_likelihood is not None
    assert abs(r1.log_likelihood - r2.log_likelihood) > 1e-6

    # A free parameter the optimiser *did* move should differ across pins
    # (otherwise the two fits collapsed to the same ignored-pin solution).
    free_vals_1 = [getattr(r1, p) for p in r1.param_names if p in _FREE_CURVE]
    free_vals_2 = [getattr(r2, p) for p in r2.param_names if p in _FREE_CURVE]
    assert not np.allclose(free_vals_1, free_vals_2, rtol=1e-5, atol=1e-5)


def test_fixing_log_c50_does_not_raise():
    data = _synthetic_data()
    cfg = _pinned_config(data, "log_c50", 0.0)
    result = SingleDrugFit(data, cfg).fit()
    assert result.log_c50 == pytest.approx(0.0)
    assert result.c50 == pytest.approx(1.0)
    assert "log_c50" not in result.param_names


def test_pinning_default_log_c50_survives_data_seeding():
    """log_c50=0 is both the FitConfig default *and* a legitimate pin
    (c50=1). Data-derived seeding must not overwrite it when it is fixed."""
    data = _synthetic_data()
    cfg = FitConfig(
        log_c50=0.0,
        fitting_parameters=["hill", "effect_0", "effect_inf"],
    )
    fitter = SingleDrugFit(data, cfg)
    assert fitter.config.log_c50 == pytest.approx(0.0)
    result = fitter.fit()
    assert result.log_c50 == pytest.approx(0.0)
    assert result.c50 == pytest.approx(1.0)


def test_all_free_single_drug_fit_still_recovers_truth():
    """Regression guard: the default (all curve params free) path is unchanged."""
    data = _synthetic_data(c50=2.0, hill=1.5)
    result = SingleDrugFit(data, FitConfig()).fit()
    assert result.success
    assert result.c50 == pytest.approx(2.0, rel=0.25)
    assert result.hill == pytest.approx(1.5, rel=0.6)
    assert result.param_names == ["log_c50", "hill", "effect_0", "effect_inf"]
    # 4 estimated curve params + 1 profiled constant-σ
    assert result.n_params == 5


def test_fixed_parameter_excluded_from_estimated_count():
    data = _synthetic_data()
    result = SingleDrugFit(data, _pinned_config(data, "hill", 2.0)).fit()
    assert "hill" not in result.param_names
    assert result.param_names == ["log_c50", "effect_0", "effect_inf"]
    # 3 estimated curve params + 1 profiled constant-σ
    assert result.n_params == 4
    if result.param_cov is not None:
        assert result.param_cov.shape == (3, 3)


def test_single_drug_fit_with_error_can_fix_log_c50():
    data = _synthetic_data(c50=1.5, length=10)
    df = data.groupby("concentration")["y"].agg(["mean", "std"]).reset_index()
    df.columns = ["concentration", "y", "y_err"]
    df["y_err"] = df["y_err"].fillna(0.01)
    base = default_fit_config(data)
    cfg = FitConfig(
        log_c50=0.0,
        hill=base.hill,
        effect_0=base.effect_0,
        effect_inf=base.effect_inf,
        bounds=base.bounds,
        fitting_parameters=["hill", "effect_0", "effect_inf"],
    )
    result = SingleDrugFitWithError(df, cfg).fit()
    assert result.log_c50 == pytest.approx(0.0)
    assert result.c50 == pytest.approx(1.0)
    assert "log_c50" not in result.param_names


def _with_error_df(data):
    """Keep every replicate so a 5-parameter WithError fit stays identifiable."""
    df = data[["concentration", "y"]].copy()
    df["y_err"] = 0.02
    return df


def test_single_drug_fit_with_error_reports_free_asymmetry():
    """Asymmetry was estimated (in param_names) but FitResult.asymmetry was
    always None, so param_ci() skipped it."""
    data = _synthetic_data(c50=1.5, length=10)
    df = _with_error_df(data)
    base = default_fit_config(data)
    cfg = FitConfig(
        log_c50=base.log_c50,
        hill=base.hill,
        effect_0=base.effect_0,
        effect_inf=base.effect_inf,
        asymmetry=1.5,
        bounds=base.bounds,
        fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf", "asymmetry"],
    )
    result = SingleDrugFitWithError(df, cfg).fit()
    assert result.success
    assert result.asymmetry is not None
    assert np.isfinite(result.asymmetry)
    assert "asymmetry" in result.param_names
    cis = result.param_ci()
    assert cis is not None
    assert "asymmetry" in cis


def test_single_drug_fit_with_error_reports_lognormal_error_model():
    data = _synthetic_data(noise_model="lognormal", noise_sigma_log=0.08)
    df = _with_error_df(data)
    result = SingleDrugFitWithError(df, FitConfig(noise=Lognormal())).fit()
    assert result.error_model == "lognormal"
    assert result.variance_model == "constant"


def test_single_drug_fit_with_error_reports_pinned_asymmetry():
    data = _synthetic_data(c50=1.5, length=10)
    df = _with_error_df(data)
    base = default_fit_config(data)
    cfg = FitConfig(
        log_c50=base.log_c50,
        hill=base.hill,
        effect_0=base.effect_0,
        effect_inf=base.effect_inf,
        asymmetry=2.0,
        bounds=base.bounds,
        fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf"],
    )
    result = SingleDrugFitWithError(df, cfg).fit()
    assert result.asymmetry == pytest.approx(2.0)
    assert "asymmetry" not in result.param_names
    cis = result.param_ci()
    if cis is not None:
        assert "asymmetry" not in cis


@pytest.mark.parametrize("fitter_cls, frame", [
    (SingleDrugFit, None),
    (SingleDrugFitWithError, "error"),
])
def test_single_drug_fitters_reject_empty_fitting_parameters(fitter_cls, frame):
    """fitting_parameters=[] used to reach the optimiser with an empty vector
    and die with 'not enough values to unpack (expected 2, got 0)'."""
    data = _synthetic_data()
    df = _with_error_df(data) if frame == "error" else data
    cfg = FitConfig(fitting_parameters=[], bounds=default_fit_config(data).bounds)
    with pytest.raises(ValueError, match="no free parameters"):
        fitter_cls(df, cfg)


def _case_ab_conc_y():
    """13-point curve shared by missing-response regression tests."""
    conc = np.logspace(-2, 2, 13)
    y = hill_curve(conc, c50=1.0, hill=1.2, effect_0=1.0, effect_inf=0.0)
    return conc, y


def test_single_drug_fit_raises_when_too_few_finite_responses():
    conc, y = _case_ab_conc_y()
    y_a = np.full(13, np.nan)
    y_a[[3, 9]] = y[[3, 9]]
    df = pd.DataFrame({"concentration": conc, "y": y_a, "replicate": 0})
    with pytest.raises(ValueError, match="valid data points"):
        SingleDrugFit(df).fit()


def test_single_drug_fit_nan_response_excluded_from_metrics():
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(0)
    y_b = y + rng.normal(0, 0.03, 13)
    y_b[5] = np.nan
    df_nan = pd.DataFrame({"concentration": conc, "y": y_b, "replicate": 0})
    df_drop = df_nan.drop(index=5).reset_index(drop=True)

    r_nan = SingleDrugFit(df_nan).fit()
    r_drop = SingleDrugFit(df_drop).fit()

    assert r_nan.n_valid == 12
    assert r_nan.n_total == 13
    assert np.isfinite(r_nan.rss)
    assert r_nan.r2 is not None and np.isfinite(r_nan.r2)

    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf", "rss", "aic", "bic"):
        np.testing.assert_allclose(
            getattr(r_nan, name), getattr(r_drop, name), rtol=1e-9, atol=1e-12,
        )


def test_single_drug_fit_non_finite_conc_on_midpoint_row_matches_drop():
    """IC50 seed row must not use a row with non-finite concentration."""
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(2)
    y_noisy = y + rng.normal(0, 0.03, 13)
    ymin, ymax = _robust_response_extrema(y_noisy)
    ym = ymin + (ymax - ymin) / 2
    midpoint_idx = int(np.argmin(np.abs(y_noisy - ym)))
    conc_bad = conc.copy()
    conc_bad[midpoint_idx] = np.nan
    df_bad = pd.DataFrame({"concentration": conc_bad, "y": y_noisy, "replicate": 0})
    df_drop = pd.DataFrame(
        {
            "concentration": np.delete(conc, midpoint_idx),
            "y": np.delete(y_noisy, midpoint_idx),
            "replicate": 0,
        },
    )
    r_bad = SingleDrugFit(df_bad).fit()
    r_drop = SingleDrugFit(df_drop).fit()
    assert r_bad.success and r_drop.success
    assert r_bad.n_valid == 12
    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf", "rss", "aic", "bic"):
        np.testing.assert_allclose(
            getattr(r_bad, name), getattr(r_drop, name), rtol=1e-9, atol=1e-12,
        )


def test_single_drug_fit_non_finite_concentration_excluded_like_missing_y():
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(1)
    y_noisy = y + rng.normal(0, 0.03, 13)
    conc_bad = conc.copy()
    conc_bad[4] = np.nan
    df_bad = pd.DataFrame({"concentration": conc_bad, "y": y_noisy, "replicate": 0})
    df_drop = pd.DataFrame(
        {"concentration": np.delete(conc, 4), "y": np.delete(y_noisy, 4), "replicate": 0},
    )
    r_bad = SingleDrugFit(df_bad).fit()
    r_drop = SingleDrugFit(df_drop).fit()
    assert r_bad.n_valid == 12
    assert r_bad.n_total == 13
    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf", "rss", "aic", "bic"):
        np.testing.assert_allclose(
            getattr(r_bad, name), getattr(r_drop, name), rtol=1e-9, atol=1e-12,
        )


def test_single_drug_fit_valids_wrong_length_raises():
    conc, y = _case_ab_conc_y()
    df = pd.DataFrame({"concentration": conc, "y": y, "replicate": 0})
    with pytest.raises(ValueError, match="valids must be a 1-D boolean array"):
        SingleDrugFit(df).fit(valids=np.ones(5, dtype=bool))


def test_single_drug_fit_with_error_nan_y_excluded_from_n_valid():
    conc, y = _case_ab_conc_y()
    df = pd.DataFrame({"concentration": conc, "y": y.copy(), "y_err": 0.02, "replicate": 0})
    df.loc[5, "y"] = np.nan
    result = SingleDrugFitWithError(df).fit()
    assert result.n_valid == 12
    assert result.n_total == 13


def test_single_drug_fit_with_error_invalid_y_err_on_excluded_row_ignored():
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(5)
    y_noisy = y + rng.normal(0, 0.03, 13)
    df = pd.DataFrame({"concentration": conc, "y": y_noisy, "y_err": 0.02})
    valids = np.ones(13, dtype=bool)
    valids[7] = False
    df_bad = df.copy()
    df_bad.loc[7, "y_err"] = np.nan
    df_drop = df.loc[valids].reset_index(drop=True)
    r_ok = SingleDrugFitWithError(df).fit(valids=valids)
    r_bad = SingleDrugFitWithError(df_bad).fit(valids=valids)
    r_drop = SingleDrugFitWithError(df_drop).fit()
    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf"):
        np.testing.assert_allclose(
            getattr(r_ok, name), getattr(r_bad, name), rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            getattr(r_bad, name), getattr(r_drop, name), rtol=1e-9, atol=1e-12,
        )


@pytest.mark.parametrize("bad_err", [np.nan, 0.0, -0.01])
def test_single_drug_fit_with_error_invalid_y_err_on_included_row_raises(bad_err):
    conc, y = _case_ab_conc_y()
    df = pd.DataFrame({"concentration": conc, "y": y, "y_err": 0.02})
    df.loc[4, "y_err"] = bad_err
    with pytest.raises(ValueError, match="Invalid y_err on observations included"):
        SingleDrugFitWithError(df).fit()


def test_single_drug_fit_with_error_likelihood_uses_weights():
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(6)
    y_noisy = y + rng.normal(0, 0.03, 13)
    df = pd.DataFrame({"concentration": conc, "y": y_noisy, "y_err": 0.02})
    fitter = SingleDrugFitWithError(df)
    mask = effective_observation_mask(
        None, len(df), df["concentration"].values, df["y"].values,
    )
    x0, bounds = fitter._get_x0_and_bounds()
    x_opt, _, _, _ = fitter._run_minimize(x0, bounds, valids=mask)
    fitter._likelihood_mask = mask
    ll_weighted = fitter._log_prob_data(x_opt, valids=mask)
    kwargs = fitter._x_to_kwargs(x_opt)
    y_pred = hill_curve(conc, **kwargs)
    ll_unweighted = noise_log_prob(
        df["y"].values, y_pred, fitter.config.noise, mask=mask,
    )
    assert ll_weighted != ll_unweighted


def test_single_drug_gaussian_linear_nan_concentration_rows_match_drop():
    conc, y = _case_ab_conc_y()
    rng = np.random.default_rng(7)
    y_noisy = y + rng.normal(0, 0.05, 13)
    df_base = pd.DataFrame({"concentration": conc, "y": y_noisy, "replicate": 0})
    cfg = FitConfig(noise=GaussianLinear())
    extra = pd.DataFrame(
        {"concentration": [np.nan, np.nan], "y": [1e6, 2e6], "replicate": [0, 0]},
    )
    df_bad = pd.concat([df_base, extra], ignore_index=True)
    fit_bad = SingleDrugFit(df_bad, cfg)
    fit_drop = SingleDrugFit(df_base, cfg)
    assert fit_bad._response_scale == pytest.approx(fit_drop._response_scale)
    assert fit_bad.config.bounds.var_a == fit_drop.config.bounds.var_a
    assert fit_bad.config.bounds.var_b == fit_drop.config.bounds.var_b
    assert fit_bad.config.bounds.var_c == fit_drop.config.bounds.var_c
    r_bad = fit_bad.fit()
    r_drop = fit_drop.fit()
    assert r_bad.success and r_drop.success
    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf"):
        np.testing.assert_allclose(
            getattr(r_bad, name), getattr(r_drop, name), rtol=1e-9, atol=1e-12,
        )


def _outlier_dose_response():
    """Issue #24: two saturated wells that pull automatic defaults off the curve."""
    conc = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0, 1000.0])
    # Fixed small noise: on an exact curve the profiled sigma collapses towards
    # zero and L-BFGS-B's convergence flag becomes platform-dependent.
    response = 1.0 / (1.0 + conc) + np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0, 0])
    response[-2:] = 5.0
    frame = pd.DataFrame({"concentration": conc, "y": response, "replicate": 0})
    included = np.array([True] * 6 + [False, False])
    return frame, included


def _assert_curve_matches(masked, dropped):
    assert masked.n_valid == dropped.n_valid
    assert masked.success == dropped.success
    for name in ("c50", "log_c50", "hill", "effect_0", "effect_inf", "response_scale"):
        np.testing.assert_allclose(
            getattr(masked, name), getattr(dropped, name), rtol=1e-9, atol=1e-12,
        )


@pytest.mark.parametrize("fitter_cls", [SingleDrugFit, SingleDrugFitWithError])
def test_valids_outliers_match_dropped_rows(fitter_cls):
    """Rows excluded by fit(valids=...) must not move automatic c50."""
    frame, included = _outlier_dose_response()
    if fitter_cls is SingleDrugFitWithError:
        frame = frame.assign(y_err=0.05)
    dropped = frame.loc[included].reset_index(drop=True)
    masked_fit = fitter_cls(frame)
    dropped_fit = fitter_cls(dropped)
    # Defaults are still resolved at construction, before valids exists.
    assert abs(float(masked_fit.config.log_c50) - float(dropped_fit.config.log_c50)) > 0.1
    result_masked = masked_fit.fit(valids=included)
    result_dropped = dropped_fit.fit()
    _assert_curve_matches(result_masked, result_dropped)
    assert result_masked.success
    assert result_masked.c50 == pytest.approx(1.0, abs=0.05)
    assert masked_fit.config.log_c50 == pytest.approx(dropped_fit.config.log_c50)
    assert masked_fit.config.effect_0 == pytest.approx(dropped_fit.config.effect_0)
    assert masked_fit.config.effect_inf == pytest.approx(dropped_fit.config.effect_inf)
    assert masked_fit.config.bounds == dropped_fit.config.bounds


def test_valids_heteroscedastic_scale_matches_dropped_rows():
    # Linear noise adds two variance parameters, so the included curve needs
    # more than the six wells in the constant-noise reproduction.
    conc = np.array([
        0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1000.0,
    ])
    response = 1.0 / (1.0 + conc) + 0.01 * np.array([1, -1] * 6)
    response[-2:] = 5.0
    frame = pd.DataFrame({"concentration": conc, "y": response, "replicate": 0})
    included = np.array([True] * 10 + [False, False])
    dropped = frame.loc[included].reset_index(drop=True)
    noise = GaussianLinear()
    masked_fit = SingleDrugFit(frame, FitConfig(noise=noise))
    dropped_fit = SingleDrugFit(dropped, FitConfig(noise=noise))
    result_masked = masked_fit.fit(valids=included)
    result_dropped = dropped_fit.fit()
    _assert_curve_matches(result_masked, result_dropped)
    assert masked_fit._response_scale == pytest.approx(dropped_fit._response_scale)
    assert masked_fit.config.noise.a_init == pytest.approx(dropped_fit.config.noise.a_init)
    assert masked_fit.config.bounds.var_a == dropped_fit.config.bounds.var_a
    assert masked_fit.config.bounds.var_b == dropped_fit.config.bounds.var_b
    for key in result_dropped.variance_params:
        assert result_masked.variance_params[key] == pytest.approx(
            result_dropped.variance_params[key], rel=1e-9, abs=1e-12,
        )


def test_unmasked_fit_does_not_rewrite_automatic_config():
    data = _synthetic_data()
    fitter = SingleDrugFit(data)
    before = copy.deepcopy(fitter.config)
    fitter.fit()
    assert fitter.config.log_c50 == before.log_c50
    assert fitter.config.effect_0 == before.effect_0
    assert fitter.config.effect_inf == before.effect_inf
    assert fitter.config.bounds == before.bounds
    assert fitter.config.noise == before.noise


def test_explicit_config_survives_valids_that_would_change_defaults():
    frame, included = _outlier_dose_response()
    caller = FitConfig(
        log_c50=-0.3,
        hill=1.4,
        effect_0=0.8,
        effect_inf=0.2,
        bounds=FitBounds(
            log_c50=(-2.0, 2.0),
            hill=(0.5, 3.0),
            effect_0=(0.4, 1.2),
            effect_inf=(0.0, 0.5),
        ),
    )
    before = copy.deepcopy(caller)
    fitter = SingleDrugFit(frame, caller)
    config_before = copy.deepcopy(fitter.config)
    fitter.fit(valids=included)
    assert caller.log_c50 == before.log_c50
    assert caller.bounds == before.bounds
    assert fitter.config.log_c50 == config_before.log_c50
    assert fitter.config.hill == config_before.hill
    assert fitter.config.effect_0 == config_before.effect_0
    assert fitter.config.effect_inf == config_before.effect_inf
    assert fitter.config.bounds == config_before.bounds


def test_post_construction_edits_survive_valids_and_other_defaults_follow_mask():
    frame, included = _outlier_dose_response()
    dropped = frame.loc[included].reset_index(drop=True)
    fitter = SingleDrugFit(frame)
    fitter.config.log_c50 = -0.7
    fitter.config.bounds.hill = (0.5, 3.0)
    fitter.fit(valids=included)
    dropped_fit = SingleDrugFit(dropped)
    assert fitter.config.log_c50 == -0.7
    assert fitter.config.bounds.hill == (0.5, 3.0)
    # Untouched asymptote bounds still follow the included rows.
    assert fitter.config.bounds.effect_0 == dropped_fit.config.bounds.effect_0
    assert fitter.config.bounds.effect_inf == dropped_fit.config.bounds.effect_inf
    assert fitter.config.effect_0 == pytest.approx(dropped_fit.config.effect_0)
    assert fitter.config.effect_inf == pytest.approx(dropped_fit.config.effect_inf)


def test_with_error_explicit_initial_kept_and_automatic_bounds_follow_mask():
    frame, included = _outlier_dose_response()
    frame = frame.assign(y_err=0.05)
    dropped = frame.loc[included].reset_index(drop=True)
    caller = FitConfig(log_c50=-0.4)
    fitter = SingleDrugFitWithError(frame, caller)
    fitter.fit(valids=included)
    dropped_fit = SingleDrugFitWithError(dropped, FitConfig(log_c50=-0.4))
    assert caller.log_c50 == -0.4
    assert caller.bounds is None
    assert fitter.config.log_c50 == -0.4
    assert fitter.config.bounds == dropped_fit.config.bounds


def test_explicit_variance_initial_not_reclamped_when_valids_changes_scale():
    data = pd.DataFrame({
        "concentration": [0, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000.0],
        "y": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "replicate": 0,
    })
    fitter = SingleDrugFit(
        data, FitConfig(noise=GaussianLinear(a_init=1.5e-12, b_init=0.0)),
    )
    a_init = fitter.config.noise.a_init
    b_init = fitter.config.noise.b_init
    valids = np.ones(len(data), dtype=bool)
    valids[0] = False
    fitter.fit(valids=valids)
    assert fitter.config.noise.a_init == a_init
    assert fitter.config.noise.b_init == b_init
    assert a_init < 4e-12  # the included-row floor would have raised this


def test_later_unmasked_fit_restores_defaults_from_all_finite_rows():
    frame, included = _outlier_dose_response()
    fitter = SingleDrugFit(frame)
    original = fitter.config.log_c50
    fitter.fit(valids=included)
    assert abs(fitter.config.log_c50 - original) > 0.1
    fitter.fit()
    assert fitter.config.log_c50 == pytest.approx(original)


def test_valids_zero_response_range_raises_like_dropped_rows():
    conc = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 1000.0, 1000.0])
    response = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 9.0])
    frame = pd.DataFrame({"concentration": conc, "y": response, "replicate": 0})
    included = np.array([True] * 6 + [False, False])
    with pytest.raises(ValueError, match="too close together"):
        SingleDrugFit(frame).fit(valids=included)
    with pytest.raises(ValueError, match="too close together"):
        SingleDrugFit(frame.loc[included].reset_index(drop=True))
