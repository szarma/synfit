import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import norm as _norm
from synfit.data import FitBounds, FitConfig, FitResult
from synfit.hill import hill_curve
from synfit.single import SingleDrugFit, SingleDrugFitWithError
from synfit.synthetic import generate_single_drug


def _synthetic_data(c50=1.0, hill=1.5, effect_0=0.9, effect_inf=0.05,
                    noise_model="gaussian", noise_sigma=0.02, noise_sigma_log=0.02,
                    seed=42):
    """Generate synthetic dose-response data via the canonical generator."""
    config = {
        "seed": seed,
        "hill_params": {"c50": c50, "hill": hill, "effect_0": effect_0, "effect_inf": effect_inf},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10**(2/7), "length": 8, "has_zero": False},
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


def test_single_drug_fit_with_error():
    data = _synthetic_data(c50=1.5)
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
