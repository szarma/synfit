"""Tests for JointMarginalFit — two drugs with shared top/bottom."""
import numpy as np
import pandas as pd
import pytest

from synfit.joint_marginal import JointMarginalFit, fit_joint_marginal_auto, default_joint_marginal_config
from synfit.noise import CompoundAddMult, GaussianLinear, GaussianQuadratic
from synfit.synthetic import generate_single_drug
from tests.helpers import matrix_from_config


TOP = 1.0
BOTTOM = 0.05


@pytest.mark.parametrize("noise", ["gaussian_constant", "lognormal"])
@pytest.mark.parametrize("zero_last", [False, True])
def test_from_matrix_matches_fit_of_distinct_edge_wells(noise, zero_last):
    reps, ch, cv = matrix_from_config(synergy_factor=1.5)
    # A rectangular matrix also makes swapped axes visible.
    cv = cv[:-1]
    reps = np.asarray(reps)[:, :-1, :]
    if zero_last:
        ch, cv = ch[::-1], cv[::-1]
        reps = reps[:, ::-1, ::-1]
    zero_h = list(ch).index(0)
    zero_v = list(cv).index(0)
    data_a = pd.DataFrame({
        "concentration": np.tile(ch, len(reps)),
        "y": reps[:, zero_v, :].ravel(),
        "replicate": np.repeat(np.arange(len(reps)), len(ch)),
    })
    data_b = pd.DataFrame({
        "concentration": np.tile(cv[cv > 0], len(reps)),
        "y": reps[:, :, zero_h][:, cv > 0].ravel(),
        "replicate": np.repeat(np.arange(len(reps)), len(cv) - 1),
    })
    expected = JointMarginalFit(data_a, data_b, noise=noise).fit()
    fitter = JointMarginalFit.from_matrix(reps, ch, cv, noise=noise)
    result = fitter.fit()

    assert result.success and expected.success
    assert len(fitter.data_a) == len(reps) * len(ch)
    assert len(fitter.data_b) == len(reps) * len(cv)
    assert result.n_data == len(reps) * (len(ch) + len(cv) - 1)
    assert result.n_data == expected.n_data
    assert result.log_likelihood == pytest.approx(expected.log_likelihood, abs=1e-7)
    assert result.aic == pytest.approx(expected.aic, abs=1e-7)
    assert result.sigma == pytest.approx(expected.sigma, rel=1e-5)
    assert result.drug_a.c50 == pytest.approx(expected.drug_a.c50, rel=1e-5)
    assert result.drug_b.c50 == pytest.approx(expected.drug_b.c50, rel=1e-5)


def test_from_matrix_exclusions_apply_to_wells_without_mutating_the_mask():
    reps, ch, cv = matrix_from_config()
    reps = np.asarray(reps)
    valids = np.ones(reps.shape, dtype=bool)
    valids[0, 0, 0] = False  # shared control
    valids[1, 0, 2] = False  # horizontal edge
    valids[2, 3, 0] = False  # vertical edge
    valids[0, 2, 2] = False  # interior: never enters the marginal fit
    original = valids.copy()
    masked = JointMarginalFit.from_matrix(reps, ch, cv, valids=valids).fit()
    missing = reps.copy()
    missing[~valids] = np.nan
    missing[:, 1:, 1:] = 1000  # combination responses must not affect this fit
    omitted = JointMarginalFit.from_matrix(missing, ch, cv).fit()

    np.testing.assert_array_equal(valids, original)
    assert masked.success and omitted.success
    assert masked.n_data == len(reps) * (len(ch) + len(cv) - 1) - 3
    assert masked.n_data == omitted.n_data
    assert masked.aic == pytest.approx(omitted.aic, abs=1e-8)


def test_from_matrix_requires_single_agent_edges_and_matching_shapes():
    reps, ch, cv = matrix_from_config()
    with pytest.raises(ValueError, match="zero dose"):
        JointMarginalFit.from_matrix(reps, ch + 1, cv)
    with pytest.raises(ValueError, match="zero dose"):
        JointMarginalFit.from_matrix(reps, np.zeros_like(ch), cv)
    with pytest.raises(ValueError, match="replicates must have shape"):
        JointMarginalFit.from_matrix(np.asarray(reps)[:, :-1, :], ch, cv)
    with pytest.raises(ValueError, match="valids must have the same shape"):
        JointMarginalFit.from_matrix(reps, ch, cv, valids=np.ones((len(cv), len(ch))))


def _two_drug_data(
    c50_a=2.0, hill_a=1.5,
    c50_b=10.0, hill_b=1.0,
    noise_model="gaussian", noise_sigma=0.02,
    seed_a=11, seed_b=22,
):
    """Generate two synthetic single-drug curves that share top/bottom."""
    def cfg(c50, hill, seed):
        return {
            "seed": seed,
            "hill_params": {"c50": c50, "hill": hill, "effect_0": TOP, "effect_inf": BOTTOM},
            "concentration_series": {
                "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
                "length": 8, "has_zero": False,
            },
            "n_replicates": 3,
            "noise_model": noise_model,
            "noise_sigma": noise_sigma,
            "noise_sigma_log": noise_sigma,
        }

    return (
        generate_single_drug(cfg(c50_a, hill_a, seed_a)),
        generate_single_drug(cfg(c50_b, hill_b, seed_b)),
    )


def test_joint_fit_recovers_shared_top_bottom_4p():
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(data_a, data_b)
    result = fitter.fit()

    assert result.success
    assert result.top == pytest.approx(TOP, rel=0.1)
    assert result.bottom == pytest.approx(BOTTOM, abs=0.05)
    # Per-drug params recovered
    assert result.drug_a.c50 == pytest.approx(2.0, rel=0.3)
    assert result.drug_b.c50 == pytest.approx(10.0, rel=0.3)
    # Shared asymptotes identical across drug objects
    assert result.drug_a.effect_0 == result.drug_b.effect_0
    assert result.drug_a.effect_inf == result.drug_b.effect_inf
    # 4p: no asymmetry
    assert result.drug_a.asymmetry is None
    assert result.drug_b.asymmetry is None


def test_joint_fit_mixed_4p_5p():
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(data_a, data_b, model_a="4p", model_b="5p")
    result = fitter.fit()

    assert result.success
    assert result.drug_a.asymmetry is None
    assert result.drug_b.asymmetry is not None
    # Shared top/bottom still in a reasonable neighbourhood
    assert result.top == pytest.approx(TOP, rel=0.15)
    assert result.bottom == pytest.approx(BOTTOM, abs=0.1)
    # 4p drug: 4 free params (log_c50_a, hill_a). 5p drug: 3 (log_c50_b, hill_b, asym_b).
    # Plus 2 shared → 7 params total.
    assert len(result.param_names) == 7
    assert "asymmetry_b" in result.param_names
    assert "asymmetry_a" not in result.param_names


def test_joint_fit_covariance_shape():
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(data_a, data_b)
    result = fitter.fit()

    # 4p + 4p: 2 shared + 2 + 2 = 6 params
    assert len(result.param_names) == 6
    if result.param_cov is not None:
        assert result.param_cov.shape == (6, 6)


def test_joint_fit_populates_shared_sigma():
    """σ̂ is pooled across both drugs and stamped onto each per-drug FitResult."""
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(data_a, data_b)
    result = fitter.fit()

    assert result.sigma is not None
    assert result.sigma > 0
    # The two per-drug FitResults share the same σ̂ since they share a plate.
    assert result.drug_a.sigma == result.sigma
    assert result.drug_b.sigma == result.sigma
    # Surfaced through to_dict for DB persistence.
    assert result.to_dict()["sigma"] == result.sigma


def test_joint_fit_shared_top_differs_from_independent_fits():
    """Sanity check: when the two drugs have different noise realisations,
    sharing top/bottom yields a single pair of asymptotes (not two)."""
    data_a, data_b = _two_drug_data(seed_a=1, seed_b=999)
    fitter = JointMarginalFit(data_a, data_b)
    result = fitter.fit()

    assert result.drug_a.effect_0 == result.drug_b.effect_0
    assert result.drug_a.effect_inf == result.drug_b.effect_inf


def test_joint_fit_validates_columns():
    import pandas as pd
    bad = pd.DataFrame({"concentration": [1.0], "y": [0.5]})  # missing replicate
    good = generate_single_drug({
        "seed": 0,
        "hill_params": {"c50": 1.0, "hill": 1.0, "effect_0": 1.0, "effect_inf": 0.0},
        "concentration_series": {"initial_conc": 10.0, "fold_dilutions": 3.0, "length": 5, "has_zero": False},
        "n_replicates": 2,
        "noise_model": "gaussian", "noise_sigma": 0.01,
    })
    with pytest.raises(ValueError, match="columns"):
        JointMarginalFit(bad, good)


def test_joint_fit_mixed_directions_activator_and_inhibitor():
    """Mixed activator+inhibitor pair shares plate-level top/bottom.

    Build drug_a (inhibitor: response falls from TOP to BOTTOM) and drug_b
    (activator: response rises from BOTTOM to TOP). Physically both drugs
    live on the same plate, so their high-signal asymptote and low-signal
    asymptote are the same numbers. The fit should recover TOP and BOTTOM
    correctly, and each drug's effect_0/effect_inf should swap accordingly.
    """
    # Inhibitor: effect_0 = TOP (hi), effect_inf = BOTTOM (lo)
    data_a = generate_single_drug({
        "seed": 31,
        "hill_params": {"c50": 2.0, "hill": 1.5, "effect_0": TOP, "effect_inf": BOTTOM},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7), "length": 8, "has_zero": False},
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02, "noise_sigma_log": 0.02,
    })
    # Activator: effect_0 = BOTTOM (lo), effect_inf = TOP (hi)
    data_b = generate_single_drug({
        "seed": 41,
        "hill_params": {"c50": 10.0, "hill": 1.0, "effect_0": BOTTOM, "effect_inf": TOP},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7), "length": 8, "has_zero": False},
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02, "noise_sigma_log": 0.02,
    })

    fitter = JointMarginalFit(
        data_a, data_b,
        direction_a="inhibition", direction_b="activation",
    )
    result = fitter.fit()

    assert result.success
    assert result.top == pytest.approx(TOP, rel=0.1)
    assert result.bottom == pytest.approx(BOTTOM, abs=0.05)
    # Inhibitor: effect_0 = top (high), effect_inf = bottom (low)
    assert result.drug_a.effect_0 == result.top
    assert result.drug_a.effect_inf == result.bottom
    # Activator: effect_0 = bottom (low), effect_inf = top (high)
    assert result.drug_b.effect_0 == result.bottom
    assert result.drug_b.effect_inf == result.top
    # C50 recovery still sane
    assert result.drug_a.c50 == pytest.approx(2.0, rel=0.3)
    assert result.drug_b.c50 == pytest.approx(10.0, rel=0.3)


def test_joint_fit_honours_fixed_top():
    """Fixing ``top`` at its known value should reduce the optimiser's free
    parameters by one and leave ``top`` exactly where we pinned it."""
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(
        data_a, data_b,
        param_config={"top": {"init": TOP, "fit": False}},
    )
    result = fitter.fit()

    assert result.success
    # Fixed params do not appear in the optimised param space.
    assert "top" not in result.param_names
    # And ``top`` came back as exactly what we pinned.
    assert result.top == pytest.approx(TOP, abs=1e-9)
    # Other params still recovered within tolerance.
    assert result.bottom == pytest.approx(BOTTOM, abs=0.05)
    assert result.drug_a.c50 == pytest.approx(2.0, rel=0.3)
    assert result.drug_b.c50 == pytest.approx(10.0, rel=0.3)


def test_joint_fit_honours_fixed_per_drug_hill():
    """Fixing one drug's hill leaves the other drugs' params free."""
    data_a, data_b = _two_drug_data(hill_a=1.5)
    fitter = JointMarginalFit(
        data_a, data_b,
        param_config={"hill_a": {"init": 1.5, "fit": False}},
    )
    result = fitter.fit()

    assert result.success
    assert "hill_a" not in result.param_names
    assert "hill_b" in result.param_names
    assert result.drug_a.hill == pytest.approx(1.5, abs=1e-9)


def test_joint_fit_all_fixed_raises():
    data_a, data_b = _two_drug_data()
    full_fixed = {
        name: {"fit": False}
        for name in ("top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b")
    }
    with pytest.raises(ValueError, match="fit=True"):
        JointMarginalFit(data_a, data_b, param_config=full_fixed)


def test_joint_fit_auto_picks_lognormal_for_multiplicative_noise():
    """Auto error-model should prefer lognormal when data has multiplicative noise.

    Needs enough heteroscedasticity for the Jacobian-corrected AIC comparison
    to favour lognormal; at very low σ both models fit similarly well.
    """
    data_a, data_b = _two_drug_data(noise_model="lognormal", noise_sigma=0.2)
    result = fit_joint_marginal_auto(data_a, data_b)

    assert result.success
    assert result.error_model == "lognormal"
    # Under heavy lognormal noise (σ_log=0.2) point estimates drift — we're
    # only asserting that the model-selection logic picked lognormal, not
    # that parameter recovery is tight.


def test_joint_fit_auto_picks_gaussian_for_additive_noise():
    data_a, data_b = _two_drug_data(noise_model="gaussian", noise_sigma=0.02)
    result = fit_joint_marginal_auto(data_a, data_b)

    assert result.success
    assert result.error_model == "gaussian"


def test_joint_fit_aic_populated():
    data_a, data_b = _two_drug_data()
    fitter = JointMarginalFit(data_a, data_b)
    result = fitter.fit()

    assert result.aic is not None
    assert result.log_likelihood is not None
    assert result.n_params == 6  # 4p/4p: top, bottom, log_c50_a, hill_a, log_c50_b, hill_b
    assert result.aic == pytest.approx(-2 * result.log_likelihood + 2 * 6)


def test_joint_fit_rejects_bad_model():
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match="4p"):
        JointMarginalFit(data_a, data_b, model_a="6p")


# ───────────────────── Variance models ──────────────────────


def _two_drug_hetero_data(a: float, b: float = 0.0, c: float = 0.0):
    """Generate two-drug data with σ²(μ) = a + b·μ + c·μ²."""
    def cfg(c50, hill, seed):
        return {
            "seed": seed,
            "hill_params": {"c50": c50, "hill": hill, "effect_0": TOP, "effect_inf": BOTTOM},
            "concentration_series": {
                "initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7),
                "length": 10, "has_zero": False,
            },
            "n_replicates": 6,
            "noise_model": "gaussian",
            "noise_var_a": a,
            "noise_var_b": b,
            "noise_var_c": c,
        }
    return (
        generate_single_drug(cfg(2.0, 1.5, seed=11)),
        generate_single_drug(cfg(10.0, 1.0, seed=22)),
    )


def test_joint_fit_constant_variance_default_unchanged():
    """variance_model defaults to constant — sigma populated, variance_params None."""
    data_a, data_b = _two_drug_data()
    result = JointMarginalFit(data_a, data_b).fit()
    assert result.variance_model == "constant"
    assert result.variance_params is None
    assert result.sigma is not None
    # Drug-level fields mirror the joint result.
    assert result.drug_a.variance_model == "constant"
    assert result.drug_a.variance_params is None
    assert result.drug_a.sigma == pytest.approx(result.sigma)


def test_joint_fit_linear_variance_recovers_coefficients():
    """Linear σ²(μ) = a + b·μ should be recovered (loosely) on synthetic data."""
    a_true, b_true = 1e-4, 0.02
    data_a, data_b = _two_drug_hetero_data(a=a_true, b=b_true)
    result = JointMarginalFit(
        data_a, data_b, variance_model="linear",
    ).fit()

    assert result.success
    assert result.variance_model == "linear"
    assert result.variance_params is not None
    assert "a" in result.variance_params and "b" in result.variance_params
    assert "c" not in result.variance_params
    # Loose tolerance — single-experiment recovery on small N.
    assert result.variance_params["b"] == pytest.approx(b_true, abs=0.04)
    # σ̂ is None for non-constant — variance_params is the answer.
    assert result.sigma is None
    assert result.drug_a.sigma is None
    assert result.drug_a.variance_params == result.variance_params


def test_joint_fit_quadratic_variance_recovers_coefficients():
    c_true = 0.04  # constant CV ≈ 0.2
    data_a, data_b = _two_drug_hetero_data(a=0.0, b=0.0, c=c_true)
    result = JointMarginalFit(
        data_a, data_b, variance_model="quadratic",
    ).fit()

    assert result.success
    assert result.variance_model == "quadratic"
    assert set(result.variance_params.keys()) == {"a", "b", "c"}
    # Polynomial coefficients on a single small dataset are noisy — sanity-
    # check the order of magnitude rather than tight tolerance.
    assert result.variance_params["c"] > 0
    assert result.variance_params["c"] == pytest.approx(c_true, abs=0.15)


def test_joint_fit_lognormal_with_non_constant_rejected():
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match="gaussian"):
        JointMarginalFit(
            data_a, data_b, error_model="lognormal", variance_model="linear",
        )


def test_joint_fit_auto_skips_lognormal_under_non_constant_variance():
    """Auto path must not attempt lognormal when variance_model is non-constant."""
    data_a, data_b = _two_drug_hetero_data(a=1e-4, b=0.02)
    result = fit_joint_marginal_auto(data_a, data_b, variance_model="linear")
    assert result.error_model == "gaussian"
    assert result.variance_model == "linear"


def test_joint_fit_to_dict_exposes_shared_and_per_drug_cis():
    """to_dict surfaces top/bottom CIs at top level and c50/hill CIs per drug.

    Drug-level FitResults don't carry param_cov of their own, so without this
    derivation the matrix params table would render empty 95% CI columns.
    """
    data_a, data_b = _two_drug_data()
    result = JointMarginalFit(data_a, data_b).fit()
    d = result.to_dict()

    # Shared CIs at top level.
    assert "ci" in d
    assert "top" in d["ci"] and "bottom" in d["ci"]
    top_lo, top_hi = d["ci"]["top"]
    assert top_lo < result.top < top_hi
    # Top/bottom should NOT bleed into the per-drug ci dicts.
    assert "top" not in d["horizontal"]["ci"]
    assert "bottom" not in d["horizontal"]["ci"]

    # Per-drug CIs derived from the joint covariance.
    h_ci = d["horizontal"]["ci"]
    for k in ("c50", "log_c50", "hill"):
        assert k in h_ci
        lo, hi = h_ci[k]
        assert lo < hi
    # c50 is consistent with log_c50 via the monotonic transform.
    log_lo, log_hi = h_ci["log_c50"]
    c_lo, c_hi = h_ci["c50"]
    assert c_lo == pytest.approx(10 ** log_lo, rel=1e-3)
    assert c_hi == pytest.approx(10 ** log_hi, rel=1e-3)


def test_joint_fit_to_dict_serialises_variance_fields():
    data_a, data_b = _two_drug_hetero_data(a=1e-4, b=0.02)
    result = JointMarginalFit(
        data_a, data_b, variance_model="linear",
    ).fit()
    d = result.to_dict()
    assert d["variance_model"] == "linear"
    assert d["variance_params"]["b"] == pytest.approx(result.variance_params["b"])
    assert d["sigma"] is None


# ---------------------------------------------------------------------------
# default_joint_marginal_config — public helper that mirrors what the fitter
# uses when no param_config overrides are provided.
# ---------------------------------------------------------------------------

def _large_magnitude_data(seed_a=50, seed_b=60):
    """Two synthetic curves with asymptotes around 5000/50 (large-magnitude)."""
    def cfg(seed):
        return {
            "seed": seed,
            "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 5000.0, "effect_inf": 50.0},
            "concentration_series": {
                "initial_conc": 100.0,
                "fold_dilutions": 10 ** (2 / 7),
                "length": 8,
                "has_zero": False,
            },
            "n_replicates": 3,
            "noise_model": "gaussian",
            "noise_sigma": 10.0,
        }
    return generate_single_drug(cfg(seed_a)), generate_single_drug(cfg(seed_b))


def test_default_joint_marginal_config_large_magnitude_bounds():
    """Asymptote bounds must track data scale, not be pinned near [0, 2]."""
    data_a, data_b = _large_magnitude_data()
    result = default_joint_marginal_config(data_a, data_b)
    # Top reaches the ~5000 scale. Top and bottom no longer share one wide
    # range — the bottom's upper bound is capped near the per-curve midpoint, so
    # it tracks the data scale (≫ the old [0, 2]) without climbing to the top.
    assert result["top"]["hi"] > 4000
    assert result["bottom"]["hi"] > 1000
    assert result["bottom"]["hi"] < result["top"]["hi"]


def test_default_joint_marginal_config_lognormal_floors_positive():
    """Lognormal noise must floor top/bottom lo strictly > 0."""
    data_a, data_b = _large_magnitude_data()
    result = default_joint_marginal_config(data_a, data_b, noise="lognormal")
    assert result["top"]["lo"] > 0.0
    assert result["bottom"]["lo"] > 0.0


def test_default_joint_marginal_config_4p_omits_asymmetry():
    """4p models must not include asymmetry keys."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, model_a="4p", model_b="4p")
    assert "asymmetry_a" not in result
    assert "asymmetry_b" not in result
    # Core keys are always present.
    for key in ("top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b"):
        assert key in result


def test_default_joint_marginal_config_5p_includes_asymmetry():
    """5p models must include asymmetry keys."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, model_a="5p", model_b="5p")
    assert "asymmetry_a" in result
    assert "asymmetry_b" in result


def test_default_joint_marginal_config_mixed_models():
    """model_a=5p, model_b=4p → asymmetry_a present, asymmetry_b absent."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, model_a="5p", model_b="4p")
    assert "asymmetry_a" in result
    assert "asymmetry_b" not in result


def test_default_joint_marginal_config_mixed_directions():
    """Mixed inhibition/activation pair must not crash and returns all core keys."""
    # Inhibitor: effect_0 = TOP, effect_inf = BOTTOM
    data_a = generate_single_drug({
        "seed": 71,
        "hill_params": {"c50": 3.0, "hill": 1.5, "effect_0": TOP, "effect_inf": BOTTOM},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7), "length": 8, "has_zero": False},
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02,
    })
    # Activator: effect_0 = BOTTOM, effect_inf = TOP
    data_b = generate_single_drug({
        "seed": 81,
        "hill_params": {"c50": 8.0, "hill": 1.0, "effect_0": BOTTOM, "effect_inf": TOP},
        "concentration_series": {"initial_conc": 100.0, "fold_dilutions": 10 ** (2 / 7), "length": 8, "has_zero": False},
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02,
    })
    result = default_joint_marginal_config(
        data_a, data_b,
        direction_a="inhibition", direction_b="activation",
    )
    for key in ("top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b"):
        assert key in result


def test_default_joint_marginal_config_entry_shape():
    """Every returned entry must have exactly init/lo/hi and lo <= hi."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, model_a="5p", model_b="5p")
    for name, entry in result.items():
        assert set(entry.keys()) == {"init", "lo", "hi"}, f"bad keys for {name}"
        assert entry["lo"] <= entry["hi"], f"lo > hi for {name}"


def test_default_joint_marginal_config_invalid_model():
    """Invalid model strings must raise ValueError."""
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match="4p.*5p|5p.*4p"):
        default_joint_marginal_config(data_a, data_b, model_a="3p")


_CURVE_KEYS = {"top", "bottom", "log_c50_a", "hill_a", "log_c50_b", "hill_b"}


@pytest.mark.parametrize(
    "noise, extra_keys",
    [
        ("gaussian_constant", ()),
        ("gaussian_linear", ("var_a", "var_b")),
        ("gaussian_quadratic", ("var_a", "var_b", "var_c")),
        ("lognormal", ()),
        ({"kind": "gaussian_linear"}, ("var_a", "var_b")),
        ({"kind": "gaussian_quadratic"}, ("var_a", "var_b", "var_c")),
        (GaussianLinear(), ("var_a", "var_b")),
        (GaussianQuadratic(), ("var_a", "var_b", "var_c")),
    ],
)
def test_default_joint_marginal_config_variance_coefficients_match_fitter(noise, extra_keys):
    """Returned keys and init/lo/hi match JointMarginalFit with no param_config."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, noise=noise)
    assert set(result) == _CURVE_KEYS | set(extra_keys)

    fitter = JointMarginalFit(data_a, data_b, noise=noise)
    assert set(result) == set(fitter._param_names)
    for name, init, (lo, hi) in zip(fitter._param_names, fitter._x0, fitter._bounds):
        assert result[name]["init"] == init
        assert result[name]["lo"] == lo
        assert result[name]["hi"] == hi


def test_default_joint_marginal_config_noise_none_matches_fitter_default():
    """noise=None matches JointMarginalFit with no noise argument; no var_* keys."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, noise=None)
    fitter = JointMarginalFit(data_a, data_b)
    assert set(result) == set(fitter._param_names) == _CURVE_KEYS
    assert not any(name.startswith("var_") for name in result)
    for name, init, (lo, hi) in zip(fitter._param_names, fitter._x0, fitter._bounds):
        assert result[name]["init"] == init
        assert result[name]["lo"] == lo
        assert result[name]["hi"] == hi


@pytest.mark.parametrize(
    "noise, var_inits",
    [
        (
            {"kind": "gaussian_linear", "a_init": 0.25, "b_init": 0.08},
            {"var_a": 0.25, "var_b": 0.08},
        ),
        (
            {"kind": "gaussian_quadratic", "a_init": 0.4, "b_init": 0.05, "c_init": 0.02},
            {"var_a": 0.4, "var_b": 0.05, "var_c": 0.02},
        ),
    ],
)
def test_default_joint_marginal_config_honours_nondefault_variance_inits(noise, var_inits):
    """Tagged-dict a_init/b_init/c_init must round-trip into returned var_* inits."""
    data_a, data_b = _two_drug_data()
    result = default_joint_marginal_config(data_a, data_b, noise=noise)
    fitter = JointMarginalFit(data_a, data_b, noise=noise)
    for name, value in var_inits.items():
        assert result[name]["init"] == value
        ix = fitter._param_names.index(name)
        assert fitter._x0[ix] == value
        assert result[name]["init"] == fitter._x0[ix]
        assert result[name]["lo"] == fitter._bounds[ix][0]
        assert result[name]["hi"] == fitter._bounds[ix][1]


_COMPOUND_NOISE_FORMS = (
    "compound_add_mult",
    {"kind": "compound_add_mult"},
    CompoundAddMult(),
)


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_default_joint_marginal_config_rejects_compound_noise(noise):
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        default_joint_marginal_config(data_a, data_b, noise=noise)


@pytest.mark.parametrize("noise", _COMPOUND_NOISE_FORMS)
def test_joint_marginal_fit_rejects_compound_noise(noise):
    data_a, data_b = _two_drug_data()
    with pytest.raises(ValueError, match=r"constant/linear/quadratic gaussian and lognormal"):
        JointMarginalFit(data_a, data_b, noise=noise)


def test_joint_marginal_nan_response_excluded_from_likelihood_and_counts():
    data_a, data_b = _two_drug_data()
    data_a = data_a.copy()
    drop_idx = 4
    data_a.loc[drop_idx, "y"] = np.nan
    data_a_drop = data_a.drop(index=drop_idx).reset_index(drop=True)

    r_nan = JointMarginalFit(data_a, data_b).fit()
    r_drop = JointMarginalFit(data_a_drop, data_b).fit()

    assert r_nan.drug_a.n_valid == len(data_a) - 1
    assert r_nan.drug_a.n_total == len(data_a)
    assert r_nan.n_data == r_drop.n_data
    assert r_nan.aic == pytest.approx(r_drop.aic, rel=1e-9, abs=1e-12)
    assert r_nan.drug_a.c50 == pytest.approx(r_drop.drug_a.c50, rel=1e-9, abs=1e-12)


def test_joint_marginal_valids_mask_excluded_like_nan():
    data_a, data_b = _two_drug_data()
    mask_idx = 3
    valids_a = np.ones(len(data_a), dtype=bool)
    valids_a[mask_idx] = False
    data_a_drop = data_a.drop(index=mask_idx).reset_index(drop=True)

    r_mask = JointMarginalFit(data_a, data_b, valids_a=valids_a).fit()
    r_drop = JointMarginalFit(data_a_drop, data_b).fit()

    assert r_mask.drug_a.n_valid == len(data_a) - 1
    assert r_mask.aic == pytest.approx(r_drop.aic, rel=1e-9, abs=1e-12)
