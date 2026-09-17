"""Tests for the centralised parameter-role taxonomy.

These predicates are the single source of truth used by the optimiser
preconditioning, curve-kwarg unpacking, and variance handling across the
single-drug, matrix, and joint-marginal fits — so they must agree with the
names those fits actually emit.
"""
from synfit.param_roles import (
    ASYMPTOTE_PARAM_NAMES,
    VARIANCE_PARAM_NAMES,
    is_asymptote_param,
    is_log_param,
    is_variance_param,
)


def test_is_log_param_matches_log_prefixed_names():
    # single-drug / joint per-drug / matrix per-axis log locations
    for name in ("log_c50", "log_c50_a", "log_c50_b", "log_c50_hor", "log_c50_ver"):
        assert is_log_param(name)
    for name in ("hill", "effect_0", "top", "var_a", "asymmetry", "c50"):
        assert not is_log_param(name)


def test_is_variance_param_matches_polynomial_coefficients():
    for name in VARIANCE_PARAM_NAMES:
        assert is_variance_param(name)
    for name in ("log_c50", "hill", "effect_inf", "top", "var_d"):
        assert not is_variance_param(name)


def test_is_asymptote_param_covers_both_naming_conventions():
    # single-drug / matrix use effect_0/effect_inf; joint-marginal uses top/bottom
    for name in ("effect_0", "effect_inf", "top", "bottom"):
        assert is_asymptote_param(name)
    for name in ("log_c50", "hill", "asymmetry", "var_a", "var_b"):
        assert not is_asymptote_param(name)


def test_role_sets_are_disjoint():
    assert not (set(ASYMPTOTE_PARAM_NAMES) & set(VARIANCE_PARAM_NAMES))
    for name in (*ASYMPTOTE_PARAM_NAMES, *VARIANCE_PARAM_NAMES):
        assert not is_log_param(name)


# --- Cross-check against the names the fitters actually emit ----------------
# Pin the predicates to the real parameter-name sources, not a hand-maintained
# list, so adding a parameter to a fit can't silently fall outside the taxonomy.

def _classify(name: str) -> set[str]:
    roles = set()
    if is_log_param(name):
        roles.add("log")
    if is_variance_param(name):
        roles.add("variance")
    if is_asymptote_param(name):
        roles.add("asymptote")
    return roles


def test_single_drug_param_names_each_have_at_most_one_role():
    from synfit.data import FitConfig

    names = FitConfig().fitting_parameters  # log_c50, hill, effect_0, effect_inf
    assert is_log_param("log_c50")
    assert is_asymptote_param("effect_0") and is_asymptote_param("effect_inf")
    # hill is a shape exponent: no log/variance/asymptote role.
    assert _classify("hill") == set()
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"


def test_matrix_param_names_classified_by_taxonomy():
    from synfit.matrix import MatrixFit

    names = MatrixFit._PARAM_NAMES
    asymptotes = {n for n in names if is_asymptote_param(n)}
    logs = {n for n in names if is_log_param(n)}
    assert asymptotes == {"effect_0", "effect_inf"}
    assert logs == {"log_c50_hor", "log_c50_ver"}
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"


def test_joint_marginal_param_names_classified_by_taxonomy():
    from synfit.joint_marginal import JointMarginalFit

    names = JointMarginalFit._full_param_names("4p", "5p")
    asymptotes = {n for n in names if is_asymptote_param(n)}
    logs = {n for n in names if is_log_param(n)}
    assert asymptotes == {"top", "bottom"}  # joint shares asymptotes as top/bottom
    assert logs == {"log_c50_a", "log_c50_b"}
    for name in names:
        assert len(_classify(name)) <= 1, f"{name} has overlapping roles"


# --- Cross-check real fitted parameter names against expected roles -----------

def test_single_drug_4p_params_have_expected_roles():
    """All param names produced by SingleDrugFit (4p) must match their expected roles."""
    from synfit.single import SingleDrugFit
    from synfit.synthetic import generate_single_drug
    
    # Generate small synthetic data for a 4p curve
    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10.0,
            "length": 8, "has_zero": True,
        },
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02,
    }
    df = generate_single_drug(cfg)
    fit = SingleDrugFit(df)
    result = fit.fit()
    
    # Verify every parameter name has the expected role
    assert result.param_names is not None
    for name in result.param_names:
        roles = _classify(name)
        if name == "log_c50":
            assert roles == {"log"}, f"{name} should have log role only"
        elif name in ("effect_0", "effect_inf"):
            assert roles == {"asymptote"}, f"{name} should have asymptote role only"
        elif name == "hill":
            assert roles == set(), f"{name} should have no role"
        else:
            raise AssertionError(f"Unexpected parameter name: {name}")


def test_single_drug_5p_params_have_expected_roles():
    """All param names produced by SingleDrugFit (5p) must match their expected roles."""
    from synfit.data import FitConfig
    from synfit.single import SingleDrugFit
    from synfit.synthetic import generate_single_drug
    
    # Generate small synthetic data for a 5p curve
    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "asymmetry": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10.0,
            "length": 8, "has_zero": True,
        },
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_sigma": 0.02,
    }
    df = generate_single_drug(cfg)
    config = FitConfig(fitting_parameters=["log_c50", "hill", "effect_0", "effect_inf", "asymmetry"])
    fit = SingleDrugFit(df, config)
    result = fit.fit()
    
    # Verify every parameter name has the expected role
    assert result.param_names is not None
    for name in result.param_names:
        roles = _classify(name)
        if name == "log_c50":
            assert roles == {"log"}, f"{name} should have log role only"
        elif name in ("effect_0", "effect_inf"):
            assert roles == {"asymptote"}, f"{name} should have asymptote role only"
        elif name in ("hill", "asymmetry"):
            assert roles == set(), f"{name} should have no role"
        else:
            raise AssertionError(f"Unexpected parameter name: {name}")


def test_single_drug_linear_gaussian_params_include_variance():
    """SingleDrugFit with linear Gaussian noise includes var_a and var_b parameters."""
    from synfit.data import FitConfig
    from synfit.single import SingleDrugFit
    from synfit.synthetic import generate_single_drug
    
    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10.0,
            "length": 8, "has_zero": True,
        },
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_var_a": 0.001, "noise_var_b": 0.05,
    }
    df = generate_single_drug(cfg)
    config = FitConfig(noise="gaussian_linear")
    fit = SingleDrugFit(df, config)
    result = fit.fit()
    
    assert result.param_names is not None
    variance_params = {n for n in result.param_names if is_variance_param(n)}
    assert "var_a" in variance_params, "var_a should be in linear Gaussian fit"
    assert "var_b" in variance_params, "var_b should be in linear Gaussian fit"
    assert "var_c" not in variance_params, "var_c should not be in linear Gaussian fit"
    
    for name in result.param_names:
        roles = _classify(name)
        if name in ("var_a", "var_b"):
            assert roles == {"variance"}, f"{name} should have variance role only"
        elif name in ("log_c50", "effect_0", "effect_inf"):
            assert roles != {"variance"}, f"{name} should not have variance role"


def test_single_drug_quadratic_gaussian_params_include_variance():
    """SingleDrugFit with quadratic Gaussian noise includes var_a, var_b, and var_c."""
    from synfit.data import FitConfig
    from synfit.single import SingleDrugFit
    from synfit.synthetic import generate_single_drug
    
    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10.0,
            "length": 8, "has_zero": True,
        },
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_var_a": 0.001, "noise_var_b": 0.05, "noise_var_c": 0.001,
    }
    df = generate_single_drug(cfg)
    config = FitConfig(noise="gaussian_quadratic")
    fit = SingleDrugFit(df, config)
    result = fit.fit()
    
    assert result.param_names is not None
    variance_params = {n for n in result.param_names if is_variance_param(n)}
    assert "var_a" in variance_params, "var_a should be in quadratic Gaussian fit"
    assert "var_b" in variance_params, "var_b should be in quadratic Gaussian fit"
    assert "var_c" in variance_params, "var_c should be in quadratic Gaussian fit"
    
    for name in result.param_names:
        if is_variance_param(name):
            roles = _classify(name)
            assert roles == {"variance"}, f"{name} should have variance role only"


def test_joint_marginal_params_have_expected_roles():
    """All param names produced by JointMarginalFit must match their expected roles."""
    from synfit.data import FitConfig
    from synfit.joint_marginal import JointMarginalFit
    from synfit.synthetic import generate_single_drug
    
    # Generate small synthetic data for two drugs
    cfg = {
        "seed": 42,
        "hill_params": {"c50": 5.0, "hill": 1.5, "effect_0": 1.0, "effect_inf": 0.05},
        "concentration_series": {
            "initial_conc": 100.0, "fold_dilutions": 10.0,
            "length": 8, "has_zero": True,
        },
        "n_replicates": 3,
        "noise_model": "gaussian", "noise_var_a": 0.001, "noise_var_b": 0.05, "noise_var_c": 0.001,
    }
    data_a = generate_single_drug(cfg)
    data_b = generate_single_drug({**cfg, "seed": 43})
    
    config = FitConfig(noise="gaussian_quadratic")
    fitter = JointMarginalFit(
        data_a, data_b, 
        model_a="4p", model_b="4p",
        noise=config.noise,
    )
    result = fitter.fit()
    
    assert result.param_names is not None
    for name in result.param_names:
        roles = _classify(name)
        if is_log_param(name):
            assert roles == {"log"}, f"{name} should have log role only"
        elif is_variance_param(name):
            assert roles == {"variance"}, f"{name} should have variance role only"
        elif is_asymptote_param(name):
            assert roles == {"asymptote"}, f"{name} should have asymptote role only"
        elif name in ("hill_a", "hill_b"):
            assert roles == set(), f"{name} should have no role"
        else:
            raise AssertionError(f"Unexpected parameter name: {name}")


def test_matrix_fit_params_have_expected_roles():
    """All param names produced by MatrixFit must match their expected roles."""
    from synfit.matrix import MatrixFit
    from tests.helpers import matrix_from_config
    
    replicates, conc_h, conc_v = matrix_from_config(noise_model="gaussian")
    fit = MatrixFit(replicates, conc_h, conc_v, noise="gaussian_constant")
    result = fit.fit()
    
    assert result.param_names is not None
    for name in result.param_names:
        roles = _classify(name)
        if is_log_param(name):
            assert roles == {"log"}, f"{name} should have log role only"
        elif is_asymptote_param(name):
            assert roles == {"asymptote"}, f"{name} should have asymptote role only"
        elif name in ("hill_hor", "hill_ver"):
            assert roles == set(), f"{name} should have no role"
        else:
            raise AssertionError(f"Unexpected parameter name: {name}")
