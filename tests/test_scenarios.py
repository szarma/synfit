"""The shipped scenarios must be reachable from an installed package.

These assertions are what downstream consumers rely on: the files travel
with the wheel, and the accessors find them without knowing where the
package landed on disk.
"""
import numpy as np
import pytest

from synfit.bliss import bliss_deviation, bliss_reference
from synfit.hill import hill_curve
from synfit.loewe import loewe_ci
from synfit.synthetic import (
    SCENARIOS_DIR,
    generate_matrix,
    generate_single_drug,
    list_scenarios,
    scenario_config,
    scenario_dir,
)


def test_scenarios_dir_is_inside_the_package():
    import synfit

    assert SCENARIOS_DIR.is_dir()
    assert SCENARIOS_DIR.parent.samefile(synfit.__path__[0])


def test_list_scenarios_finds_the_shipped_set():
    names = list_scenarios()
    assert len(names) >= 20
    assert names == sorted(names)
    for expected in (
        "single_drug_clean",
        "single_drug_activation",
        "matrix_independent",
        "matrix_loewe_sham",
        "matrix_synergy",
    ):
        assert expected in names


@pytest.mark.parametrize("name", list_scenarios())
def test_every_scenario_has_a_config_and_its_csvs(name):
    path = scenario_dir(name)
    config = scenario_config(name)
    assert config["kind"] in ("single_drug", "matrix")
    csvs = sorted(p.name for p in path.glob("*.csv"))
    assert csvs, f"{name} ships no CSV"
    if config["kind"] == "matrix":
        assert all(n.startswith("rep") for n in csvs)
    else:
        assert csvs == [f"{name}.csv"]


def test_unknown_scenario_raises_keyerror():
    with pytest.raises(KeyError):
        scenario_dir("no_such_scenario")
    with pytest.raises(KeyError):
        scenario_config("no_such_scenario")


@pytest.mark.parametrize("name", list_scenarios())
def test_every_config_still_drives_its_generator(name):
    """A scenario whose config no longer generates is a broken scenario."""
    config = scenario_config(name)
    if config["kind"] == "single_drug":
        df = generate_single_drug(config)
        assert not df.empty
        assert {"concentration", "y", "replicate"} <= set(df.columns)
    else:
        reps, conc_hor, conc_ver = generate_matrix(config)
        assert reps
        assert all(r.shape == (conc_ver.size, conc_hor.size) for r in reps)


def _noise_free(config: dict, **overrides) -> tuple:
    cfg = dict(config, noise_model="gaussian", noise_var_a=0.0, noise_var_b=0.0,
               noise_var_c=0.0, n_replicates=1, **overrides)
    reps, conc_hor, conc_ver = generate_matrix(cfg)
    return reps[0], conc_hor, conc_ver


@pytest.mark.parametrize("reference_model", ["bliss", "loewe"])
@pytest.mark.parametrize("synergy_factor", [-0.5, 0.3, 0.8])
def test_synergy_injection_leaves_single_drug_edges_on_their_hill_curves(
    reference_model, synergy_factor
):
    """The configured c50 / hill must be the true marginals at any synergy.

    Regression: the injection used to be a function of the reference level
    alone, which also shifted the zero-concentration row and column.
    """
    config = scenario_config("matrix_synergy")
    surface, conc_hor, conc_ver = _noise_free(
        config, reference_model=reference_model, synergy_factor=synergy_factor,
    )
    e0, einf = config["effect_0"], config["effect_inf"]
    h, v = config["horizontal_drug"], config["vertical_drug"]
    assert conc_hor[0] == 0 and conc_ver[0] == 0
    np.testing.assert_allclose(
        surface[0, :], hill_curve(conc_hor, h["c50"], h["hill"], e0, einf), atol=1e-12,
    )
    np.testing.assert_allclose(
        surface[:, 0], hill_curve(conc_ver, v["c50"], v["hill"], e0, einf), atol=1e-12,
    )


@pytest.mark.parametrize("synergy_factor, sign", [(0.3, -1), (-0.3, 1)])
def test_injected_synergy_is_visible_against_observed_edge_bliss(synergy_factor, sign):
    """Positive synergy_factor lowers every interior cell below the
    observed-edge Bliss reference (negative deviation = synergy)."""
    config = scenario_config("matrix_synergy")
    surface, conc_hor, conc_ver = _noise_free(config, synergy_factor=synergy_factor)
    e0, einf = config["effect_0"], config["effect_inf"]
    ref = bliss_reference(surface[0, :], surface[:, 0], e0, einf)
    dev = bliss_deviation(surface, ref, e0, einf, conc_hor, conc_ver)[1:, 1:]
    assert np.all(sign * dev > 0)


@pytest.mark.parametrize("synergy_factor, synergistic", [(0.3, True), (-0.3, False)])
def test_injected_synergy_is_visible_to_loewe_ci(synergy_factor, synergistic):
    """Under a Loewe reference the injection moves CI off 1 in the right
    direction. CI here inverts the configured marginals, which the edge test
    above guarantees are the true ones."""
    config = scenario_config("matrix_loewe_additive")
    surface, conc_hor, conc_ver = _noise_free(config, synergy_factor=synergy_factor)
    h, v = config["horizontal_drug"], config["vertical_drug"]
    ci = loewe_ci(
        conc_hor, conc_ver, h["c50"], v["c50"], h["hill"], v["hill"],
        surface, config["effect_0"], config["effect_inf"],
    )[1:, 1:]
    assert np.all(np.isfinite(ci))
    assert np.all(ci < 1) if synergistic else np.all(ci > 1)


def test_synergy_factor_at_or_below_minus_one_is_rejected():
    with pytest.raises(ValueError, match="synergy_factor"):
        generate_matrix(dict(scenario_config("matrix_synergy"), synergy_factor=-1.0))
