"""The shipped scenarios must be reachable from an installed package.

These assertions are what downstream consumers rely on: the files travel
with the wheel, and the accessors find them without knowing where the
package landed on disk.
"""
import pytest

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
