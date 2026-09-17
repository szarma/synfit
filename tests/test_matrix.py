import numpy as np
import pytest
from synfit.matrix import MatrixFit

from tests.helpers import matrix_from_config


@pytest.fixture(scope="module", params=[
    ("lognormal", "lognormal"),
    ("gaussian", "gaussian"),
], ids=["lognormal", "gaussian"])
def independent_fit(request):
    noise_model, error_model = request.param
    replicates, conc_hor, conc_ver = matrix_from_config(seed=42, noise_model=noise_model)
    fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model=error_model)
    return fitter.fit()


class TestMatrixFitParameterRecovery:
    def test_succeeds(self, independent_fit):
        assert independent_fit.success

    def test_recovers_horizontal_ic50(self, independent_fit):
        assert independent_fit.horizontal.c50 == pytest.approx(5.0, rel=0.25)

    def test_recovers_vertical_ic50(self, independent_fit):
        assert independent_fit.vertical.c50 == pytest.approx(8.0, rel=0.25)

    def test_recovers_horizontal_hill(self, independent_fit):
        assert independent_fit.horizontal.hill == pytest.approx(1.5, rel=0.25)

    def test_recovers_vertical_hill(self, independent_fit):
        assert independent_fit.vertical.hill == pytest.approx(1.2, rel=0.25)

    def test_recovers_top(self, independent_fit):
        assert independent_fit.effect_0 == pytest.approx(1.0, rel=0.15)

    def test_recovers_bottom(self, independent_fit):
        assert independent_fit.effect_inf == pytest.approx(0.02, abs=0.05)

    def test_no_warnings_with_zero_edge(self, independent_fit):
        assert independent_fit.warnings is None

    def test_sigma_populated_and_shared_across_edges(self, independent_fit):
        # σ̂ pooled over all replicates and cells; same value stamped onto edges.
        assert independent_fit.sigma is not None
        assert independent_fit.sigma > 0
        assert independent_fit.horizontal.sigma == independent_fit.sigma
        assert independent_fit.vertical.sigma == independent_fit.sigma

    def test_to_dict_keys(self, independent_fit):
        d = independent_fit.to_dict()
        assert set(d.keys()) == {
            "horizontal", "vertical", "effect_0", "effect_inf",
            "success", "message", "direction_horizontal", "direction_vertical",
            "sigma", "param_cov", "param_names",
        }
        assert "warnings" not in d
        assert set(d["horizontal"].keys()) >= {"ic50", "c50", "hill", "effect_0", "effect_inf", "success"}


class TestMatrixFitEdgeCases:
    def test_with_valids(self):
        replicates, conc_hor, conc_ver = matrix_from_config()
        valids = [np.ones(r.shape, dtype=bool) for r in replicates]
        fitter = MatrixFit(replicates, conc_hor, conc_ver, valids=valids, error_model="lognormal")
        result = fitter.fit()
        assert result.success

    @pytest.mark.parametrize("error_model", ["gaussian", "lognormal"])
    def test_error_model_propagated_to_edge_fits(self, error_model):
        replicates, conc_hor, conc_ver = matrix_from_config()
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model=error_model)
        assert fitter.synfit_hor.config.error_model == error_model
        assert fitter.synfit_ver.config.error_model == error_model

    def test_synergy_scenario_succeeds(self):
        replicates, conc_hor, conc_ver = matrix_from_config(synergy_factor=0.3, seed=99)
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model="lognormal")
        result = fitter.fit()
        assert result.success

    def test_no_zero_edge_warns_and_succeeds(self):
        replicates, conc_hor, conc_ver = matrix_from_config(has_zero=False)
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model="lognormal")
        assert fitter.zero_edge_missing
        result = fitter.fit()
        assert result.success
        assert result.warnings is not None
        assert any("zero-concentration" in w.lower() for w in result.warnings)
        d = result.to_dict()
        assert "warnings" in d


class TestMatrixSharedAsymptoteBounds:
    """The shared effect_0/effect_inf bounds come from the data-derived
    single-drug derivation (per-asymptote, rounded, lognormal-floored), not a
    fresh ±(0.5–2)x envelope around the pre-fit results."""

    def test_lognormal_effect_inf_lower_bound_is_strictly_positive(self):
        """Regression: under lognormal noise the bottom asymptote bound must be
        floored strictly positive (the old code hardcoded it to 0.0, which let
        the Bliss surface bottom out at zero and break the likelihood)."""
        replicates, conc_hor, conc_ver = matrix_from_config(noise_model="lognormal")
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model="lognormal")
        effect_inf_bounds = fitter._bounds[5]
        assert effect_inf_bounds[0] > 0.0

    def test_gaussian_allows_nonpositive_bottom_bound(self):
        """Gaussian noise has no positivity constraint, so the bottom bound may
        extend at or below zero — proving the floor is lognormal-specific, not a
        blanket clamp."""
        replicates, conc_hor, conc_ver = matrix_from_config(noise_model="gaussian")
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model="gaussian")
        effect_inf_bounds = fitter._bounds[5]
        assert effect_inf_bounds[0] <= 0.0

    def test_shared_bounds_match_merged_edge_config_bounds(self):
        """Each shared asymptote bound is the (min lo, max hi) merge of the two
        edge configs' same-named bound — the JointMarginalFit-style derivation."""
        replicates, conc_hor, conc_ver = matrix_from_config(noise_model="lognormal")
        fitter = MatrixFit(replicates, conc_hor, conc_ver, error_model="lognormal")
        hor, ver = fitter.synfit_hor.config.bounds, fitter.synfit_ver.config.bounds
        assert fitter._bounds[4] == (
            min(hor.effect_0[0], ver.effect_0[0]),
            max(hor.effect_0[1], ver.effect_0[1]),
        )
        assert fitter._bounds[5] == (
            min(hor.effect_inf[0], ver.effect_inf[0]),
            max(hor.effect_inf[1], ver.effect_inf[1]),
        )


class TestMatrixMixedDirectionGuard:
    """The Bliss surface uses one shared (effect_0, effect_inf) pair for both
    marginals, so it cannot represent one drug activating while the other
    inhibits. Such plates are rejected up front (use JointMarginalFit instead),
    rather than silently merging one drug's top bound with the other's bottom."""

    @pytest.mark.parametrize("dir_h,dir_v", [
        ("inhibition", "activation"),
        ("activation", "inhibition"),
    ])
    def test_mixed_directions_rejected(self, dir_h, dir_v):
        replicates, conc_hor, conc_ver = matrix_from_config()
        with pytest.raises(ValueError, match="share a direction"):
            MatrixFit(
                replicates, conc_hor, conc_ver,
                direction_horizontal=dir_h, direction_vertical=dir_v,
            )

    @pytest.mark.parametrize("direction", ["inhibition", "activation"])
    def test_matching_directions_allowed(self, direction):
        # Same direction on both axes constructs fine (the well-defined case).
        replicates, conc_hor, conc_ver = matrix_from_config(
            noise_model="gaussian" if direction == "activation" else "lognormal",
        )
        fitter = MatrixFit(
            replicates, conc_hor, conc_ver,
            direction_horizontal=direction, direction_vertical=direction,
        )
        assert fitter.direction_horizontal == fitter.direction_vertical == direction
