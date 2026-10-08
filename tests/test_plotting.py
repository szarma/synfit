import numpy as np
import pandas as pd
import pytest
from synfit.data import FitResult
from synfit.joint_marginal import JointMarginalResult
from synfit.plotting import (
    deviation_heatmap,
    dose_response_plot,
    matrix_heatmap,
    raw_replicate_heatmap,
    synergy_heatmaps,
)
from synfit.hill import hill_curve
from synfit.bliss import bliss_independence

from tests.helpers import PNG_MAGIC


@pytest.fixture
def single_drug_data():
    concs = np.geomspace(0.1, 100, 8)
    rows = []
    for rep in ["A", "B", "C"]:
        y = hill_curve(concs, c50=5.0, hill=1.5, effect_0=1.0, effect_inf=0.02)
        for c, yi in zip(concs, y):
            rows.append({"concentration": c, "y": yi, "replicate": rep})
    return pd.DataFrame(rows)


@pytest.fixture
def fit_result():
    return FitResult(
        c50=5.0, log_c50=np.log10(5.0),
        hill=1.5, effect_0=1.0, effect_inf=0.02,
        success=True, n_valid=24, n_total=24,
    )


@pytest.fixture
def matrix_data():
    conc_hor = np.array([50.0, 16.7, 5.56, 1.85, 0.617, 0.206])
    conc_ver = np.array([80.0, 26.7, 8.89, 2.96, 0.988, 0.329])
    bliss = bliss_independence(
        conc_hor, conc_ver,
        c50_hor=5.0, c50_ver=8.0,
        hill_hor=1.5, hill_ver=1.2,
        effect_0=1.0, effect_inf=0.02,
    )
    rng = np.random.default_rng(42)
    matrix = bliss * np.exp(rng.normal(0, 0.05, size=bliss.shape))
    return matrix, conc_hor, conc_ver


class TestDoseResponsePlot:
    def test_produces_png(self, single_drug_data, fit_result):
        result = dose_response_plot(single_drug_data, fit_result)
        assert result[:4] == PNG_MAGIC

    def test_with_excluded_points(self, single_drug_data, fit_result):
        valids = np.ones(len(single_drug_data), dtype=bool)
        valids[:3] = False
        result = dose_response_plot(single_drug_data, fit_result, valids=valids)
        assert result[:4] == PNG_MAGIC

    def test_excluded_points_get_one_legend_entry(self, single_drug_data, fit_result, monkeypatch):
        from synfit import plotting

        captured = {}

        def capture(fig):
            captured["labels"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
            return PNG_MAGIC

        monkeypatch.setattr(plotting, "_save_fig_png", capture)
        valids = np.ones(len(single_drug_data), dtype=bool)
        valids[[0, 8, 16]] = False  # one excluded well in each of the three replicates
        dose_response_plot(single_drug_data, fit_result, valids=valids)
        assert captured["labels"].count("excluded") == 1

    def test_with_reference_curve_and_labels(self, single_drug_data, fit_result):
        result = dose_response_plot(
            single_drug_data,
            fit_result,
            reference={"c50": 5, "hill": 1.5, "effect_0": 1, "effect_inf": 0.02},
            reference_label="Ground truth",
            x_label="Test compound concentration [µM]",
            title="Synthetic fit",
        )
        assert result[:4] == PNG_MAGIC

    def test_rejects_incomplete_reference(self, single_drug_data, fit_result):
        with pytest.raises(ValueError, match="missing effect_inf"):
            dose_response_plot(single_drug_data, fit_result, reference={"c50": 5, "hill": 1.5, "effect_0": 1})

    def test_lognormal_fit_with_zero_control_produces_png(self, single_drug_data, fit_result):
        data = pd.concat([
            single_drug_data,
            pd.DataFrame({"concentration": [0.0], "y": [1.0], "replicate": ["A"]}),
        ], ignore_index=True)
        fit_result.error_model = "lognormal"
        result = dose_response_plot(data, fit_result)
        assert result[:4] == PNG_MAGIC


class TestMatrixHeatmap:
    def test_produces_png(self, matrix_data):
        matrix, conc_hor, conc_ver = matrix_data
        result = matrix_heatmap(matrix, conc_hor, conc_ver, vmin=0.0, vmax=1.0)
        assert result[:4] == PNG_MAGIC

    def test_raw_replicate_heatmap_accepts_string_replicate_labels(self, single_drug_data):
        result = raw_replicate_heatmap(single_drug_data)
        assert result[:4] == PNG_MAGIC


class TestDeviationHeatmap:
    def test_produces_png(self, matrix_data):
        matrix, conc_hor, conc_ver = matrix_data
        result = deviation_heatmap(
            matrix, conc_hor, conc_ver,
            c50_hor=5.0, c50_ver=8.0,
            hill_hor=1.5, hill_ver=1.2,
            effect_0=1.0, effect_inf=0.02,
        )
        assert result[:4] == PNG_MAGIC


class TestSynergyHeatmaps:
    def test_produces_four_score_png_without_zero_dose_edges(self):
        conc_hor = np.array([0.0, 0.5, 2.0, 8.0])
        conc_ver = np.array([0.0, 1.0, 4.0, 16.0])
        matrix = bliss_independence(
            conc_hor, conc_ver,
            c50_hor=2.0, c50_ver=4.0,
            hill_hor=1.2, hill_ver=1.1,
            effect_0=1.0, effect_inf=0.05,
        )
        result = synergy_heatmaps(
            matrix, conc_hor, conc_ver,
            c50_hor=2.0, c50_ver=4.0,
            hill_hor=1.2, hill_ver=1.1,
            effect_0=1.0, effect_inf=0.05,
        )
        assert result[:4] == PNG_MAGIC


@pytest.mark.parametrize("direction", ["inhibition", "activation"])
def test_joint_result_plot_uses_5p_curves_and_direction(direction, monkeypatch):
    from synfit import plotting

    ch = np.array([0.0, 0.5, 2.0, 8.0, 32.0])
    cv = np.array([0.0, 1.0, 4.0, 16.0, 64.0])
    e0, einf = (1.0, 0.05) if direction == "inhibition" else (0.05, 1.0)
    drugs = [FitResult(
        c50=c50, log_c50=np.log10(c50), hill=hill, asymmetry=asymmetry,
        effect_0=e0, effect_inf=einf, direction=direction,
        success=True, n_valid=5, n_total=5,
    ) for c50, hill, asymmetry in [(2.0, 1.2, 0.7), (4.0, 1.1, 1.5)]]
    result = JointMarginalResult(
        drug_a=drugs[0], drug_b=drugs[1], top=1.0, bottom=0.05, success=True,
    )
    matrix = bliss_independence(
        ch, cv, c50_hor=2.0, c50_ver=4.0, hill_hor=1.2, hill_ver=1.1,
        effect_0=e0, effect_inf=einf, asymmetry_hor=0.7, asymmetry_ver=1.5,
    )
    scores = []

    def capture_scores(fig):
        scores.extend(ax.images[0].get_array() for ax in fig.axes[:4])
        assert fig.axes[0].get_xlabel() == "Drug A [µM]"
        assert fig.axes[0].get_ylabel() == "Drug B [µM]"
        plotting.plt.close(fig)
        return PNG_MAGIC

    monkeypatch.setattr(plotting, "_save_fig_png", capture_scores)
    png = result.plot_synergy(matrix, ch, cv, x_label="Drug A [µM]", y_label="Drug B [µM]")
    assert png == PNG_MAGIC
    np.testing.assert_allclose(scores[0], 0, atol=1e-12)  # Bliss null
    np.testing.assert_allclose(scores[3], 0, atol=1e-4)  # ZIP null, with 5p slices

    result.drug_b.direction = "activation" if direction == "inhibition" else "inhibition"
    with pytest.raises(ValueError, match="same direction"):
        result.plot_synergy(matrix, ch, cv)
