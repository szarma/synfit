import numpy as np
import pandas as pd
import pytest
from synfit.data import FitResult
from synfit.plotting import dose_response_plot, matrix_heatmap, deviation_heatmap
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


class TestMatrixHeatmap:
    def test_produces_png(self, matrix_data):
        matrix, conc_hor, conc_ver = matrix_data
        result = matrix_heatmap(matrix, conc_hor, conc_ver, vmin=0.0, vmax=1.0)
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
