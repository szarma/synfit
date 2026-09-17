import numpy as np
import pytest
from synfit.bliss import bliss_deviation, bliss_independence
from synfit.hill import calculate_concentration_series
from synfit.matrix import MatrixFit
from synfit.plotting import zip_heatmap
from synfit.synthetic import generate_matrix
from synfit.zip import _fit_slice, zip_delta, zip_reference

from tests.helpers import PNG_MAGIC, matrix_from_config


class TestZipDelta:
    def test_zip_delta_shape(self):
        replicates, conc_hor, conc_ver = matrix_from_config(noise_sigma_log=0.0)
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        d = zip_delta(
            mean_m,
            conc_hor,
            conc_ver,
            p.horizontal.c50,
            p.vertical.c50,
            p.horizontal.hill,
            p.vertical.hill,
            p.effect_0,
            p.effect_inf,
        )
        assert d.shape == mean_m.shape

    def test_zip_delta_independent_data(self):
        replicates, conc_hor, conc_ver = matrix_from_config(
            synergy_factor=0.0, seed=42, noise_sigma_log=0.0
        )
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        d = zip_delta(
            mean_m,
            conc_hor,
            conc_ver,
            p.horizontal.c50,
            p.vertical.c50,
            p.horizontal.hill,
            p.vertical.hill,
            p.effect_0,
            p.effect_inf,
        )
        # Edge cells (zero concentration) are NaN by design; use nanmean for interior.
        assert np.nanmean(np.abs(d)) < 0.1

    def test_zip_delta_synergistic_data(self):
        replicates, conc_hor, conc_ver = matrix_from_config(
            synergy_factor=0.3, seed=99, noise_sigma_log=0.0
        )
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        bliss = bliss_independence(
            conc_hor,
            conc_ver,
            c50_hor=5.0,
            c50_ver=8.0,
            hill_hor=1.5,
            hill_ver=1.2,
            effect_0=1.0,
            effect_inf=0.02,
        )
        bliss_mean_dev = np.mean(bliss_deviation(mean_m, bliss, 1.0, 0.02))
        assert bliss_mean_dev < -0.02

        d = zip_delta(
            mean_m,
            conc_hor,
            conc_ver,
            p.horizontal.c50,
            p.vertical.c50,
            p.horizontal.hill,
            p.vertical.hill,
            p.effect_0,
            p.effect_inf,
        )
        assert np.nanmean(d) < 0

    def test_zip_delta_handles_flat_slices(self):
        conc_hor = np.array([50.0, 16.7, 5.56, 1.85, 0.617, 0.206])
        conc_ver = np.array([80.0, 26.7, 8.89, 2.96, 0.988, 0.329])
        bliss = bliss_independence(
            conc_hor,
            conc_ver,
            c50_hor=5.0,
            c50_ver=8.0,
            hill_hor=1.5,
            hill_ver=1.2,
            effect_0=1.0,
            effect_inf=0.02,
        )
        m = bliss.copy()
        m[3, :] = 0.5
        d = zip_delta(
            m,
            conc_hor,
            conc_ver,
            5.0,
            8.0,
            1.5,
            1.2,
            1.0,
            0.02,
        )
        assert np.all(np.isfinite(d))

    def test_zip_delta_zero_dose_in_conc(self):
        conc_hor = calculate_concentration_series(
            initial_conc=50.0,
            fold_dilutions=3.0,
            length=6,
            has_zero=True,
        )
        conc_ver = calculate_concentration_series(
            initial_conc=80.0,
            fold_dilutions=3.0,
            length=6,
            has_zero=False,
        )
        bliss = bliss_independence(
            conc_hor,
            conc_ver,
            c50_hor=5.0,
            c50_ver=8.0,
            hill_hor=1.5,
            hill_ver=1.2,
            effect_0=1.0,
            effect_inf=0.02,
        )
        d = zip_delta(
            bliss,
            conc_hor,
            conc_ver,
            5.0,
            8.0,
            1.5,
            1.2,
            1.0,
            0.02,
        )
        assert bliss.shape == d.shape
        # conc_hor has a zero entry — the corresponding column is NaN by design.
        assert np.any(np.isnan(d))
        assert np.all(np.isfinite(d[:, conc_hor > 0]))

    def test_zip_delta_activation(self):
        effect_0, effect_inf = 0.1, 1.0
        rng = np.random.default_rng(0)
        replicates, conc_hor, conc_ver = generate_matrix(
            {
                "seed": 1,
                "horizontal_drug": {
                    "c50": 5.0,
                    "hill": 1.5,
                    "concentration_series": {
                        "initial_conc": 50.0,
                        "fold_dilutions": 3.0,
                        "length": 6,
                        "has_zero": True,
                    },
                },
                "vertical_drug": {
                    "c50": 8.0,
                    "hill": 1.2,
                    "concentration_series": {
                        "initial_conc": 80.0,
                        "fold_dilutions": 3.0,
                        "length": 6,
                        "has_zero": True,
                    },
                },
                "effect_0": effect_0,
                "effect_inf": effect_inf,
                "n_replicates": 1,
                "noise_sigma_log": 0.0,
                "synergy_factor": 0.25,
            },
            rng=rng,
        )
        mean_m = np.mean(np.stack(replicates), axis=0)
        # This is activation data (effect_0 < effect_inf), so fit it as such.
        # The per-asymptote default bounds meet at the response midpoint, so a
        # mismatched (default inhibition) direction pins both asymptotes near
        # the midpoint and recovers a wrong surface — the loose pre-merge bounds
        # used to mask this.
        p = MatrixFit(
            replicates, conc_hor, conc_ver,
            direction_horizontal="activation", direction_vertical="activation",
        ).fit()
        d = zip_delta(
            mean_m,
            conc_hor,
            conc_ver,
            p.horizontal.c50,
            p.vertical.c50,
            p.horizontal.hill,
            p.vertical.hill,
            p.effect_0,
            p.effect_inf,
        )
        assert np.nanmean(d) < 0

    def test_zip_delta_equal_asymptotes(self):
        m = np.ones((2, 2))
        with pytest.raises(ValueError):
            zip_delta(m, np.ones(2), np.ones(2), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def test_fit_slice_equal_asymptotes(self):
        with pytest.raises(ValueError):
            _fit_slice(np.ones(3), np.ones(3), 1.0, 1.0)

    def test_zip_delta_masks_zero_concentration_edges(self):
        conc_hor = np.array([0.0, 1.0, 4.0])
        conc_ver = np.array([0.0, 2.0, 6.0])
        bliss = bliss_independence(
            conc_hor, conc_ver,
            c50_hor=5.0, c50_ver=8.0,
            hill_hor=1.5, hill_ver=1.2,
            effect_0=1.0, effect_inf=0.02,
        )
        d = zip_delta(
            bliss, conc_hor, conc_ver,
            5.0, 8.0, 1.5, 1.2, 1.0, 0.02,
        )
        assert np.all(np.isnan(d[0, :]))    # zero-conc_ver row
        assert np.all(np.isnan(d[:, 0]))    # zero-conc_hor column
        assert np.any(np.isfinite(d[1:, 1:]))  # interior is finite


class TestZipReference:
    def test_reference_shape(self):
        replicates, conc_hor, conc_ver = matrix_from_config(noise_sigma_log=0.0)
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        ref = zip_reference(
            mean_m, conc_hor, conc_ver,
            p.horizontal.c50, p.vertical.c50,
            p.horizontal.hill, p.vertical.hill,
            p.effect_0, p.effect_inf,
        )
        assert ref.shape == mean_m.shape

    def test_delta_equals_observed_minus_reference(self):
        replicates, conc_hor, conc_ver = matrix_from_config(noise_sigma_log=0.0)
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        ref = zip_reference(
            mean_m, conc_hor, conc_ver,
            p.horizontal.c50, p.vertical.c50,
            p.horizontal.hill, p.vertical.hill,
            p.effect_0, p.effect_inf,
        )
        d = zip_delta(
            mean_m, conc_hor, conc_ver,
            p.horizontal.c50, p.vertical.c50,
            p.horizontal.hill, p.vertical.hill,
            p.effect_0, p.effect_inf,
        )
        scale = p.effect_0 - p.effect_inf
        ch = np.asarray(conc_hor)
        cv = np.asarray(conc_ver)
        interior = (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)
        assert np.allclose(d[interior], ((mean_m - ref) / scale)[interior])


class TestZipHeatmap:
    def test_zip_heatmap_returns_png(self):
        replicates, conc_hor, conc_ver = matrix_from_config(noise_sigma_log=0.0)
        mean_m = np.mean(np.stack(replicates), axis=0)
        p = MatrixFit(replicates, conc_hor, conc_ver).fit()
        png = zip_heatmap(
            mean_m,
            conc_hor,
            conc_ver,
            c50_hor=p.horizontal.c50,
            c50_ver=p.vertical.c50,
            hill_hor=p.horizontal.hill,
            hill_ver=p.vertical.hill,
            effect_0=p.effect_0,
            effect_inf=p.effect_inf,
        )
        assert png[:4] == PNG_MAGIC
