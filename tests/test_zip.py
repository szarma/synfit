import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from synfit.bliss import bliss_independence
from synfit.hill import calculate_concentration_series, hill_curve
from synfit.matrix import MatrixFit
from synfit.plotting import zip_heatmap
from synfit.synthetic import generate_matrix
from synfit.zip import zip_delta, zip_fitted_surface, zip_reference

from tests.helpers import PNG_MAGIC, matrix_from_config

REF_JSON = Path(__file__).resolve().parents[1] / "tests" / "data" / "zip_reference_values.json"


def _bliss_exact_surface(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    effect_0: float,
    effect_inf: float,
    asymmetry_hor: float | None = None,
    asymmetry_ver: float | None = None,
) -> np.ndarray:
    fh = hill_curve(
        ch, c50_hor, hill_hor, effect_0, effect_inf, asymmetry=asymmetry_hor
    )
    fv = hill_curve(
        cv, c50_ver, hill_ver, effect_0, effect_inf, asymmetry=asymmetry_ver
    )
    return np.outer(fv, fh)


class TestZipZeroInteraction:
    @pytest.fixture
    def grid(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        return ch, cv

    def test_inhibition_fraction(self, grid):
        ch, cv = grid
        # effect_0=1, effect_inf=0: Bliss is the outer product of marginal Hill curves.
        m = _bliss_exact_surface(ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        assert np.nanmax(np.abs(d)) < 1e-3

    def test_inhibition_elisa_scale(self, grid):
        ch, cv = grid
        e0, e_inf = 5000.0, 200.0
        m = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf
        )
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        assert np.nanmax(np.abs(d)) < 1e-3

    def test_activation(self, grid):
        ch, cv = grid
        e0, e_inf = 0.1, 3.0
        m = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf
        )
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        assert np.nanmax(np.abs(d)) < 1e-3

    def test_five_parameter_drug(self, grid):
        ch, cv = grid
        shape = np.zeros((len(cv), len(ch)))
        m = zip_reference(
            shape, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0, asymmetry_hor=0.6
        )
        d = zip_delta(
            m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0, asymmetry_hor=0.6
        )
        # 3-parameter slice Hills do not exactly reproduce every row of a 5p
        # zero-interaction surface; residual δ is small but above 1e-3 at 0.6 asymmetry.
        assert np.nanmax(np.abs(d)) < 3e-3


class TestZipSign:
    def _synergistic_surface(
        self, ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf
    ):
        """Stronger combination effect than Bliss (horizontal IC50 drops as vertical rises)."""
        m = np.empty((len(cv), len(ch)))
        for i, v in enumerate(cv):
            for j, h in enumerate(ch):
                if h == 0.0:
                    m[i, j] = hill_curve(
                        v, c50_ver, hill_ver, effect_0, effect_inf
                    )
                elif v == 0.0:
                    m[i, j] = hill_curve(
                        h, c50_hor, hill_hor, effect_0, effect_inf
                    )
                else:
                    c50_eff = c50_hor / (1.0 + 0.35 * v)
                    yh = hill_curve(
                        h, c50_eff, hill_hor, effect_0, effect_inf
                    )
                    yv = hill_curve(
                        v, c50_ver, hill_ver, effect_0, effect_inf
                    )
                    scale = effect_0 - effect_inf
                    fh = (effect_0 - yh) / scale
                    fv = (effect_0 - yv) / scale
                    f = fh + fv - fh * fv + 0.08 * fh * fv
                    m[i, j] = effect_0 - f * scale
        return m

    def _antagonistic_surface(
        self, ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf
    ):
        bliss = bliss_independence(
            ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, effect_0, effect_inf
        )
        scale = effect_0 - effect_inf
        out = bliss.copy()
        interior = (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)
        out[interior] = effect_0 - 0.85 * (effect_0 - bliss[interior])
        return out

    def test_synergy_and_antagonism_inhibition(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        kw = dict(
            c50_hor=1.0, c50_ver=2.0, hill_hor=1.5, hill_ver=1.0,
            effect_0=1.0, effect_inf=0.0,
        )
        syn_m = self._synergistic_surface(ch, cv, **kw)
        ant_m = self._antagonistic_surface(ch, cv, **kw)
        d_syn = zip_delta(syn_m, ch, cv, **kw)
        d_ant = zip_delta(ant_m, ch, cv, **kw)
        assert np.nanmean(d_syn) < -0.02
        assert np.nanmean(d_ant) > 0.02

    def test_synergy_and_antagonism_activation(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        e0, e_inf = 0.1, 3.0
        kw = dict(
            c50_hor=1.0, c50_ver=2.0, hill_hor=1.5, hill_ver=1.0,
            effect_0=e0, effect_inf=e_inf,
        )
        syn_m = self._synergistic_surface(ch, cv, **kw)
        ant_m = self._antagonistic_surface(ch, cv, **kw)
        d_syn = zip_delta(syn_m, ch, cv, **kw)
        d_ant = zip_delta(ant_m, ch, cv, **kw)
        assert np.nanmean(d_syn) < -0.02
        assert np.nanmean(d_ant) > 0.02


class TestZipUnitInvariance:
    def test_scale_invariance(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        e0, e_inf = 1.0, 0.0
        m = bliss_independence(ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        m[2, 3] *= 0.92
        m[3, 2] *= 1.05
        base = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        for scale in (1e-3, 1e3):
            d = zip_delta(
                m * scale,
                ch,
                cv,
                1.0,
                2.0,
                1.5,
                1.0,
                e0 * scale,
                e_inf * scale,
            )
            interior = np.isfinite(base) & np.isfinite(d)
            assert np.nanmax(np.abs(base[interior] - d[interior])) < 1e-6


class TestZipReference:
    def test_matches_bliss_independence(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        m = np.zeros((len(cv), len(ch)))
        ref = zip_reference(
            m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0
        )
        bliss = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0
        )
        assert np.allclose(ref, bliss, atol=1e-12, rtol=0)


class TestZipFailedSliceFallback:
    def test_two_finite_positive_doses_warns_once(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        m = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0
        )
        # Row 2: only two finite responses at positive horizontal doses.
        m[2, :] = np.nan
        m[2, 1] = m[0, 1]
        m[2, 2] = m[0, 2]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        user = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user) == 1
        assert "slice fit" in str(user[0].message).lower()
        assert np.all(np.isfinite(d[np.isfinite(d)]))


class TestZipSynergyReferenceJson:
    # synfit uses negative=synergy; synergy package reports reference - fit (positive=synergy).
    # Remaining per-cell gaps vs curve_fit-based slice fits are below 0.01 for these surfaces.

    @pytest.fixture(scope="class")
    def cases(self):
        with REF_JSON.open() as f:
            return json.load(f)["cases"]

    def test_agrees_with_synergy_reference(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            ref_delta = -np.array(case["synergy_delta"])
            d = zip_delta(
                m,
                ch,
                cv,
                case["c50_hor"],
                case["c50_ver"],
                case["hill_hor"],
                case["hill_ver"],
                case["effect_0"],
                case["effect_inf"],
            )
            ch_g = ch[np.newaxis, :]
            cv_g = cv[:, np.newaxis]
            interior = (ch_g > 0) & (cv_g > 0)
            diff = np.abs(d - ref_delta)
            assert np.nanmax(diff[interior]) < 0.01


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

    def test_zip_delta_equal_asymptotes(self):
        m = np.ones((2, 2))
        with pytest.raises(ValueError):
            zip_delta(m, np.ones(2), np.ones(2), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

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
        assert np.all(np.isnan(d[0, :]))
        assert np.all(np.isnan(d[:, 0]))
        assert np.any(np.isfinite(d[1:, 1:]))

    def test_fitted_minus_reference_is_delta(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        m = _bliss_exact_surface(ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        m[2, 3] *= 0.9
        ref = zip_reference(m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        fit = zip_fitted_surface(m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0)
        scale = 1.0
        interior = (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)
        expected = (fit - ref) / scale
        assert np.allclose(d[interior], expected[interior], rtol=1e-9, atol=1e-9)


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
