import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from synfit.bliss import bliss_independence
from synfit.hill import hill_curve
from synfit.matrix import MatrixFit
from synfit.plotting import zip_heatmap
from synfit.zip import zip_delta, zip_fitted_surface, zip_reference, zip_scores

from tests.helpers import PNG_MAGIC, matrix_from_config

REF_JSON = Path(__file__).resolve().parents[1] / "tests" / "data" / "zip_reference_values.json"


def _edge_mask(ch: np.ndarray, cv: np.ndarray) -> np.ndarray:
    return (ch[np.newaxis, :] == 0) | (cv[:, np.newaxis] == 0)


def _interior_mask(ch: np.ndarray, cv: np.ndarray) -> np.ndarray:
    return (ch[np.newaxis, :] > 0) & (cv[:, np.newaxis] > 0)


def _assert_zip_delta_mask(d: np.ndarray, ch: np.ndarray, cv: np.ndarray) -> None:
    assert d.shape == (len(cv), len(ch))
    assert np.all(np.isnan(d[_edge_mask(ch, cv)]))
    interior = _interior_mask(ch, cv)
    assert np.all(np.isfinite(d[interior]))


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
        _assert_zip_delta_mask(d, ch, cv)
        assert np.max(np.abs(d[_interior_mask(ch, cv)])) < 1e-3

    def test_inhibition_elisa_scale(self, grid):
        ch, cv = grid
        e0, e_inf = 5000.0, 200.0
        m = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf
        )
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        _assert_zip_delta_mask(d, ch, cv)
        assert np.max(np.abs(d[_interior_mask(ch, cv)])) < 1e-3

    def test_activation(self, grid):
        ch, cv = grid
        e0, e_inf = 0.1, 3.0
        m = bliss_independence(
            ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf
        )
        d = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        _assert_zip_delta_mask(d, ch, cv)
        assert np.max(np.abs(d[_interior_mask(ch, cv)])) < 1e-3

    @pytest.mark.parametrize(
        "asymmetry_hor,asymmetry_ver",
        [
            (0.2, None),
            (None, 0.6),
            (10.0, None),
            (None, 10.0),
        ],
    )
    def test_five_parameter_null_surface(self, asymmetry_hor, asymmetry_ver):
        ch = np.concatenate([[0.0], np.logspace(-3, 3, 20)])
        cv = np.concatenate([[0.0], np.logspace(-3, 3, 20)])
        m = _bliss_exact_surface(
            ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0,
            asymmetry_hor=asymmetry_hor, asymmetry_ver=asymmetry_ver,
        )
        d = zip_delta(
            m, ch, cv, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0,
            asymmetry_hor=asymmetry_hor, asymmetry_ver=asymmetry_ver,
        )
        _assert_zip_delta_mask(d, ch, cv)
        assert np.max(np.abs(d[_interior_mask(ch, cv)])) < 1e-4


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
        interior = _interior_mask(ch, cv)
        assert np.mean(d_syn[interior]) < -0.02
        assert np.mean(d_ant[interior]) > 0.02

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
        interior = _interior_mask(ch, cv)
        assert np.mean(d_syn[interior]) < -0.02
        assert np.mean(d_ant[interior]) > 0.02


class TestZipUnitInvariance:
    def test_scale_invariance(self):
        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        e0, e_inf = 1.0, 0.0
        m = bliss_independence(ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        m[2, 3] *= 0.92
        m[3, 2] *= 1.05
        base = zip_delta(m, ch, cv, 1.0, 2.0, 1.5, 1.0, e0, e_inf)
        interior = _interior_mask(ch, cv)
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
            assert np.max(np.abs(base[interior] - d[interior])) < 1e-6


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
    def test_one_failed_row_uses_column_only(self):
        """A failed row slice must be replaced by the column slice alone, checked
        quantitatively on a surface with real interaction (a null surface would let a
        zero or halved fallback pass)."""
        from scipy.optimize import least_squares

        ch = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
        cv = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
        kw = dict(
            c50_hor=1.0, c50_ver=2.0, hill_hor=1.5, hill_ver=1.0,
            effect_0=1.0, effect_inf=0.0,
        )
        # Synergistic surface: the vertical drug's c50 drops up to 4x as the
        # horizontal dose rises (responses are surviving fractions, effect_0=1).
        m = np.empty((len(cv), len(ch)))
        for j, h in enumerate(ch):
            c50_v = 2.0 / (1.0 + 3.0 * h / (h + 1.0))
            m[:, j] = hill_curve(cv, c50_v, 1.0, 1.0, 0.0) * hill_curve(h, 1.0, 1.5, 1.0, 0.0)
        fail_row = 2
        m[fail_row, 3:] = np.nan  # two finite positive doses left: the row slice fails

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = zip_scores(m, ch, cv, **kw)
        assert len([x for x in w if issubclass(x.category, UserWarning)]) == 1
        assert result.failed_rows == (fail_row,)
        assert result.failed_cols == ()
        assert result.n_unscored == 0
        interior = _interior_mask(ch, cv)
        assert np.all(np.isfinite(result.delta[interior]))

        # Independent column-only expectation: fit each column slice here with
        # scipy.least_squares (not synfit's optimiser) and score it against the
        # zero-interaction expectation.
        f_obs = 1.0 - m
        f_h = 1.0 - hill_curve(ch, 1.0, 1.5, 1.0, 0.0)
        f_v = 1.0 - hill_curve(cv, 2.0, 1.0, 1.0, 0.0)
        expected = []
        for j in range(1, len(ch)):
            ok = (cv > 0) & np.isfinite(f_obs[:, j])
            x, y, f0 = cv[ok], f_obs[ok, j], f_h[j]

            def resid(p):
                return f0 + (p[2] - f0) * (1 - 1 / (1 + (x / 10 ** p[0]) ** p[1])) - y

            fit = least_squares(resid, [np.log10(2.0), 1.0, 1.0],
                                bounds=([-3, 0.1, -0.5], [3, 10, 1.0]), xtol=1e-14, ftol=1e-14)
            lm, lam, emax = fit.x
            pred = f0 + (emax - f0) * (1 - 1 / (1 + (cv[fail_row] / 10 ** lm) ** lam))
            f_zip = f_h[j] + f_v[fail_row] - f_h[j] * f_v[fail_row]
            expected.append(-(pred - f_zip))
        got = result.delta[fail_row, 1:]
        expected = np.array(expected)
        assert np.max(np.abs(expected)) > 0.02  # real interaction on this row
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=0)

    def test_all_slices_fail_3x3(self):
        ch = np.array([0.0, 0.1, 1.0])
        cv = np.array([0.0, 0.2, 2.0])
        kw = dict(
            c50_hor=1.0, c50_ver=2.0, hill_hor=1.5, hill_ver=1.0,
            effect_0=1.0, effect_inf=0.0,
        )
        m = bliss_independence(ch, cv, **kw)
        # Strong synergy in the interior so data are not on the null surface.
        interior = _interior_mask(ch, cv)
        m[interior] *= 0.5

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = zip_scores(m, ch, cv, **kw)
        user = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user) == 1

        assert result.n_unscored == 4
        assert np.all(np.isnan(result.delta[interior]))
        assert np.all(np.isnan(result.fitted[interior]))
        assert len(result.failed_rows) == 2
        assert len(result.failed_cols) == 2


class TestZipSynergyReferenceJson:
    # synfit uses negative=synergy; synergy package reports reference - fit (positive=synergy).
    # Independent implementation of the same definition (GPL-3, run offline by
    # scripts/zip_reference_values.py). Both fit the same 3-parameter slice model
    # with Emax capped at full effect, so they agree to optimiser precision
    # (observed max 5e-4 against deltas of 0.04-0.09).

    # Mirrors the strict loader in test_synergy_additive_reference.py: a truncated
    # or empty fixture must fail loudly rather than pass by iterating over nothing.
    EXPECTED_CASE_NAMES = frozenset({"synergistic", "potentiated"})
    REQUIRED_VERSION_KEYS = ("synergy_package_version", "numpy_version", "scipy_version")
    REQUIRED_CASE_KEYS = frozenset({
        "name", "conc_hor", "conc_ver", "c50_hor", "c50_ver", "hill_hor", "hill_ver",
        "effect_0", "effect_inf", "mean_matrix", "synergy_delta", "zip_reference",
        "zip_fitted",
    })

    @pytest.fixture(scope="class")
    def cases(self):
        with REF_JSON.open() as f:
            payload = json.load(f)
        for key in self.REQUIRED_VERSION_KEYS:
            assert key in payload and payload[key], f"fixture missing version field {key!r}"
        cases = payload["cases"]
        assert isinstance(cases, list) and cases, "fixture cases must be a non-empty list"
        names = [case["name"] for case in cases]
        assert len(names) == len(set(names)), "fixture case names must be unique"
        assert frozenset(names) == self.EXPECTED_CASE_NAMES, (
            f"unexpected case set: got {frozenset(names)!r}"
        )
        for case in cases:
            missing = self.REQUIRED_CASE_KEYS - case.keys()
            assert not missing, f"{case['name']}: missing required keys {missing!r}"
        return cases

    def test_agrees_with_synergy_reference(self, cases):
        for case in cases:
            ch = np.array(case["conc_hor"])
            cv = np.array(case["conc_ver"])
            m = np.array(case["mean_matrix"])
            kw = dict(
                c50_hor=case["c50_hor"],
                c50_ver=case["c50_ver"],
                hill_hor=case["hill_hor"],
                hill_ver=case["hill_ver"],
                effect_0=case["effect_0"],
                effect_inf=case["effect_inf"],
            )
            ref_delta = -np.array(case["synergy_delta"])
            d = zip_delta(m, ch, cv, **kw)
            ref_syn = zip_reference(m, ch, cv, **kw)
            fit_syn = zip_fitted_surface(m, ch, cv, **kw)
            ref_pkg = np.array(case["zip_reference"])
            fit_pkg = np.array(case["zip_fitted"])
            interior = _interior_mask(ch, cv)
            _assert_zip_delta_mask(d, ch, cv)
            diff = np.abs(d - ref_delta)
            assert np.max(np.abs(ref_delta[interior])) > 0.03  # real signal
            assert np.max(diff[interior]) < 1e-3
            assert np.max(np.abs(ref_syn - ref_pkg)) < 1e-3
            assert np.max(np.abs(fit_syn - fit_pkg)[interior]) < 1e-3


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
        conc_hor = np.array([0.0, 1.0, 2.0, 4.0])
        conc_ver = np.array([0.0, 2.0, 4.0, 6.0])
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
        interior = _interior_mask(ch, cv)
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
