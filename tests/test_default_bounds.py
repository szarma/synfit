"""Default bounds derivation: the single source of truth, and literal honouring.

``SingleDrugFit`` derives data-driven bounds when the caller does not supply
them (so a bare ``FitConfig`` fits any-magnitude data), and honours supplied
bounds literally (no equality-based override). ``default_fit_config`` /
``default_bounds`` expose the same derivation for UIs to consume.
"""
import numpy as np
import pandas as pd

from synfit import default_bounds, default_fit_config
from synfit.data import FitConfig, FitBounds
from synfit.noise import GaussianConstant, Lognormal
from synfit.single import SingleDrugFit, _round_asymptote_bounds


def _elisa_activation(seed: int = 3) -> pd.DataFrame:
    """Large-magnitude (RFU-scale) activation curve: top ~1.1e4, bottom ~7."""
    rng = np.random.default_rng(seed)
    c50, hill, e0, einf = 20.0, 1.1, 7.0, 11000.0
    conc = np.array([0.0098, 0.039, 0.156, 0.625, 2.5, 10.0, 40.0, 160.0])
    frac = conc ** hill / (conc ** hill + c50 ** hill)
    y = e0 + (einf - e0) * frac
    return pd.concat(
        [
            pd.DataFrame(
                {"concentration": conc, "y": y * np.exp(rng.normal(0, 0.05, conc.shape)), "replicate": r}
            )
            for r in ("A", "B", "C")
        ],
        ignore_index=True,
    )


def test_default_bounds_track_data_magnitude():
    df = _elisa_activation()
    b = default_bounds(df, direction="activation", noise=GaussianConstant())
    # Top asymptote (effect_inf for activation) reaches the ~1.1e4 top.
    assert b.effect_inf[1] > 11_000
    # Bottom and top no longer share one wide range: each is its own bound and
    # they meet at the midpoint. The bottom's upper bound tracks the data scale
    # (far above the old [0, 2] clamp) but is capped at the midpoint, below the
    # top — and the bottom's ceiling equals the top's floor.
    assert b.effect_0[1] > 2.0
    assert b.effect_0[1] < b.effect_inf[1]
    assert b.effect_0[1] == b.effect_inf[0]
    # log_c50 bounds are data-derived, not the hardcoded (-5, 5).
    assert b.log_c50 != (-5.0, 5.0)


def test_default_bounds_lognormal_floors_lower_positive():
    df = _elisa_activation()
    b = default_bounds(df, direction="activation", noise=Lognormal())
    assert b.effect_0[0] > 0.0
    assert b.effect_inf[0] > 0.0


def test_outward_rounding_preserves_narrow_high_offset_bound_hierarchy():
    """Rounding must not collapse a narrow response range on a large offset."""
    df = pd.DataFrame(
        {
            "concentration": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "y": [1000.0, 1000.1, 1000.2, 1000.3, 1000.4, 1000.5],
            "replicate": ["A"] * 6,
        }
    )

    activation = default_bounds(df, direction="activation", noise=GaussianConstant())
    bottom = activation.effect_0
    top = activation.effect_inf
    assert bottom[0] < bottom[1] == top[0] < top[1]
    # Outermost edges are rounded away from the raw feasible interval.
    assert bottom[0] <= 999.95
    assert top[1] >= 1001.0

    inhibition = default_bounds(df, direction="inhibition", noise=GaussianConstant())
    assert inhibition.effect_inf == bottom
    assert inhibition.effect_0 == top


def test_outward_rounding_keeps_lognormal_lower_bound_positive():
    """Outward rounding must not turn a small positive lognormal floor into zero."""
    df = pd.DataFrame(
        {
            "concentration": [0.01, 0.1, 1.0, 10.0, 100.0],
            "y": [0.01, 0.2, 0.5, 0.8, 1.0],
            "replicate": ["A"] * 5,
        }
    )
    b = default_bounds(df, direction="activation", noise=Lognormal())
    assert 0.0 < b.effect_0[0] < b.effect_0[1] == b.effect_inf[0] < b.effect_inf[1]


def test_outward_rounding_preserves_float_resolution_at_extreme_offset():
    """Cleanup must not merge bounds separated by only a few float units."""
    bottom = (-81228082.64515305, -81228082.64515302)
    top = (-81228082.64515302, -81228082.64515294)
    rounded_bottom, rounded_top = _round_asymptote_bounds(
        bottom,
        top,
        scale=2.493817517783659e-08,
    )
    assert rounded_bottom[0] <= bottom[0]
    assert rounded_top[1] >= top[1]
    assert (
        rounded_bottom[0]
        < rounded_bottom[1]
        == rounded_top[0]
        < rounded_top[1]
    )


def test_default_fit_config_returns_concrete_bounds():
    df = _elisa_activation()
    cfg = default_fit_config(df, direction="activation", noise=GaussianConstant())
    assert isinstance(cfg, FitConfig)
    assert cfg.bounds is not None  # derived bounds are concrete, honoured literally


def test_string_and_dict_lognormal_floor_positive():
    """Public helpers accept a kind string / tagged dict, not just a Lognormal()."""
    df = _elisa_activation()
    for noise in ("lognormal", {"kind": "lognormal"}, Lognormal()):
        b = default_bounds(df, direction="activation", noise=noise)
        assert b.effect_0[0] > 0.0, f"lower bound not floored for noise={noise!r}"
        assert b.effect_inf[0] > 0.0, f"lower bound not floored for noise={noise!r}"


def test_single_drug_with_error_resolves_none_bounds():
    """Regression: SingleDrugFitWithError(FitConfig()) (bounds=None) must derive
    bounds, not dereference None in FitBase._get_x0_and_bounds."""
    from synfit.single import SingleDrugFitWithError

    df = _elisa_activation()  # concentration, y, replicate
    df = df.assign(y_err=(df["y"].abs() * 0.05 + 1.0))  # WithError needs y_err
    result = SingleDrugFitWithError(df, FitConfig(direction="activation")).fit()
    assert result is not None
    assert result.effect_inf > 1_000  # derived bounds reached the data


def test_bounds_none_survives_dataclass_replace_and_asdict():
    """bounds=None must persist through dataclasses.replace / asdict so the
    derive-vs-literal semantic can't be corrupted into a spurious clamp."""
    import dataclasses

    df = _elisa_activation()
    base = FitConfig(direction="activation")  # no bounds → derive
    assert base.bounds is None
    # replace() must not resurrect concrete (explicit) bounds.
    replaced = dataclasses.replace(base, direction="activation")
    assert replaced.bounds is None
    r = SingleDrugFit(df, replaced).fit()
    assert r.effect_inf > 1_000, "replace() must not reintroduce the [0,2] clamp"
    # asdict round-trip preserves the None signal.
    assert dataclasses.asdict(base)["bounds"] is None


def test_unspecified_bounds_are_derived_not_clamped():
    """A FitConfig with no bounds must not clamp a large-magnitude fit to [0, 2]."""
    df = _elisa_activation()
    result = SingleDrugFit(df, FitConfig(direction="activation")).fit()
    assert result.effect_inf > 1_000, "top must reach the data, not the old [0,2] default"
    assert result.param_cov is not None


def test_explicit_bounds_are_honoured_literally():
    """Explicitly-supplied bounds are used as-is — no silent data-derived override."""
    df = _elisa_activation()
    # Deliberately absurd tight asymptote bounds: a literal fit must respect them.
    result = SingleDrugFit(
        df,
        FitConfig(
            direction="activation",
            effect_0=7.0,
            effect_inf=11000.0,
            bounds=FitBounds(
                log_c50=(-5.0, 5.0),
                hill=(0.1, 4.0),
                effect_0=(0.0, 2.0),
                effect_inf=(0.0, 2.0),
            ),
        ),
    ).fit()
    # Clamped to the literal upper bound of 2 — proves bounds are not overridden.
    assert result.effect_inf <= 2.0 + 1e-6


def _small_curve(n: int) -> pd.DataFrame:
    """A clean n-point inhibition curve (single replicate) for small-sample tests."""
    conc = np.logspace(-2, 2, n)
    frac = conc ** 1.2 / (conc ** 1.2 + 5.0 ** 1.2)
    y = 0.9 - 0.85 * frac
    return pd.DataFrame({"concentration": conc, "y": y, "replicate": "A"})


def test_default_bounds_small_samples_do_not_raise():
    """Adaptive extrema: 2–4 point datasets derive bounds instead of being
    rejected. The old blanket ``_MIN_POINTS_FOR_DEFAULTS = 5`` guard 500'd the
    matrix-defaults endpoint on 2x2 plates and refused four-point single curves.
    """
    for n in (2, 3, 4, 5, 8):
        b = default_bounds(_small_curve(n))
        lo, hi = b.effect_inf
        assert hi > lo, f"degenerate asymptote bound at n={n}: {b.effect_inf}"
        # log_c50 spans the concentration decades, never the hardcoded fallback.
        assert b.log_c50 != (-5.0, 5.0)


def test_default_bounds_small_sample_skips_outlier_trim():
    """When the trimmed range collapses, bounds fall back to the true min/max."""
    df = _small_curve(3)
    b = default_bounds(df)
    ys = df["y"].to_numpy()
    # effect_0 (top, low-conc) upper bound must reach toward the true max; the
    # bottom floor must reach toward the true min. With trimming on n=3 these
    # would collapse to the single middle point.
    assert b.effect_0[1] > b.effect_inf[1]  # top sits above bottom
    assert float(ys.min()) <= b.effect_inf[1]  # bottom bound informed by true min
    assert float(ys.max()) >= b.effect_0[0]    # top bound informed by true max


def test_default_bounds_falls_back_when_trimmed_extrema_collapse():
    """Four-point sparse curves can have one plateau represented by a single
    point. Trimming both ends would collapse the remaining dynamic range even
    though the true min/max are usable."""
    df = pd.DataFrame(
        {
            "concentration": [0.01, 0.1, 1.0, 10.0],
            "y": [0.9, 0.9, 0.9, 0.05],
            "replicate": ["A"] * 4,
        }
    )
    b = default_bounds(df)
    assert b.effect_inf[0] < b.effect_inf[1] == b.effect_0[0] < b.effect_0[1]
    assert b.effect_inf[1] > 0.05
    assert b.effect_0[0] < 0.9


def test_default_bounds_single_point_still_raises():
    """One point can't define a dynamic range; the guard must still fire."""
    import pytest

    with pytest.raises(ValueError, match="at least 2"):
        default_bounds(_small_curve(1))
