import numpy as np
import pytest
from synfit.bliss import (
    bliss_independence,
    bliss_reference,
    bliss_deviation,
    hsa_deviation,
    hsa_reference,
)


def test_bliss_shape():
    hor = np.array([0.5, 1.0, 2.0])
    ver = np.array([0.5, 1.0])
    result = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0)
    assert result.shape == (2, 3)


def test_bliss_respects_top_bottom():
    hor = np.array([1e-6, 1e6])
    ver = np.array([1e-6, 1e6])
    result = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0, effect_0=0.9, effect_inf=0.1)
    # all values should be within [effect_inf, effect_0]
    assert result.min() >= 0.1 - 1e-6
    assert result.max() <= 0.9 + 1e-6


def test_bliss_independence_zero_deviation():
    # if data == bliss prediction, deviation should be 0
    hor = np.array([0.5, 1.0, 2.0])
    ver = np.array([0.5, 1.0, 2.0])
    bliss = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0, effect_0=1.0, effect_inf=0.0)
    dev = bliss_deviation(bliss, bliss, effect_0=1.0, effect_inf=0.0)
    assert np.allclose(dev, 0.0, atol=1e-10)


def test_bliss_deviation_synergy():
    # more inhibition than predicted = positive deviation
    hor = np.array([1.0])
    ver = np.array([1.0])
    bliss = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0)
    # suppress response below Bliss prediction
    synergistic = bliss * 0.7
    dev = bliss_deviation(synergistic, bliss, effect_0=1.0, effect_inf=0.0)
    assert dev[0, 0] < 0  # data below bliss = negative deviation by our convention


def test_bliss_deviation_invalid_scale():
    with pytest.raises(ValueError):
        bliss_deviation(np.ones((2, 2)), np.ones((2, 2)), effect_0=1.0, effect_inf=1.0)


def test_hsa_deviation_shape():
    resp_hor = np.array([0.9, 0.5, 0.1])   # n_hor = 3
    resp_ver = np.array([0.8, 0.3])          # n_ver = 2
    mean_matrix = np.ones((2, 3)) * 0.4
    dev = hsa_deviation(mean_matrix, resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert dev.shape == (2, 3)


def test_hsa_deviation_zero_when_equal_to_hsa():
    # if observed == HSA prediction, deviation should be 0
    resp_hor = np.array([0.8, 0.4])
    resp_ver = np.array([0.6, 0.2])
    hsa = np.minimum(resp_ver[:, None], resp_hor[None, :])
    dev = hsa_deviation(hsa, resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert np.allclose(dev, 0.0, atol=1e-10)


def test_hsa_deviation_synergy():
    # more inhibition than HSA = data < hsa = negative deviation
    resp_hor = np.array([0.8])
    resp_ver = np.array([0.6])
    # hsa[0,0] = min(0.6, 0.8) = 0.6; synergistic data is below that
    data = np.array([[0.3]])
    dev = hsa_deviation(data, resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert dev[0, 0] < 0


def test_hsa_deviation_invalid_scale():
    with pytest.raises(ValueError):
        hsa_deviation(np.ones((2, 2)), np.ones(2), np.ones(2), effect_0=1.0, effect_inf=1.0)


def test_hsa_reference_min_of_marginals():
    resp_hor = np.array([0.8, 0.4, 0.1])
    resp_ver = np.array([0.6, 0.2])
    ref = hsa_reference(resp_hor, resp_ver)
    assert ref.shape == (2, 3)
    expected = np.array([[0.6, 0.4, 0.1], [0.2, 0.2, 0.1]])
    assert np.allclose(ref, expected)


def test_bliss_reference_shape_and_range():
    resp_hor = np.array([0.9, 0.5, 0.1])
    resp_ver = np.array([0.8, 0.3])
    ref = bliss_reference(resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert ref.shape == (2, 3)
    assert ref.min() >= 0.0 - 1e-12
    assert ref.max() <= 1.0 + 1e-12


def test_bliss_reference_agrees_with_marginals_at_edges():
    # At the single-drug edges, Bliss(obs) must equal the observed marginal
    # (the probabilistic identity: multiplying by survival=1 of the absent drug).
    resp_hor = np.array([1.0, 0.7, 0.3])   # first element = no vertical drug baseline
    resp_ver = np.array([1.0, 0.5])
    ref = bliss_reference(resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    # Top row (vertical at "no effect" = 1.0) reproduces horizontal marginal
    assert np.allclose(ref[0, :], resp_hor)
    # Left column (horizontal at "no effect" = 1.0) reproduces vertical marginal
    assert np.allclose(ref[:, 0], resp_ver)


def test_bliss_reference_clips_out_of_range_inputs():
    # Observed marginals can exceed [effect_inf, effect_0] due to noise or
    # super-inhibition. Clipping keeps the reference inside the response window.
    resp_hor = np.array([1.2, 0.5, -0.1])  # above effect_0 and below effect_inf
    resp_ver = np.array([1.1, 0.3])
    ref = bliss_reference(resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert ref.min() >= 0.0 - 1e-12
    assert ref.max() <= 1.0 + 1e-12


def test_bliss_reference_invalid_scale():
    with pytest.raises(ValueError):
        bliss_reference(np.ones(2), np.ones(2), effect_0=1.0, effect_inf=1.0)


def test_bliss_independence_clips_to_response_window():
    # effect_0=0.9, effect_inf=0.1 — Hill values at extreme conc stay within
    # the window so clipping is a no-op here; verify the result is bounded.
    hor = np.array([1e-9, 1e9])
    ver = np.array([1e-9, 1e9])
    result = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0,
                                effect_0=0.9, effect_inf=0.1)
    assert result.min() >= 0.1 - 1e-9
    assert result.max() <= 0.9 + 1e-9


def test_bliss_independence_accepts_asymmetry():
    # 5p Hill: asymmetry != 1.0 must not be silently dropped.
    hor = np.array([0.5, 1.0, 2.0])
    ver = np.array([0.5, 1.0])
    sym = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0)
    asym = bliss_independence(hor, ver, c50_hor=1.0, c50_ver=1.0,
                              asymmetry_hor=2.0, asymmetry_ver=0.5)
    assert asym.shape == sym.shape
    # Different asymmetry must produce a different surface.
    assert not np.allclose(sym, asym)


def test_bliss_independence_asymmetry_matches_hill_outer_product():
    # Exact identity: bliss_independence with 5p kwargs must equal the Bliss
    # product of the corresponding 5p hill_curve marginals. Pins semantics so
    # a future refactor cannot silently drop asymmetry again (the bug fixed
    # in edaeec3 was exactly that — asymmetry accepted but not forwarded).
    from synfit.hill import hill_curve

    hor = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    ver = np.array([0.2, 0.8, 2.0, 5.0])
    c50_h, hill_h, S_h = 1.0, 1.2, 2.3
    c50_v, hill_v, S_v = 2.0, 0.9, 0.5
    effect_0, effect_inf = 1.0, 0.0

    result = bliss_independence(
        hor, ver,
        c50_hor=c50_h, c50_ver=c50_v,
        hill_hor=hill_h, hill_ver=hill_v,
        effect_0=effect_0, effect_inf=effect_inf,
        asymmetry_hor=S_h, asymmetry_ver=S_v,
    )

    resp_hor = hill_curve(hor, c50=c50_h, hill=hill_h,
                          effect_0=effect_0, effect_inf=effect_inf, asymmetry=S_h)
    resp_ver = hill_curve(ver, c50=c50_v, hill=hill_v,
                          effect_0=effect_0, effect_inf=effect_inf, asymmetry=S_v)
    scale = effect_0 - effect_inf
    norm_h = np.clip((resp_hor - effect_inf) / scale, 0.0, 1.0)
    norm_v = np.clip((resp_ver - effect_inf) / scale, 0.0, 1.0)
    expected = effect_inf + scale * np.outer(norm_v, norm_h)

    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_hsa_deviation_uses_reference():
    # deviation should equal (obs - ref) / scale
    resp_hor = np.array([0.7, 0.3])
    resp_ver = np.array([0.5, 0.2])
    mean = np.array([[0.4, 0.25], [0.18, 0.15]])
    ref = hsa_reference(resp_hor, resp_ver)
    dev = hsa_deviation(mean, resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0)
    assert np.allclose(dev, mean - ref)


def test_bliss_deviation_masks_zero_concentration_edges():
    conc_hor = np.array([0.0, 1.0, 3.0])
    conc_ver = np.array([0.0, 2.0, 5.0])
    bliss = bliss_independence(conc_hor, conc_ver, c50_hor=1.0, c50_ver=2.0,
                               effect_0=1.0, effect_inf=0.0)
    dev = bliss_deviation(bliss, bliss, effect_0=1.0, effect_inf=0.0,
                          conc_hor=conc_hor, conc_ver=conc_ver)
    assert np.all(np.isnan(dev[0, :]))    # zero-conc_ver row
    assert np.all(np.isnan(dev[:, 0]))    # zero-conc_hor column
    assert np.any(np.isfinite(dev[1:, 1:]))  # interior is finite


def test_hsa_deviation_masks_zero_concentration_edges():
    conc_hor = np.array([0.0, 1.0, 3.0])
    conc_ver = np.array([0.0, 2.0, 5.0])
    resp_hor = np.array([1.0, 0.6, 0.2])
    resp_ver = np.array([1.0, 0.7, 0.3])
    mean = np.array([
        [1.0, 0.6, 0.2],
        [0.7, 0.45, 0.18],
        [0.3, 0.22, 0.10],
    ])
    dev = hsa_deviation(mean, resp_hor, resp_ver, effect_0=1.0, effect_inf=0.0,
                        conc_hor=conc_hor, conc_ver=conc_ver)
    assert np.all(np.isnan(dev[0, :]))    # zero-conc_ver row
    assert np.all(np.isnan(dev[:, 0]))    # zero-conc_hor column
    assert np.any(np.isfinite(dev[1:, 1:]))  # interior is finite
