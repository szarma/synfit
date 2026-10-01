"""Generate Bliss, HSA, and Loewe reference values via GPL-3 ``synergy`` (run offline).

    uv run --no-project \\
        --with synergy==1.0.0 \\
        --with mpmath==1.4.1 \\
        --with numpy==2.4.3 \\
        --with scipy==1.17.1 \\
        python scripts/bliss_hsa_loewe_reference_values.py
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from synergy.combination import Bliss, HSA, Loewe
from synergy.single import Hill
from synergy.single.dose_response_model_1d import DoseResponseModel1D

OUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "bliss_hsa_loewe_reference_values.json"
)

REQUIRED_VERSIONS = {
    "synergy": "1.0.0",
    "numpy": "2.4.3",
    "scipy": "1.17.1",
    "mpmath": "1.4.1",
}

CH = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
CV = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
CH_RECT = np.array([0.0, 0.15, 0.5, 2.0, 7.0])
CV_RECT = np.array([0.0, 0.3, 1.2, 5.0, 15.0, 40.0])
C50_HOR, C50_VER = 1.0, 2.0
C50_HOR_RECT, C50_VER_RECT = 1.0, 3.0
HILL_HOR, HILL_VER = 1.5, 1.0
E0, EMAX = 1.0, 0.0
E0_OFFSET, EMAX_OFFSET = 2.5, 0.4
E0_ACT, EMAX_ACT = 0.2, 1.8


def _check_required_versions() -> None:
    import numpy

    found = {
        "synergy": importlib.metadata.version("synergy"),
        "numpy": numpy.__version__,
        "scipy": scipy_version,
        "mpmath": importlib.metadata.version("mpmath"),
    }
    mismatches = [
        f"{name}: need {want}, got {found[name]}"
        for name, want in REQUIRED_VERSIONS.items()
        if found[name] != want
    ]
    if mismatches:
        print("Refusing to regenerate with mismatched dependency versions:", file=sys.stderr)
        for line in mismatches:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


class TabulatedMarginal(DoseResponseModel1D):
    """Tabulated single-agent response at committed doses (for Bliss reference only)."""

    def __init__(self, doses: np.ndarray, effects: np.ndarray) -> None:
        self._doses = np.asarray(doses, dtype=float)
        self._effects = np.asarray(effects, dtype=float)
        if self._doses.shape != self._effects.shape:
            raise ValueError("doses and effects must match")

    @property
    def is_specified(self) -> bool:
        return True

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, d, E, **kwargs) -> None:
        raise NotImplementedError("TabulatedMarginal is pre-specified")

    def E(self, d):
        d_arr = np.asarray(d, dtype=float)
        return np.interp(
            d_arr,
            self._doses,
            self._effects,
            left=float(self._effects[0]),
            right=float(self._effects[-1]),
        )

    def E_inv(self, E):
        raise NotImplementedError("TabulatedMarginal has no analytic inverse")


def _hill_survival(
    conc: np.ndarray,
    c50: float,
    hill: float,
    *,
    e0: float,
    emax: float,
) -> np.ndarray:
    c = np.asarray(conc, dtype=float)
    return e0 + (emax - e0) * c**hill / (c50**hill + c**hill)


def _bliss_surface(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    *,
    e0: float,
    emax: float,
) -> np.ndarray:
    """Bliss independence in synfit convention (normalised survival, then map back)."""
    fh = _hill_survival(ch, c50_hor, hill_hor, e0=e0, emax=emax)
    fv = _hill_survival(cv, c50_ver, hill_ver, e0=e0, emax=emax)
    scale = e0 - emax
    nh = np.clip((fh - emax) / scale, 0.0, 1.0)
    nv = np.clip((fv - emax) / scale, 0.0, 1.0)
    return emax + scale * np.outer(nv, nh)


def _synergistic_surface(bliss: np.ndarray, ch: np.ndarray, cv: np.ndarray) -> np.ndarray:
    out = bliss.copy()
    for i in range(1, len(cv)):
        for j in range(1, len(ch)):
            out[i, j] = bliss[i, j] ** 1.25
    return out


def _antagonistic_surface(bliss: np.ndarray, ch: np.ndarray, cv: np.ndarray) -> np.ndarray:
    out = bliss.copy()
    for i in range(1, len(cv)):
        for j in range(1, len(ch)):
            out[i, j] = np.minimum(bliss[i, j] ** 0.8, 1.0)
    return out


def _dose_grids(ch: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ch_grid = np.tile(ch, (len(cv), 1))
    cv_grid = np.repeat(cv, len(ch)).reshape(len(cv), len(ch))
    return ch_grid, cv_grid, ch_grid.ravel(), cv_grid.ravel()


def _loewe_reference_exact_mpmath(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    e0: float,
    emax: float,
) -> np.ndarray:
    """Loewe additive reference at 50-digit precision (mpmath, offline only)."""
    from mpmath import findroot, mp

    mp.dps = 50
    e0_m, einf_m = mp.mpf(e0), mp.mpf(emax)
    scale = e0_m - einf_m
    e_lo, e_hi = min(e0_m, einf_m), max(e0_m, einf_m)
    span = e_hi - e_lo
    eps = mp.mpf("1e-45") * span
    lo, hi = e_lo + eps, e_hi - eps

    def invert_hill(y: mp.mpf, c50: mp.mpf, hill: mp.mpf) -> mp.mpf:
        if not (y > e_lo and y < e_hi):
            return mp.nan
        u = y - einf_m
        inner = (scale / u) - mp.mpf(1)
        if inner <= 0:
            return mp.nan
        return c50 * inner ** (mp.mpf(1) / hill)

    def hill_at(conc: mp.mpf, c50: mp.mpf, hill: mp.mpf) -> mp.mpf:
        return einf_m + scale / (mp.mpf(1) + (conc / c50) ** hill)

    c50_h, c50_v = mp.mpf(c50_hor), mp.mpf(c50_ver)
    h_h, h_v = mp.mpf(hill_hor), mp.mpf(hill_ver)
    n_ver, n_hor = len(cv), len(ch)
    out = np.full((n_ver, n_hor), np.nan, dtype=float)

    for i, cv_i in enumerate(cv):
        cv_m = mp.mpf(cv_i)
        for j, ch_j in enumerate(ch):
            ch_m = mp.mpf(ch_j)
            if ch_j == 0 and cv_i == 0:
                out[i, j] = float(e0_m)
                continue
            if ch_j == 0:
                out[i, j] = float(hill_at(cv_m, c50_v, h_v))
                continue
            if cv_i == 0:
                out[i, j] = float(hill_at(ch_m, c50_h, h_h))
                continue

            def residual(e: mp.mpf) -> mp.mpf:
                a = invert_hill(e, c50_h, h_h)
                b = invert_hill(e, c50_v, h_v)
                if not (mp.isfinite(a) and mp.isfinite(b)) or a == 0 or b == 0:
                    return mp.nan
                return ch_m / a + cv_m / b - mp.mpf(1)

            try:
                r_lo, r_hi = residual(lo), residual(hi)
                if not (mp.isfinite(r_lo) and mp.isfinite(r_hi)) or r_lo * r_hi > 0:
                    continue
                e_star = findroot(residual, (lo, hi), solver="bisect", maxsteps=200)
                out[i, j] = float(e_star)
            except (ValueError, RuntimeError):
                continue
    return out


def _scores(
    mean_matrix: np.ndarray,
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    e0: float,
    emax: float,
    *,
    resp_hor_obs: np.ndarray | None = None,
    resp_ver_obs: np.ndarray | None = None,
) -> dict[str, list]:
    _, _, d1, d2 = _dose_grids(ch, cv)
    drug1 = Hill(E0=e0, Emax=emax, h=hill_hor, C=c50_hor)
    drug2 = Hill(E0=e0, Emax=emax, h=hill_ver, C=c50_ver)
    e_flat = mean_matrix.ravel()
    n_ver, n_hor = mean_matrix.shape

    stronger = np.maximum if emax > e0 else np.minimum

    bliss = Bliss(drug1_model=drug1, drug2_model=drug2)
    bliss_ref = bliss.E_reference(d1, d2).reshape(n_ver, n_hor)
    bliss_syn = bliss.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    hsa = HSA(drug1_model=drug1, drug2_model=drug2, stronger_orientation=stronger)
    hsa_ref = hsa.E_reference(d1, d2).reshape(n_ver, n_hor)
    hsa_syn = hsa.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    loewe = Loewe(mode="CI", drug1_model=drug1, drug2_model=drug2)
    loewe_ref = loewe.E_reference(d1, d2).reshape(n_ver, n_hor)
    loewe_ci = loewe.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    loewe_exact = _loewe_reference_exact_mpmath(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, e0, emax
    )

    out: dict[str, list] = {
        "bliss_reference": bliss_ref.tolist(),
        "hsa_reference": hsa_ref.tolist(),
        "loewe_reference": loewe_ref.tolist(),
        "bliss_synergy": bliss_syn.tolist(),
        "hsa_synergy": hsa_syn.tolist(),
        "loewe_ci": loewe_ci.tolist(),
        "loewe_reference_exact": loewe_exact.tolist(),
    }

    if resp_hor_obs is not None and resp_ver_obs is not None:
        tab1 = TabulatedMarginal(ch, resp_hor_obs)
        tab2 = TabulatedMarginal(cv, resp_ver_obs)
        bliss_tab = Bliss(drug1_model=tab1, drug2_model=tab2)
        out["bliss_reference_tabulated"] = (
            bliss_tab.E_reference(d1, d2).reshape(n_ver, n_hor).tolist()
        )
        hsa_tab = HSA(
            drug1_model=tab1,
            drug2_model=tab2,
            stronger_orientation=stronger,
        )
        out["hsa_reference_tabulated"] = (
            hsa_tab.E_reference(d1, d2).reshape(n_ver, n_hor).tolist()
        )
        out["hsa_synergy_tabulated"] = (
            hsa_tab.fit(d1, d2, e_flat).reshape(n_ver, n_hor).tolist()
        )
        out["resp_hor_observed"] = np.asarray(resp_hor_obs, dtype=float).tolist()
        out["resp_ver_observed"] = np.asarray(resp_ver_obs, dtype=float).tolist()

    return out


def _case(
    name: str,
    mean_matrix: np.ndarray,
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    e0: float,
    emax: float,
    *,
    resp_hor_obs: np.ndarray | None = None,
    resp_ver_obs: np.ndarray | None = None,
) -> dict:
    scores = _scores(
        mean_matrix,
        ch,
        cv,
        c50_hor,
        c50_ver,
        hill_hor,
        hill_ver,
        e0,
        emax,
        resp_hor_obs=resp_hor_obs,
        resp_ver_obs=resp_ver_obs,
    )
    return {
        "name": name,
        "conc_hor": ch.tolist(),
        "conc_ver": cv.tolist(),
        "c50_hor": c50_hor,
        "c50_ver": c50_ver,
        "hill_hor": hill_hor,
        "hill_ver": hill_ver,
        "effect_0": e0,
        "effect_inf": emax,
        "mean_matrix": mean_matrix.tolist(),
        **scores,
    }


def _perturbed_marginals(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
    e0: float,
    emax: float,
) -> tuple[np.ndarray, np.ndarray]:
    exact_h = _hill_survival(ch, c50_hor, hill_hor, e0=e0, emax=emax)
    exact_v = _hill_survival(cv, c50_ver, hill_ver, e0=e0, emax=emax)
    scale = abs(e0 - emax)
    bump_h = scale * 0.035 * np.sin(np.arange(len(ch)) * 1.7)
    bump_v = scale * 0.028 * np.cos(np.arange(len(cv)) * 1.3)
    lo, hi = min(e0, emax), max(e0, emax)
    resp_h = np.clip(exact_h + bump_h, lo + 0.02 * scale, hi - 0.02 * scale)
    resp_v = np.clip(exact_v + bump_v, lo + 0.02 * scale, hi - 0.02 * scale)
    return resp_h, resp_v


def _surface_with_observed_edges(
    interior: np.ndarray,
    resp_hor: np.ndarray,
    resp_ver: np.ndarray,
) -> np.ndarray:
    m = np.array(interior, dtype=float, copy=True)
    m[0, :] = resp_hor
    m[:, 0] = resp_ver
    m[0, 0] = resp_hor[0]
    return m


def main() -> None:
    _check_required_versions()
    cases = []

    bliss = _bliss_surface(
        CH, CV, C50_HOR, C50_VER, HILL_HOR, HILL_VER, e0=E0, emax=EMAX
    )
    cases.append(
        _case("bliss_additive", bliss, CH, CV, C50_HOR, C50_VER, HILL_HOR, HILL_VER, E0, EMAX)
    )
    cases.append(
        _case(
            "synergistic",
            _synergistic_surface(bliss, CH, CV),
            CH,
            CV,
            C50_HOR,
            C50_VER,
            HILL_HOR,
            HILL_VER,
            E0,
            EMAX,
        )
    )
    cases.append(
        _case(
            "antagonistic",
            _antagonistic_surface(bliss, CH, CV),
            CH,
            CV,
            C50_HOR,
            C50_VER,
            HILL_HOR,
            HILL_VER,
            E0,
            EMAX,
        )
    )

    hill_steep, hill_mild = 2.5, 0.6
    bliss_slopes = _bliss_surface(
        CH, CV, C50_HOR, C50_VER, hill_steep, hill_mild, e0=E0, emax=EMAX
    )
    cases.append(
        _case(
            "mismatched_slopes_synergistic",
            _synergistic_surface(bliss_slopes, CH, CV),
            CH,
            CV,
            C50_HOR,
            C50_VER,
            hill_steep,
            hill_mild,
            E0,
            EMAX,
        )
    )

    drug1 = Hill(E0=E0, Emax=EMAX, h=1.0, C=C50_HOR)
    drug2 = Hill(E0=E0, Emax=EMAX, h=1.0, C=C50_VER)
    _, _, d1, d2 = _dose_grids(CH, CV)
    loewe = Loewe(mode="CI", drug1_model=drug1, drug2_model=drug2)
    loewe_additive = loewe.E_reference(d1, d2).reshape(len(CV), len(CH))
    cases.append(
        _case(
            "loewe_additive",
            loewe_additive,
            CH,
            CV,
            C50_HOR,
            C50_VER,
            1.0,
            1.0,
            E0,
            EMAX,
        )
    )

    bliss_offset = _bliss_surface(
        CH,
        CV,
        C50_HOR,
        C50_VER,
        HILL_HOR,
        HILL_VER,
        e0=E0_OFFSET,
        emax=EMAX_OFFSET,
    )
    cases.append(
        _case(
            "offset_inhibition_synergistic",
            _synergistic_surface(bliss_offset, CH, CV),
            CH,
            CV,
            C50_HOR,
            C50_VER,
            HILL_HOR,
            HILL_VER,
            E0_OFFSET,
            EMAX_OFFSET,
        )
    )

    bliss_rect = _bliss_surface(
        CH_RECT,
        CV_RECT,
        C50_HOR_RECT,
        C50_VER_RECT,
        HILL_HOR,
        HILL_VER,
        e0=E0,
        emax=EMAX,
    )
    syn_rect = _synergistic_surface(bliss_rect, CH_RECT, CV_RECT)
    resp_h, resp_v = _perturbed_marginals(
        CH_RECT,
        CV_RECT,
        C50_HOR_RECT,
        C50_VER_RECT,
        HILL_HOR,
        HILL_VER,
        E0,
        EMAX,
    )
    cases.append(
        _case(
            "rectangular_perturbed_marginals",
            _surface_with_observed_edges(syn_rect, resp_h, resp_v),
            CH_RECT,
            CV_RECT,
            C50_HOR_RECT,
            C50_VER_RECT,
            HILL_HOR,
            HILL_VER,
            E0,
            EMAX,
            resp_hor_obs=resp_h,
            resp_ver_obs=resp_v,
        )
    )

    bliss_act = _bliss_surface(
        CH,
        CV,
        C50_HOR,
        C50_VER,
        HILL_HOR,
        HILL_VER,
        e0=E0_ACT,
        emax=EMAX_ACT,
    )
    cases.append(
        _case(
            "activation_synergistic",
            _synergistic_surface(bliss_act, CH, CV),
            CH,
            CV,
            C50_HOR,
            C50_VER,
            HILL_HOR,
            HILL_VER,
            E0_ACT,
            EMAX_ACT,
        )
    )

    payload = {
        "synergy_package_version": REQUIRED_VERSIONS["synergy"],
        "numpy_version": REQUIRED_VERSIONS["numpy"],
        "scipy_version": REQUIRED_VERSIONS["scipy"],
        "mpmath_version": REQUIRED_VERSIONS["mpmath"],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
