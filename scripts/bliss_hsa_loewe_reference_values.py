"""Generate Bliss, HSA, and Loewe reference values via GPL-3 ``synergy`` (run offline).

    uv run --no-project --with synergy --with mpmath \\
        python scripts/bliss_hsa_loewe_reference_values.py
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
from synergy.combination import Bliss, HSA, Loewe
from synergy.single import Hill

OUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "bliss_hsa_loewe_reference_values.json"
)

CH = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
CV = np.array([0.0, 0.2, 0.6, 2.0, 6.0, 20.0])
C50_HOR, C50_VER = 1.0, 2.0
HILL_HOR, HILL_VER = 1.5, 1.0
E0, EMAX = 1.0, 0.0


def _hill_survival(conc: np.ndarray, c50: float, hill: float) -> np.ndarray:
    c = np.asarray(conc, dtype=float)
    return E0 + (EMAX - E0) * c**hill / (c50**hill + c**hill)


def _bliss_surface(
    ch: np.ndarray,
    cv: np.ndarray,
    c50_hor: float,
    c50_ver: float,
    hill_hor: float,
    hill_ver: float,
) -> np.ndarray:
    fh = _hill_survival(ch, c50_hor, hill_hor)
    fv = _hill_survival(cv, c50_ver, hill_ver)
    return np.outer(fv, fh)


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
    """Loewe additive reference at 50-digit precision (mpmath, offline only).

    Solves ``d_h / A(E) + d_v / B(E) = 1`` with the four-parameter Hill inverse
    matching synfit ``_invert_hill`` (``asymmetry=1``).
    """
    from mpmath import findroot, mp

    mp.dps = 50
    e0_m, emax_m = mp.mpf(e0), mp.mpf(emax)
    scale = e0_m - emax_m
    e_lo, e_hi = min(e0_m, emax_m), max(e0_m, emax_m)
    eps = mp.mpf("1e-45") * scale
    lo, hi = e_lo + eps, e_hi - eps

    def invert_hill(y: mp.mpf, c50: mp.mpf, hill: mp.mpf) -> mp.mpf:
        u = y - emax_m
        inner = (scale / u) - mp.mpf(1)
        if inner <= 0:
            return mp.nan
        return c50 * inner ** (mp.mpf(1) / hill)

    def hill_at(conc: mp.mpf, c50: mp.mpf, hill: mp.mpf) -> mp.mpf:
        return emax_m + scale / (mp.mpf(1) + (conc / c50) ** hill)

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
) -> dict[str, list]:
    _, _, d1, d2 = _dose_grids(ch, cv)
    drug1 = Hill(E0=e0, Emax=emax, h=hill_hor, C=c50_hor)
    drug2 = Hill(E0=e0, Emax=emax, h=hill_ver, C=c50_ver)
    e_flat = mean_matrix.ravel()
    n_ver, n_hor = mean_matrix.shape

    bliss = Bliss(drug1_model=drug1, drug2_model=drug2)
    bliss_ref = bliss.E_reference(d1, d2).reshape(n_ver, n_hor)
    bliss_syn = bliss.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    hsa = HSA(drug1_model=drug1, drug2_model=drug2)
    hsa_ref = hsa.E_reference(d1, d2).reshape(n_ver, n_hor)
    hsa_syn = hsa.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    loewe = Loewe(mode="CI", drug1_model=drug1, drug2_model=drug2)
    loewe_ref = loewe.E_reference(d1, d2).reshape(n_ver, n_hor)
    loewe_ci = loewe.fit(d1, d2, e_flat).reshape(n_ver, n_hor)

    loewe_exact = _loewe_reference_exact_mpmath(
        ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, e0, emax
    )

    return {
        # From synergy 1.x API (Hill model marginals):
        "bliss_reference": bliss_ref.tolist(),
        "hsa_reference": hsa_ref.tolist(),
        "loewe_reference": loewe_ref.tolist(),
        # From synergy fit() on the fixture surface:
        "bliss_synergy": bliss_syn.tolist(),
        "hsa_synergy": hsa_syn.tolist(),
        "loewe_ci": loewe_ci.tolist(),
        # High-precision ground truth (mpmath dps=50, offline generator only):
        "loewe_reference_exact": loewe_exact.tolist(),
    }


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
) -> dict:
    scores = _scores(mean_matrix, ch, cv, c50_hor, c50_ver, hill_hor, hill_ver, e0, emax)
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


def main() -> None:
    synergy_version = importlib.metadata.version("synergy")
    cases = []

    bliss = _bliss_surface(CH, CV, C50_HOR, C50_VER, HILL_HOR, HILL_VER)
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
    bliss_slopes = _bliss_surface(CH, CV, C50_HOR, C50_VER, hill_steep, hill_mild)
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

    payload = {"synergy_package_version": synergy_version, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT} (synergy {synergy_version})")


if __name__ == "__main__":
    main()
