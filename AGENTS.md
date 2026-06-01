# AGENTS.md — synfit

`README.md` covers what synfit is and how to install it; `CHANGELOG.md` covers
the version history and the SemVer rule. This file is only what those don't.

**Constraint:** pure-Python numerical library — no Django, web, or filesystem /
network I/O (beyond reading the bundled seed CSVs). It ships as a standalone
public package; keep it that way.

**Running it:** `just check` is the gate, `just --list` for the rest. Tests run
under `uv run pytest` with `pythonpath = ["src", "."]` already configured — no
`PYTHONPATH=` needed, and `from tests.helpers import ...` resolves.

## The FitBase contract (load-bearing)

Fitters subclass `FitBase` and implement `_log_prob_data(x, **kwargs)`;
`FitBase` owns `_run_minimize` and `_estimate_covariance`.

- Both run in **scale-relative coordinates** (`z = x / scale`,
  `scale = parameter_scale(x0, bounds)` — uses bound magnitude only when
  ``|x0|`` is small vs the feasible range; variance coeffs use ``|x0|`` only)
  and back-transform (`Σ_x = D Σ_z D`). This is what makes fits
  scale-invariant and keeps covariances valid for large-magnitude (ELISA/RFU)
  data. **Never reintroduce a raw `minimize()` or an absolute-step Hessian in a
  fitter** — that was the 0.2.0 bug.
- Multi-parameter fitters (`JointMarginalFit`, `MatrixFit`) don't match the
  single-drug `FitConfig` / `FitBounds` schema, so they pass explicit
  `self._x0` / `self._bounds` to `_run_minimize` and override `_log_prior_prob`,
  bypassing `_get_x0_and_bounds` / `_x_to_kwargs`. Mirror that for new fitters.

## Gotchas

1. `param_cov` can be `None` (singular / ill-conditioned / non-finite Hessian),
   and bound-active parameters get a degenerate (0) row/col. Never assume a CI
   exists.
2. `MatrixFit` (6-param Bliss surface) is a diagnostic null model; the default
   matrix path is `JointMarginalFit`. Both are scale-invariant when edge /
   asymptote bounds match the data magnitude.
3. Preconditioning uses `parameter_scale(x0, bounds)` — unrepresentative `x0`
   with bounds that do not span the true parameter scale can still break
   invariance.
4. Lognormal noise needs strictly-positive predictions (lower asymptote bound
   floored at `1e-6`) and supports constant variance only.
5. Parameters are direction-neutral: `effect_0` / `effect_inf` are the low- /
   high-concentration asymptotes, `log_c50` is fitted in log space; the
   `direction` field swaps which asymptote is the displayed IC₅₀ / EC₅₀ top.
6. Seed CSVs in `data/synthetic/` are committed: edit `config.json` →
   `just generate-seed-data`; CI's `just check-seed-data` catches drift.

## Versioning

A change that moves fitted values is ≥ MINOR (see `CHANGELOG.md`). Tag `vX.Y.Z`
on `main` after merge — never a feature-branch tip.
