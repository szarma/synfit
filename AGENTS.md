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
 ``|x0|`` is small vs the feasible range for the magnitude-bearing
 asymptotes; `response_scale` is passed only for linear/quadratic Gaussian
 noise, so constant/lognormal/compound keep the historical scale vector.
 Variance coeffs then use ``max(|x0|, typical(s))`` without flooring at 1,
 where ``s`` is the robust response range, so response-scaled ``var_*`` stay
 equivariant at tiny and ELISA scales)
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
6. Scenario CSVs in `src/synfit/scenarios/` are committed *and* shipped as
   package data: edit `config.json` → `just generate-seed-data`; CI's
   `just check-seed-data` catches drift. They live inside the package so an
   installed wheel can reach them — keep the `package-data` globs in
   `pyproject.toml` in step with any new file type added there.

## Versioning

A change that moves fitted values is ≥ MINOR (see `CHANGELOG.md`). Tag `vX.Y.Z`
on `main` after merge — never a feature-branch tip.

## Commit messages

Write for a reader of this repository alone. `synfit` is a standalone library
here; it is not "the core" of anything, and a scope like `release(core):` only
means something to a project that vendors it. Use plain conventional-commit
scopes that name what changed — `fit`, `zip`, `loewe`, `ci`, `docs`, `build` —
or no scope at all. Earlier commits use `(core)`; that is history, not a pattern
to follow.

Releasing, in one commit on `main`, then the tag:

1. `pyproject.toml` `version`
2. `CITATION.cff` `version` **and** `date-released` — stale values are what
   people paste into a methods section
3. `CHANGELOG.md`: promote `[Unreleased]` to a dated `[X.Y.Z]` heading
4. `uv lock`, then tag `vX.Y.Z` once CI is green on that commit

Publishing to PyPI is a separate, deliberate step: `.github/workflows/publish.yml`
uploads only when a **GitHub release is published** for a tag, using PyPI trusted
publishing (no stored token). It refuses to upload when the tag does not match
`pyproject.toml`'s version, when the tagged commit is not an ancestor of `main`,
when there is no successful `ci.yml` run for that exact commit, or when
`twine check --strict` fails.

Artifact verification itself lives in `ci.yml`'s `dist-smoke` job, not in
`publish.yml`: installing the wheel in a clean venv, running the README examples,
installing *from the sdist* and running the suite it ships, and asserting the
`py.typed` marker is present in both artifacts. `publish.yml` does not repeat
those checks — it requires that the run which performed them succeeded for the
released SHA. Ancestry alone would not establish that, which is why the explicit
run check exists.

Manual dispatch can only target TestPyPI — production upload is release-only on
purpose, because a dispatch can run from any ref and would bypass those guards.
The `pypi` GitHub environment must exist with required reviewers before the
trusted publisher is enabled (an unreferenced environment is auto-created with no
protection), and tag `v*` should be protected by a ruleset so a tag cannot be
moved after its release is published. Actions are pinned to commit SHAs;
Dependabot proposes updates.
