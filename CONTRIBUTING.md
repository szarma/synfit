# Contributing to synfit

## Setup

```bash
git clone https://github.com/szarma/synfit.git
cd synfit
uv sync --extra dev
```

## Commands

```bash
just check               # full gate: README examples + all tests
just test                # tests only
just test-coverage       # tests with a coverage report
just check-readme-examples   # execute the README's python blocks
just generate-seed-data  # regenerate the shipped scenario CSVs from their configs
just check-seed-data     # verify those CSVs match their configs (CI runs this)
just help                # every recipe
```

CI runs the suite on Python 3.11, 3.12 and 3.13, and separately installs the
built wheel in a clean environment to check that the README's examples work
against an installed package and that the shipped scenarios are reachable from
it.

## Layout

```
src/synfit/
├── hill.py            # Hill equation, log_wall prior, concentration series
├── fitting.py         # FitBase: scipy minimize, log-prior, bounded optimisation
├── single.py          # SingleDrugFit, SingleDrugFitWithError
├── matrix.py          # MatrixFit: drug-drug interaction fitting
├── joint_marginal.py  # JointMarginalFit
├── noise.py           # NoiseSpec union, serde, profiled/weighted likelihood
├── data.py            # FitConfig, FitBounds, FitResult
├── bliss.py           # Bliss independence, HSA
├── loewe.py           # Loewe reference, combination index
├── zip.py             # ZIP reference and δ-scores
├── synthetic.py       # data generation and the shipped scenario accessors
├── plotting.py        # Matplotlib dose-response curves and heatmaps
└── scenarios/         # the shipped datasets; */config.json is the ground truth
tests/                 # mirrors the modules above
scripts/
├── generate_seed_datasets.py  # scenario generator (--check verifies)
└── check_readme_examples.py   # executes the README's python blocks
```

`AGENTS.md` documents the `FitBase` contract and the gotchas that are easy to
break — read it before changing a fitter.

## Scenarios

The datasets under `src/synfit/scenarios/` are committed *and* shipped as
package data. `config.json` is the source of truth; the CSVs are derived.
Edit a config, run `just generate-seed-data`, and commit both. Adding a new
file type there means updating the `package-data` globs in `pyproject.toml`,
or it will be missing from the wheel without anything failing to import.

## Versioning

SemVer, with one domain-specific rule: **a change that moves fitted values**
(optimiser, objective, priors, bounds, noise model) **is at least MINOR, never
a PATCH** — downstream results shift even when the public API does not.

A release promotes `CHANGELOG.md`'s `[Unreleased]` to a dated `[X.Y.Z]`
heading, bumps `pyproject.toml` and `CITATION.cff`, and is tagged `vX.Y.Z` on
`main` after CI passes there. The tag triggers the publish workflow, which
uploads to PyPI through trusted publishing — never a local token.

## Commit messages

Write for a reader of this repository alone. `synfit` is a standalone library;
a scope like `release(core):` only means something to a project that vendors
it. Use scopes that name what changed — `fit`, `zip`, `loewe`, `ci`, `docs`,
`build` — or no scope at all.
