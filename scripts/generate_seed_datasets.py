"""
Generate (or verify) committed synthetic seed CSVs from config.json files.

Usage:
    uv run python scripts/generate_seed_datasets.py              # regenerate all
    uv run python scripts/generate_seed_datasets.py --check      # CI: verify up-to-date
    uv run python scripts/generate_seed_datasets.py --scenario single_drug_clean

Each scenario lives in src/synfit/scenarios/<name>/config.json and ships
with the package.
For single_drug: generates <name>.csv (concentration, y, replicate).
For matrix:      generates rep1.csv, rep2.csv, ... (whitespace-separated 2D arrays).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Locate package relative to this script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from synfit.synthetic import SCENARIOS_DIR, generate_single_drug, generate_matrix


def _csv_float_format(value) -> str:
    """Stable decimal text for float columns (portable across OS / pandas builds)."""
    return np.format_float_positional(float(value), precision=17, unique=True, trim="-")


def load_config(scenario_dir: Path) -> dict:
    return json.loads((scenario_dir / "config.json").read_text())


def render_single_drug(config: dict) -> bytes:
    df = generate_single_drug(config)
    buf = io.StringIO()
    df.to_csv(
        buf,
        index=False,
        lineterminator="\n",
        float_format=_csv_float_format,
    )
    return buf.getvalue().encode()


def render_matrix_reps(config: dict) -> list[tuple[str, bytes]]:
    replicates, conc_hor, conc_ver = generate_matrix(config)
    files = []
    for i, rep in enumerate(replicates, start=1):
        buf = io.StringIO()
        pd.DataFrame(rep).to_csv(
            buf,
            index=False,
            header=False,
            sep="\t",
            lineterminator="\n",
            float_format=_csv_float_format,
        )
        files.append((f"rep{i}.csv", buf.getvalue().encode()))
    return files


def generate_scenario(scenario_dir: Path) -> dict[str, bytes]:
    """Return {filename: content} for all output files of a scenario."""
    config = load_config(scenario_dir)
    kind = config.get("kind")
    if kind == "single_drug":
        name = scenario_dir.name
        return {f"{name}.csv": render_single_drug(config)}
    elif kind == "matrix":
        return {name: content for name, content in render_matrix_reps(config)}
    else:
        raise ValueError(f"Unknown scenario kind: {kind!r} in {scenario_dir}")


def write_scenario(scenario_dir: Path, files: dict[str, bytes]) -> None:
    for filename, content in files.items():
        (scenario_dir / filename).write_bytes(content)


def _df_bytes_equivalent(kind: str, disk: bytes, expected: bytes) -> bool:
    """
    True if disk and expected decode to the same numeric/text grid.

    Used in --check when raw bytes differ due to benign CSV formatting
    (line endings, float text) or tiny float round-off between runners.
    """
    if disk == expected:
        return True

    def _norm(b: bytes) -> bytes:
        return b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    disk_n, exp_n = _norm(disk), _norm(expected)
    if disk_n == exp_n:
        return True

    buf_d, buf_e = io.BytesIO(disk_n), io.BytesIO(exp_n)
    try:
        if kind == "single_drug":
            df_d = pd.read_csv(buf_d)
            df_e = pd.read_csv(buf_e)
        elif kind == "matrix":
            df_d = pd.read_csv(buf_d, sep="\t", header=None)
            df_e = pd.read_csv(buf_e, sep="\t", header=None)
        else:
            return False
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False

    if df_d.shape != df_e.shape or list(df_d.columns) != list(df_e.columns):
        return False

    for col in df_d.columns:
        s_d, s_e = df_d[col], df_e[col]
        if pd.api.types.is_numeric_dtype(s_d) and pd.api.types.is_numeric_dtype(s_e):
            a_d = s_d.to_numpy(dtype=np.float64)
            a_e = s_e.to_numpy(dtype=np.float64)
            if not np.allclose(a_d, a_e, rtol=1e-12, atol=1e-12, equal_nan=True):
                return False
        else:
            if not s_d.astype(str).equals(s_e.astype(str)):
                return False
    return True


def check_scenario(scenario_dir: Path, files: dict[str, bytes], *, kind: str) -> list[str]:
    """Return list of issues (empty = OK)."""
    issues = []
    for filename, expected in files.items():
        path = scenario_dir / filename
        if not path.exists():
            issues.append(f"  Missing: {path.relative_to(REPO_ROOT)}")
        else:
            disk = path.read_bytes()
            if disk != expected and not _df_bytes_equivalent(kind, disk, expected):
                issues.append(f"  Stale:   {path.relative_to(REPO_ROOT)}")
    return issues


def discover_scenarios() -> list[Path]:
    return sorted(
        d for d in SCENARIOS_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Verify CSVs are up-to-date with configs (CI mode, exits non-zero on failure)")
    parser.add_argument("--scenario", metavar="NAME",
                        help="Process only this scenario (default: all)")
    args = parser.parse_args()

    scenarios = discover_scenarios()
    if not scenarios:
        print(f"No scenarios found under {SCENARIOS_DIR}")
        sys.exit(1)

    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            print(f"Scenario not found: {args.scenario}")
            sys.exit(1)

    all_issues = []

    for scenario_dir in scenarios:
        try:
            files = generate_scenario(scenario_dir)
        except Exception as exc:
            print(f"ERROR generating {scenario_dir.name}: {exc}")
            all_issues.append(f"  Error: {scenario_dir.name}: {exc}")
            continue

        if args.check:
            issues = check_scenario(scenario_dir, files, kind=load_config(scenario_dir).get("kind", ""))
            if issues:
                print(f"FAIL  {scenario_dir.name}")
                all_issues.extend(issues)
            else:
                print(f"OK    {scenario_dir.name}")
        else:
            write_scenario(scenario_dir, files)
            filenames = ", ".join(files)
            print(f"Generated {scenario_dir.name}: {filenames}")

    if args.check:
        if all_issues:
            print("\nSeed data check FAILED. Run: uv run python scripts/generate_seed_datasets.py")
            for issue in all_issues:
                print(issue)
            sys.exit(1)
        else:
            print("\nAll seed data up-to-date.")
    else:
        print(f"\nDone. {len(scenarios)} scenario(s) generated.")


if __name__ == "__main__":
    main()
