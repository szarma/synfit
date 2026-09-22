"""
Execute every ```python fenced block in README.md (smoke test for docs).

Usage:
    uv run python scripts/check_readme_examples.py
    python scripts/check_readme_examples.py   # against whichever synfit is on PYTHONPATH
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

_BLOCK_RE = re.compile(r"^```python\s*\n(.*?)```", re.MULTILINE | re.DOTALL)


def extract_python_blocks(markdown: str) -> list[str]:
    return [m.group(1) for m in _BLOCK_RE.finditer(markdown)]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    readme = README
    if argv:
        readme = Path(argv[0]).resolve()

    if not readme.is_file():
        print(f"README not found: {readme}", file=sys.stderr)
        return 1

    blocks = extract_python_blocks(readme.read_text(encoding="utf-8"))
    if not blocks:
        print(f"No python code blocks in {readme}", file=sys.stderr)
        return 1

    for i, code in enumerate(blocks, start=1):
        print(f"Running README python block {i}/{len(blocks)}...")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
            print(f"README python block {i} failed (exit {proc.returncode})", file=sys.stderr)
            return proc.returncode

    print(f"All {len(blocks)} README python block(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
