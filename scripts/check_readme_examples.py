"""
Execute every ```python fenced block in README.md and docs/*.md.

Usage:
    uv run python scripts/check_readme_examples.py
    python scripts/check_readme_examples.py   # against the installed synfit
    python scripts/check_readme_examples.py docs/tutorials.md
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

_BLOCK_RE = re.compile(r"^```python\s*\n(.*?)```", re.MULTILINE | re.DOTALL)


def extract_python_blocks(markdown: str) -> list[str]:
    return [m.group(1) for m in _BLOCK_RE.finditer(markdown)]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = (
        [Path(arg).resolve() for arg in argv]
        if argv else [README, *sorted((REPO_ROOT / "docs").glob("*.md"))]
    )
    total = 0
    for path in paths:
        if not path.is_file():
            print(f"Documentation not found: {path}", file=sys.stderr)
            return 1

        blocks = extract_python_blocks(path.read_text(encoding="utf-8"))
        # Reference pages can contain signatures without executable examples.
        if not blocks and (argv or path == README):
            print(f"No python code blocks in {path}", file=sys.stderr)
            return 1

        for i, code in enumerate(blocks, start=1):
            print(f"Running {path.name} python block {i}/{len(blocks)}...")
            # Examples must work with the installed package, without checkout
            # paths. Keep generated images out of the working tree during checks.
            with tempfile.TemporaryDirectory(prefix="synfit-doc-") as workdir:
                proc = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                )
            if proc.returncode != 0:
                print(proc.stdout, end="")
                print(proc.stderr, end="", file=sys.stderr)
                print(f"{path.name} python block {i} failed (exit {proc.returncode})", file=sys.stderr)
                return proc.returncode
        total += len(blocks)

    print(f"All {total} documentation python block(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
