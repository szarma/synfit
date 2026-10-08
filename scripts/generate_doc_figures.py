"""Regenerate committed figures from the Python blocks displayed in docs.

The blocks build figures only; this script saves and closes the active figure.

Usage: uv run python scripts/generate_doc_figures.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from check_readme_examples import REPO_ROOT, extract_python_blocks


def main() -> None:
    figures = {
        "README.md": ("synthetic-fit.png", 160),
        "docs/tutorials.md": ("synergy-analysis.png", 150),
    }
    destination = REPO_ROOT / "docs" / "images"
    destination.mkdir(parents=True, exist_ok=True)
    for markdown, (filename, dpi) in figures.items():
        blocks = extract_python_blocks((REPO_ROOT / markdown).read_text(encoding="utf-8"))
        matches = [code for code in blocks if code.startswith(f"# Generates {filename}\n")]
        if len(matches) != 1:
            raise ValueError(f"Expected one '# Generates {filename}' block in {markdown}")
        with tempfile.TemporaryDirectory(prefix="synfit-figure-") as workdir:
            code = matches[0] + (
                "\nfrom matplotlib import pyplot as _doc_pyplot\n"
                f"_doc_pyplot.gcf().savefig({filename!r}, dpi={dpi})\n"
                "_doc_pyplot.close('all')\n"
            )
            subprocess.run([sys.executable, "-c", code], cwd=workdir, check=True)
            image = (Path(workdir) / filename).read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{filename} is not a PNG image")
        (destination / filename).write_bytes(image)
        print(f"Generated docs/images/{filename} from {markdown}")


if __name__ == "__main__":
    main()
