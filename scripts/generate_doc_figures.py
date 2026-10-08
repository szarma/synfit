"""Regenerate committed figures from the Python blocks displayed in docs.

Each block must end with a synfit plotting call; the PNG bytes it returns are
written unchanged.

Usage: uv run python scripts/generate_doc_figures.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from check_readme_examples import REPO_ROOT, extract_python_blocks


def pypi_readme() -> str:
    """Return the package README with version-pinned public image URLs."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    image_root = f"https://raw.githubusercontent.com/szarma/synfit/v{version}/"
    return readme.replace("](docs/images/", f"]({image_root}docs/images/")


def write_pypi_readme(*, check: bool) -> None:
    destination = REPO_ROOT / "README.pypi.md"
    content = pypi_readme()
    if check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            raise ValueError("README.pypi.md is out of date; run just generate-doc-figures")
        return
    destination.write_text(content, encoding="utf-8")
    print("Generated README.pypi.md from README.md")


def main() -> None:
    figures = {
        "README.md": "synthetic-fit.png",
        "docs/tutorials.md": "synergy-analysis.png",
    }
    destination = REPO_ROOT / "docs" / "images"
    destination.mkdir(parents=True, exist_ok=True)
    for markdown, filename in figures.items():
        blocks = extract_python_blocks((REPO_ROOT / markdown).read_text(encoding="utf-8"))
        matches = [code for code in blocks if code.startswith(f"# Generates {filename}\n")]
        if len(matches) != 1:
            raise ValueError(f"Expected one '# Generates {filename}' block in {markdown}")
        with tempfile.TemporaryDirectory(prefix="synfit-figure-") as workdir:
            # Rewrite the final expression so a PNG-returning helper can be
            # captured without adding generator plumbing to the documentation.
            code = (
                "import ast\n"
                f"_doc_source = {matches[0]!r}\n"
                "_doc_tree = ast.parse(_doc_source)\n"
                "_doc_last = _doc_tree.body[-1]\n"
                "if not isinstance(_doc_last, ast.Expr):\n"
                "    raise TypeError('The figure example must end with a plotting expression')\n"
                "_doc_tree.body[-1] = ast.Assign(\n"
                "    targets=[ast.Name(id='_doc_output', ctx=ast.Store())], value=_doc_last.value\n"
                ")\n"
                "ast.fix_missing_locations(_doc_tree)\n"
                f"exec(compile(_doc_tree, {markdown!r}, 'exec'))\n"
                "if not isinstance(_doc_output, bytes):\n"
                "    raise TypeError('The plotting expression must return PNG bytes')\n"
                f"open({filename!r}, 'wb').write(_doc_output)\n"
            )
            subprocess.run([sys.executable, "-c", code], cwd=workdir, check=True)
            image = (Path(workdir) / filename).read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{filename} is not a PNG image")
        target = destination / filename
        target.write_bytes(image)
        print(f"Generated docs/images/{filename} from {markdown}")
    write_pypi_readme(check=False)


if __name__ == "__main__":
    if "--generate-pypi-readme" in sys.argv[1:]:
        write_pypi_readme(check=False)
    else:
        main()
