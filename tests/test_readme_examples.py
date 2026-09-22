import subprocess
import sys
from pathlib import Path

from scripts.check_readme_examples import extract_python_blocks

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_readme_has_runnable_python_blocks():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    blocks = extract_python_blocks(text)
    assert blocks, "README should contain at least one ```python block"


def test_readme_python_blocks_execute():
    script = REPO_ROOT / "scripts" / "check_readme_examples.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
