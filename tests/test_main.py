"""Tests for running the package as a Python module."""

import os
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJ_ROOT / "src"


def test_package_can_be_run_as_python_module() -> None:
    run_env = os.environ.copy()

    old_pythonpath = run_env.get("PYTHONPATH", "")

    if old_pythonpath:
        run_env["PYTHONPATH"] = (
            str(SRC_DIR)
            + os.pathsep
            + old_pythonpath
        )
    else:
        run_env["PYTHONPATH"] = str(SRC_DIR)

    run_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_chunking_utility",
            "--help",
        ],
        cwd=PROJ_ROOT,
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )

    help_text = run_res.stdout + run_res.stderr

    assert run_res.returncode == 0
    assert (
        "Split text and Markdown documents into metadata-rich chunks."
        in help_text
    )
    assert "chunk" in help_text
    assert "inspect" in help_text