"""Smoke tests for the bundled CLI package."""

import subprocess
import sys
from pathlib import Path

import deepagents_cli
from deepagents_cli import __version__, cli_main


def test_cli_module_is_available() -> None:
    """`rlmagents` should bundle the CLI module."""
    assert callable(cli_main)


def test_cli_assets_are_packaged() -> None:
    """Bundled CLI assets should exist in the installed package tree."""
    package_dir = Path(deepagents_cli.__file__).resolve().parent
    assert (package_dir / "app.tcss").is_file()
    assert (package_dir / "default_agent_prompt.md").is_file()
    assert (package_dir / "system_prompt.md").is_file()


def test_cli_version_flag_uses_rlmagents_name() -> None:
    """`--version` should report the rlmagents package name."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagents_cli.main", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"rlmagents {__version__}" in result.stdout
