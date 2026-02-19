"""Utilities for project root detection and project-specific configuration."""

from pathlib import Path


def find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific agent.md file(s).

    Checks locations and returns ALL that exist:
    1. project_root/.rlmagents/agent.md
    2. project_root/agent.md

    Files will be loaded and combined in that order.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of paths to project agent.md files (may contain 0, 1, or 2 paths).
    """
    paths = []

    # Check .rlmagents/agent.md (preferred)
    rlmagents_md = project_root / ".rlmagents" / "agent.md"
    if rlmagents_md.exists():
        paths.append(rlmagents_md)

    # Check root agent.md (fallback, but also include if both exist)
    root_md = project_root / "agent.md"
    if root_md.exists():
        paths.append(root_md)

    return paths
