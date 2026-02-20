"""Version information for the bundled rlmagents CLI."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version


def _resolve_version() -> str:
    for package_name in ("rlmagents", "rlmagents-cli"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue
    return "0.0.4"


__version__ = _resolve_version()
