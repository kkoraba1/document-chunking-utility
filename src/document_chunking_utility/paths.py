"""Cross-platform path formatting helpers.

Filesystem operations should use pathlib.Path, while exported paths should use
forward slashes so the output stays consistent across operating systems.
"""

from pathlib import Path


def to_portable_path(path: str | Path) -> str:
    """Convert a path to a consistent forward-slash format."""
    return str(path).replace("\\", "/")