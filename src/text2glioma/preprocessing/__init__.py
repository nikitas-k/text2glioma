"""Preprocessing utilities for text2glioma."""

from importlib.resources import files as _pkg_files
from pathlib import Path

_ATLAS_PKG_DIR = Path(_pkg_files("text2glioma.preprocessing").joinpath("atlas_masks"))


def get_atlas_dir(space: str = "sri24") -> str:
    """Return the absolute path to the bundled atlas masks for a given space.

    Parameters
    ----------
    space : str
        ``"sri24"`` (default, for BraTS data) or ``"mni152"`` (original
        MNI-152 registered masks).

    The masks are shipped as package data under
    ``text2glioma/preprocessing/atlas_masks/{space}/``.
    """
    d = _ATLAS_PKG_DIR / space
    if not d.exists():
        raise FileNotFoundError(f"Atlas space '{space}' not found at {d}")
    return str(d)
