"""Preprocessing utilities for text2glioma."""

from importlib.resources import files as _pkg_files
from pathlib import Path


def get_atlas_dir() -> str:
    """Return the absolute path to the bundled atlas_masks directory.

    The masks are shipped as package data under
    ``text2glioma/preprocessing/atlas_masks/`` and are copied into the
    user's site-packages on ``pip install``.
    """
    return str(Path(_pkg_files("text2glioma.preprocessing").joinpath("atlas_masks")))
