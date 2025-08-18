import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
from typing import Optional


def localise_pathology(
    healthy_img: np.ndarray,
    diseased_img: np.ndarray,
    threshold: float,
    *,
    refine: bool = True,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Locate pathological regions by comparing two images.

    Parameters
    ----------
    healthy_img: np.ndarray
        Baseline image representing healthy anatomy.
    diseased_img: np.ndarray
        Image containing potential pathology.
    threshold: float
        Absolute intensity difference threshold used to create the binary mask.
    refine: bool, optional
        When ``True`` (default), apply morphological opening and closing to
        remove small spurious regions and fill holes.
    iterations: int, optional
        Number of iterations for the morphological operations.
    structure: np.ndarray, optional
        Structuring element used for morphology. By default a cross-shaped
        element is generated based on the input dimensionality.

    Returns
    -------
    np.ndarray
        Boolean array indicating the lesion mask.
    """
    diff = np.abs(np.asarray(diseased_img, dtype=float) - np.asarray(healthy_img, dtype=float))
    mask = diff > float(threshold)

    if refine:
        if structure is None:
            structure = generate_binary_structure(mask.ndim, 1)
        mask = binary_opening(mask, structure=structure, iterations=iterations)
        mask = binary_closing(mask, structure=structure, iterations=iterations)
    return mask
