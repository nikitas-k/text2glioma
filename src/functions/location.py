import numpy as np
from typing import Sequence, Tuple, Optional


def tumour_lobe_hemisphere(
    lab: np.ndarray,
    vox: Sequence[float],
    tumour_labels: Sequence[int] = (1, 2, 3, 4),
    bilateral_frac: float = 0.10,
    midline_mm: float = 5.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Determine tumour laterality and lobe.

    Parameters
    ----------
    lab: np.ndarray
        Label volume in LPS orientation.
    vox: Sequence[float]
        Voxel spacing ``(vx, vy, vz)`` in millimetres.
    tumour_labels: Sequence[int], optional
        Label ids considered part of the tumour.
    bilateral_frac: float, optional
        Minimum fraction of voxels on each side to call the lesion bilateral.
    midline_mm: float, optional
        Maximum distance from midline to be considered a midline lesion.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        ``(hemisphere, lobe)`` where each element is one of
        ``{'left','right','bilateral','midline'}`` for hemisphere and
        ``{'frontal','temporal','parietal','occipital'}`` for lobe. ``None`` is
        returned when no tumour voxels are present.
    """
    mask = np.isin(lab, tumour_labels)
    if not np.any(mask):
        return None, None

    X, Y, Z = mask.shape
    x_mid = (X - 1) / 2.0

    left_vox = mask[int(np.floor(x_mid)) + 1 :, :, :].sum()
    right_vox = mask[: int(np.ceil(x_mid)), :, :].sum()
    total_vox = mask.sum()
    lf = float(left_vox) / float(total_vox)
    rf = float(right_vox) / float(total_vox)

    com = np.mean(np.argwhere(mask), axis=0)
    x_dist_mm = abs(com[0] - x_mid) * float(vox[0])

    if lf > bilateral_frac and rf > bilateral_frac:
        hemisphere = "bilateral"
    elif x_dist_mm <= midline_mm:
        hemisphere = "midline"
    else:
        hemisphere = "left" if lf >= rf else "right"

    y_frac = float(com[1] / max(Y - 1, 1))
    z_frac = float(com[2] / max(Z - 1, 1))
    if y_frac <= 1 / 3:
        lobe = "frontal"
    elif y_frac >= 2 / 3:
        lobe = "occipital"
    else:
        lobe = "temporal" if z_frac < 0.45 else "parietal"

    return hemisphere, lobe
