import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Sequence, Optional, Tuple


def voxel_volume_ml(vox: Sequence[float]) -> float:
    """Return the volume of a voxel in millilitres.

    Parameters
    ----------
    vox: Sequence[float]
        Voxel spacing ``(vx, vy, vz)`` in millimetres.
    """
    vx, vy, vz = map(float, vox)
    return (vx * vy * vz) / 1000.0


def detect_ring_enhancement(
    lab: np.ndarray,
    core_label: int = 1,
    enh_label: int = 3,
    shell_radius_vox: int = 3,
    inner_clear_vox: int = 1,
    min_ring_frac: float = 0.30,
    min_touch_frac: float = 0.50,
) -> Tuple[bool, float, float]:
    """Detect ring enhancement around a tumour core.

    Parameters
    ----------
    lab: np.ndarray
        Label volume.
    core_label: int, optional
        Label id for the non‑enhancing core.
    enh_label: int, optional
        Label id for the enhancing tumour.
    shell_radius_vox: int, optional
        Radius of the shell around the core in voxels.
    inner_clear_vox: int, optional
        Erosion radius for the core when forming the shell.
    min_ring_frac: float, optional
        Minimum fraction of the shell that must be enhancing to call a ring.
    min_touch_frac: float, optional
        Minimum fraction of enhancing voxels that touch the core.

    Returns
    -------
    Tuple[bool, float, float]
        ``(is_ring, ring_fraction, touch_fraction)``.
    """
    core = lab == core_label
    enh = lab == enh_label
    if not np.any(enh):
        return False, 0.0, 0.0

    dil_core = binary_dilation(core, iterations=shell_radius_vox)
    if inner_clear_vox > 0:
        ero_core = binary_erosion(core, iterations=inner_clear_vox)
    else:
        ero_core = core

    shell = dil_core & (~ero_core)
    shell_vox = shell.sum()
    if shell_vox == 0:
        return False, 0.0, 0.0

    ring_frac_shell = float(np.logical_and(enh, shell).sum()) / float(shell_vox)
    touch_frac = float(np.logical_and(enh, dil_core).sum()) / float(enh.sum())

    is_ring = (ring_frac_shell >= min_ring_frac) and (touch_frac >= min_touch_frac)
    return bool(is_ring), float(ring_frac_shell), float(touch_frac)


def edema_severity_from_labels(
    lab: np.ndarray,
    vox: Sequence[float],
    edema_label: int = 2,
    brain_volume_ml: Optional[float] = None,
) -> Tuple[str, float, Optional[float]]:
    """Classify edema severity given pre‑loaded labels and voxel size.

    Parameters
    ----------
    lab: np.ndarray
        Label volume.
    vox: Sequence[float]
        Voxel spacing ``(vx, vy, vz)`` in millimetres.
    edema_label: int, optional
        Label id representing edema.
    brain_volume_ml: Optional[float], optional
        Total brain volume in millilitres for relative fractions.

    Returns
    -------
    Tuple[str, float, Optional[float]]
        ``(severity, edema_ml, edema_fraction)``.  The fraction is ``None``
        when ``brain_volume_ml`` is not provided.
    """
    ed_vox = int(np.sum(lab == edema_label))
    ed_ml = ed_vox * voxel_volume_ml(vox)

    if brain_volume_ml is not None and brain_volume_ml > 0:
        frac = ed_ml / brain_volume_ml
        if frac >= 0.20:
            sev = "severe"
        elif frac >= 0.10:
            sev = "moderate"
        elif frac >= 0.03:
            sev = "mild"
        else:
            sev = "none"
        return sev, ed_ml, frac

    if ed_ml >= 100.0:
        sev = "severe"
    elif ed_ml >= 50.0:
        sev = "moderate"
    elif ed_ml >= 20.0:
        sev = "mild"
    else:
        sev = "none"
    return sev, ed_ml, None


def estimate_midline_shift_mm(
    image: np.ndarray,
    lab: np.ndarray,
    vox: Sequence[float],
    tumour_label_any: Sequence[int] = (1, 2, 4),
    search_vox: int = 15,
) -> Tuple[float, str, Optional[int]]:
    """Estimate the midline shift (MLS) in millimetres.

    Parameters
    ----------
    image: np.ndarray
        3‑D image volume in LPS orientation.
    lab: np.ndarray
        Corresponding label volume.
    vox: Sequence[float]
        Voxel spacing ``(vx, vy, vz)`` in millimetres.
    tumour_label_any: Sequence[int], optional
        Labels considered tumour.
    search_vox: int, optional
        Number of voxels either side of centre to search.

    Returns
    -------
    Tuple[float, str, Optional[int]]
        ``(mls_mm, direction, z_index)``.  ``z_index`` is ``None`` if no
        tumour is present.
    """
    X, Y, Z = image.shape
    vx = float(vox[0])

    tumour = np.isin(lab, tumour_label_any)
    if not np.any(tumour):
        return 0.0, "none", None

    areas = tumour.sum(axis=(0, 1))
    z = int(np.argmax(areas))
    I = image[:, :, z].astype(np.float32)

    p1, p99 = np.percentile(I, [1, 99])
    I = np.clip((I - p1) / (p99 - p1 + 1e-6), 0, 1)

    x_mid0 = (X - 1) / 2.0
    best_cc, best_cx = -1.0, x_mid0
    for cx in range(int(x_mid0 - search_vox), int(x_mid0 + search_vox) + 1):
        cx = int(np.clip(cx, 1, X - 2))
        w = int(min(cx, X - cx))
        if w < 8:
            continue
        L = I[cx - w:cx, :]
        R = I[cx:cx + w, :]
        Lf = L[::-1, :]
        L0 = Lf - Lf.mean()
        R0 = R - R.mean()
        denom = np.linalg.norm(L0) * np.linalg.norm(R0) + 1e-6
        cc = float((L0 * R0).sum() / denom)
        if cc > best_cc:
            best_cc, best_cx = cc, cx

    delta_vox = best_cx - x_mid0
    mls_mm = abs(delta_vox) * vx
    direction = "leftward" if delta_vox > 0 else ("rightward" if delta_vox < 0 else "none")
    return float(mls_mm), direction, z
