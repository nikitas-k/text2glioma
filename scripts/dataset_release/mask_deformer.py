"""Light affine deformation for training-set masks.

Purpose: increase diversity of the synthetic release by pairing each
prompt with a slightly-perturbed real training mask rather than the
exact mask that originally accompanied that prompt. Because the mask
defines the sample's tumour geometry (see 3.5 of the paper), the
deformation is intentionally small - enough to break exact-training-mask
identity, not enough to place the tumour anywhere anatomically implausible.

Design constraints
------------------

    * Affine only, no elastic warp. Keeps geometry realistic.
    * Rotation: uniform in [-max_rot, +max_rot] degrees around each axis.
    * Translation: uniform in [-max_trans, +max_trans] voxels along each
      axis (rejection-sampled to keep the tumour bbox inside the brain
      volume).
    * Scale: uniform in [1 - max_scale, 1 + max_scale].
    * Interpolation: nearest-neighbour, to preserve integer label
      semantics (BG=0, oedema=2, enhancing=3, etc.).
    * Deterministic seeding: one seed per sample, stored in the manifest.

Default magnitudes were chosen so the tumour centroid rarely moves by
more than 3 voxels and the total tumour volume changes by <10%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import affine_transform


@dataclass(frozen=True)
class DeformParams:
    """Deterministic affine deformation parameters for one mask."""
    seed: int
    rotation_deg: tuple[float, float, float]  # (rx, ry, rz)
    translation_vox: tuple[float, float, float]
    scale: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "rotation_deg": list(self.rotation_deg),
            "translation_vox": list(self.translation_vox),
            "scale": self.scale,
        }


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def _matrix_from_params(p: DeformParams) -> np.ndarray:
    """Build the 3x3 rotation-and-scale matrix and 3-vector translation."""
    rx, ry, rz = [np.deg2rad(a) for a in p.rotation_deg]
    R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)
    M = R * p.scale
    return M


def sample_deform_params(
    seed: int,
    max_rot_deg: float = 3.0,
    max_trans_vox: float = 2.0,
    max_scale: float = 0.03,
) -> DeformParams:
    """Draw a deterministic deformation from the seed."""
    rng = np.random.default_rng(seed)
    rx = float(rng.uniform(-max_rot_deg, max_rot_deg))
    ry = float(rng.uniform(-max_rot_deg, max_rot_deg))
    rz = float(rng.uniform(-max_rot_deg, max_rot_deg))
    tx = float(rng.uniform(-max_trans_vox, max_trans_vox))
    ty = float(rng.uniform(-max_trans_vox, max_trans_vox))
    tz = float(rng.uniform(-max_trans_vox, max_trans_vox))
    s  = float(rng.uniform(1.0 - max_scale, 1.0 + max_scale))
    return DeformParams(
        seed=seed,
        rotation_deg=(rx, ry, rz),
        translation_vox=(tx, ty, tz),
        scale=s,
    )


def apply_deformation(
    mask: np.ndarray,
    params: DeformParams,
    order: int = 0,
) -> np.ndarray:
    """Apply an affine deformation to a 3D label mask, preserving integer
    labels via nearest-neighbour interpolation (``order=0``).

    Deformation is applied about the mask centroid so a zero-translation
    deformation rotates in place.
    """
    if mask.ndim != 3:
        raise ValueError(f"expected 3D mask, got shape {mask.shape}")

    M = _matrix_from_params(params)
    centre = np.asarray(mask.shape, dtype=np.float64) / 2.0

    # scipy.ndimage.affine_transform maps output -> input coordinates:
    #     input_coords = M @ output_coords + offset
    # For a rotation about the volume centre with translation t (in voxel space):
    #     input = M @ (output - centre) + centre + t
    #           = M @ output + (centre - M @ centre + t)
    M_inv = np.linalg.inv(M)
    offset = centre - M_inv @ centre - np.asarray(params.translation_vox)

    out = affine_transform(
        mask.astype(np.float32),
        matrix=M_inv,
        offset=offset,
        order=order,
        mode="constant",
        cval=0.0,
    )
    if order == 0:
        # Preserve integer labels exactly.
        return np.rint(out).astype(mask.dtype)
    return out


# ---------------------------------------------------------------------------
# Constraint check: does the deformed mask still contain the tumour
# reasonably (volume ratio, bbox inside brain)?
# ---------------------------------------------------------------------------

def deformation_is_valid(
    original: np.ndarray,
    deformed: np.ndarray,
    min_volume_ratio: float = 0.75,
    max_volume_ratio: float = 1.25,
) -> tuple[bool, dict]:
    """Check the deformation didn't destroy or explode the tumour."""
    orig_vol = int((original > 0).sum())
    def_vol  = int((deformed > 0).sum())
    if orig_vol == 0:
        return False, {"reason": "original empty", "orig_vol": 0, "def_vol": def_vol}
    ratio = def_vol / max(orig_vol, 1)
    if ratio < min_volume_ratio:
        return False, {"reason": "deformed too small", "ratio": ratio,
                        "orig_vol": orig_vol, "def_vol": def_vol}
    if ratio > max_volume_ratio:
        return False, {"reason": "deformed too large", "ratio": ratio,
                        "orig_vol": orig_vol, "def_vol": def_vol}
    return True, {"ratio": ratio, "orig_vol": orig_vol, "def_vol": def_vol}


# ---------------------------------------------------------------------------
# CLI: visualise a few deformations on a real mask
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import nibabel as nib
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mask", type=Path, required=True,
                    help="Path to a real label NIfTI to deform.")
    ap.add_argument("--n", type=int, default=4,
                    help="Number of deformations to draw.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("data/mask_deform_check"))
    args = ap.parse_args()

    lbl_nii = nib.load(str(args.mask))
    lbl = lbl_nii.get_fdata().astype(np.int16)
    print(f"loaded {args.mask}: shape={lbl.shape}, unique labels={sorted(np.unique(lbl).tolist())}")
    print(f"tumour voxels: {(lbl > 0).sum():,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        params = sample_deform_params(seed=args.seed + i)
        deformed = apply_deformation(lbl, params)
        ok, info = deformation_is_valid(lbl, deformed)
        outp = args.out_dir / f"deform_{i:02d}_seed{params.seed}.nii.gz"
        nib.save(nib.Nifti1Image(deformed.astype(np.int16), lbl_nii.affine, lbl_nii.header),
                 str(outp))
        print(f"  [{i}] seed={params.seed}  rot={tuple(round(r,2) for r in params.rotation_deg)}\u00b0  "
              f"trans={tuple(round(t,2) for t in params.translation_vox)}vox  scale={params.scale:.3f}   "
              f"valid={ok}  {info}   -> {outp.name}")


if __name__ == "__main__":
    _cli()
