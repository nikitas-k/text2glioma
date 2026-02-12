import pathlib
import nibabel as nib
import numpy as np
import torch

class NiftiSaver:
    """
    Minimal, robust NIfTI saver for MONAI-style batches.
    - uses meta['affine'] when provided (recommended),
    - saves float32 by default (no quantisation),
    - supports labels (uint16, no rescale),
    - works with shapes [B,C,D,H,W] / [C,D,H,W] / [D,H,W].
    """
    def __init__(self, output_dir: str, default_affine: np.ndarray = None,
                 rescale: bool = True, dtype: str = "float32") -> None:
        """
        rescale: if True, min-max to [0,1] before save (ignored for labels).
        dtype: 'float32' (recommended) or 'uint8' if you must quantise.
        """
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_affine = np.array(
            [
                [-1., 0.0, 0.0, 96.48149872],
                [0.0, 1., 0.0, -141.47715759],
                [0.0, 0.0, 1., -156.55375671],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.rescale = bool(rescale)
        assert dtype in ("float32", "uint8"), "dtype must be 'float32' or 'uint8'"
        self.dtype = dtype

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """Convert to numpy. Returns [C,D,H,W] for multi-channel or [D,H,W] for single."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 5:   # [B,C,D,H,W]
            x = x[0]      # [C,D,H,W]
        # If single-channel [1,D,H,W], squeeze to [D,H,W]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        # x is now [C,D,H,W] (multi-ch) or [D,H,W] (single / raw 3D)
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _affine_from_meta(meta: dict, fallback: np.ndarray) -> np.ndarray:
        if isinstance(meta, dict) and "affine" in meta and meta["affine"] is not None:
            A = np.array(meta["affine"], dtype=np.float32)
            if A.shape == (4, 4):
                return A
        return fallback

    def save(self, image, file_name: str,
             meta: dict = None, is_label: bool = False) -> str:
        """Save image as NIfTI. Multi-channel [C,D,H,W] → 4D NIfTI [D,H,W,C]."""
        vol = self._to_numpy(image)  # [C,D,H,W] or [D,H,W]
        A = self._affine_from_meta(meta, self.default_affine)

        if is_label:
            data = vol.astype(np.uint16, copy=False)
        else:
            data = vol.copy()
            if self.rescale:
                if data.ndim == 4:  # [C,D,H,W] → rescale per channel
                    for c in range(data.shape[0]):
                        vmin, vmax = float(data[c].min()), float(data[c].max())
                        denom = max(vmax - vmin, 1e-8)
                        data[c] = (data[c] - vmin) / denom
                else:
                    vmin, vmax = float(data.min()), float(data.max())
                    denom = max(vmax - vmin, 1e-8)
                    data = (data - vmin) / denom
            data = (data * 255).astype(np.float32 if self.dtype == "float32" else np.uint8)

        # Crop border artefacts
        if data.ndim == 4:  # [C,D,H,W]
            data = data[:, 5:-5, 5:-5, :-10]
            # NIfTI convention: channels in last dim → [D,H,W,C]
            data = np.transpose(data, (1, 2, 3, 0))
        else:  # [D,H,W]
            data = data[5:-5, 5:-5, :-10]

        img = nib.Nifti1Image(data, A)
        hdr = img.header
        hdr.set_xyzt_units("mm")
        img.set_qform(A, code=1)
        img.set_sform(A, code=1)

        out_path = self.output_dir / f"{file_name}"
        nib.save(img, str(out_path))
        return str(out_path)