import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
import torch


class NiftiSaver:
    """Utility for saving 3D volumes in NIfTI format."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.affine = np.array(
            [
                [-1.0, 0.0, 0.0, 96.48149872],
                [0.0, 1.0, 0.0, -141.47715759],
                [0.0, 0.0, 1.0, -156.55375671],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._executor = ThreadPoolExecutor()

    def save(self, image_data: torch.Tensor, file_name: str) -> None:
        """Save a tensor synchronously as a NIfTI file."""
        image_data = image_data.cpu().numpy()
        image_data = image_data[0, 0, :, :, :]
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
        image_data = (image_data * 255).astype(np.uint8)

        empty_header = nib.Nifti1Header()
        sample_nii = nib.Nifti1Image(image_data, self.affine, empty_header)
        nib.save(sample_nii, str(self.output_dir / f"{file_name}.nii.gz"))

    async def save_async(self, image_data: torch.Tensor, file_name: str) -> Any:
        """Asynchronously save a tensor by offloading to a background thread."""
        loop = asyncio.get_running_loop()
        image_cpu = image_data.cpu()
        return await loop.run_in_executor(
            self._executor, self.save, image_cpu, file_name
        )
