#!/usr/bin/env python
"""Thin launcher for torchrun_nccl.sh compatibility (matches launch_inpainting_ddp.py)."""
from text2glioma.inpainting.train_pixel_inpainting_ddp import main

if __name__ == "__main__":
    main()
