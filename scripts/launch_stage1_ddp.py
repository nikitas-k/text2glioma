#!/usr/bin/env python
"""Thin launcher for torchrun_nccl.sh compatibility."""
from text2glioma.training.train_stage1_ddp import main

if __name__ == "__main__":
    main()
