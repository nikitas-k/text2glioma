"""Inpainting task package.

Self-contained 3D latent-diffusion inpainting model trained on BraTS-GLI 2025
longitudinal pairs. Categorical conditioning only (no text encoder):

  - trajectory      : response / stable / progression
  - treatment_a/b   : pre / post

Public entrypoint: ``train_inpainting_ddp.main``.

This package deliberately does NOT depend on the Stage-2 text-conditioned
trainer (``text2glioma.training.train_stage2_ddp``) — it forks only the data
plumbing and the diffusion loop, dropping the RadBERT pipeline entirely.
"""
