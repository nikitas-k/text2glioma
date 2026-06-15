"""Categorical conditioning encoder for the inpainting LDM.

Replaces the RadBERT text-encoding path. The DiffusionModelUNet's cross-
attention layers expect a context tensor of shape ``[B, seq_len, dim]``; we
synthesise that from three small ``nn.Embedding`` tables (one per categorical
conditioner) plus a learned NULL token used for classifier-free guidance (CFG)
dropout.

Token layout (seq_len = 3):
    position 0  -> trajectory      (3 classes:  response/stable/progression)
    position 1  -> treatment_a     (2 classes:  pre/post)
    position 2  -> treatment_b     (2 classes:  pre/post)

CFG dropout
-----------
``forward_with_dropout`` applies a separate Bernoulli per conditioner per
sample. When a dropout fires, that position is replaced by its learned NULL
embedding. The three streams are independent so we can guide on any subset at
inference time (e.g. fix treatment_a but classifier-free-guide on trajectory).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


N_TRAJECTORY = 3       # response, stable, progression
N_TREATMENT = 2        # pre, post

# Per-position NULL token indices are appended past the real class indices in
# each embedding table. With size = (real_classes + 1) the last row stores the
# learned NULL token used for CFG dropout.
NULL_IDX_TRAJECTORY = N_TRAJECTORY
NULL_IDX_TREATMENT = N_TREATMENT


class CategoricalConditioningEncoder(nn.Module):
    """Map ``(trajectory, treatment_a, treatment_b)`` triples to a UNet context.

    Output shape ``[B, 3, embed_dim]`` matches the ``context`` argument of
    ``generative.networks.nets.DiffusionModelUNet.forward`` whose cross-
    attention layers were configured with ``cross_attention_dim = embed_dim``.

    The class also exposes ``get_uncond_context(batch_size, device)`` which
    returns an all-NULL context, used by ``log_inpainting_sample`` for the
    unconditional branch of CFG sampling.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        # Each table has one extra row for the learned NULL token used by CFG.
        self.traj_embed = nn.Embedding(N_TRAJECTORY + 1, self.embed_dim)
        self.treat_a_embed = nn.Embedding(N_TREATMENT + 1, self.embed_dim)
        self.treat_b_embed = nn.Embedding(N_TREATMENT + 1, self.embed_dim)
        # Small init so the early-training context magnitudes don't dwarf the
        # zero-initialised mask/masked-image channels (cf. conv_in zero-init).
        nn.init.normal_(self.traj_embed.weight, std=0.02)
        nn.init.normal_(self.treat_a_embed.weight, std=0.02)
        nn.init.normal_(self.treat_b_embed.weight, std=0.02)

    def forward(
        self,
        trajectory: torch.Tensor,
        treatment_a: torch.Tensor,
        treatment_b: torch.Tensor,
    ) -> torch.Tensor:
        """No-dropout path. Inputs are integer index tensors of shape ``[B]``."""
        t = self.traj_embed(trajectory)
        a = self.treat_a_embed(treatment_a)
        b = self.treat_b_embed(treatment_b)
        return torch.stack([t, a, b], dim=1)  # [B, 3, embed_dim]

    def forward_with_dropout(
        self,
        trajectory: torch.Tensor,
        treatment_a: torch.Tensor,
        treatment_b: torch.Tensor,
        p_traj: float = 0.0,
        p_treat: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Training path: independent CFG dropout per conditioner.

        ``p_treat`` is shared between treatment_a and treatment_b because
        they are essentially one joint conditioner (the treatment-direction
        pair). Using two independent Bernoullis would generate impossible
        triples like ``(pre_a, NULL_b)`` that the model would never see at
        inference.
        """
        device = trajectory.device
        B = trajectory.shape[0]

        def _draw(p: float) -> torch.Tensor:
            if p <= 0.0:
                return torch.zeros(B, dtype=torch.bool, device=device)
            r = torch.rand(B, generator=generator, device=device)
            return r < p

        drop_t = _draw(p_traj)
        drop_treat = _draw(p_treat)

        traj_idx = torch.where(drop_t, torch.full_like(trajectory, NULL_IDX_TRAJECTORY), trajectory)
        a_idx = torch.where(drop_treat, torch.full_like(treatment_a, NULL_IDX_TREATMENT), treatment_a)
        b_idx = torch.where(drop_treat, torch.full_like(treatment_b, NULL_IDX_TREATMENT), treatment_b)

        return self.forward(traj_idx, a_idx, b_idx)

    @torch.no_grad()
    def get_uncond_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return an all-NULL context of shape ``[batch_size, 3, embed_dim]``."""
        traj = torch.full((batch_size,), NULL_IDX_TRAJECTORY, dtype=torch.long, device=device)
        treat = torch.full((batch_size,), NULL_IDX_TREATMENT, dtype=torch.long, device=device)
        return self.forward(traj, treat, treat)


def downsample_binary_mask_to_latent(
    mask: torch.Tensor,
    latent_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Trilinear-downsample a (B, 1, D, H, W) binary mask to the latent grid.

    Output stays soft in ``[0, 1]`` rather than re-binarising; this preserves
    sub-voxel boundary information that helps the UNet localise the ROI.
    """
    import torch.nn.functional as F
    return F.interpolate(mask, size=latent_shape, mode="trilinear", align_corners=False)
