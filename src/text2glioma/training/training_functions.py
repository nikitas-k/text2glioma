from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional
from collections import OrderedDict
from copy import deepcopy
import math
import warnings

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from generative.losses import PatchAdversarialLoss

from text2glioma.utils import print_gpu_memory_report, get_lr, log_reconstructions, log_ldm_sample_unconditioned, prepare_mask_conditioning, get_text_encoder_hidden_states


# ---------------------------------------------------------------------------
# 3-D wavelet decomposition helpers  (pywt for filter coefficients)
# ---------------------------------------------------------------------------

# Module-level cache: (wavelet_name, device) → (kernels, labels)
_WAVELET_KERNEL_CACHE: dict[tuple[str, torch.device], tuple[torch.Tensor, list[str]]] = {}


def _get_wavelet_kernels_3d(
    wavelet_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Build 8 separable 3-D DWT kernels from *pywt* filter coefficients.

    Returns ``(kernels, labels)`` where *kernels* has shape
    ``(8, 1, K, K, K)`` and *labels* is ``['LLL', 'LLH', …, 'HHH']``.

    Kernels are outer products of 1-D decomposition filters, which is
    exact for Haar (``'haar'`` / ``'db1'``) and a close approximation
    for longer wavelets (``'db2'``, ``'sym4'``, ``'coif1'``, …).
    See `PyWavelets docs <https://pywavelets.readthedocs.io>`_ for the
    full list of available wavelet families.
    """
    cache_key = (wavelet_name, device)
    if cache_key in _WAVELET_KERNEL_CACHE:
        return _WAVELET_KERNEL_CACHE[cache_key]

    import pywt

    w = pywt.Wavelet(wavelet_name)
    lo = torch.tensor(w.dec_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(w.dec_hi, dtype=torch.float32, device=device)

    filt = [lo, hi]
    names = ["L", "H"]
    kernels: list[torch.Tensor] = []
    labels: list[str] = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                kern = torch.einsum("a,b,c->abc", filt[i], filt[j], filt[k])
                kernels.append(kern)
                labels.append(names[i] + names[j] + names[k])

    stacked = torch.stack(kernels).unsqueeze(1)  # (8, 1, K, K, K)
    _WAVELET_KERNEL_CACHE[cache_key] = (stacked, labels)
    return stacked, labels


def _wavelet_dwt_3d(
    x: torch.Tensor,
    wavelet_name: str = "haar",
) -> dict[str, torch.Tensor]:
    """Single-level 3-D DWT using *pywt* filter coefficients.

    Input : ``(N, C, D, H, W)`` tensor
    Output: dict with keys ``'LLL'``, ``'LLH'``, …, ``'HHH'``
            each of shape ``(N, C, D', H', W')`` where ``D' ≈ D//2``.

    The decomposition is implemented as a grouped 3-D convolution at
    stride ``(2, 2, 2)`` using the outer-product kernels from
    `_get_wavelet_kernels_3d`.  Fully differentiable and GPU-accelerated.
    """
    kernels, labels = _get_wavelet_kernels_3d(wavelet_name, x.device)
    K = kernels.shape[2]
    N, C, D, H, W = x.shape

    # Pad so spatial dims are even and accommodate filter support
    p = (K - 2) // 2
    pd = p + (D % 2)
    ph = p + (H % 2)
    pw = p + (W % 2)
    if pd or ph or pw:
        x = F.pad(x, (p, pw, p, ph, p, pd))

    # Grouped conv: each of C input channels → 8 sub-band channels
    w = kernels.to(x.dtype).repeat(C, 1, 1, 1, 1)   # (8C, 1, K, K, K)
    out = F.conv3d(x, w, stride=2, groups=C)          # (N, 8C, D', H', W')

    D2, H2, W2 = out.shape[2], out.shape[3], out.shape[4]
    out = out.reshape(N, C, 8, D2, H2, W2)

    return {label: out[:, :, idx] for idx, label in enumerate(labels)}


def _wavelet_l1_loss_3d(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    detail_weight: float = 2.0,
    wavelet_name: str = "haar",
) -> torch.Tensor:
    """Multi-scale wavelet L1 loss for 3-D volumes.

    Decomposes both *reconstruction* and *target* into 8 wavelet
    sub-bands and computes a weighted L1 loss.  The low-frequency band
    (LLL) gets weight 1.0; all 7 high-frequency detail bands get
    *detail_weight*.  This emphasises edge / texture fidelity without
    changing the model architecture.

    Parameters
    ----------
    reconstruction, target : (N, C, D, H, W) tensors
    detail_weight : float
        Multiplier for detail (non-LLL) sub-bands.  Higher values
        penalise blurriness more aggressively.
    wavelet_name : str
        Any wavelet recognised by ``pywt.Wavelet`` (e.g. ``'haar'``,
        ``'db2'``, ``'sym4'``, ``'coif1'``).

    Returns
    -------
    Scalar loss tensor.
    """
    bands_rec = _wavelet_dwt_3d(reconstruction.float(), wavelet_name)
    bands_tgt = _wavelet_dwt_3d(target.float(), wavelet_name)

    loss = torch.tensor(0.0, device=reconstruction.device)
    for key in bands_rec:
        w = 1.0 if key == "LLL" else detail_weight
        loss = loss + w * F.l1_loss(bands_rec[key], bands_tgt[key])
    # Normalise by number of sub-bands so the overall magnitude stays
    # comparable to the plain L1 loss regardless of detail_weight.
    loss = loss / (1.0 + 7.0 * detail_weight)
    return loss


# ---------------------------------------------------------------------------
# Multi-scale PatchDiscriminator wrapper
# ---------------------------------------------------------------------------

class MultiScalePatchDiscriminator(nn.Module):
    """Wraps 1–3 PatchDiscriminators at different spatial scales.

    Scale 0 = original resolution (same as the existing single-scale D).
    Scale 1 = 2× average-pooled (half resolution).
    Scale 2 = 4× average-pooled (quarter resolution, optional).

    Each discriminator is an independent copy with its own parameters.
    During forward, the input is progressively downsampled and fed to
    each sub-discriminator.

    Returns
    -------
    list of logits tensors (one per scale, following MONAI
    PatchDiscriminator convention of returning a list where [-1] is
    the final logits tensor).  For convenience the [-1] element is a
    *concatenation* of the final logits from all scales (flattened and
    cat'd along dim=0) so that ``PatchAdversarialLoss`` works unchanged.
    """

    def __init__(self, disc_params: dict, n_scales: int = 2):
        super().__init__()
        from generative.networks.nets.patchgan_discriminator import PatchDiscriminator

        self.n_scales = n_scales
        self.discs = nn.ModuleList([
            PatchDiscriminator(**disc_params) for _ in range(n_scales)
        ])
        # Average-pool kernel for 3-D downsampling
        self._pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        all_logits = []
        cur = x
        for i, disc in enumerate(self.discs):
            if i > 0:
                cur = self._pool(cur)
            logits = disc(cur)[-1]  # PatchDiscriminator returns [layer_outs..., final_logits]
            all_logits.append(logits)
        # Flatten each scale's spatial dims and cat along feature dim
        # so the result preserves the batch dimension.  This allows
        # torch.chunk(logits, 2, dim=0) to split fake/real halves
        # when using the concatenated-batch DDP trick.
        combined = torch.cat([lg.flatten(1) for lg in all_logits], dim=1)
        return all_logits + [combined]

    def per_scale_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-scale logits tensors (for independent loss computation)."""
        results = []
        cur = x
        for i, disc in enumerate(self.discs):
            if i > 0:
                cur = self._pool(cur)
            results.append(disc(cur)[-1])
        return results


def _r1_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R1 gradient penalty (Mescheder et al., 2018).

    Penalises ||∇D(real)||² to prevent the discriminator from creating
    overly sharp decision boundaries.  This stabilises GAN training
    without needing to weaken D architecturally or skip its updates.

    The penalty is computed in float32 for numerical stability.

    If *condition* is given (conditional D), the input to D is
    ``cat([condition, real], dim=1)`` and the gradient is taken
    w.r.t. *real* only.
    """
    real = real.detach().requires_grad_(True)
    if condition is not None:
        disc_in = torch.cat([condition.detach(), real], dim=1).float()
    else:
        disc_in = real.float()
    logits = discriminator(disc_in)[-1]
    grad, = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=real,
        create_graph=True,
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def _get_last_decoder_weight(model: nn.Module) -> torch.Tensor:
    """Return the weight tensor of the last Conv layer in the VAE decoder.

    Used by ``_compute_adaptive_weight`` to equalize reconstruction and
    adversarial gradient magnitudes (VQGAN / Taming Transformers).
    Walks *model.decoder* (or *model.module.decoder* for DDP) and returns
    the ``weight`` of the last ``Conv3d`` / ``Conv2d`` leaf module found.
    """
    decoder = getattr(model, "module", model).decoder
    last_weight = None
    for mod in decoder.modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv3d)):
            last_weight = mod.weight
    if last_weight is None:
        raise RuntimeError("Could not find any Conv layer in model.decoder")
    return last_weight


def _ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute mean SSIM for a single-channel 3D volume (NCDHW or NCHW).

    Uses a uniform sliding window (no Gaussian weighting) for simplicity
    and speed.  Returns a scalar per batch element, averaged over spatial
    locations.
    """
    ndim = pred.ndim - 2  # spatial dims (2 or 3)
    if ndim == 3:
        kernel = torch.ones(1, 1, window_size, window_size, window_size,
                            device=pred.device, dtype=pred.dtype)
        conv_fn = F.conv3d
    else:
        kernel = torch.ones(1, 1, window_size, window_size,
                            device=pred.device, dtype=pred.dtype)
        conv_fn = F.conv2d
    kernel = kernel / kernel.numel()
    pad = window_size // 2

    mu_p = conv_fn(pred, kernel, padding=pad)
    mu_t = conv_fn(target, kernel, padding=pad)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = conv_fn(pred * pred, kernel, padding=pad) - mu_pp
    sigma_tt = conv_fn(target * target, kernel, padding=pad) - mu_tt
    sigma_pt = conv_fn(pred * target, kernel, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))
    return ssim_map.mean()


def _ms_ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    n_levels: int = 4,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute multi-scale SSIM for a single-channel 3-D volume.

    Follows Wang et al. (2003) but adapted for 3-D:
    - At each scale, compute contrast/structure (cs) and luminance (l).
    - Downsample by 2× average-pooling.
    - Final MS-SSIM = prod(cs_i ** weight_i) * l_last ** weight_last.

    Uses *n_levels* = 4 by default (scales 1×, 0.5×, 0.25×, 0.125×).
    With input 160×224×160, the smallest scale is 20×28×20 — still
    large enough for the 7×7×7 sliding window.

    Parameters
    ----------
    pred, target : (N, 1, D, H, W) tensors
    n_levels : int
        Number of scales.  Must satisfy min(D,H,W) / 2^(n_levels-1) >= window_size.

    Returns
    -------
    Scalar MS-SSIM value in [0, 1].
    """
    # Weights from Wang et al. 2003 (normalised to sum to 1)
    weights_5 = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = weights_5[:n_levels]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)

    kernel = torch.ones(1, 1, window_size, window_size, window_size,
                        device=pred.device, dtype=pred.dtype)
    kernel = kernel / kernel.numel()
    pad = window_size // 2
    pool = nn.AvgPool3d(kernel_size=2, stride=2)

    cs_list = []
    for i in range(n_levels):
        mu_p = F.conv3d(pred, kernel, padding=pad)
        mu_t = F.conv3d(target, kernel, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_pp = F.conv3d(pred * pred, kernel, padding=pad) - mu_pp
        sigma_tt = F.conv3d(target * target, kernel, padding=pad) - mu_tt
        sigma_pt = F.conv3d(pred * target, kernel, padding=pad) - mu_pt

        l_map = (2 * mu_pt + C1) / (mu_pp + mu_tt + C1)
        cs_map = (2 * sigma_pt + C2) / (sigma_pp + sigma_tt + C2)

        if i < n_levels - 1:
            cs_list.append(cs_map.mean().clamp(min=1e-8))
            pred = pool(pred)
            target = pool(target)
        else:
            # Last level: use full SSIM (luminance × contrast-structure)
            cs_list.append((l_map * cs_map).mean().clamp(min=1e-8))

    ms_ssim = torch.ones(1, device=pred.device, dtype=pred.dtype)
    for i, cs in enumerate(cs_list):
        ms_ssim = ms_ssim * cs.pow(weights[i])
    return ms_ssim.squeeze()


def _compute_adaptive_weight(
    rec_loss: torch.Tensor,
    g_loss: torch.Tensor,
    last_layer_weight: torch.Tensor,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """Compute adaptive adversarial weight (Esser et al., VQGAN 2021).

    Balances reconstruction and adversarial gradients at the last decoder
    layer so that neither dominates regardless of discriminator strength::

        d_weight = ‖∂L_rec/∂w_last‖ / ‖∂L_adv/∂w_last‖

    This replaces the fixed ``adv_weight`` hyperparameter with a
    dynamically computed scalar, eliminating the D collapse / G starvation
    failure mode caused by a mis-tuned constant.

    Returns
    -------
    d_weight : scalar tensor
        The adaptive weight, clamped to [0, max_weight] for safety.
    """
    rec_grads = torch.autograd.grad(
        rec_loss, last_layer_weight, retain_graph=True,
    )[0]
    g_grads = torch.autograd.grad(
        g_loss, last_layer_weight, retain_graph=True,
    )[0]
    d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_weight).detach()
    return d_weight


def _safe_perceptual_loss(
    perceptual_loss_fn: nn.Module,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-channel MedicalNet perceptual loss, guarding against NaN.

    ``medicalnet_intensity_normalisation`` inside MONAI does ``(x - mean) / std``
    with **no epsilon**.  When a single channel has near-zero variance (common
    early in multi-channel training before the model differentiates modalities),
    ``std -> 0`` and the normalisation produces inf/NaN.

    This helper computes the loss per-channel (MedicalNet expects 1-ch input)
    and skips channels whose reconstruction std is below *eps*.
    """
    n_ch = reconstruction.shape[1]
    total = torch.tensor(0.0, device=reconstruction.device)
    counted = 0
    for c in range(n_ch):
        recon_ch = reconstruction[:, c : c + 1]
        target_ch = target[:, c : c + 1]
        # Guard: skip channels that would cause div-by-zero in
        # medicalnet_intensity_normalisation  ((x - mean) / std).
        if recon_ch.std() < eps or target_ch.std() < eps:
            continue
        total = total + perceptual_loss_fn(recon_ch, target_ch)
        counted += 1
    if counted == 0:
        return torch.tensor(0.0, device=reconstruction.device, requires_grad=True)
    return total / counted

@torch.no_grad()
def encode_text(tokenizer, text_encoder, texts, pad_to_max=True, device='cpu'):
    """Encode a list of texts into text embeddings using the provided tokenizer and text encoder."""
    tokens = tokenizer(
        text=texts,
        max_length=tokenizer.model_max_length if pad_to_max else None,
        padding="max_length" if pad_to_max else True,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    out = text_encoder(**tokens)
    return get_text_encoder_hidden_states(out).to(device)

def get_uncond(tokenizer, text_encoder, batch_size, device):
    return encode_text(tokenizer, text_encoder, [""] * batch_size, device=device)

def prepare_conditioning(tokenizer, text_encoder, texts, batch_size, dropout_p=0.2, uncond_cache=None, device='cpu'):
    B = len(texts)
    cond = encode_text(tokenizer, text_encoder, texts, device=device)
    uncond = uncond_cache if (uncond_cache is not None and uncond_cache.size(0) == B) \
        else get_uncond(tokenizer, text_encoder, batch_size, device=device)
    # text dropout for classifier-free guidance
    drop = (torch.rand(B) < dropout_p).float().to(device).view(B, 1, 1)
    context = cond * (1 - drop) + uncond * drop
    return context, uncond

def train_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    train_loader,
    val_loader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int,
    start_epoch: int = 0,
    best_loss: float = float("inf"),
    val_interval: int = 1,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str  = "./runs",
    kl_weight: float = 1e-6,
    perceptual_weight: float = 2e-3,
    adversarial_weight: float = 1e-3,
    l1_weight: float = 1.0,
    autoencoder_warm_up_n_epochs: int = 0,
    d_skip_threshold: float = 0.0,
    r1_gamma: float = 0.0,
    kl_warmup_epochs: int = 0,
    kl_max: float = 0.0,
    adaptive_adv_weight: bool = False,
    wavelet_loss_weight: float = 0.0,
    wavelet_detail_weight: float = 2.0,
    wavelet_name: str = "haar",
    grad_accum_steps: int = 1,
    l2sp_weight: float = 0.0,
    pretrained_decoder_weights: Optional[dict] = None,
    conditional_disc: bool = False,
    scheduler_g: Optional[Any] = None,
    scheduler_d: Optional[Any] = None,
    # Deprecated — kept for backwards compatibility with queued jobs.
    # GradScaler is no longer used (bf16 doesn't need loss scaling).
    scaler_g=None,
    scaler_d=None,
):
    if scaler_g is not None or scaler_d is not None:
        warnings.warn(
            "scaler_g/scaler_d are deprecated and ignored (bf16 training "
            "does not use GradScaler). Remove them from your call site.",
            DeprecationWarning,
            stacklevel=2,
        )
    raw_model = model.module if hasattr(model, "module") else model

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_loss = eval_autoencoder(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        kl_weight=kl_weight,
        adversarial_weight=adversarial_weight,
        perceptual_weight=perceptual_weight,
        l1_weight=l1_weight,
        kl_max=kl_max,
        wavelet_loss_weight=wavelet_loss_weight,
        wavelet_detail_weight=wavelet_detail_weight,
        wavelet_name=wavelet_name,
        conditional_disc=conditional_disc,
    )

    for epoch in range(start_epoch, n_epochs):
        train_epoch_autoencoder(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            kl_weight=kl_weight,
            adversarial_weight=adversarial_weight,
            perceptual_weight=perceptual_weight,
            l1_weight=l1_weight,
            autoencoder_warm_up_n_epochs=autoencoder_warm_up_n_epochs,
            d_skip_threshold=d_skip_threshold,
            r1_gamma=r1_gamma,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_max=kl_max,
            adaptive_adv_weight=adaptive_adv_weight,
            wavelet_loss_weight=wavelet_loss_weight,
            wavelet_detail_weight=wavelet_detail_weight,
            wavelet_name=wavelet_name,
            grad_accum_steps=grad_accum_steps,
            l2sp_weight=l2sp_weight,
            pretrained_decoder_weights=pretrained_decoder_weights,
            conditional_disc=conditional_disc,
        )

        # Step LR schedulers (per-epoch)
        if scheduler_g is not None:
            scheduler_g.step()
        if scheduler_d is not None:
            scheduler_d.step()

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_autoencoder(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                kl_weight=kl_weight,
                adversarial_weight=adversarial_weight,
                perceptual_weight=perceptual_weight,
                l1_weight=l1_weight,
                kl_max=kl_max,
                wavelet_loss_weight=wavelet_loss_weight,
                wavelet_detail_weight=wavelet_detail_weight,
                wavelet_name=wavelet_name,
                conditional_disc=conditional_disc,
            )
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }
            if scheduler_g is not None:
                checkpoint["scheduler_g"] = scheduler_g.state_dict()
            if scheduler_d is not None:
                checkpoint["scheduler_d"] = scheduler_d.state_dict()
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

    print(f"[rank-0] [INFO] Training finished!")
    print(f"[rank-0] [INFO] Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss

def train_epoch_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
    l1_weight: float = 1.0,
    autoencoder_warm_up_n_epochs: int = 0,
    d_skip_threshold: float = 0.0,
    r1_gamma: float = 0.0,
    kl_warmup_epochs: int = 0,
    kl_max: float = 0.0,
    adaptive_adv_weight: bool = False,
    wavelet_loss_weight: float = 0.0,
    wavelet_detail_weight: float = 2.0,
    wavelet_name: str = "haar",
    grad_accum_steps: int = 1,
    l2sp_weight: float = 0.0,
    pretrained_decoder_weights: Optional[dict] = None,
    conditional_disc: bool = False,
    # Deprecated — kept for backwards compatibility with queued jobs.
    scaler_g=None,
    scaler_d=None,
) -> None:
    if scaler_g is not None or scaler_d is not None:
        warnings.warn(
            "scaler_g/scaler_d are deprecated and ignored (bf16 training "
            "does not use GradScaler). Remove them from your call site.",
            DeprecationWarning,
            stacklevel=2,
        )
    warming_up = epoch < autoencoder_warm_up_n_epochs
    if warming_up and epoch == 0:
        print(f"[WARMUP] Training generator only for the first "
              f"{autoencoder_warm_up_n_epochs} epochs (D disabled).")
    if warming_up:
        print(f"[WARMUP] Epoch {epoch}/{autoencoder_warm_up_n_epochs - 1} "
              f"— discriminator skipped.")
    elif epoch == autoencoder_warm_up_n_epochs and autoencoder_warm_up_n_epochs > 0:
        print(f"[WARMUP] Warmup complete — enabling discriminator at epoch {epoch}.")

    # KL warmup: linearly ramp kl_weight from 0 → full over the first
    # kl_warmup_epochs.  Prevents catastrophic KL spikes at init
    # (latent space is unconstrained; early KL can reach billions).
    if kl_warmup_epochs > 0:
        kl_ramp = min(1.0, (epoch + 1) / kl_warmup_epochs)
        kl_weight = kl_weight * kl_ramp
        if epoch < kl_warmup_epochs:
            print(f"[KL-WARMUP] epoch {epoch}: kl_weight ramped to "
                  f"{kl_weight:.2e} ({kl_ramp:.0%} of target)")
        elif epoch == kl_warmup_epochs:
            print(f"[KL-WARMUP] Warmup complete — kl_weight at full {kl_weight:.2e}")

    if d_skip_threshold > 0 and epoch == 0:
        print(f"[D-SKIP] Adaptive D skipping enabled: "
              f"D update skipped when d_loss < {d_skip_threshold}")

    d_skips = 0  # count how many steps D was skipped this epoch

    model.train()
    discriminator.train()

    # Underlying module (bypasses DDP wrapper for the generator's
    # adversarial-loss forward — avoids DDP gradient-sync hooks and
    # the in-place buffer-version errors introduced in PyTorch ≥ 2.6).
    disc_module = getattr(discriminator, "module", discriminator)

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    # For adaptive adversarial weight: cache the last decoder conv weight.
    last_layer_weight = _get_last_decoder_weight(model) if adaptive_adv_weight else None
    if adaptive_adv_weight and epoch == 0:
        print(f"[ADAPTIVE-ADV] Using VQGAN-style adaptive adversarial weight "
              f"(loss += adv_weight × d_weight × g_loss, "
              f"adv_weight={adversarial_weight})")

    if wavelet_loss_weight > 0 and epoch == 0:
        print(f"[WAVELET-L1] Wavelet L1 loss enabled: "
              f"weight={wavelet_loss_weight}, detail_weight={wavelet_detail_weight}, "
              f"wavelet='{wavelet_name}'")

    # Multi-scale discriminator: detect whether D is a MultiScalePatchDiscriminator
    is_multiscale_disc = isinstance(
        getattr(discriminator, "module", discriminator),
        MultiScalePatchDiscriminator,
    )
    if is_multiscale_disc and epoch == 0:
        n_sc = getattr(discriminator, "module", discriminator).n_scales
        print(f"[MULTISCALE-D] Multi-scale discriminator enabled ({n_sc} scales)")

    if grad_accum_steps > 1 and epoch == 0:
        print(f"[GRAD-ACCUM] Gradient accumulation enabled: {grad_accum_steps} "
              f"micro-steps per optimizer step")

    # L2-SP: build {name → pretrained_tensor} map for decoder params.
    # Only decoder params that require grad are penalised.
    _l2sp_ref: dict[str, torch.Tensor] = {}
    if l2sp_weight > 0 and pretrained_decoder_weights is not None:
        raw = getattr(model, "module", model)
        for name, p in raw.named_parameters():
            if p.requires_grad and name in pretrained_decoder_weights:
                _l2sp_ref[name] = pretrained_decoder_weights[name].to(device)
        if epoch == 0:
            print(f"[L2-SP] Weight-space regularisation enabled: "
                  f"lambda={l2sp_weight:.1e}, {len(_l2sp_ref)} decoder tensors")

    pbar = tqdm(enumerate(loader), total=len(loader))
    n_steps = len(loader)
    for step, x in pbar:
        images = x["image"].to(device)

        # Gradient accumulation control
        accum_start = (step % grad_accum_steps == 0)
        should_step = ((step + 1) % grad_accum_steps == 0) or (step == n_steps - 1)

        # Shared forward pass — kept in the graph so the generator
        # backward can reach the autoencoder parameters.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reconstruction, z_mu, z_sigma = model(x=images)

        # -------- DISCRIMINATOR --------
        # Concatenate fake + real into one batch for a SINGLE
        # discriminator forward.  This avoids the PyTorch ≥ 2.6
        # in-place version-mismatch error: BatchNorm updates
        # running_mean/running_var in-place during forward, so two
        # separate forwards through the same BN layer create two
        # saved-variable versions and the second backward fails.
        # One forward = one BN update = no conflict.
        # During warmup, skip D entirely so G learns a stable
        # reconstruction baseline before adversarial training begins.
        #
        # Adaptive D skipping: when d_loss drops below d_skip_threshold
        # the discriminator is "too good" — skip its update to let G
        # catch up.  D still runs a forward pass so we get the loss
        # value for logging, but we skip backward + step.
        if adversarial_weight > 0 and not warming_up:
            if accum_start:
                optimizer_d.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if conditional_disc:
                    # Pix2pix-style: D sees (input, output) channel-concat pairs.
                    fake_pair = torch.cat([images.detach(), reconstruction.contiguous().detach()], dim=1)
                    real_pair = torch.cat([images.detach(), images.detach()], dim=1)
                    disc_input = torch.cat([fake_pair, real_pair], dim=0)
                else:
                    disc_input = torch.cat(
                        [reconstruction.contiguous().detach(),
                         images.contiguous().detach()],
                        dim=0,
                    )
                logits_all = discriminator(disc_input.float())[-1]
                logits_fake, logits_real = torch.chunk(logits_all, 2, dim=0)

                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                d_loss = (adversarial_weight * discriminator_loss).mean()

            # R1 gradient penalty — computed outside autocast because
            # grad computation requires float32 for stability.
            # Use disc_module (unwrapped) to avoid DDP's _sync_buffers
            # triggering a second in-place copy_() on BN running stats
            # (same two-forward version-mismatch bug as the GAN pattern).
            r1_loss = torch.tensor(0.0, device=device)
            if r1_gamma > 0:
                r1_loss = _r1_penalty(
                    disc_module, images,
                    condition=images if conditional_disc else None,
                )
                d_loss = d_loss + 0.5 * r1_gamma * r1_loss

            # Adaptive skip: only update D if it hasn't collapsed yet.
            # CRITICAL: synchronize the skip decision across all DDP ranks.
            # Each rank sees a different mini-batch so d_loss can differ.
            # If some ranks skip backward() and others don't, the DDP
            # gradient allreduce will hang (NCCL timeout).
            if d_skip_threshold > 0 and dist.is_initialized():
                d_loss_avg = discriminator_loss.detach().clone()
                dist.all_reduce(d_loss_avg, op=dist.ReduceOp.AVG)
                skip_d = d_loss_avg.item() < d_skip_threshold
            elif d_skip_threshold > 0:
                skip_d = discriminator_loss.item() < d_skip_threshold
            else:
                skip_d = False
            if skip_d:
                d_skips += 1
            else:
                _d_ctx = nullcontext() if should_step else getattr(discriminator, 'no_sync', nullcontext)()
                with _d_ctx:
                    (d_loss / grad_accum_steps).backward()
                if should_step:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                    optimizer_d.step()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        # -------- GENERATOR --------
        # Freeze discriminator — no need to track its params/buffers
        # during the generator backward pass.  Use unwrapped module
        # to bypass DDP's in-place buffer broadcast.
        for p in disc_module.parameters():
            p.requires_grad_(False)

        if accum_start:
            optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # MedicalNet perceptual loss: per-channel (MedicalNet expects 1-ch)
            # with div-by-zero guard for near-constant channels.
            p_loss = _safe_perceptual_loss(perceptual_loss, reconstruction, images)

            # Wavelet L1 loss: penalise high-frequency sub-band errors
            if wavelet_loss_weight > 0:
                w_loss = _wavelet_l1_loss_3d(
                    reconstruction, images,
                    detail_weight=wavelet_detail_weight,
                    wavelet_name=wavelet_name,
                )
            else:
                w_loss = torch.tensor(0.0, device=device)

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # Clamp KL to prevent individual-batch spikes from
            # overwhelming the loss (common at init or with large kl_weight).
            if kl_max > 0:
                kl_loss = torch.clamp(kl_loss, max=kl_max)

            if adversarial_weight > 0 and not warming_up:
                if conditional_disc:
                    g_disc_in = torch.cat([images, reconstruction.contiguous()], dim=1).float()
                else:
                    g_disc_in = reconstruction.contiguous().float()
                logits_fake_g = disc_module(g_disc_in)[-1]
                generator_loss = adv_loss(logits_fake_g, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # Reconstruction loss (everything except adversarial)
            rec_loss = (l1_weight * l1_loss
                        + kl_weight * kl_loss
                        + perceptual_weight * p_loss
                        + wavelet_loss_weight * w_loss)

            # L2-SP: penalise decoder weights drifting from pretrained values
            l2sp_loss = torch.tensor(0.0, device=device)
            if l2sp_weight > 0 and _l2sp_ref:
                raw = getattr(model, "module", model)
                for name, p in raw.named_parameters():
                    if name in _l2sp_ref:
                        l2sp_loss = l2sp_loss + (p - _l2sp_ref[name]).pow(2).sum()
                rec_loss = rec_loss + l2sp_weight * l2sp_loss

            # Adversarial contribution: adaptive or fixed weight
            if adaptive_adv_weight and adversarial_weight > 0 and not warming_up:
                d_weight = _compute_adaptive_weight(
                    rec_loss, generator_loss, last_layer_weight,
                )
                # VQGAN-style: d_weight = ‖∂rec/∂w‖ / ‖∂adv/∂w‖
                # scaled by adv_weight as a global multiplier to
                # moderate D's influence (prevents D-dominance / grid
                # artifacts when adv_weight=1.0).
                loss = rec_loss + adversarial_weight * d_weight * generator_loss
            else:
                d_weight = torch.tensor(adversarial_weight, device=device)
                loss = rec_loss + adversarial_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            w_loss = w_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )
            if wavelet_loss_weight > 0:
                losses["w_loss"] = w_loss
            if l2sp_weight > 0:
                losses["l2sp_loss"] = l2sp_loss

        _g_ctx = nullcontext() if should_step else getattr(model, 'no_sync', nullcontext)()
        with _g_ctx:
            (losses["loss"] / grad_accum_steps).backward()
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer_g.step()

        # Unfreeze discriminator for the next iteration.
        for p in disc_module.parameters():
            p.requires_grad_(True)

        losses["d_loss"] = discriminator_loss

        if writer is not None:
            global_step = epoch * len(loader) + step
            writer.add_scalar("lr_g", get_lr(optimizer_g), global_step)
            writer.add_scalar("lr_d", get_lr(optimizer_d), global_step)
            for k, v in losses.items():
                writer.add_scalar(f"{k}", v.item(), global_step)
            if r1_gamma > 0:
                writer.add_scalar("r1_penalty", r1_loss.item(), global_step)
            if d_skip_threshold > 0:
                writer.add_scalar("d_skip_rate",
                                  d_skips / (step + 1), global_step)
            if kl_warmup_epochs > 0 and epoch < kl_warmup_epochs:
                writer.add_scalar("kl_weight_eff", kl_weight, global_step)
            if adaptive_adv_weight:
                writer.add_scalar("d_weight", d_weight.item(), global_step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )

    if d_skip_threshold > 0:
        skip_pct = 100.0 * d_skips / max(len(loader), 1)
        print(f"[D-SKIP] Epoch {epoch}: skipped {d_skips}/{len(loader)} "
              f"D updates ({skip_pct:.1f}%)")

@torch.no_grad()
def eval_autoencoder(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    kl_weight: float,
    adversarial_weight: float,
    perceptual_weight: float,
    l1_weight: float = 1.0,
    kl_max: float = 0.0,
    wavelet_loss_weight: float = 0.0,
    wavelet_detail_weight: float = 2.0,
    wavelet_name: str = "haar",
    conditional_disc: bool = False,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    n_samples = 0
    for x in loader:
        images = x["image"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            # MedicalNet perceptual loss: per-channel (MedicalNet expects 1-ch)
            # with div-by-zero guard for near-constant channels.
            p_loss = _safe_perceptual_loss(perceptual_loss, reconstruction, images)
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if kl_max > 0:
                kl_loss = torch.clamp(kl_loss, max=kl_max)

            if adversarial_weight > 0:
                if conditional_disc:
                    g_disc_in = torch.cat([images, reconstruction.contiguous()], dim=1).float()
                else:
                    g_disc_in = reconstruction.contiguous().float()
                logits_fake = discriminator(g_disc_in)[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adversarial_weight > 0:
                if conditional_disc:
                    fake_pair = torch.cat([images.detach(), reconstruction.contiguous().detach()], dim=1)
                    real_pair = torch.cat([images.detach(), images.detach()], dim=1)
                    logits_fake = discriminator(fake_pair.float())[-1]
                    logits_real = discriminator(real_pair.float())[-1]
                else:
                    logits_fake = discriminator(reconstruction.contiguous().detach().float())[-1]
                    logits_real = discriminator(images.contiguous().detach().float())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_weight * l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adversarial_weight * generator_loss

            # Wavelet L1 loss (eval)
            if wavelet_loss_weight > 0:
                w_loss = _wavelet_l1_loss_3d(
                    reconstruction, images,
                    detail_weight=wavelet_detail_weight,
                    wavelet_name=wavelet_name,
                )
                loss = loss + wavelet_loss_weight * w_loss
            else:
                w_loss = torch.tensor(0.0, device=device)

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()
            w_loss_val = w_loss.mean()

            # Per-channel L1 (T1, T1CE, T2, FLAIR)
            ch_names = ["T1", "T1CE", "T2", "FLAIR"]
            rec_f = reconstruction.float()
            img_f = images.float()
            per_ch_l1 = {}
            for ci, cn in enumerate(ch_names):
                per_ch_l1[f"l1_{cn}"] = F.l1_loss(
                    rec_f[:, ci : ci + 1], img_f[:, ci : ci + 1]
                ).mean()

            # Per-channel SSIM (3D, sliding-window)
            per_ch_ssim = {}
            for ci, cn in enumerate(ch_names):
                per_ch_ssim[f"ssim_{cn}"] = _ssim_3d(
                    rec_f[:, ci : ci + 1], img_f[:, ci : ci + 1]
                ).mean()
            ssim_mean = sum(per_ch_ssim.values()) / len(per_ch_ssim)

            # Per-channel MS-SSIM (3D, multi-scale)
            per_ch_ms_ssim = {}
            for ci, cn in enumerate(ch_names):
                per_ch_ms_ssim[f"ms_ssim_{cn}"] = _ms_ssim_3d(
                    rec_f[:, ci : ci + 1], img_f[:, ci : ci + 1]
                ).mean()
            ms_ssim_mean = sum(per_ch_ms_ssim.values()) / len(per_ch_ms_ssim)

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
                ssim=ssim_mean,
                ms_ssim=ms_ssim_mean,
                **per_ch_l1,
                **per_ch_ssim,
                **per_ch_ms_ssim,
            )
            if wavelet_loss_weight > 0:
                losses["w_loss"] = w_loss_val

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

        n_samples += images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= max(n_samples, 1)

    if writer is not None:
        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]

# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: Any,
    tokenizer: Any,
    text_encoder: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
    text_field: str = "impression",
    start_epoch: int = 0,
    val_interval: int = 1,
    dropout_p: float = 0.2,
    model_dir: str = "./models",
    writer_train: Any = None,
    writer_val: Any = None,
    run_dir: str = "./runs",
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    mask_dropout_p: float = 0.2,
    latent_channels: int = 3,
    warmup_epochs: int = 10,
    ema_decay: float = 0.9999,
    ema_state_dict: dict = None,
    snr_gamma: float = 5.0,
    # Deprecated — kept for backwards compatibility with queued jobs.
    scaler=None,
) -> float:
    if scaler is not None:
        warnings.warn(
            "scaler is deprecated and ignored (bf16 training does not use "
            "GradScaler). Remove it from your call site.",
            DeprecationWarning,
            stacklevel=2,
        )
    raw_model = model.module if hasattr(model, "module") else model

    best_loss = float("inf")

    # ── EMA (exponential moving average) ─────────────────────────────
    ema_model = deepcopy(raw_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if ema_state_dict is not None:
        ema_model.load_state_dict(ema_state_dict)
        if _is_main:
            print("[rank-0] [INFO] Loaded EMA state from checkpoint.")

    # ── Pre-compute min-SNR weights (Hang et al., 2023) ────────────
    # SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
    # For v-prediction the per-timestep weight is:
    #   w(t) = min(SNR(t), gamma) / SNR(t)  where gamma = snr_gamma (typically 5)
    # This down-weights high-noise timesteps that produce noisy gradients.
    alphas_cumprod = scheduler.alphas_cumprod.to(device)     # [T]
    snr = alphas_cumprod / (1.0 - alphas_cumprod)            # [T]
    # For v-prediction: weight = min(SNR, gamma) / SNR
    # For epsilon-prediction: weight = min(SNR, gamma) / SNR
    # (both reduce to clamping the effective weight at high-noise steps)
    min_snr_weights = torch.clamp(snr, max=snr_gamma) / snr  # [T]
    # ── LR schedule: linear warmup → cosine decay with min_lr floor ──
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = 0.01  # LR never drops below 1% of base_lr

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Fast-forward scheduler if resuming
    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            lr_scheduler.step()

    val_loss = eval_ldm(
        model=model,
        stage1=stage1,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_field=text_field,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        sample=False,
        scale_factor=scale_factor,
        num_mask_classes=num_mask_classes,
        latent_channels=latent_channels,
    )

    # Determine rank for gated printing
    _is_main = writer_train is not None  # only rank 0 has a writer
    if _is_main:
        print(f"[rank-0] [INFO] epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_epoch_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_field=text_field,
            dropout_p=dropout_p,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scale_factor=scale_factor,
            num_mask_classes=num_mask_classes,
            mask_dropout_p=mask_dropout_p,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            min_snr_weights=min_snr_weights,
        )

        if (epoch + 1) % val_interval == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                text_field=text_field,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True if (epoch + 1) % (val_interval * 2) == 0 else False,
                scale_factor=scale_factor,
                num_mask_classes=num_mask_classes,
                latent_channels=latent_channels,
            )

            if _is_main:
                print(f"[rank-0] [INFO] epoch {epoch + 1} val loss: {val_loss:.4f}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": raw_model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))
                torch.save(ema_model.state_dict(), str(run_dir / "best_model_ema.pth"))

    if _is_main:
        print(f"[rank-0] [INFO] Training finished!")
        print(f"[rank-0] [INFO] Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))
    torch.save(ema_model.state_dict(), str(run_dir / "final_model_ema.pth"))

    return val_loss


def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    tokenizer: Any,
    text_encoder: Any,
    text_field: str,
    dropout_p: float,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    mask_dropout_p: float = 0.2,
    max_grad_norm: float = 1.0,
    lr_scheduler: Any = None,
    ema_model: Any = None,
    ema_decay: float = 0.9999,
    min_snr_weights: torch.Tensor = None,
    # Deprecated — kept for backwards compatibility with queued jobs.
    scaler=None,
) -> None:
    if scaler is not None:
        warnings.warn(
            "scaler is deprecated and ignored (bf16 training does not use "
            "GradScaler). Remove it from your call site.",
            DeprecationWarning,
            stacklevel=2,
        )
    model.train()
    raw_model = model.module if hasattr(model, "module") else model

    # Only show progress bar on rank 0 to avoid interleaved output
    is_main = not (torch.distributed.is_available() and torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    rank = 0 if not (torch.distributed.is_available() and torch.distributed.is_initialized()) else torch.distributed.get_rank()
    pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_main)
    for step, x in pbar:
        images = x["image"].to(device)
        labels = x["label"].to(device)
        reports = x[text_field]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                e = stage1(images) * scale_factor
                latent_spatial = e.shape[2:]  # (D', H', W')

                # Prepare mask conditioning: one-hot → downsample → dropout
                mask_cond = prepare_mask_conditioning(
                    labels, latent_spatial,
                    num_classes=num_mask_classes,
                    dropout_p=mask_dropout_p,
                ).to(device)

            # Prepare text conditioning (with independent text dropout)
            cond, _ = prepare_conditioning(tokenizer, text_encoder, reports, images.size(0), dropout_p=dropout_p, device=device)

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            # Concatenate noisy latent with mask conditioning: [B, latent_ch + num_classes, D', H', W']
            model_input = torch.cat([noisy_e, mask_cond], dim=1)

            noise_pred = model(x=model_input, timesteps=timesteps, context=cond)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

            # Per-sample MSE (reduce over all dims except batch)
            mse = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            mse = mse.mean(dim=list(range(1, mse.ndim)))  # [B]

            # Min-SNR-γ weighting (Hang et al., 2023) — down-weights
            # high-noise timesteps that produce noisy, unhelpful gradients.
            if min_snr_weights is not None:
                snr_w = min_snr_weights[timesteps]  # [B]
                loss = (mse * snr_w).mean()
            else:
                loss = mse.mean()

        losses = OrderedDict(loss=loss)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # EMA update
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), raw_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1.0 - ema_decay)

        if writer is not None:
            writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

            for k, v in losses.items():
                writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix({"epoch": epoch, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"})


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    tokenizer: Any,
    text_encoder: Any,
    text_field: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample: bool = False,
    scale_factor: float = 1.0,
    num_mask_classes: int = 4,
    latent_channels: int = 3,
) -> float:
    model.eval()
    total_losses = OrderedDict()
    n_samples = 0

    for x in loader:
        images = x["image"].to(device)
        labels = x["label"].to(device)
        reports = x[text_field]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e = stage1(images) * scale_factor
            latent_spatial = e.shape[2:]

            # Prepare mask conditioning (no dropout during eval)
            mask_cond = prepare_mask_conditioning(
                labels, latent_spatial,
                num_classes=num_mask_classes,
                dropout_p=0.0,
            ).to(device)

            cond, _ = prepare_conditioning(tokenizer, text_encoder, reports, images.size(0), dropout_p=0.0, device=device)

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            # Concatenate noisy latent with mask conditioning
            model_input = torch.cat([noisy_e, mask_cond], dim=1)

            noise_pred = model(x=model_input, timesteps=timesteps, context=cond)

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        n_samples += images.shape[0]
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    # Normalise by samples this rank actually processed (not full dataset
    # size, which over-counts under DDP sharded loaders).
    for k in total_losses.keys():
        total_losses[k] /= max(n_samples, 1)

    if writer is not None:
        for k, v in total_losses.items():
            writer.add_scalar(f"{k}", v, step)

    if sample:
        log_ldm_sample_unconditioned(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            latent_channels=latent_channels,
            num_mask_classes=num_mask_classes,
        )

    return total_losses["loss"]
