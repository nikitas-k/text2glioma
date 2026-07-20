"""One-call inference engine for the released text2glioma models.

Wraps the same model-loading and sampling primitives used by
``scripts/offline_sample_stage2_compare.py`` so that behaviour matches
the paper's evaluation pipeline byte-for-byte, including the
deterministic uncond-first CFG order and the ``cfg_mode="text_only"``
mask routing. Intended for interactive UIs (Gradio, notebooks) and
lightweight API endpoints where the caller doesn't want to touch
individual samplers, schedulers, tokenisers, or NIfTI I/O.

Typical use::

    from text2glioma.inference.engine import Text2GliomaEngine

    engine = Text2GliomaEngine.from_paths(
        stage1_config="configs/stage1.yaml",
        stage2_config="configs/ldm_radbert.yaml",
        stage1_ckpt="/runs/stage1_kl1e6_freebits_lc6/autoencoder_stage1/checkpoint.pth",
        stage2_ckpt="/runs/stage1_kl1e6_freebits_lc6/ldm_stage2/best_model.pth",
        device="cuda",
    )
    result = engine.generate(
        prompt="Right frontal enhancing mass with surrounding oedema",
        mask_nifti_path="/data/subj0001/seg.nii.gz",
        cfg=7.0, seed=42, steps=50, mode="text+mask",
    )
    result.save_nifti("out_dir")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms as T

from text2glioma.utils import (
    MODALITY_NAMES,
    get_model,
    get_text_encoder_hidden_states,
    load_config,
    load_text_encoder_and_tokenizer,
    prepare_mask_conditioning,
    stage1_ify,
)


# Default target spatial size the models were trained on. Overridable via
# the stage-2 config ``mask.spatial_size`` key.
_DEFAULT_SPATIAL = (160, 224, 160)


# ----------------------------------------------------------------------
# Helpers copied verbatim from scripts/offline_sample_stage2_compare.py
# (private in that file; duplicated here to keep the engine importable
#  without shelling out to the CLI). Keep in sync if the CLI changes.
# ----------------------------------------------------------------------

def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "diffusion" in checkpoint_obj:
            return checkpoint_obj["diffusion"]
        if "ldm_state_dict" in checkpoint_obj:
            return checkpoint_obj["ldm_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format for Stage-2 model.")


def _extract_stage1_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return checkpoint_obj["state_dict"]
        if "autoencoder" in checkpoint_obj and isinstance(checkpoint_obj["autoencoder"], dict):
            return checkpoint_obj["autoencoder"]
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return checkpoint_obj["model"]
    raise ValueError("Unsupported checkpoint format for Stage-1 model.")


def _infer_stage1_latent_channels(checkpoint_path: str) -> Optional[int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_stage1_state_dict(checkpoint)
    for key in (
        "quant_conv_mu.conv.weight",
        "module.quant_conv_mu.conv.weight",
        "model.quant_conv_mu.conv.weight",
    ):
        weight = state_dict.get(key)
        if torch.is_tensor(weight):
            return int(weight.shape[0])
    return None


def _resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if requested == "auto" and mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if mps_available else "cpu")
    return torch.device(requested)


@torch.no_grad()
def _sample_latent(
    model,
    scheduler,
    latent0: torch.Tensor,
    mask_cond: torch.Tensor,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    uncond_embeds: Optional[torch.Tensor] = None,
    uncond_mask_cond: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    cfg_mode: str = "text_only",
) -> torch.Tensor:
    """Deterministic DDIM sampling with text-only CFG (mask kept in both
    branches). Mirrors ``_sample_latent`` in the offline sampler."""
    latent = latent0.clone()
    do_cfg = (
        uncond_embeds is not None
        and uncond_mask_cond is not None
        and guidance_scale != 1.0
    )
    if do_cfg:
        mask_for_uncond = mask_cond if cfg_mode == "text_only" else uncond_mask_cond

    for t in scheduler.timesteps:
        ts = torch.as_tensor((t,)).to(device)
        if do_cfg:
            cond_input = torch.cat([latent, mask_cond], dim=1)
            uncond_input = torch.cat([latent, mask_for_uncond], dim=1)
            x_in = torch.cat([uncond_input, cond_input], dim=0)
            ts_in = torch.cat([ts, ts], dim=0)
            ctx_in = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            model_output = model(x=x_in, timesteps=ts_in, context=ctx_in)
            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            model_input = torch.cat([latent, mask_cond], dim=1)
            noise_pred = model(x=model_input, timesteps=ts, context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)
    return latent


def _encode_text(tokenizer, text_encoder, text: str, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return get_text_encoder_hidden_states(text_encoder(**tokens))


def _load_mask_to_target_space(
    mask_nifti_path: str,
    reference_image_path: Optional[str],
    target_spatial: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load a segmentation NIfTI and bring it into the model's canonical
    ``(160, 224, 160)`` LPS-oriented space.

    Returns:
        labels tensor ``(1, 1, D, H, W)`` on ``device`` with integer classes,
        and the original affine used for output-space alignment.
    """
    keys = ["label"]
    xforms = [
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        T.EnsureTyped(keys=keys, dtype=torch.float32),
        T.Orientationd(keys=keys, axcodes="LPS"),
    ]
    if reference_image_path is not None:
        # Match the training pipeline: crop to the reference image FG.
        xforms = [
            T.LoadImaged(keys=["image", "label"]),
            T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
            T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            T.EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            T.Orientationd(keys=["image", "label"], axcodes="LPS"),
            T.CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
        xforms.extend([
            T.SpatialPadd(keys=["image", "label"], spatial_size=target_spatial, mode="constant"),
            T.CenterSpatialCropd(keys=["image", "label"], roi_size=target_spatial),
            T.ToTensord(keys=["image", "label"]),
        ])
        batch = T.Compose(xforms)({"image": reference_image_path, "label": mask_nifti_path})
    else:
        xforms.extend([
            T.SpatialPadd(keys=keys, spatial_size=target_spatial, mode="constant"),
            T.CenterSpatialCropd(keys=keys, roi_size=target_spatial),
            T.ToTensord(keys=keys),
        ])
        batch = T.Compose(xforms)({"label": mask_nifti_path})

    label = batch["label"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
    affine = np.asarray(
        getattr(batch["label"], "affine", np.eye(4)), dtype=np.float64
    )
    return label, affine


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class GenerationResult:
    """A single generated 4-modality volume, plus enough metadata to
    reconstruct it and to render a preview."""

    images: torch.Tensor                    # (1, C, D, H, W) in [0, 1]
    affine: np.ndarray                       # 4x4
    modality_names: list[str] = field(default_factory=lambda: list(MODALITY_NAMES))
    prompt: str = ""
    cfg: float = 1.0
    seed: int = 0
    steps: int = 50
    mode: str = "text+mask"
    idh: Optional[int] = None
    mgmt: Optional[int] = None

    def to_nifti_list(self) -> list[nib.Nifti1Image]:
        arr = self.images[0].detach().cpu().numpy().astype(np.float32)  # (C, D, H, W)
        return [
            nib.Nifti1Image(arr[c], affine=self.affine)
            for c in range(arr.shape[0])
        ]

    def save_nifti(self, out_dir: str | Path, prefix: str = "sample") -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, img in zip(self.modality_names, self.to_nifti_list()):
            p = out_dir / f"{prefix}_{name}.nii.gz"
            nib.save(img, str(p))
            paths.append(p)
        return paths

    def preview_slices(self, axis: int = -1) -> np.ndarray:
        """Return a ``(C, H, W)`` mid-slice array for a per-modality preview.
        ``axis=-1`` = axial for LPS-oriented volumes."""
        vol = self.images[0].detach().cpu().numpy()      # (C, D, H, W)
        mid = vol.shape[axis] // 2
        if axis == -1 or axis == vol.ndim - 1:
            return vol[..., mid]
        if axis == -2 or axis == vol.ndim - 2:
            return vol[..., mid, :]
        return vol[:, mid]


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class Text2GliomaEngine:
    """Load the released text+mask-conditioned LDM once, then serve
    per-request samples via :meth:`generate`."""

    def __init__(
        self,
        stage1_config: dict,
        stage2_config: dict,
        stage1_ckpt: str,
        stage2_ckpt: str,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        local_text_encoder: bool = True,
        molecular_head_ckpt: Optional[str] = None,
    ):
        self.device = _resolve_device(device)
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config

        # -- Stage 1 (autoencoder) --
        stage1_params = stage1_config.setdefault("model", {}).setdefault("params", {})
        inferred = _infer_stage1_latent_channels(stage1_ckpt)
        if inferred is not None:
            stage1_params["latent_channels"] = inferred
        if self.device.type != "cuda":
            stage1_params["use_flash_attention"] = False
        self.stage1 = stage1_ify(
            get_model("AutoencoderKL", stage1_config, from_file=stage1_ckpt)
        ).to(self.device).eval()
        for p in self.stage1.parameters():
            p.requires_grad = False

        # Probe true latent-channel count (whitening / channel-wise wrappers).
        with torch.no_grad():
            probe = torch.zeros(
                1, 4, *_DEFAULT_SPATIAL, device=self.device
            )
            try:
                z_probe = self.stage1(probe)
                self.stage1_latent_ch = int(z_probe.shape[1])
                self.latent_spatial = tuple(int(v) for v in z_probe.shape[2:])
            except Exception:
                # Fallback: read from config.
                self.stage1_latent_ch = int(
                    stage2_config.get("model", {}).get("latent_channels", 3)
                )
                # 16x downsampling assumed if we can't probe.
                self.latent_spatial = tuple(v // 16 for v in _DEFAULT_SPATIAL)

        # -- Stage 2 (diffusion U-Net) --
        self.num_mask_classes = int(
            stage2_config.get("mask", {}).get("num_classes", 4)
        )
        self.target_spatial = tuple(
            int(v) for v in
            stage2_config.get("mask", {}).get("spatial_size", _DEFAULT_SPATIAL)
        )

        model_cfg = stage2_config.setdefault("model", {})
        params = model_cfg.setdefault("params", {})
        model_cfg["latent_channels"] = self.stage1_latent_ch
        params["in_channels"] = self.stage1_latent_ch + self.num_mask_classes
        params["out_channels"] = self.stage1_latent_ch

        self.model = get_model(model_cfg.get("name", "DiffusionModelUNet"), stage2_config)
        checkpoint = torch.load(stage2_ckpt, map_location="cpu")
        self.model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
        self.model = self.model.to(self.device).eval()

        # -- Scheduler --
        sched_name = stage2_config.get("scheduler", {}).get("name", "DDIMScheduler")
        sched_params = stage2_config.get("scheduler", {}).get("params", {})
        if sched_name == "DDIMScheduler":
            from generative.networks.schedulers import DDIMScheduler
            self.scheduler = DDIMScheduler(**sched_params)
        elif sched_name == "DDPMScheduler":
            from generative.networks.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler(**sched_params)
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        # -- Text encoder --
        self.tokenizer, self.text_encoder = load_text_encoder_and_tokenizer(
            stage2_config["conditioning"],
            cache_dir=cache_dir,
            local_files_only=local_text_encoder,
        )
        cfg_max_len = stage2_config["conditioning"].get("max_length")
        if cfg_max_len is not None:
            self.tokenizer.model_max_length = int(cfg_max_len)
        self.text_encoder = self.text_encoder.to(self.device).eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Deterministic kernels so the CFG-uncond output is stable across
        # calls. See docs/lessons_learned.md §"CFG determinism".
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # -- Optional molecular class-conditioning head --
        # If a sibling ``best_molecular_head.pth`` (or explicit
        # ``molecular_head_ckpt``) exists, load a
        # ``MolecularClassConditioning`` module and route two learnable
        # pseudo-tokens (IDH, MGMT) alongside the RadBERT text embeddings.
        # Silently absent when the stage-2 model was trained without it.
        self.molecular_head: Optional[torch.nn.Module] = None
        mh_ckpt: Optional[Path] = None
        if molecular_head_ckpt is not None:
            mh_ckpt = Path(molecular_head_ckpt)
        else:
            sibling = Path(stage2_ckpt).parent / "best_molecular_head.pth"
            if sibling.is_file():
                mh_ckpt = sibling
        if mh_ckpt is not None and mh_ckpt.is_file():
            from text2glioma.training.molecular_conditioning import (
                MolecularClassConditioning,
            )
            hidden_dim = int(getattr(self.text_encoder.config, "hidden_size",
                                      getattr(self.text_encoder.config, "dim", 768)))
            self.molecular_head = MolecularClassConditioning(
                hidden_dim=hidden_dim,
                dropout_to_unknown_p=0.0,   # inference: no dropout
            )
            self.molecular_head.load_state_dict(
                torch.load(str(mh_ckpt), map_location="cpu"), strict=True,
            )
            self.molecular_head = self.molecular_head.to(self.device).eval()

    # ------------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        stage1_config: str | Path,
        stage2_config: str | Path,
        stage1_ckpt: str | Path,
        stage2_ckpt: str | Path,
        **kwargs,
    ) -> "Text2GliomaEngine":
        return cls(
            stage1_config=load_config(str(stage1_config)),
            stage2_config=load_config(str(stage2_config)),
            stage1_ckpt=str(stage1_ckpt),
            stage2_ckpt=str(stage2_ckpt),
            **kwargs,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        mask_nifti_path: Optional[str | Path] = None,
        reference_nifti_path: Optional[str | Path] = None,
        cfg: float = 7.0,
        seed: int = 42,
        steps: int = 50,
        mode: str = "text+mask",
        idh: Optional[int] = None,
        mgmt: Optional[int] = None,
    ) -> GenerationResult:
        """Generate one 4-modality sample.

        Args:
            prompt: Radiology-style description. Empty string for the
                mask-only mode (drives the model to its unconditional
                text baseline).
            mask_nifti_path: Path to an integer segmentation ``.nii.gz``
                (0=background, 1=nCET, 2=oedema, 3=enhancing). Required
                for ``mode`` in ``{"text+mask", "mask-only"}``.
            reference_nifti_path: Optional 4-modality image used to
                match training-time CropForeground behaviour. Improves
                spatial alignment when the mask NIfTI has generous
                margins.
            cfg: Text CFG scale (>= 1). CFG=1 disables guidance.
            seed: PyTorch RNG seed. Fixed seed + deterministic cuDNN =
                reproducible output.
            steps: DDIM sampling steps. Paper default = 50.
            mode: One of ``"text+mask"``, ``"mask-only"``, ``"text-only"``.
                ``text-only`` is the experimental / unsupported path
                that reverts to the healthy-brain baseline; see §3.5.
            idh: Target IDH status for molecular class conditioning.
                ``0`` = wildtype, ``1`` = mutant, ``2`` = unknown (or
                pass ``None`` for the UNKNOWN / null direction). Only
                used when the engine was loaded with a
                ``molecular_head_ckpt`` (or a sibling
                ``best_molecular_head.pth``) \u2014 silently ignored
                otherwise.
            mgmt: Target MGMT status. ``0`` = unmethylated, ``1`` =
                methylated, ``2`` = unknown / ``None``.

        Returns:
            :class:`GenerationResult` with per-modality volumes in
            ``[0, 1]`` and the affine of the input mask.
        """
        if mode not in {"text+mask", "mask-only", "text-only"}:
            raise ValueError(f"unknown mode: {mode!r}")
        if mode != "text-only" and mask_nifti_path is None:
            raise ValueError(
                f"mode={mode!r} requires a mask NIfTI (mask_nifti_path)."
            )

        # ---- Mask conditioning ----
        if mode == "text-only":
            mask_cond = torch.zeros(
                (1, self.num_mask_classes) + self.latent_spatial,
                device=self.device,
            )
            affine = np.eye(4, dtype=np.float64)
        else:
            label, affine = _load_mask_to_target_space(
                mask_nifti_path=str(mask_nifti_path),
                reference_image_path=(
                    str(reference_nifti_path) if reference_nifti_path else None
                ),
                target_spatial=self.target_spatial,
                device=self.device,
            )
            mask_cond = prepare_mask_conditioning(
                labels=label,
                latent_shape=self.latent_spatial,
                num_classes=self.num_mask_classes,
                dropout_p=0.0,
            ).to(self.device)
        mask_uncond = torch.zeros_like(mask_cond)

        # ---- Text conditioning ----
        prompt_text = "" if mode == "mask-only" else prompt
        cond_embeds = _encode_text(self.tokenizer, self.text_encoder, prompt_text, self.device)
        uncond_embeds = _encode_text(self.tokenizer, self.text_encoder, "", self.device)

        # ---- Optional molecular class conditioning ----
        # If the engine loaded a molecular head, append the two IDH/MGMT
        # pseudo-tokens to both cond and uncond sequences. The single CFG
        # scale then guides over the combined (text + molecular) direction.
        if self.molecular_head is not None:
            from text2glioma.training.molecular_conditioning import (
                IDH_UNKNOWN, MGMT_UNKNOWN,
            )
            idh_int  = IDH_UNKNOWN  if idh  is None else int(idh)
            mgmt_int = MGMT_UNKNOWN if mgmt is None else int(mgmt)
            idh_t  = torch.tensor([idh_int],  dtype=torch.long, device=self.device)
            mgmt_t = torch.tensor([mgmt_int], dtype=torch.long, device=self.device)
            mol_tokens = self.molecular_head(idh_t, mgmt_t).to(cond_embeds.dtype)
            mol_null   = self.molecular_head.null_tokens(
                batch_size=1, device=self.device, dtype=uncond_embeds.dtype,
            )
            cond_embeds   = torch.cat([cond_embeds,   mol_tokens], dim=1)  # (1, 128+2, D)
            uncond_embeds = torch.cat([uncond_embeds, mol_null],   dim=1)  # (1, 128+2, D)

        # ---- Sampling ----
        self.scheduler.set_timesteps(min(steps, self.scheduler.num_train_timesteps))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        latent0 = torch.randn(
            (1, self.stage1_latent_ch) + self.latent_spatial, device=self.device
        )

        # Uncond trajectory first so CUDA non-determinism can't couple
        # it to CFG scale — see docs/lessons_learned.md.
        torch.manual_seed(int(seed))
        _ = _sample_latent(
            self.model, self.scheduler, latent0, mask_uncond, uncond_embeds,
            self.device,
        )
        self.scheduler.set_timesteps(min(steps, self.scheduler.num_train_timesteps))
        torch.manual_seed(int(seed))
        latent_cond = _sample_latent(
            self.model, self.scheduler, latent0, mask_cond, cond_embeds,
            self.device,
            uncond_embeds=uncond_embeds,
            uncond_mask_cond=mask_uncond,
            guidance_scale=float(cfg),
            cfg_mode="text_only",
        )

        # ---- Decode ----
        scale_factor = 1.0 / max(latent_cond.std().item(), 1e-8)
        images = self.stage1.decode(latent_cond / scale_factor).float().clamp(0.0, 1.0)

        return GenerationResult(
            images=images,
            affine=affine,
            modality_names=list(MODALITY_NAMES),
            prompt=prompt_text,
            cfg=float(cfg),
            seed=int(seed),
            steps=int(steps),
            mode=mode,
            idh=(int(idh) if idh is not None else None),
            mgmt=(int(mgmt) if mgmt is not None else None),
        )
