"""Interactive inpainting inference — VS Code cell script.

Run cells with `# %%` markers using VS Code's "Run Cell" (Shift+Enter) or
Jupyter extension. Works as a normal .py file too. Designed for iterating on
samples, conditioning, and guidance without restarting the model.

Usage on Gadi (interactive PBS session)::

    qsub -I -q gpuhopper -P vp06 -l ngpus=1,ncpus=12,mem=96GB,walltime=4:00:00,storage=gdata/dk92+scratch/vp06+gdata/vp06+gdata/hl36+scratch/hl36 -l wd
    module use /g/data/dk92/apps/Modules/modulefiles
    module load NCI-ai-ml/23.05 python3/3.9.2
    source /g/data/hl36/nk9793/venv/monai/bin/activate
    cd ~/text2glioma
    # then open this file in VS Code (Remote-SSH to Gadi) and run cells.

Cell layout
-----------
  1. Imports + paths       — edit PATHS dict to point at your run
  2. Load Stage-1 + UNet   — slow (~30s); only re-run if you change checkpoints
  3. Load dataset          — fast
  4. Pick a sample         — change SAMPLE_IDX to scrub through the test set
  5. Visualise input       — masked image_a + mask overlay
  6. Single conditional    — Task A
  7. Trajectory sweep      — Task B
  8. Unconditional         — Task C
  9. Guidance-scale sweep  — same sample, varying CFG strength
 10. SSIM diagnostics      — per-modality + ROI numbers for whatever's in `last_pred`
"""

# %% [markdown]
# # Inpainting inference — interactive
# Edit the `PATHS` block below to point at your trained checkpoint and run cells top-to-bottom on first launch. After that, cells 4 onwards can be re-run independently.

# %% Imports + paths
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism

# Make ``src/`` importable when running as a plain .py from the repo root.
ROOT = Path("/home/561/nk9793/text2glioma")  # change if your clone lives elsewhere
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from text2glioma.inpainting.conditioning import (
    CategoricalConditioningEncoder, NULL_IDX_TRAJECTORY, NULL_IDX_TREATMENT,
)
from text2glioma.inpainting.run_inference import _load_inpainting_state_dict
from text2glioma.inpainting.sampling import (
    compute_ssim_per_modality, sample_inpainting,
)
from text2glioma.inpainting.training_functions import InpaintingModel
from text2glioma.preprocessing.inpainting_dataset import (
    TRAJECTORY_TO_IDX, TREATMENT_TO_IDX,
    build_pair_transforms, prepare_pair_records,
)
from text2glioma.utils import get_model, load_config, stage1_ify

IDX_TO_TRAJECTORY = {v: k for k, v in TRAJECTORY_TO_IDX.items()}
IDX_TO_TREATMENT = {v: k for k, v in TREATMENT_TO_IDX.items()}
MODALITY_NAMES = ["t1c", "t1n", "t2f", "t2w"]

USER = "nk9793"
RUN_BASE = Path(f"/g/data/vp06/{USER}/text2glioma_train/runs/stage1_overfit_ablate_kl1e6")

PATHS = {
    "config":          ROOT / "configs/inpainting.yaml",
    "stage1_config":   ROOT / "configs/stage1.yaml",
    "stage1_uri":      RUN_BASE / "autoencoder_stage1/checkpoint.pth",
    "inpainting_ckpt": RUN_BASE / "inpainting_ldm/best_model_ema.pth",  # or checkpoint.pth
    "datalist":        ROOT / "datalist_brats_gli_2025_pairs_split.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {DEVICE}")
for k, v in PATHS.items():
    print(f"  {k:18s} exists={v.exists()}  {v}")

# %% Load Stage-1 + UNet (slow — only re-run on checkpoint change)
set_determinism(42)

config = load_config(str(PATHS["config"]))
stage1_config = load_config(str(PATHS["stage1_config"]))

print("Loading Stage-1 VAE …")
stage1 = stage1_ify(
    get_model(model_type="AutoencoderKL", config=stage1_config,
              from_file=str(PATHS["stage1_uri"]))
).eval().to(DEVICE)
for p in stage1.parameters():
    p.requires_grad = False

# %% Load dataset (fast)
with open(PATHS["datalist"]) as f:
    datalist = json.load(f)

FOLD = "testing"  # change to 'validation' or 'training' if you want
records = prepare_pair_records(datalist[FOLD])
print(f"  {FOLD}: {len(records)} pairs")

# Optional: filter by treatment direction.
# Note: This datalist is 100% longitudinal pairs (different timepoints).
# To filter, set FILTER_DIRECTION to one of: 'pre->post', 'post->post', 'pre->pre', 'post->pre'.
# Set to None to keep all pairs.
FILTER_DIRECTION = None   # e.g. 'pre->post', 'post->post', 'pre->pre', 'post->pre', or None
if FILTER_DIRECTION:
    dir_map = {
        'pre->post': (0, 1),
        'post->post': (1, 1),
        'pre->pre': (0, 0),
        'post->pre': (1, 0),
    }
    if FILTER_DIRECTION in dir_map:
        ta_target, tb_target = dir_map[FILTER_DIRECTION]
        records = [r for r in records 
                  if int(r['treatment_a']) == ta_target and int(r['treatment_b']) == tb_target]
        print(f"  after filtering (direction={FILTER_DIRECTION}): {len(records)} pairs")

DILATION_MM = 18.0
SPATIAL_SIZE = tuple(config.get("data", {}).get("spatial_size", (160, 224, 160)))
xforms = build_pair_transforms(training=False, dilation_mm=DILATION_MM,
                               spatial_size=SPATIAL_SIZE)
ds = Dataset(data=records, transform=xforms)

# %% Build UNet + load weights (after first batch so we can probe latent channels)
# We need one batch to probe Stage-1's latent_ch before instantiating the UNet
# with the correct in/out channels. This mirrors the launcher's startup logic.
probe_batch = ds[0]
with torch.no_grad():
    probe_z = stage1(probe_batch["image_a"].unsqueeze(0).to(DEVICE))
LATENT_CH = int(probe_z.shape[1])
print(f"  latent_ch={LATENT_CH}  latent_shape={tuple(probe_z.shape[2:])}")

cfg_params = config.setdefault("model", {}).setdefault("params", {})
cfg_params["in_channels"] = 2 * LATENT_CH + 1
cfg_params["out_channels"] = LATENT_CH
config["model"]["latent_channels"] = LATENT_CH

print("Building UNet + cond encoder …")
unet = get_model(config["model"].get("name", "DiffusionModelUNet"), config)
embed_dim = int(cfg_params.get("cross_attention_dim", 256))
cond_encoder = CategoricalConditioningEncoder(embed_dim=embed_dim)
inpainting = InpaintingModel(unet=unet, cond_encoder=cond_encoder)

payload = _load_inpainting_state_dict(str(PATHS["inpainting_ckpt"]))
missing, unexpected = inpainting.load_state_dict(payload["state_dict"], strict=False)
print(f"  load_state_dict: missing={len(missing)}  unexpected={len(unexpected)}")
inpainting = inpainting.eval().to(DEVICE)
for p in inpainting.parameters():
    p.requires_grad = False

SCALE_FACTOR = payload["scale_factor"]
if SCALE_FACTOR is None:
    raise RuntimeError(
        "Checkpoint has no scale_factor; provide one manually via SCALE_FACTOR = <float>."
    )
print(f"  scale_factor = {SCALE_FACTOR:.4f}")

# %% Scheduler
sch_name = config["scheduler"].get("name", "DDIMScheduler")
sch_params = config["scheduler"].get("params", {})
scheduler = (DDPMScheduler(**sch_params) if sch_name == "DDPMScheduler"
             else DDIMScheduler(**sch_params))
print(f"  scheduler = {sch_name}")

# %% [markdown]
# ## Pick a sample
# Change `SAMPLE_IDX` and re-run from here. The chosen pair's metadata is
# printed so you know what conditioning the model was trained to expect.

# %% Pick a sample
SAMPLE_IDX = 0   # 0 .. len(records)-1

batch = ds[SAMPLE_IDX]
# MONAI Dataset returns a dict of tensors with shape (C, D, H, W); add batch dim.
masked_a = batch["masked_image_a"].unsqueeze(0).to(DEVICE)
image_b_real = batch["image_b"].unsqueeze(0).to(DEVICE)
image_a_real = batch["image_a"].unsqueeze(0).to(DEVICE)
mask = batch["mask"].unsqueeze(0).to(DEVICE)
true_traj = torch.tensor([int(batch["trajectory"])], device=DEVICE).long()
true_ta = torch.tensor([int(batch["treatment_a"])], device=DEVICE).long()
true_tb = torch.tensor([int(batch["treatment_b"])], device=DEVICE).long()

rec = records[SAMPLE_IDX]
print(f"pair_id      : {rec.get('pair_id', 'n/a')}")
print(f"subject_id   : {rec['subject_id']}")
print(f"trajectory   : {IDX_TO_TRAJECTORY[int(true_traj)]}")
print(f"treatment    : {IDX_TO_TREATMENT[int(true_ta)]} -> {IDX_TO_TREATMENT[int(true_tb)]}")
print(f"image shape  : {tuple(masked_a.shape)}")
print(f"mask voxels  : {int(mask.sum().item())} / {int(mask.numel())}  "
      f"({100 * float(mask.float().mean()):.2f}%)")

# %% Visualise input (centre slice of each modality + mask overlay)
def _to_np(t):  # (1, C, D, H, W) -> (C, D, H, W) numpy
    return t.detach().float().cpu().numpy()[0]

def show_slices(volumes_dict, slice_axis=2, slice_idx=None, modality=0, mask_np=None,
                vmin=None, vmax=None, suptitle=None):
    """Show the same axial/coronal/sagittal slice across several volumes.

    volumes_dict : {label: np.ndarray (C, D, H, W)}
    slice_axis   : 0=sagittal (D), 1=coronal (H), 2=axial (W)  -> indexing happens on spatial axis
    """
    n = len(volumes_dict)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (label, vol) in zip(axes, volumes_dict.items()):
        c = vol[modality]
        sl_idx = slice_idx if slice_idx is not None else c.shape[slice_axis] // 2
        if slice_axis == 0:
            sl = c[sl_idx, :, :]
            msl = mask_np[sl_idx, :, :] if mask_np is not None else None
        elif slice_axis == 1:
            sl = c[:, sl_idx, :]
            msl = mask_np[:, sl_idx, :] if mask_np is not None else None
        else:
            sl = c[:, :, sl_idx]
            msl = mask_np[:, :, sl_idx] if mask_np is not None else None
        ax.imshow(sl.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        if msl is not None and msl.sum() > 0:
            ax.contour(msl.T, levels=[0.5], colors="red", linewidths=0.7)
        ax.set_title(f"{label}\n[{MODALITY_NAMES[modality]} sl={sl_idx}]")
        ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout(); plt.show()

mask_np = _to_np(mask)[0]
show_slices(
    {"image_a (input)": _to_np(image_a_real),
     "masked_image_a":  _to_np(masked_a),
     "image_b (target)": _to_np(image_b_real)},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"sample {SAMPLE_IDX}  ({IDX_TO_TRAJECTORY[int(true_traj)]})",
)

# %% [markdown]
# ## Knobs
# Sampling settings used by all the cells below. Cheap to re-run a cell after
# changing one of these — no need to reload the model.

# %% Sampling settings
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.0
SEED = 0          # reseeded per sample below so cell re-runs are reproducible

def _gen():
    return torch.Generator(device=DEVICE).manual_seed(SEED)

# %% Task A — single conditional sample (true labels)
last_pred = sample_inpainting(
    inpainting_model=inpainting, stage1=stage1, scheduler=scheduler,
    masked_image_a=masked_a, mask=mask,
    trajectory=true_traj, treatment_a=true_ta, treatment_b=true_tb,
    scale_factor=SCALE_FACTOR,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    use_uncond=False, generator=_gen(),
)
ssim = compute_ssim_per_modality(last_pred[0], image_b_real[0], mask=mask[0])
print(f"SSIM global = {ssim['ssim_global_mean']:.4f}   ROI = {ssim['ssim_roi_mean']:.4f}")
print(f"  per-modality global = {['%.3f' % v for v in ssim['ssim_global_perch']]}")
print(f"  per-modality ROI    = {['%.3f' % v for v in ssim['ssim_roi_perch']]}")

show_slices(
    {"masked_image_a": _to_np(masked_a),
     "predicted_b":    _to_np(last_pred),
     "real_b":         _to_np(image_b_real)},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"Task A (conditional)  traj={IDX_TO_TRAJECTORY[int(true_traj)]}  "
             f"CFG={GUIDANCE_SCALE}",
)

# %% Task B — trajectory sweep
# Same masked input + true treatments; sweep trajectory ∈ {response, stable, progression}.
# The off-diagonal entries probe whether the model actually uses the trajectory token.
sweep_preds = {}
sweep_ssim = {}
for name, idx in TRAJECTORY_TO_IDX.items():
    traj_in = torch.full_like(true_traj, fill_value=int(idx))
    pred = sample_inpainting(
        inpainting_model=inpainting, stage1=stage1, scheduler=scheduler,
        masked_image_a=masked_a, mask=mask,
        trajectory=traj_in, treatment_a=true_ta, treatment_b=true_tb,
        scale_factor=SCALE_FACTOR,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        use_uncond=False, generator=_gen(),
    )
    s = compute_ssim_per_modality(pred[0], image_b_real[0], mask=mask[0])
    sweep_preds[name] = pred
    sweep_ssim[name] = s
    diag = " (TRUE)" if int(idx) == int(true_traj) else ""
    print(f"  sweep[{name:12s}]  global={s['ssim_global_mean']:.4f}  "
          f"roi={s['ssim_roi_mean']:.4f}{diag}")

# Side-by-side: masked_a | response | stable | progression | real_b
panels = {"masked_a": _to_np(masked_a)}
for name in ("response", "stable", "progression"):
    panels[f"sweep[{name}]"] = _to_np(sweep_preds[name])
panels["real_b"] = _to_np(image_b_real)
show_slices(panels, slice_axis=2, modality=0, mask_np=mask_np,
            suptitle=f"Task B (trajectory sweep)  true={IDX_TO_TRAJECTORY[int(true_traj)]}  "
                     f"CFG={GUIDANCE_SCALE}")
last_pred = sweep_preds[IDX_TO_TRAJECTORY[int(true_traj)]]

# %% Task C — unconditional baseline (all NULL tokens)
uncond_pred = sample_inpainting(
    inpainting_model=inpainting, stage1=stage1, scheduler=scheduler,
    masked_image_a=masked_a, mask=mask,
    trajectory=true_traj, treatment_a=true_ta, treatment_b=true_tb,  # ignored
    scale_factor=SCALE_FACTOR,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,  # ignored
    use_uncond=True, generator=_gen(),
)
ssim_uncond = compute_ssim_per_modality(uncond_pred[0], image_b_real[0], mask=mask[0])
print(f"unconditional  global={ssim_uncond['ssim_global_mean']:.4f}  "
      f"roi={ssim_uncond['ssim_roi_mean']:.4f}")

show_slices(
    {"masked_a":           _to_np(masked_a),
     "conditional (true)": _to_np(last_pred),
     "unconditional":      _to_np(uncond_pred),
     "real_b":             _to_np(image_b_real)},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"Task C (uncond) vs Task A (cond)  CFG={GUIDANCE_SCALE}",
)
last_pred = uncond_pred

# %% Guidance-scale sweep — same sample, varying CFG strength
# Useful for tuning the headline CFG value. Higher = sharper but more
# saturated; lower = blurrier but stays closer to the masked-image prior.
GS_VALUES = [0.0, 1.5, 3.0, 5.0, 7.5]
gs_preds = {}
gs_ssim = {}
for gs in GS_VALUES:
    pred = sample_inpainting(
        inpainting_model=inpainting, stage1=stage1, scheduler=scheduler,
        masked_image_a=masked_a, mask=mask,
        trajectory=true_traj, treatment_a=true_ta, treatment_b=true_tb,
        scale_factor=SCALE_FACTOR,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=float(gs),
        use_uncond=False, generator=_gen(),
    )
    s = compute_ssim_per_modality(pred[0], image_b_real[0], mask=mask[0])
    gs_preds[gs] = pred
    gs_ssim[gs] = s
    print(f"  CFG={gs:>4.1f}  global={s['ssim_global_mean']:.4f}  roi={s['ssim_roi_mean']:.4f}")

show_slices(
    {f"CFG={gs}": _to_np(gs_preds[gs]) for gs in GS_VALUES},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"Guidance-scale sweep  true traj={IDX_TO_TRAJECTORY[int(true_traj)]}",
)
last_pred = gs_preds[GUIDANCE_SCALE if GUIDANCE_SCALE in gs_preds else GS_VALUES[len(GS_VALUES)//2]]

# %% [markdown]
# ## Diagnostics
# Quantitative numbers for whatever's currently in `last_pred`. Doesn't run
# inference — just metrics — so cheap to re-run.

# %% SSIM diagnostics
s = compute_ssim_per_modality(last_pred[0], image_b_real[0], mask=mask[0])
print("Per-modality SSIM (global / ROI):")
for c, name in enumerate(MODALITY_NAMES):
    g = s["ssim_global_perch"][c]
    r = s["ssim_roi_perch"][c]
    print(f"  {name:>3s}   global={g:.4f}   roi={r:.4f}")
print(f"\nROI bbox (d_lo, d_hi, h_lo, h_hi, w_lo, w_hi) = {s['bbox']}")

# %% Difference map — abs(pred - real) inside the dilated mask
diff = (last_pred - image_b_real).abs()
diff_np = _to_np(diff)
show_slices(
    {"abs(pred - real)": diff_np},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle="Residual",
)
print(f"max abs diff  = {diff.max().item():.4f}")
print(f"mean abs diff (in-mask)  = {(diff * mask).sum().item() / mask.sum().item() / diff.shape[1]:.4f}")
print(f"mean abs diff (out-mask) = {(diff * (1 - mask)).sum().item() / (1 - mask).sum().item() / diff.shape[1]:.4f}")

# %% [markdown]
# ## Pix2pix baseline
# Train a small 3D pix2pix (U-Net generator + PatchGAN discriminator) on the
# same masked → image_b task, **without** trajectory / treatment conditioning.
# This is a deterministic, supervised baseline — the LDM should beat it on
# diversity (it can't sample variants) and ideally also on ROI fidelity once
# you've trained the LDM long enough. Pix2pix typically wins early because
# L1 + adversarial is a *much* easier optimisation than score matching.
#
# **Compute budget**: this is a toy baseline. We use a half-resolution U-Net
# and a few hundred iterations so you can iterate inside one interactive
# session. For a paper-worthy pix2pix comparison, train it the same way you
# train the LDM (multi-GPU, full epochs) — that's a separate `train_pix2pix_ddp.py`
# job, not this script.

# %% Pix2pix — model definitions
import torch.nn as nn
import torch.nn.functional as F

class Pix2PixGenerator(nn.Module):
    """Small 3D U-Net. Input 5ch (4 modalities + 1 mask) -> 4ch output.

    Skip connections at every resolution. tanh-free output (regressed in the
    same intensity range as the masked input, which has already been
    percentile-normalised by build_pair_transforms).
    """
    def __init__(self, in_ch: int = 5, out_ch: int = 4, base: int = 32):
        super().__init__()
        def cbr(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1, bias=False),
                nn.InstanceNorm3d(o, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.e1 = nn.Sequential(cbr(in_ch, base), cbr(base, base))
        self.e2 = nn.Sequential(cbr(base, base * 2), cbr(base * 2, base * 2))
        self.e3 = nn.Sequential(cbr(base * 2, base * 4), cbr(base * 4, base * 4))
        self.bottleneck = nn.Sequential(cbr(base * 4, base * 8), cbr(base * 8, base * 8))
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.d3 = nn.Sequential(cbr(base * 8, base * 4), cbr(base * 4, base * 4))
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(cbr(base * 4, base * 2), cbr(base * 2, base * 2))
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(cbr(base * 2, base), cbr(base, base))
        self.out = nn.Conv3d(base, out_ch, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class PatchGAN3D(nn.Module):
    """3D PatchGAN discriminator. Sees (masked_image_a + mask, image_b_or_pred)
    concatenated along channels => 4 + 1 + 4 = 9 input channels."""
    def __init__(self, in_ch: int = 9, base: int = 32):
        super().__init__()
        def block(i, o, stride=2, norm=True):
            layers = [nn.Conv3d(i, o, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm3d(o, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            block(in_ch, base, norm=False),       # /2
            block(base, base * 2),                # /4
            block(base * 2, base * 4),            # /8
            block(base * 4, base * 8, stride=1),  # patch logits
            nn.Conv3d(base * 8, 1, 4, stride=1, padding=1),
        )
    def forward(self, x):
        return self.net(x)


# Downsample to a friendlier spatial size for the U-Net (must be divisible by 8).
# Full res 160x224x160 in fp32 is ~5GB activations through this U-Net;
# half-res 80x112x80 keeps a single H100 comfortable at batch 2.
P2P_SPATIAL = tuple(s // 2 for s in SPATIAL_SIZE)
P2P_SPATIAL = tuple(8 * (s // 8) for s in P2P_SPATIAL)   # snap to multiple of 8
print(f"pix2pix spatial size = {P2P_SPATIAL}  (downsampled from {SPATIAL_SIZE})")

def _resize_volumes(*tensors, size):
    """Trilinear resize for image volumes, nearest for masks (last positional)."""
    *imgs, mask_t = tensors
    out_imgs = [F.interpolate(t, size=size, mode="trilinear", align_corners=False) for t in imgs]
    out_mask = F.interpolate(mask_t, size=size, mode="nearest")
    return (*out_imgs, out_mask)

# %% Pix2pix — instantiate G, D, optimisers
P2P_BASE = 24                # generator base width
P2P_D_BASE = 16              # discriminator base width
P2P_L1_WEIGHT = 100.0        # standard pix2pix L1 weight
P2P_LR = 2e-4

G = Pix2PixGenerator(in_ch=5, out_ch=4, base=P2P_BASE).to(DEVICE)
D = PatchGAN3D(in_ch=9, base=P2P_D_BASE).to(DEVICE)
opt_G = torch.optim.Adam(G.parameters(), lr=P2P_LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=P2P_LR, betas=(0.5, 0.999))

print(f"Generator params:     {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M")
print(f"Discriminator params: {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M")

# %% Pix2pix — training loader (re-uses the inpainting dataset transforms)
P2P_BATCH = 2
train_records = prepare_pair_records(datalist["training"])
train_xforms = build_pair_transforms(training=True, dilation_mm=DILATION_MM,
                                     spatial_size=SPATIAL_SIZE)
p2p_train_ds = Dataset(data=train_records, transform=train_xforms)
p2p_train_loader = DataLoader(
    p2p_train_ds, batch_size=P2P_BATCH, shuffle=True,
    num_workers=2, pin_memory=True, drop_last=True,
)
print(f"pix2pix training pairs = {len(train_records)}  batch_size={P2P_BATCH}")

# %% Pix2pix — training loop
# Stops at either N_ITERS or one full epoch, whichever is smaller. Re-run the
# cell to keep training (G, D, opts persist between cells).
N_ITERS = 200          # number of generator updates; bump to taste
LOG_EVERY = 20

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

G.train(); D.train()
it_count = 0
losses = {"D": [], "G_gan": [], "G_l1": []}

for batch in p2p_train_loader:
    if it_count >= N_ITERS:
        break

    masked_a_b = batch["masked_image_a"].to(DEVICE, non_blocking=True)
    image_a_b  = batch["image_a"].to(DEVICE, non_blocking=True)
    mask_b     = batch["mask"].to(DEVICE, non_blocking=True)

    masked_a_b, image_a_b, mask_b = _resize_volumes(masked_a_b, image_a_b, mask_b, size=P2P_SPATIAL)
    cond = torch.cat([masked_a_b, mask_b], dim=1)   # (B, 5, D, H, W)

    # --- D step ----------------------------------------------------
    with torch.no_grad():
        fake = G(cond)
    real_pair = torch.cat([cond, image_a_b], dim=1)
    fake_pair = torch.cat([cond, fake], dim=1)
    d_real = D(real_pair)
    d_fake = D(fake_pair)
    loss_D = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                    bce(d_fake, torch.zeros_like(d_fake)))
    opt_D.zero_grad(set_to_none=True)
    loss_D.backward()
    opt_D.step()

    # --- G step ----------------------------------------------------
    fake = G(cond)
    fake_pair = torch.cat([cond, fake], dim=1)
    d_fake_for_g = D(fake_pair)
    loss_G_gan = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
    loss_G_l1 = l1(fake, image_a_b)
    loss_G = loss_G_gan + P2P_L1_WEIGHT * loss_G_l1
    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    losses["D"].append(float(loss_D.item()))
    losses["G_gan"].append(float(loss_G_gan.item()))
    losses["G_l1"].append(float(loss_G_l1.item()))
    it_count += 1
    if it_count % LOG_EVERY == 0 or it_count == 1:
        print(f"  iter {it_count:>4d}/{N_ITERS}  "
              f"D={losses['D'][-1]:.3f}  "
              f"G_gan={losses['G_gan'][-1]:.3f}  "
              f"G_l1={losses['G_l1'][-1]:.4f}")

print(f"Finished {it_count} iterations.")

# Quick loss plot
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.plot(losses["D"], label="D", alpha=0.6)
ax.plot(losses["G_gan"], label="G_gan", alpha=0.6)
ax.plot(losses["G_l1"], label="G_l1", alpha=0.6)
ax.set_xlabel("iter"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% Pix2pix — inference on the currently-selected sample
def _match_batch_dim(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Repeat singleton batch to match ref batch; otherwise require equal batch."""
    if x.shape[0] == ref.shape[0]:
        return x
    if x.shape[0] == 1:
        reps = [ref.shape[0]] + [1] * (x.ndim - 1)
        return x.repeat(*reps)
    raise ValueError(f"Batch mismatch: got {x.shape[0]} vs ref {ref.shape[0]}")


G.eval()
with torch.no_grad():
    # Ensure all tensors share the same batch dim before concat.
    image_a_infer = _match_batch_dim(image_a_real, masked_a)
    mask_infer = _match_batch_dim(mask, masked_a)

    masked_a_lr, image_a_lr, mask_lr = _resize_volumes(
        masked_a, image_a_infer, mask_infer, size=P2P_SPATIAL,
    )
    cond_lr = torch.cat([masked_a_lr, mask_lr], dim=1)
    pred_lr = G(cond_lr)
    # Up-sample back to original spatial size for an apples-to-apples SSIM.
    pix2pix_pred = F.interpolate(pred_lr, size=SPATIAL_SIZE, mode="trilinear", align_corners=False)

ssim_p2p = compute_ssim_per_modality(pix2pix_pred[0], image_a_real[0], mask=mask[0])
ssim_ldm = compute_ssim_per_modality(last_pred[0], image_b_real[0], mask=mask[0])

print(f"\n--- Sample {SAMPLE_IDX}  ({IDX_TO_TRAJECTORY[int(true_traj)]}) ---")
print(f"  pix2pix (same-time)   global={ssim_p2p['ssim_global_mean']:.4f}  roi={ssim_p2p['ssim_roi_mean']:.4f}")
print(f"  LDM (future-time)     global={ssim_ldm['ssim_global_mean']:.4f}  roi={ssim_ldm['ssim_roi_mean']:.4f}")

show_slices(
    {"masked_a":      _to_np(masked_a),
     "pix2pix":       _to_np(pix2pix_pred),
     "LDM (future)":  _to_np(last_pred),
     "real_a":        _to_np(image_a_real)},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"pix2pix (recon a) vs LDM (pred b)   sample {SAMPLE_IDX}   "
             f"traj={IDX_TO_TRAJECTORY[int(true_traj)]}",
)

# %% [markdown]
# ## Pixel-space diffusion baseline
# Trains a small 3D diffusion model **directly in image space** (no Stage-1
# VAE). Same noise schedule + v-prediction + DDIM sampler as the LDM, but
# operating on heavily downsampled images so a single GPU can hold the
# activations.
#
# Why this baseline matters:
#   - pix2pix → LDM is a two-axis change (loss formulation + latent space).
#   - This pixel-DDM isolates **the latent axis**: same loss, same sampler,
#     just no compression. If the LDM beats this, the gain is attributable
#     to the Stage-1 VAE (smaller spatial extent ⇒ deeper UNet at fixed
#     compute ⇒ better global structure). If pixel-DDM matches or beats it,
#     the VAE is mostly costing capacity.
#
# **Memory note**: pixel-space diffusion in 3D is brutal. Even at
# 64×80×64 (quarter-res-ish) a UNet with attention OOMs on 40 GB cards at
# batch 1. We deliberately use:
#   - aggressive downsample to (64, 80, 64) — ~22× fewer voxels than 160×224×160,
#   - no attention layers (the second-largest memory consumer after activations),
#   - num_res_blocks=1 and num_channels=(32, 64, 128) (~8M params).
#
# Even so the iteration is meant to be illustrative — a paper-grade pixel-DDM
# would need a multi-GPU launcher analogous to train_inpainting_ddp.py.

# %% Pixel-DDM — UNet + scheduler
from generative.networks.nets import DiffusionModelUNet

DDM_SPATIAL = (64, 80, 64)                # multiples of 16; ~22× fewer voxels
DDM_IN_CH = 4 + 4 + 1                     # noisy_b ⊕ masked_a ⊕ mask
DDM_OUT_CH = 4                            # v / ε in image space

ddm_unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=DDM_IN_CH, out_channels=DDM_OUT_CH,
    num_res_blocks=1,
    num_channels=(32, 64, 128),
    attention_levels=(False, False, False),   # no attention — memory killer at pixel res
    with_conditioning=False,
    resblock_updown=True,
    norm_num_groups=8,
).to(DEVICE)
# Zero-init the conv_in weights for the masked-image + mask channels (same
# trick as the LDM: model starts as a vanilla diffusion in noisy_b, learns to
# use the cond channels gradually).
with torch.no_grad():
    ddm_unet.conv_in.conv.weight[:, 4:].zero_()

opt_ddm = torch.optim.AdamW(ddm_unet.parameters(), lr=1e-4)

# Re-use the inpainting scheduler exactly so the comparison isolates the
# latent vs pixel axis.
ddm_scheduler = (DDPMScheduler(**sch_params) if sch_name == "DDPMScheduler"
                 else DDIMScheduler(**sch_params))

print(f"pixel-DDM spatial = {DDM_SPATIAL}  in_ch={DDM_IN_CH}  out_ch={DDM_OUT_CH}")
print(f"  params = {sum(p.numel() for p in ddm_unet.parameters()) / 1e6:.2f}M")

# %% Pixel-DDM — training loop
# Reuses p2p_train_loader. Re-run the cell to keep training (weights + opt
# persist between cell executions).
DDM_ITERS = 200
DDM_LOG_EVERY = 20

ddm_unet.train()
ddm_it = 0
ddm_losses: list[float] = []

for batch in p2p_train_loader:
    if ddm_it >= DDM_ITERS:
        break

    masked_a_b = batch["masked_image_a"].to(DEVICE, non_blocking=True)
    image_b_b  = batch["image_b"].to(DEVICE, non_blocking=True)
    mask_b     = batch["mask"].to(DEVICE, non_blocking=True)
    masked_a_b, image_b_b, mask_b = _resize_volumes(
        masked_a_b, image_b_b, mask_b, size=DDM_SPATIAL,
    )

    B_ = image_b_b.shape[0]
    t = torch.randint(0, ddm_scheduler.num_train_timesteps, (B_,), device=DEVICE).long()
    noise = torch.randn_like(image_b_b)
    noisy_b = ddm_scheduler.add_noise(original_samples=image_b_b, noise=noise, timesteps=t)
    model_input = torch.cat([noisy_b, masked_a_b, mask_b], dim=1)

    pred = ddm_unet(x=model_input, timesteps=t)
    if ddm_scheduler.prediction_type == "v_prediction":
        target = ddm_scheduler.get_velocity(image_b_b, noise, t)
    else:
        target = noise

    # Mask-weighted MSE so in-mask voxels get the same 5× boost as the LDM.
    w = 1.0 + 4.0 * mask_b
    loss = ((pred - target).pow(2) * w).mean()

    opt_ddm.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ddm_unet.parameters(), 1.0)
    opt_ddm.step()

    ddm_losses.append(float(loss.item()))
    ddm_it += 1
    if ddm_it % DDM_LOG_EVERY == 0 or ddm_it == 1:
        print(f"  iter {ddm_it:>4d}/{DDM_ITERS}  loss={loss.item():.4f}")

print(f"Finished {ddm_it} iterations.")

fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
ax.plot(ddm_losses, alpha=0.7)
ax.set_xlabel("iter"); ax.set_ylabel("mask-weighted MSE")
ax.set_title("pixel-DDM training loss")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% Pixel-DDM — inference on the currently-selected sample
DDM_INFER_STEPS = 50

ddm_unet.eval()
with torch.no_grad():
    masked_a_lr, image_b_lr, mask_lr = _resize_volumes(
        masked_a, image_b_real, mask, size=DDM_SPATIAL,
    )
    ddm_scheduler.set_timesteps(DDM_INFER_STEPS, device=DEVICE)
    z = torch.randn(1, 4, *DDM_SPATIAL, device=DEVICE,
                    generator=_gen())
    z = z * float(getattr(ddm_scheduler, "init_noise_sigma", 1.0))
    for t in ddm_scheduler.timesteps:
        t_b = t.expand(1).to(DEVICE).long() if t.ndim == 0 else t.to(DEVICE).long()
        model_input = torch.cat([z, masked_a_lr, mask_lr], dim=1)
        pred = ddm_unet(x=model_input, timesteps=t_b)
        z = ddm_scheduler.step(pred, t, z)[0]

    # Upsample back to original spatial size for an apples-to-apples SSIM.
    ddm_pred = F.interpolate(z, size=SPATIAL_SIZE, mode="trilinear", align_corners=False)

ssim_ddm = compute_ssim_per_modality(ddm_pred[0], image_b_real[0], mask=mask[0])
ssim_p2p = compute_ssim_per_modality(pix2pix_pred[0], image_b_real[0], mask=mask[0])
ssim_ldm = compute_ssim_per_modality(last_pred[0], image_b_real[0], mask=mask[0])

print(f"\n--- Sample {SAMPLE_IDX}  ({IDX_TO_TRAJECTORY[int(true_traj)]}) ---")
print(f"  pixel-DDM   global={ssim_ddm['ssim_global_mean']:.4f}  roi={ssim_ddm['ssim_roi_mean']:.4f}")
print(f"  pix2pix     global={ssim_p2p['ssim_global_mean']:.4f}  roi={ssim_p2p['ssim_roi_mean']:.4f}")
print(f"  LDM         global={ssim_ldm['ssim_global_mean']:.4f}  roi={ssim_ldm['ssim_roi_mean']:.4f}")

show_slices(
    {"masked_a":   _to_np(masked_a),
     "pixel-DDM":  _to_np(ddm_pred),
     "pix2pix":    _to_np(pix2pix_pred),
     "LDM":        _to_np(last_pred),
     "real_b":     _to_np(image_b_real)},
    slice_axis=2, modality=0, mask_np=mask_np,
    suptitle=f"pixel-DDM vs pix2pix vs LDM   sample {SAMPLE_IDX}   "
             f"traj={IDX_TO_TRAJECTORY[int(true_traj)]}",
)


