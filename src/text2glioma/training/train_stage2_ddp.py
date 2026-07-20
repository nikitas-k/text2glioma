"""DDP-aware LDM (stage 2) training using MONAI DecathlonDataset or custom datalist.

Requires a pre-trained Stage-1 VAE checkpoint (``--stage1_uri``).

With DecathlonDataset::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage2_ddp \
        --config configs/ldm.yaml --stage1_config configs/stage1.yaml \
        --stage1_uri /path/to/best_model.pth --run_dir /runs/

With a custom datalist::

    torchrun --nproc_per_node=4 -m text2glioma.training.train_stage2_ddp \
        --config configs/ldm.yaml --stage1_config configs/stage1.yaml \
        --stage1_uri /path/to/best_model.pth --run_dir /runs/ \
        --datalist datalist_task03.json --no_channel_reorder
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai import transforms as T
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from text2glioma.training.training_functions import train_ldm
from text2glioma.utils import (
    compute_scale_factor,
    get_model,
    load_config,
    load_text_encoder_and_tokenizer,
    stage1_ify,
    WhiteningStage1Wrapper,
)

warnings.filterwarnings("ignore")

# ── Channel reorder: MSD BraTS (FLAIR/T1/T1CE/T2) → pipeline (T1/T1CE/T2/FLAIR)
MSD_TO_T2G = [1, 2, 3, 0]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-2 LDM with DDP on BraTS or a custom datalist",
    )
    p.add_argument("--config", type=str, required=True,
                    help="LDM (stage 2) YAML config (e.g. configs/ldm.yaml).")
    p.add_argument("--stage1_config", type=str, required=True,
                    help="Stage-1 VAE YAML config (e.g. configs/stage1.yaml).")
    p.add_argument("--stage1_uri", type=str, required=True,
                    help="Path to pretrained Stage-1 VAE checkpoint (.pth).")
    p.add_argument("--run_dir", type=str, required=True,
                    help="Root output directory.")
    p.add_argument("--data_dir", type=str, default="./data",
                    help="Root for DecathlonDataset download / cache.")
    p.add_argument("--datalist", type=str, default=None,
                    help="Path to a JSON datalist (overrides --data_dir / DecathlonDataset). "
                         "JSON must have 'training' and 'validation' keys, each a list of "
                         "dicts with 'image' and 'label' paths.")
    p.add_argument("--no_channel_reorder", action="store_true", default=False,
                    help="Skip MSD→pipeline channel reorder (use for non-MSD data).")
    p.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of training set reserved for validation.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true", default=False)
    p.add_argument("--num_epochs", type=int, default=250)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False,
                    help="Resume from checkpoint.pth in run_dir.")
    p.add_argument("--warm_start_stage2", type=str, default=None,
                    help="Load LDM (and EMA if present) weights from this path, then "
                         "start a fresh training run (epoch 0, fresh optimizer). Use to "
                         "seed a schedule-change experiment (e.g. joint-dropout retrain) "
                         "from an existing checkpoint without inheriting its optimizer state.")
    p.add_argument("--dist_backend", type=str, default="nccl",
                    choices=["nccl", "gloo"],
                    help="Distributed backend (nccl for GPU, gloo for CPU/fallback).")
    p.add_argument("--find_unused_parameters", action="store_true", default=False,
                    help="Pass find_unused_parameters=True to DDP.")
    p.add_argument("--scale_factor", type=float, default=None,
                    help="Latent scale factor. If omitted, auto-computed as "
                         "1/std(latents) over training data (recommended).")
    p.add_argument("--latent_whitening_path", type=str, default=None,
                    help="Path to a whitening .pt produced by "
                         "scripts/fit_latent_whitening.py. If set, the stage-1 "
                         "encoder is wrapped so it emits ZCA-whitened latents "
                         "and its decode() inverts the whitening. --scale_factor "
                         "is forced to 1.0 when this is set (whitening handles "
                         "normalisation).")
    p.add_argument("--train_spec", type=str, default="impression",
                    choices=["impression", "findings"],
                    help="Text field used for conditioning.")
    p.add_argument("--mask_dropout_p", type=float, default=None,
                    help="Override mask dropout probability (default: from config).")
    p.add_argument("--text_dropout_p", type=float, default=None,
                    help="Override text dropout probability (default: from config).")
    p.add_argument("--joint_dropout_p", type=float, default=0.0,
                    help="Probability of forcing BOTH text and mask to their uncond "
                         "state on the same sample, applied on top of the independent "
                         "dropouts. Directly trains the fully-unconditional CFG branch.")
    p.add_argument("--use_molecular_conditioning", action="store_true",
                    help="Enable learnable IDH/MGMT class conditioning "
                         "(prepends 2 learnable pseudo-tokens to the RadBERT text "
                         "embedding sequence). Requires the datalist to carry "
                         "'idh' and 'mgmt' integer fields (0/1/2 for wt/mut/unk "
                         "and unm/met/unk). See src/text2glioma/training/"
                         "molecular_conditioning.py.")
    p.add_argument("--molecular_dropout_p", type=float, default=0.2,
                    help="Independent per-field dropout-to-unknown probability "
                         "for the molecular head at training time. Drives the "
                         "CFG null direction for IDH and MGMT independently.")
    p.add_argument("--molecular_lr_multiplier", type=float, default=1.0,
                    help="LR multiplier for the molecular embedding parameters "
                         "(applied on top of the base LR). Set >1 for faster "
                         "convergence on the 4.6k new params.")
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Cache directory for HuggingFace models / tokenizers.")
    p.add_argument("--stage2_run_dir", type=str, default=None,
                    help="Explicit Stage-2 output directory. When set, this is used "
                         "verbatim and the auto-resolution from --stage1_uri is skipped. "
                         "Useful when training multiple Stage-2 variants off the same "
                         "Stage-1 run (e.g. checkpoint.pth vs final_model.pth).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed(backend: str) -> tuple[int, int, int]:
    """Initialise the process group and return (rank, world_size, local_rank).

    Works automatically when launched via ``torchrun``, which sets the
    environment variables ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, and
    ``MASTER_ADDR``/``MASTER_PORT``.  If those variables are missing the
    function falls back to single-process mode.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return rank, world_size, local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def print0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main(rank):
        print(msg)


def _resolve_stage2_run_dir(
    stage1_uri: str,
    run_dir_arg: str,
    explicit_override: Optional[str] = None,
) -> Path:
    """Resolve Stage-2 run directory.

    Resolution order:
      1. ``explicit_override`` (from ``--stage2_run_dir``) — used verbatim.
      2. Infer from Stage-1 checkpoint path so runs stay grouped by experiment:
           .../runs/v4/autoencoder_stage1/output/models/best_model.pth
             -> .../runs/v4/ldm_stage2
      3. Fallback: ``run_dir_arg`` with an ``ldm_stage2`` suffix.
    """
    if explicit_override:
        return Path(explicit_override).expanduser().resolve()

    stage1_path = Path(stage1_uri).expanduser().resolve()

    parts = stage1_path.parts
    if "autoencoder_stage1" in parts:
        idx = parts.index("autoencoder_stage1")
        run_root = Path(*parts[:idx])
        if str(run_root):
            return run_root / "ldm_stage2"

    return Path(run_dir_arg).expanduser().resolve() / "ldm_stage2"


# ---------------------------------------------------------------------------
# Transforms  (image + label for mask conditioning)
# ---------------------------------------------------------------------------

def get_train_transform(channel_reorder: bool = True) -> T.Compose:
    """Training transforms for 4-ch BraTS images + segmentation labels."""
    xforms = [
        T.LoadImaged(keys=["image", "label"]),
        # Image: (H,W,D,4) → (4,H,W,D)
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        # Label: (H,W,D,1) → (1,H,W,D) or already channel-first
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=["label"], dtype=torch.float32),
        # Spatial
        T.Orientationd(keys=["image", "label"], axcodes="LPS"),
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        T.SpatialPadd(keys=["image", "label"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 224, 160)),
        # Intensity (image only)
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        # Augmentation
        T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        T.RandAffined(
            keys=["image", "label"], prob=0.1,
            translate_range=(1, 1, 1), scale_range=(-0.02, 0.02),
            spatial_size=[160, 224, 160],
            mode=["trilinear", "nearest"],
        ),
        T.RandShiftIntensityd(
            keys=["image"], offsets=0.05, prob=0.1, channel_wise=True,
        ),
        T.RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.97, 1.03)),
        T.ToTensord(keys=["image", "label"]),
    ])
    return T.Compose([x for x in xforms if x is not None])


def get_val_transform(channel_reorder: bool = True) -> T.Compose:
    """Validation transforms (deterministic)."""
    xforms = [
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    if channel_reorder:
        xforms.append(T.Lambdad(keys=["image"], func=lambda x: x[MSD_TO_T2G]))
    xforms.extend([
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=["label"], dtype=torch.float32),
        T.Orientationd(keys=["image", "label"], axcodes="LPS"),
        T.CropForegroundd(keys=["image", "label"], source_key="image"),
        T.SpatialPadd(keys=["image", "label"], spatial_size=(160, 224, 160), mode="constant"),
        T.CenterSpatialCropd(keys=["image", "label"], roi_size=(160, 224, 160)),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1,
            channel_wise=True,
        ),
        T.ToTensord(keys=["image", "label"]),
    ])
    return T.Compose([x for x in xforms if x is not None])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_determinism(args.seed)

    rank, world_size, local_rank = setup_distributed(args.dist_backend)
    distributed = world_size > 1

    print0(f"World size: {world_size}  |  Backend: {args.dist_backend}", rank)
    if is_main(rank):
        print_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = load_config(args.config)
    stage1_config = load_config(args.stage1_config)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    channel_reorder = not args.no_channel_reorder

    if args.datalist:
        # ── Custom JSON datalist ─────────────────────────────────────
        print0(f"Loading datalist from {args.datalist}", rank)
        with open(args.datalist) as f:
            datalist = json.load(f)
        train_data = datalist["training"]
        val_data = datalist["validation"]
        print0(f"  {len(train_data)} training, {len(val_data)} validation entries", rank)

        train_ds = Dataset(data=train_data, transform=get_train_transform(channel_reorder))
        val_ds = Dataset(data=val_data, transform=get_val_transform(channel_reorder))
    else:
        # ── DecathlonDataset (BraTS) ─────────────────────────────────
        if is_main(rank):
            Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        task_dir = Path(args.data_dir) / "Task01_BrainTumour"
        need_download = not task_dir.is_dir()
        download = need_download and is_main(rank)
        if distributed:
            dist.barrier()

        train_ds = DecathlonDataset(
            root_dir=args.data_dir,
            task="Task01_BrainTumour",
            section="training",
            download=download,
            seed=args.seed,
            val_frac=args.val_frac,
            transform=get_train_transform(channel_reorder),
            num_workers=args.num_workers,
        )
        if distributed:
            dist.barrier()

        val_ds = DecathlonDataset(
            root_dir=args.data_dir,
            task="Task01_BrainTumour",
            section="validation",
            download=False,
            seed=args.seed,
            val_frac=args.val_frac,
            transform=get_val_transform(channel_reorder),
            num_workers=args.num_workers,
        )

    # Samplers
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    print0(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples", rank)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Stage-1 VAE (frozen)
    # ------------------------------------------------------------------
    print0("Loading frozen Stage-1 VAE …", rank)

    # The stage-1 config's default latent_channels can disagree with the
    # actual checkpoint (e.g. a lc=3 stage-1 loaded via the default lc=6
    # stage1.yaml). Read the true value from the checkpoint before model
    # construction so load_state_dict does not fail on channel-size
    # mismatches at quant_conv_mu / post_quant_conv / encoder.blocks[-1]
    # / decoder.blocks[0].
    try:
        _ckpt_probe = torch.load(args.stage1_uri, map_location="cpu", weights_only=False)
        _sd = _ckpt_probe
        if isinstance(_ckpt_probe, dict):
            for _k in ("state_dict", "model", "autoencoder", "vae"):
                if isinstance(_ckpt_probe.get(_k), dict) and _ckpt_probe[_k]:
                    _sd = _ckpt_probe[_k]
                    break
        _inferred_lc = None
        for _k in ("quant_conv_mu.conv.weight",
                   "module.quant_conv_mu.conv.weight",
                   "model.quant_conv_mu.conv.weight"):
            _w = _sd.get(_k) if isinstance(_sd, dict) else None
            if torch.is_tensor(_w):
                _inferred_lc = int(_w.shape[0])
                break
        if _inferred_lc is not None:
            _cfg_lc = int(
                stage1_config.get("model", {}).get("params", {}).get("latent_channels", -1)
            )
            if _cfg_lc != _inferred_lc:
                print0(
                    f"[stage1] latent_channels mismatch: config={_cfg_lc}, "
                    f"checkpoint={_inferred_lc}. Overriding config with checkpoint value.",
                    rank,
                )
                stage1_config.setdefault("model", {}).setdefault("params", {})[
                    "latent_channels"
                ] = _inferred_lc
        del _ckpt_probe, _sd
    except Exception as _e:  # pragma: no cover — best-effort probe
        print0(f"[stage1] latent_channels probe failed ({_e!r}); "
               "falling back to config value.", rank)

    stage1 = stage1_ify(
        get_model(
            model_type="AutoencoderKL",
            config=stage1_config,
            from_file=args.stage1_uri,
        )
    )
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False
    stage1 = stage1.to(device)

    # ------------------------------------------------------------------
    # Optional: wrap Stage-1 with a ZCA whitening layer so the LDM sees
    # a latent whose channels are decorrelated and unit-variance. Fitted
    # once offline via scripts/fit_latent_whitening.py.
    # ------------------------------------------------------------------
    if args.latent_whitening_path is not None:
        whit_path = Path(args.latent_whitening_path).expanduser().resolve()
        if not whit_path.is_file():
            raise FileNotFoundError(f"--latent_whitening_path not found: {whit_path}")
        print0(f"Loading latent whitening from {whit_path}", rank)
        whit = torch.load(str(whit_path), map_location="cpu")
        stage1 = WhiteningStage1Wrapper(
            stage1,
            mu=whit["mu"].to(device),
            W=whit["W"].to(device),
            W_inv=whit["W_inv"].to(device),
        ).to(device)
        stage1.eval()
        for param in stage1.parameters():
            param.requires_grad = False
        print0(f"  latent_channels = {stage1.latent_channels}, "
               f"fit on {whit.get('n_samples', '?')} voxels, kind={whit.get('kind', '?')}", rank)

    # ------------------------------------------------------------------
    # Reconcile latent channels between Stage-1 checkpoint and Stage-2
    # ------------------------------------------------------------------
    num_mask_classes = int(config.get("mask", {}).get("num_classes", 4))
    stage1_latent_ch = None

    # Preferred: run one forward pass and read latent channels directly.
    # This is robust to wrappers that expand channels (e.g., channel-wise Pinaya).
    try:
        probe_batch = next(iter(train_loader))
        probe_img = probe_batch["image"][:1].to(device)
        with torch.no_grad():
            probe_z = stage1(probe_img)
        stage1_latent_ch = int(probe_z.shape[1])
    except Exception:
        stage1_latent_ch = None

    # Fallback: read from Stage-1 model attributes.
    if stage1_latent_ch is None and hasattr(stage1, "model") and hasattr(stage1.model, "latent_channels"):
        try:
            stage1_latent_ch = int(stage1.model.latent_channels)
        except Exception:
            stage1_latent_ch = None

    # Fallback: infer from quant conv shape when available.
    if stage1_latent_ch is None and hasattr(stage1, "model") and hasattr(stage1.model, "quant_conv_mu"):
        try:
            stage1_latent_ch = int(stage1.model.quant_conv_mu.out_channels)
        except Exception:
            stage1_latent_ch = None

    cfg_latent = int(config.get("model", {}).get("latent_channels", config.get("model", {}).get("params", {}).get("out_channels", 3)))
    if stage1_latent_ch is None:
        stage1_latent_ch = cfg_latent
        print0(
            f"[WARN] Could not infer Stage-1 latent channels; using config value {cfg_latent}.",
            rank,
        )

    expected_in_channels = stage1_latent_ch + num_mask_classes
    cfg_params = config.setdefault("model", {}).setdefault("params", {})
    cfg_in = int(cfg_params.get("in_channels", expected_in_channels))
    cfg_out = int(cfg_params.get("out_channels", stage1_latent_ch))

    if cfg_latent != stage1_latent_ch or cfg_in != expected_in_channels or cfg_out != stage1_latent_ch:
        print0(
            "[WARN] Stage-2 config/channel mismatch detected. "
            f"Stage-1 latent={stage1_latent_ch}, mask_classes={num_mask_classes}, "
            f"config latent={cfg_latent}, in={cfg_in}, out={cfg_out}. "
            "Auto-aligning Stage-2 UNet channels to match Stage-1.",
            rank,
        )

    config["model"]["latent_channels"] = stage1_latent_ch
    cfg_params["in_channels"] = expected_in_channels
    cfg_params["out_channels"] = stage1_latent_ch
    print0(
        f"Stage-2 channels: latent={stage1_latent_ch}, mask={num_mask_classes}, "
        f"UNet in={expected_in_channels}, out={stage1_latent_ch}",
        rank,
    )

    # ------------------------------------------------------------------
    # LDM (diffusion model)
    # ------------------------------------------------------------------
    print0("Initialising LDM …", rank)
    model_type = config["model"].get("name", "DiffusionModelUNet")
    ldm = get_model(model_type, config)

    # ── Zero-init mask channels in the first conv layer ──────────────
    # The UNet's conv_in accepts [latent_ch + mask_ch] channels.  With
    # random init the mask channels inject noise that the model must
    # first learn to suppress, slowing convergence dramatically.  By
    # zeroing the mask-channel weights the model starts as if it were
    # a latent-only model and *gradually* learns to use the mask.
    latent_ch = stage1_latent_ch
    with torch.no_grad():
        first_conv = ldm.conv_in.conv  # nn.Conv3d(7, 256, 3)
        first_conv.weight[:, latent_ch:].zero_()
        print0(f"Zero-initialised conv_in mask channels [{latent_ch}:{first_conv.weight.shape[1]}]", rank)

    ldm = ldm.to(device)

    if distributed:
        ldm = DDP(
            ldm, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    # ------------------------------------------------------------------
    # Noise scheduler
    # ------------------------------------------------------------------
    scheduler_name = config["scheduler"].get("name", "DDIMScheduler")
    scheduler_params = config["scheduler"].get("params", {})
    if scheduler_name == "DDPMScheduler":
        scheduler = DDPMScheduler(**scheduler_params)
    elif scheduler_name == "DDIMScheduler":
        scheduler = DDIMScheduler(**scheduler_params)
    else:
        raise ValueError(f"Unsupported noise scheduler: {scheduler_name}")

    # ------------------------------------------------------------------
    # Text encoder + tokenizer (frozen)
    # ------------------------------------------------------------------
    print0("Loading text encoder …", rank)
    tokenizer, text_encoder = load_text_encoder_and_tokenizer(
        config["conditioning"],
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    # Override model_max_length if config specifies it (avoids wasteful
    # 512-token padding for BERT-family encoders)
    cfg_max_len = config["conditioning"].get("max_length")
    if cfg_max_len is not None:
        tokenizer.model_max_length = cfg_max_len
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Optional molecular class-conditioning head (IDH + MGMT)
    # ------------------------------------------------------------------
    molecular_head = None
    if args.use_molecular_conditioning:
        from text2glioma.training.molecular_conditioning import MolecularClassConditioning
        # Introspect the text-encoder hidden dim so the two branches emit
        # equal-width vectors and can be concatenated along the sequence axis.
        hidden_dim = int(getattr(text_encoder.config, "hidden_size",
                                  getattr(text_encoder.config, "dim", 768)))
        molecular_head = MolecularClassConditioning(
            hidden_dim=hidden_dim,
            dropout_to_unknown_p=float(args.molecular_dropout_p),
        ).to(device)
        print0(f"Molecular head enabled: hidden_dim={hidden_dim}, "
               f"dropout_p={args.molecular_dropout_p}, "
               f"params={sum(p.numel() for p in molecular_head.parameters())}",
               rank)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    base_lr = config["model"].get("base_lr", 1e-4)
    if molecular_head is not None and args.molecular_lr_multiplier != 1.0:
        # Separate param group so the fresh molecular embeddings can be
        # trained at a different LR than the fine-tuning LDM body.
        optimizer = optim.AdamW([
            {"params": list(ldm.parameters()),           "lr": base_lr},
            {"params": list(molecular_head.parameters()), "lr": base_lr * float(args.molecular_lr_multiplier)},
        ])
        print0(f"Optimizer: base_lr={base_lr}, molecular_lr={base_lr * args.molecular_lr_multiplier}", rank)
    elif molecular_head is not None:
        optimizer = optim.AdamW(
            list(ldm.parameters()) + list(molecular_head.parameters()),
            lr=base_lr,
        )
    else:
        optimizer = optim.AdamW(ldm.parameters(), lr=base_lr)

    # ------------------------------------------------------------------
    # Latent scale factor
    # ------------------------------------------------------------------
    if args.latent_whitening_path is not None:
        # Whitening enforces unit-variance channels; scale_factor should be 1.
        scale_factor = 1.0
        print0("Whitening active: forcing scale_factor = 1.0", rank)
    elif args.scale_factor is not None:
        scale_factor = args.scale_factor
        print0(f"Using user-specified scale_factor = {scale_factor:.4f}", rank)
    else:
        print0("Auto-computing latent scale_factor from training data …", rank)
        scale_factor = compute_scale_factor(
            stage1, train_loader, device, max_batches=50,
        )
        print0(f"Computed scale_factor = {scale_factor:.4f}", rank)

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    run_dir = _resolve_stage2_run_dir(
        args.stage1_uri,
        args.run_dir,
        explicit_override=args.stage2_run_dir,
    )
    output_dir = run_dir / "output"
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    print0(f"Resolved Stage-2 run directory: {run_dir}", rank)
    if is_main(rank):
        for d in [output_dir, model_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    writer_train = SummaryWriter(log_dir / "train") if is_main(rank) else None
    writer_val = SummaryWriter(log_dir / "val") if is_main(rank) else None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    ckpt_path = run_dir / "checkpoint.pth"

    ema_state_dict = None
    molecular_head_state_dict = None
    # ldm may already be DDP-wrapped, in which case its state_dict keys are
    # prefixed with `module.`. Checkpoints saved by this trainer use
    # raw_model.state_dict() (no prefix), so load into the underlying module
    # to keep resume and warm-start robust to wrapping.
    raw_ldm = ldm.module if hasattr(ldm, "module") else ldm
    if args.resume and ckpt_path.exists():
        print0(f"Resuming from {ckpt_path}", rank)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ldm_state = ckpt.get("diffusion", ckpt.get("ldm_state_dict"))
        if ldm_state is None:
            raise KeyError("Checkpoint missing 'diffusion' or 'ldm_state_dict' key.")
        raw_ldm.load_state_dict(ldm_state)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        ema_state_dict = ckpt.get("ema")
        if ema_state_dict is not None:
            print0("  Loaded EMA state from checkpoint.", rank)
        molecular_head_state_dict = ckpt.get("molecular_head")
        if molecular_head is not None and molecular_head_state_dict is not None:
            print0("  Loaded molecular head state from checkpoint.", rank)
        elif molecular_head is not None:
            print0("  [WARN] --use_molecular_conditioning set but checkpoint has no "
                   "'molecular_head' key; starting from fresh init.", rank)
        print0(f"Resumed at epoch {start_epoch}", rank)
    elif args.warm_start_stage2:
        seed_path = Path(args.warm_start_stage2).expanduser().resolve()
        if not seed_path.is_file():
            raise FileNotFoundError(f"--warm_start_stage2 not found: {seed_path}")
        print0(f"Warm-starting LDM weights from {seed_path} (fresh optimizer, epoch 0)", rank)
        seed = torch.load(seed_path, map_location="cpu")
        ldm_state = seed.get("diffusion", seed.get("ldm_state_dict", seed))
        if not isinstance(ldm_state, dict):
            raise KeyError("warm-start checkpoint has no recognisable state_dict payload.")
        raw_ldm.load_state_dict(ldm_state)
        ema_state_dict = seed.get("ema")
        if ema_state_dict is not None:
            print0("  Loaded EMA state from warm-start checkpoint.", rank)
        # Warm-start checkpoints predate the molecular head; leave it at fresh init.
        if molecular_head is not None:
            print0("  Molecular head: fresh init (warm-start seed has none).", rank)
    else:
        print0("Starting fresh training.", rank)

    # ------------------------------------------------------------------
    # Dropout overrides
    # ------------------------------------------------------------------
    text_dropout = (
        args.text_dropout_p
        if args.text_dropout_p is not None
        else config["conditioning"].get("dropout_p", 0.2)
    )
    mask_dropout = (
        args.mask_dropout_p
        if args.mask_dropout_p is not None
        else config.get("mask", {}).get("dropout_p", 0.2)
    )

    # ------------------------------------------------------------------
    # DDP wrap the molecular head (if enabled and distributed)
    # ------------------------------------------------------------------
    if molecular_head is not None and distributed:
        molecular_head = torch.nn.parallel.DistributedDataParallel(
            molecular_head, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print0("Starting training …", rank)
    val_loss = train_ldm(
        model=ldm,
        stage1=stage1,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        n_epochs=args.num_epochs,
        start_epoch=start_epoch,
        text_field=args.train_spec,
        val_interval=args.val_interval,
        dropout_p=text_dropout,
        model_dir=model_dir,
        writer_train=writer_train,
        writer_val=writer_val,
        run_dir=run_dir,
        scale_factor=scale_factor,
        num_mask_classes=num_mask_classes,
        mask_dropout_p=mask_dropout,
        joint_dropout_p=float(args.joint_dropout_p),
        latent_channels=stage1_latent_ch,
        ema_state_dict=ema_state_dict,
        molecular_head=molecular_head,
        molecular_head_state_dict=molecular_head_state_dict,
    )

    print0(f"Training finished.  Final val loss: {val_loss:.4f}", rank)
    cleanup()


if __name__ == "__main__":
    main()
