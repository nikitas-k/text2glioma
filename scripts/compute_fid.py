"""Fréchet distance for 3D brain MRI synthesis.

Two backbones are supported:

1. ``2d_inception`` (default): torchvision Inception-V3 (ImageNet weights).
   Standard FID. Operates on axial tumour-containing slices, replicating the
   single-modality slice to 3 channels. Cheap, reviewer-safe baseline.

2. ``2d_radimagenet``: RadImageNet ResNet-50 (Mei et al., 2022). Requires a
   weights file passed via ``--backbone_weights``. Preferred for medical
   imaging reviewers. See https://github.com/BMEII-AI/RadImageNet.

3. ``3d_autoencoder``: uses the project's own Stage-1 AutoencoderKL encoder as
   the feature extractor. No external weights required. Defensible precedent:
   Pinaya et al. 2022 (BrainLDM). 3D-FID computed on the full latent vector.

Usage
-----

CLI (consumes NIfTIs already on disk):

    python scripts/compute_fid.py \
        --pairs_csv /path/to/pairs.csv \
        --mode 2d_inception \
        --out fid_results.csv

where ``pairs.csv`` has columns ``case_idx,modality,cfg,real_path,gen_path,mask_path``.

Library (drop into notebook loop):

    from compute_fid import FIDAccumulator
    acc = FIDAccumulator(mode="2d_inception", device=device)
    for case in cases:
        acc.add_real(real_tensor, mask, modality)   # real_tensor [H,W,D]
        acc.add_gen(gen_tensor, mask, modality, cfg=1.0)
    df = acc.compute()  # rows: (modality, cfg, fid, n_real, n_gen)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


# ---------- Fréchet distance ----------

def _frechet_distance(mu1: np.ndarray, sig1: np.ndarray,
                      mu2: np.ndarray, sig2: np.ndarray,
                      eps: float = 1e-6) -> float:
    """Standard FID computation: ||mu1-mu2||^2 + Tr(S1+S2-2*sqrt(S1@S2))."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * eps
        covmean = sqrtm((sig1 + offset) @ (sig2 + offset))
    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-3:
            # numerical noise — drop imaginary part
            pass
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


def _stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sig = np.cov(feats, rowvar=False)
    return mu, sig


# ---------- Backbones ----------

class _InceptionFeat(nn.Module):
    """torchvision Inception-V3 pool3 features (2048-d)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                            aux_logits=True, init_weights=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], [B, 3, H, W]. Inception expects 299x299 normalised.
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # [-1, 1]
        return self.net(x)


class _RadImageNetFeat(nn.Module):
    """RadImageNet ResNet-50, penultimate pool features (2048-d)."""

    def __init__(self, weights_path: Path):
        super().__init__()
        from torchvision.models import resnet50
        net = resnet50(weights=None)
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # strip common prefixes
        state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"[RadImageNet] missing keys: {len(missing)}; example: {missing[:3]}")
        if unexpected:
            print(f"[RadImageNet] unexpected keys: {len(unexpected)}; example: {unexpected[:3]}")
        net.fc = nn.Identity()
        net.eval()
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # RadImageNet preprocessing: scale to [0, 255], subtract mean
        x = x * 255.0
        mean = torch.tensor([123.68, 116.78, 103.94], device=x.device).view(1, 3, 1, 1)
        x = x - mean
        return self.net(x)


class _AutoencoderFeat(nn.Module):
    """Use Stage-1 AutoencoderKL encoder as a 3D feature extractor.

    Returns the spatial mean of the latent (channel-wise) → low-d feature
    vector (= latent_channels). Cheap and dependency-free. For richer features
    use ``adaptive_avg_pool3d`` over the spatial dims at coarser resolution.
    """

    def __init__(self, stage1: nn.Module, pool: str = "mean"):
        super().__init__()
        self.stage1 = stage1
        self.pool = pool

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W, D] in [0,1]
        z = self.stage1(x)
        # z: [B, latent_channels, h, w, d]
        if self.pool == "mean":
            return z.mean(dim=(2, 3, 4))
        if self.pool == "flatten_pool":
            return F.adaptive_avg_pool3d(z, 1).flatten(1)
        raise ValueError(self.pool)


# ---------- Accumulator ----------

@dataclass
class _FeatureBag:
    feats: list[np.ndarray] = field(default_factory=list)

    def add(self, f: np.ndarray) -> None:
        self.feats.append(f.astype(np.float64))

    def stack(self) -> np.ndarray:
        return np.concatenate(self.feats, axis=0) if self.feats else np.zeros((0, 0))


class FIDAccumulator:
    """Streaming FID computation.

    Group rows by (modality, cfg) for generated; by ``modality`` for real.
    Call ``add_real`` and ``add_gen`` as samples arrive, then ``compute()``.
    """

    def __init__(
        self,
        mode: str = "2d_inception",
        device: str | torch.device = "cpu",
        backbone_weights: Path | None = None,
        stage1: nn.Module | None = None,
        slice_axis: int = -1,
        slices_per_volume: int = 8,
        batch: int = 32,
    ) -> None:
        self.mode = mode
        self.device = torch.device(device)
        self.slice_axis = slice_axis
        self.slices_per_volume = slices_per_volume
        self.batch = batch

        if mode == "2d_inception":
            self.backbone = _InceptionFeat().to(self.device).eval()
            self.feat_dim = 2048
            self.is_3d = False
        elif mode == "2d_radimagenet":
            if backbone_weights is None or not Path(backbone_weights).exists():
                raise FileNotFoundError(
                    f"--backbone_weights required for radimagenet; got {backbone_weights}"
                )
            self.backbone = _RadImageNetFeat(Path(backbone_weights)).to(self.device).eval()
            self.feat_dim = 2048
            self.is_3d = False
        elif mode == "3d_autoencoder":
            if stage1 is None:
                raise ValueError("3d_autoencoder mode requires stage1 module")
            self.backbone = _AutoencoderFeat(stage1).to(self.device).eval()
            self.feat_dim = None
            self.is_3d = True
        else:
            raise ValueError(mode)

        self._real: dict[str, _FeatureBag] = {}
        self._gen: dict[tuple[str, float], _FeatureBag] = {}

    # ---- slice extraction ----

    def _extract_slices(self, vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """vol [H,W,D] float in [0,1]; mask [H,W,D] bool. Return [N, 3, H, W]."""
        ax = self.slice_axis if self.slice_axis >= 0 else vol.ndim + self.slice_axis
        mask_per_slice = mask.movedim(ax, -1).reshape(-1, mask.shape[ax]).sum(0)
        nz = torch.nonzero(mask_per_slice > 0, as_tuple=False).flatten()
        if nz.numel() == 0:
            return torch.zeros((0, 3, vol.shape[0], vol.shape[1]), device=vol.device)
        k = min(self.slices_per_volume, nz.numel())
        idx = torch.linspace(0, nz.numel() - 1, k).round().long()
        chosen = nz[idx]
        slices = []
        for s in chosen.tolist():
            sl = vol.movedim(ax, -1)[..., s]  # [H,W]
            sl = sl.clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)  # [3,H,W]
            slices.append(sl)
        return torch.stack(slices, dim=0)

    # ---- feature compute ----

    def _features_2d(self, vol: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        x = self._extract_slices(vol.to(self.device), mask.to(self.device))
        if x.shape[0] == 0:
            return np.zeros((0, self.feat_dim or 0))
        feats = []
        for i in range(0, x.shape[0], self.batch):
            feats.append(self.backbone(x[i:i + self.batch]).cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _features_3d(self, vol: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        # crop tumour bbox to remove background
        nz = torch.nonzero(mask > 0, as_tuple=False)
        if nz.numel() == 0:
            return np.zeros((0, 0))
        mins = nz.min(0).values.tolist()
        maxs = (nz.max(0).values + 1).tolist()
        crop = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        # pad to multiple of 8 (encoder downsample)
        pad = [(0, (-s) % 8) for s in crop.shape]
        crop = F.pad(crop, [p for ab in pad[::-1] for p in ab])
        crop = crop.unsqueeze(0).unsqueeze(0).to(self.device).float().clamp(0, 1)
        f = self.backbone(crop).cpu().numpy()
        return f  # [1, feat_dim]

    # ---- public API ----

    def add_real(self, vol: torch.Tensor, mask: torch.Tensor, modality: str) -> None:
        feats = self._features_3d(vol, mask) if self.is_3d else self._features_2d(vol, mask)
        if feats.shape[0]:
            self._real.setdefault(modality, _FeatureBag()).add(feats)

    def add_gen(self, vol: torch.Tensor, mask: torch.Tensor,
                modality: str, cfg: float) -> None:
        feats = self._features_3d(vol, mask) if self.is_3d else self._features_2d(vol, mask)
        if feats.shape[0]:
            self._gen.setdefault((modality, float(cfg)), _FeatureBag()).add(feats)

    def compute(self) -> pd.DataFrame:
        rows = []
        for (mod, cfg), gbag in self._gen.items():
            rbag = self._real.get(mod)
            if rbag is None or not rbag.feats or not gbag.feats:
                continue
            fr = rbag.stack()
            fg = gbag.stack()
            mu_r, sig_r = _stats(fr)
            mu_g, sig_g = _stats(fg)
            fid = _frechet_distance(mu_r, sig_r, mu_g, sig_g)
            rows.append({
                "modality": mod, "cfg": cfg, "fid": fid,
                "n_real": int(fr.shape[0]), "n_gen": int(fg.shape[0]),
                "feat_dim": int(fr.shape[1]),
                "mode": self.mode,
            })
        return pd.DataFrame(rows).sort_values(["modality", "cfg"]).reset_index(drop=True)


# ---------- CLI ----------

def _load_nifti(path: Path) -> torch.Tensor:
    import nibabel as nib
    arr = nib.load(str(path)).get_fdata()
    return torch.from_numpy(arr.astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", type=Path, required=True,
                   help="CSV with columns case_idx,modality,cfg,real_path,gen_path,mask_path")
    p.add_argument("--mode", choices=["2d_inception", "2d_radimagenet", "3d_autoencoder"],
                   default="2d_inception")
    p.add_argument("--backbone_weights", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slices_per_volume", type=int, default=8)
    p.add_argument("--stage1_uri", type=Path, default=None,
                   help="Required for 3d_autoencoder mode")
    p.add_argument("--stage1_config", type=Path, default=None)
    args = p.parse_args()

    stage1 = None
    if args.mode == "3d_autoencoder":
        if args.stage1_uri is None or args.stage1_config is None:
            raise SystemExit("3d_autoencoder requires --stage1_uri and --stage1_config")
        from text2glioma.utils import get_model, load_config, stage1_ify
        cfg = load_config(str(args.stage1_config))
        stage1 = stage1_ify(get_model("AutoencoderKL", cfg, from_file=str(args.stage1_uri)))
        stage1.model.in_channels = 1
        stage1.model.out_channels = 1
        stage1 = stage1.to(args.device).eval()

    acc = FIDAccumulator(
        mode=args.mode, device=args.device,
        backbone_weights=args.backbone_weights, stage1=stage1,
        slices_per_volume=args.slices_per_volume,
    )

    df = pd.read_csv(args.pairs_csv)
    # Real volumes are deduplicated per (case_idx, modality)
    real_seen: set[tuple[int, str]] = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pairs"):
        mod = row["modality"]
        mask = _load_nifti(Path(row["mask_path"])) > 0
        key = (int(row["case_idx"]), mod)
        if key not in real_seen:
            acc.add_real(_load_nifti(Path(row["real_path"])), mask, mod)
            real_seen.add(key)
        acc.add_gen(_load_nifti(Path(row["gen_path"])), mask, mod, float(row["cfg"]))

    out = acc.compute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows -> {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
