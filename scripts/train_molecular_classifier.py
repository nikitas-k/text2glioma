"""Train a 3D DenseNet-121 to classify IDH or MGMT status from brain MRI.

Supports controlled real+synthetic augmentation ablations. For each run
the user specifies:

    --task       {idh, mgmt}
    --n_synthetic <int>     absolute number of synthetic samples to add
                             to the real training set (0 disables augmentation)
    --seed       <int>     RNG seed (repeat with 3+ seeds per condition)

The 30-run grid for the paper is enumerated by shell loop; each run is
one invocation of this script and is fully deterministic given
(task, n_synthetic, seed) plus a fixed --real_datalist and --synth_root.

Data
----
* Real training subjects come from ``--real_datalist`` which must carry
  integer ``idh`` and ``mgmt`` fields per sample (see
  ``scripts/build_datalist_with_molecular.py``). Rows whose task-specific
  status is ``UNKNOWN`` (2) are dropped both from training and validation
  because they contribute no labelled signal.
* Synthetic samples come from ``--synth_root`` (the release directory)
  paired with its ``manifest_release.csv``. Only samples with a known
  status for the current task are eligible; the first ``n_synthetic`` such
  samples (ordered by ``sample_id`` for reproducibility) are added to the
  training pool.
* Validation is always on the real ``validation`` split of the datalist.

Metrics
-------
Primary: AUROC on the real validation set. Secondary: F1, balanced
accuracy. Reported per-epoch and best across training. Best-model checkpoint
saved at the epoch with the highest val AUROC.

Usage
-----
::

    python scripts/train_molecular_classifier.py \\
        --task idh \\
        --real_datalist datalist_N494_molecular.json \\
        --synth_root /path/to/synth_release \\
        --synth_manifest /path/to/synth_release/manifest_release.csv \\
        --n_synthetic 1000 \\
        --seed 0 \\
        --n_epochs 200 \\
        --out_dir runs/cls_idh_synth1000_seed0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import PersistentDataset
from monai import transforms as T
from monai.networks import nets
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from text2glioma.training.molecular_conditioning import (  # noqa: E402
    IDH_MUTANT, IDH_UNKNOWN, IDH_WILDTYPE,
    MGMT_METHYLATED, MGMT_UNKNOWN, MGMT_UNMETHYLATED,
)


# ── Constants ──────────────────────────────────────────────────────────

_TARGET_SPATIAL = (160, 224, 160)
_UNKNOWN_BY_TASK = {"idh": IDH_UNKNOWN, "mgmt": MGMT_UNKNOWN}


# ── Data assembly ──────────────────────────────────────────────────────

def _filter_known(items: list[dict], task: str) -> list[dict]:
    """Drop entries whose task-specific status is UNKNOWN."""
    key = task
    unk = _UNKNOWN_BY_TASK[task]
    return [it for it in items if int(it.get(key, unk)) != unk]


def _normalise_item(item: dict, source: str) -> dict:
    """Project a heterogeneous datalist / manifest entry to the minimal
    schema the trainer's collate expects.

    All samples in a batch must share the exact same set of keys or
    PyTorch's ``default_collate`` raises ``KeyError``. Real items from
    the datalist carry ``label``, ``subject_id``, ``impression``,
    ``findings`` etc; synth items carry ``sample_id``. Only the four
    fields below are actually consumed downstream.
    """
    return {
        "image":  str(item["image"]),
        "idh":    int(item.get("idh",  IDH_UNKNOWN)),
        "mgmt":   int(item.get("mgmt", MGMT_UNKNOWN)),
        "source": source,
    }


def load_real_split(datalist_path: Path, task: str
                    ) -> tuple[list[dict], list[dict]]:
    """Return (train_items, val_items) filtered to the labelled subset."""
    with datalist_path.open() as fh:
        dl = json.load(fh)
    train = [_normalise_item(it, "real") for it in _filter_known(dl.get("training",   []), task)]
    val   = [_normalise_item(it, "real") for it in _filter_known(dl.get("validation", []), task)]
    return train, val


def load_synth_items(synth_manifest: Path, synth_root: Path, task: str,
                     n_synthetic: int) -> list[dict]:
    """Read the release manifest and select the first ``n_synthetic``
    samples with a KNOWN status for the requested task."""
    if n_synthetic <= 0:
        return []

    import pandas as pd
    df = pd.read_csv(synth_manifest)

    # Enrich from metadata.json if the manifest lacks the task column.
    if task not in df.columns:
        def _fetch_label(row, task=task):
            meta_path = synth_root / row["relpath_image"].replace("image.nii.gz", "metadata.json")
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                return int(meta.get(task, _UNKNOWN_BY_TASK[task]))
            except Exception:
                return _UNKNOWN_BY_TASK[task]
        df[task] = df.apply(_fetch_label, axis=1)

    # We also need the OTHER task label so the item schema matches real
    # samples (which always carry both idh and mgmt).
    other_task = "mgmt" if task == "idh" else "idh"
    if other_task not in df.columns:
        def _fetch_other(row, ot=other_task):
            meta_path = synth_root / row["relpath_image"].replace("image.nii.gz", "metadata.json")
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                return int(meta.get(ot, _UNKNOWN_BY_TASK[ot]))
            except Exception:
                return _UNKNOWN_BY_TASK[ot]
        df[other_task] = df.apply(_fetch_other, axis=1)

    unk = _UNKNOWN_BY_TASK[task]
    df = df[df[task] != unk].reset_index(drop=True)
    df = df.sort_values("sample_id").reset_index(drop=True)

    take = df.head(n_synthetic)
    items: list[dict] = []
    for _, r in take.iterrows():
        image_path = synth_root / r["relpath_image"]
        items.append(_normalise_item(
            {
                "image": str(image_path),
                "idh":   int(r["idh"]),
                "mgmt":  int(r["mgmt"]),
            },
            source="synth",
        ))
    return items


def _tag_source(items: list[dict], source: str) -> list[dict]:
    """Deprecated: kept only for backwards compatibility. Use _normalise_item."""
    return [{**it, "source": source} for it in items]


# ── Transforms ────────────────────────────────────────────────────────

def _build_transforms() -> tuple[T.Compose, T.Compose]:
    """MONAI transforms shared between real and synthetic samples.

    Deliberately *skips* ``CropForegroundd`` — the synthetic samples are
    already in the canonical (160, 224, 160) preprocessed space, and
    running CropForeground on them would crop to the tumour ROI. The
    ``SpatialPadd`` + ``CenterSpatialCropd`` combination is idempotent
    on already-preprocessed volumes and safely brings the raw real
    volumes to the same shape.

    Real training NIfTIs are stored as 4-channel float with the channel
    axis as the last dim; ``EnsureChannelFirstd(channel_dim=3)`` moves
    it to axis 0. Synthetic release NIfTIs have the same layout by
    construction (see ``scripts/dataset_release/generate_dataset.py``).
    """
    common = [
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"], channel_dim=3),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),
        T.Orientationd(keys=["image"], axcodes="LPS"),
        T.SpatialPadd(keys=["image"], spatial_size=_TARGET_SPATIAL, mode="constant"),
        T.CenterSpatialCropd(keys=["image"], roi_size=_TARGET_SPATIAL),
        T.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0,
            channel_wise=True, clip=True,
        ),
    ]

    train_tail = [
        T.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        T.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1, channel_wise=True),
        T.RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.97, 1.03)),
        T.ToTensord(keys=["image"]),
    ]
    val_tail = [T.ToTensord(keys=["image"])]

    return T.Compose(common + train_tail), T.Compose(common + val_tail)


# ── Metrics ────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUROC, F1, balanced accuracy from raw scores + hard labels."""
    from sklearn.metrics import (
        roc_auc_score, f1_score, balanced_accuracy_score,
        precision_recall_fscore_support,
    )
    y_pred = (y_score >= 0.5).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
    return {
        "auroc":            auroc,
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n":                int(len(y_true)),
        "n_pos":            n_pos,
        "n_neg":            n_neg,
    }


# ── Training loop ─────────────────────────────────────────────────────

def _make_model(device: torch.device, dropout_prob: float = 0.3) -> torch.nn.Module:
    return nets.DenseNet121(
        spatial_dims=3, in_channels=4, out_channels=2,
        dropout_prob=dropout_prob,
    ).to(device)


def _class_weights(items: list[dict], task: str, device: torch.device) -> torch.Tensor:
    labels = np.array([int(it[task]) for it in items])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = n_pos + n_neg
    # Inverse-frequency weighting normalised so the two weights average to 1.
    w_neg = total / (2 * max(n_neg, 1))
    w_pos = total / (2 * max(n_pos, 1))
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


def evaluate(model, loader, task, device) -> dict:
    model.eval()
    y_true_all: list[int] = []
    y_score_all: list[float] = []
    total_loss = 0.0
    n_batches = 0
    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch[task].to(device).long()
            logits = model(images)
            loss = ce(logits, labels)
            probs = F.softmax(logits, dim=1)[:, 1]
            y_true_all.extend(labels.cpu().numpy().tolist())
            y_score_all.extend(probs.cpu().numpy().tolist())
            total_loss += float(loss.item())
            n_batches += 1
    metrics = _compute_metrics(np.array(y_true_all), np.array(y_score_all))
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=["idh", "mgmt"], required=True)
    ap.add_argument("--real_datalist", type=Path, required=True)
    ap.add_argument("--synth_root",    type=Path, default=None,
                    help="Release root containing shard_XXXX/sample_YYYYYYY/.")
    ap.add_argument("--synth_manifest", type=Path, default=None,
                    help="Release manifest CSV. Defaults to <synth_root>/manifest_release.csv.")
    ap.add_argument("--n_synthetic", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_epochs", type=int, default=200,
                    help="Upper bound on epochs. Effective epoch count is "
                         "min(n_epochs, ceil(total_grad_updates / batches_per_epoch)).")
    ap.add_argument("--total_grad_updates", type=int, default=None,
                    help="Optional hard cap on total optimizer steps across the "
                         "run. When set, keeps compute budget roughly constant "
                         "across n_synthetic conditions (larger datasets get "
                         "proportionally fewer epochs).")
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stop after this many validation cycles with no "
                         "AUROC improvement. Default: disabled (train to n_epochs).")
    ap.add_argument("--resume", action="store_true",
                    help="If out_dir already contains best_model.pth + resume_state.pt, "
                         "load them and continue training from the recorded epoch.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore an existing metrics.json in out_dir and retrain "
                         "from scratch (or from --resume state if present).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--val_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--val_interval", type=int, default=5)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="MONAI PersistentDataset cache. Defaults to <out_dir>/cache.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ── Setup ──
    set_determinism(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip-if-done: if a metrics.json already exists in out_dir and looks
    # complete, exit early. Prevents redundant compute when the whole grid
    # is requeued after some jobs already finished. Pass --force to override.
    done_marker = out_dir / "metrics.json"
    if done_marker.exists() and not args.force:
        try:
            with done_marker.open() as fh:
                prev = json.load(fh)
            if prev.get("best_epoch", -1) > 0 and "final_metrics" in prev:
                print(f"[skip-if-done] {done_marker} already exists with "
                      f"best_auroc={prev.get('best_auroc'):.4f} at epoch "
                      f"{prev.get('best_epoch')}. Pass --force to retrain.",
                      file=sys.stderr)
                return
        except Exception as e:
            print(f"[warn] existing metrics.json unreadable ({e}); retraining.",
                  file=sys.stderr)

    cache_dir = args.cache_dir or (out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer_train = SummaryWriter(log_dir / "train")
    writer_val   = SummaryWriter(log_dir / "val")

    # ── Data ──
    real_train, real_val = load_real_split(args.real_datalist, args.task)
    # (real items are already normalised with source="real" by load_real_split)
    print(f"Real: {len(real_train)} train / {len(real_val)} val (task={args.task})",
          file=sys.stderr, flush=True)

    synth_items: list[dict] = []
    if args.n_synthetic > 0:
        if args.synth_root is None:
            raise ValueError("--synth_root required when --n_synthetic > 0")
        synth_manifest = args.synth_manifest or (args.synth_root / "manifest_release.csv")
        synth_items = load_synth_items(
            synth_manifest, args.synth_root, args.task, args.n_synthetic,
        )
        print(f"Synthetic: {len(synth_items)} of {args.n_synthetic} requested",
              file=sys.stderr, flush=True)
        if len(synth_items) < args.n_synthetic:
            print(f"[warn] fewer synthetic samples available with a known "
                  f"{args.task} label than requested; using all {len(synth_items)}",
                  file=sys.stderr, flush=True)

    train_items = real_train + synth_items
    val_items   = real_val

    # Report class balance
    train_labels = Counter(int(it[args.task]) for it in train_items)
    val_labels   = Counter(int(it[args.task]) for it in val_items)
    print(f"Train class counts: {dict(train_labels)}", file=sys.stderr)
    print(f"Val   class counts: {dict(val_labels)}",   file=sys.stderr)

    # ── Transforms + loaders ──
    train_tf, val_tf = _build_transforms()
    train_ds = PersistentDataset(data=train_items, transform=train_tf,
                                  cache_dir=cache_dir / "train")
    val_ds   = PersistentDataset(data=val_items,   transform=val_tf,
                                  cache_dir=cache_dir / "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True, drop_last=False)

    # ── Model ──
    model = _make_model(device)
    weights = _class_weights(train_items, args.task, device)
    print(f"Class weights: {weights.tolist()}", file=sys.stderr)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Effective epoch budget ──
    batches_per_epoch = max(len(train_loader), 1)
    effective_n_epochs = args.n_epochs
    if args.total_grad_updates is not None:
        cap_epochs = max(1, math.ceil(args.total_grad_updates / batches_per_epoch))
        effective_n_epochs = min(args.n_epochs, cap_epochs)
        print(f"Total-gradient-updates cap: {args.total_grad_updates} steps -> "
              f"{cap_epochs} epochs at {batches_per_epoch} batches/epoch "
              f"(effective n_epochs = {effective_n_epochs})", file=sys.stderr)

    # ── Resume ──
    history: list[dict] = []
    best_auroc = -1.0
    best_epoch = -1
    start_epoch = 0
    epochs_without_improvement = 0
    resume_state_path = out_dir / "resume_state.pt"
    if args.resume and resume_state_path.exists() and (out_dir / "best_model.pth").exists():
        rs = torch.load(str(resume_state_path), map_location="cpu")
        model.load_state_dict(torch.load(str(out_dir / "best_model.pth"),
                                          map_location="cpu"))
        model.to(device)
        optimizer.load_state_dict(rs["optimizer"])
        start_epoch     = int(rs["next_epoch"])
        best_auroc      = float(rs["best_auroc"])
        best_epoch      = int(rs["best_epoch"])
        history         = list(rs.get("history", []))
        epochs_without_improvement = int(rs.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start_epoch}, best_auroc={best_auroc:.4f} "
              f"@ epoch {best_epoch}", file=sys.stderr)

    # ── Train ──
    for epoch in range(start_epoch, effective_n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch[args.task].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        writer_train.add_scalar("loss", train_loss, epoch)
        print(f"[{epoch+1:03d}/{effective_n_epochs}] train_loss={train_loss:.4f}",
              file=sys.stderr, flush=True)

        # Release cached memory at end of epoch. Prevents accumulation of
        # transient tensor allocations from the DataLoader workers +
        # pinned-memory pool + CUDA caching allocator, which can otherwise
        # OOM-kill long-running jobs even when steady-state usage is fine.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Val ──
        do_val = (epoch + 1) % args.val_interval == 0 or (epoch + 1) == effective_n_epochs
        if do_val:
            metrics = evaluate(model, val_loader, args.task, device)
            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = train_loss
            history.append(metrics)
            writer_val.add_scalar("loss",   metrics["loss"],  epoch)
            writer_val.add_scalar("auroc",  metrics["auroc"], epoch)
            writer_val.add_scalar("f1",     metrics["f1"],    epoch)
            writer_val.add_scalar("balanced_accuracy", metrics["balanced_accuracy"], epoch)
            print(f"    val_loss={metrics['loss']:.4f} "
                  f"AUROC={metrics['auroc']:.4f} "
                  f"F1={metrics['f1']:.4f} "
                  f"balAcc={metrics['balanced_accuracy']:.4f}",
                  file=sys.stderr, flush=True)

            improved = False
            if not math.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), out_dir / "best_model.pth")
                improved = True

            # Persist resume state (optimizer + counters) at every val cycle.
            torch.save({
                "next_epoch": epoch + 1,
                "best_auroc": best_auroc,
                "best_epoch": best_epoch,
                "history":    history,
                "optimizer":  optimizer.state_dict(),
                "epochs_without_improvement": epochs_without_improvement,
            }, out_dir / "resume_state.pt")

            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience is not None and epochs_without_improvement >= args.patience:
                    print(f"Early stop: {epochs_without_improvement} val cycles "
                          f"without AUROC improvement (patience={args.patience})",
                          file=sys.stderr, flush=True)
                    break

    # ── Save results ──
    torch.save(model.state_dict(), out_dir / "final_model.pth")

    final_metrics = evaluate(model, val_loader, args.task, device)
    summary = {
        "task": args.task,
        "n_synthetic": args.n_synthetic,
        "seed": args.seed,
        "n_train_real": len(real_train),
        "n_train_synth": len(synth_items),
        "n_val": len(real_val),
        "class_weights": weights.tolist(),
        "best_epoch": best_epoch,
        "best_auroc": best_auroc,
        "final_metrics": final_metrics,
        "history": history,
    }
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_dir / 'metrics.json'}", file=sys.stderr)
    print(f"Best AUROC = {best_auroc:.4f} at epoch {best_epoch}", file=sys.stderr)


if __name__ == "__main__":
    main()
