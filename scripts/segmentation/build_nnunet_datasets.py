"""Build nnU-Net v2 raw datasets that mix real and synthetic Text2Glioma training data.

Produces one Dataset5XX_T2G_* directory per n_synth value. Each dataset contains:
    imagesTr/{caseID}_{0000..0003}.nii.gz   T1, T1CE, T2, FLAIR
    labelsTr/{caseID}.nii.gz                4-class BraTS mask
    dataset.json                             nnU-Net v2 dataset descriptor

Real subjects come from datalist_N1510.json (training split). Synthetic subjects come
from data/synth_release_10k/manifest.csv (first n_synth rows after shuffling with
--seed). Existing 4-channel stacked NIfTIs are split into per-modality volumes as
nnU-Net v2 requires one file per channel.

Usage:
    python scripts/segmentation/build_nnunet_datasets.py \
        --nnUNet-raw $nnUNet_raw \
        --datalist datalist_N1510.json \
        --synth-manifest data/synth_release_10k/manifest.csv \
        --n-synth 0 500 1000 5000 10000 \
        --seed 42

On Gadi the real NIfTIs live under /g/data/hl36/mhf/monai/Task03_BrainTumourDx/;
override the resolved paths with --real-root if the datalist points elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITIES = ("T1", "T1CE", "T2", "FLAIR")
DATASET_ID_BASE = 510  # Dataset510_T2GRealOnly, 511, 512, 513, 514

@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path   # 4-channel stacked NIfTI
    label_path: Path


def _load_datalist(datalist_json: Path, real_root: Path | None) -> list[Case]:
    d = json.loads(datalist_json.read_text())
    cases: list[Case] = []
    for entry in d["training"]:
        img = Path(entry["image"])
        lbl = Path(entry["label"])
        if real_root is not None:
            img = real_root / img.name
            lbl = (real_root.parent / "labelsTr" / lbl.name)
        cases.append(Case(case_id=entry["subject_id"], image_path=img, label_path=lbl))
    return cases


def _load_synth_manifest(manifest_csv: Path) -> list[Case]:
    root = manifest_csv.parent
    out: list[Case] = []
    with manifest_csv.open() as f:
        for row in csv.DictReader(f):
            # manifest.csv is expected to have columns pointing at
            # shard_XXXX/sample_YYYYYYY/{image,mask}.nii.gz relative to root.
            sid = row.get("sample_id") or row.get("case_id") or row["image"]
            img = root / row["image"] if "image" in row else Path(row["image_path"])
            lbl = root / row["mask"]  if "mask"  in row else Path(row["mask_path"])
            out.append(Case(case_id=sid, image_path=img, label_path=lbl))
    return out


def _split_channels(src: Path, out_prefix: Path) -> None:
    """Split a 4-channel NIfTI into four single-channel files with nnU-Net suffixes."""
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    # allow (H,W,D,C) or (C,H,W,D); nnU-Net expects (H,W,D) per channel file
    if data.ndim == 4 and data.shape[0] == 4:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 4 or data.shape[-1] != 4:
        raise ValueError(f"Expected 4-channel NIfTI at {src}; got shape {data.shape}")
    for c in range(4):
        out = out_prefix.with_name(out_prefix.name + f"_{c:04d}.nii.gz")
        nib.save(nib.Nifti1Image(data[..., c].astype(np.float32), img.affine, img.header), str(out))


def _write_label(src: Path, dst: Path) -> None:
    img = nib.load(str(src))
    data = np.asarray(img.dataobj).astype(np.uint8)
    # Some pipelines store label 4 for enhancing (BraTS legacy); normalise to 3.
    data[data == 4] = 3
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(dst))


def _write_dataset_json(dataset_dir: Path, n_train: int, description: str) -> None:
    payload = {
        "name": dataset_dir.name,
        "description": description,
        "reference": "Text2Glioma",
        "licence": "CC-BY-4.0",
        "channel_names": {str(i): m for i, m in enumerate(MODALITIES)},
        "labels": {
            "background": 0,
            "necrotic_non_enhancing": 1,
            "peritumoral_edema": 2,
            "enhancing_tumor": 3,
        },
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(payload, indent=2))


def _emit_case(dataset_dir: Path, case: Case, case_id: str) -> None:
    _split_channels(case.image_path, dataset_dir / "imagesTr" / case_id)
    _write_label(case.label_path, dataset_dir / "labelsTr" / f"{case_id}.nii.gz")


def build_one(dataset_root: Path, name: str, real: list[Case], synth: list[Case]) -> None:
    dataset_dir = dataset_root / name
    for sub in ("imagesTr", "labelsTr"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(real):
        _emit_case(dataset_dir, case, f"REAL_{i:05d}")
    for i, case in enumerate(synth):
        _emit_case(dataset_dir, case, f"SYNTH_{i:06d}")
    n = len(real) + len(synth)
    _write_dataset_json(dataset_dir, n, f"Text2Glioma nnU-Net segmentation. Real={len(real)}, Synth={len(synth)}.")
    print(f"[wrote] {dataset_dir}  (real={len(real)}, synth={len(synth)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnUNet-raw", required=True, type=Path,
                    help="Destination root (typically $nnUNet_raw).")
    ap.add_argument("--datalist", required=True, type=Path)
    ap.add_argument("--synth-manifest", required=True, type=Path)
    ap.add_argument("--n-synth", nargs="+", type=int, default=[0, 500, 1000, 5000, 10000])
    ap.add_argument("--real-root", type=Path, default=None,
                    help="Override /g/data/... image root in datalist entries.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    real = _load_datalist(args.datalist, args.real_root)
    synth = _load_synth_manifest(args.synth_manifest)
    rng = random.Random(args.seed)
    rng.shuffle(synth)

    for i, n in enumerate(sorted(args.n_synth)):
        did = DATASET_ID_BASE + i
        tag = "RealOnly" if n == 0 else f"RealSynth{n}"
        name = f"Dataset{did}_T2G{tag}"
        build_one(args.nnUNet_raw, name, real, synth[:n])


if __name__ == "__main__":
    main()
