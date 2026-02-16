import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from importlib.resources import files as _pkg_files
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split

from .utils import compose_radiology_prompts

# Resolve the atlas_masks directory shipped inside the installed package.
_ATLAS_PKG_DIR = Path(_pkg_files("text2glioma.preprocessing").joinpath("atlas_masks"))
_ATLAS_SPACES = {"sri24": str(_ATLAS_PKG_DIR / "sri24"), "mni152": str(_ATLAS_PKG_DIR / "mni152")}
_ATLAS_DIR_DEFAULT = _ATLAS_SPACES["sri24"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create datalist and prompts. Assumes flat directory "
        "structure with images in input_dir and labels in label_dir (or "
        "input_dir if label_dir is not given). For Decathlon-style datasets "
        "with separate Tr/Ts folders, use --test_input_dir and --test_label_dir."
    )
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing training image files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output datalist.json.")
    parser.add_argument("--label_dir", type=str, required=False, default=None, help="Directory containing label files (optional, defaults to input_dir).")
    parser.add_argument("--test_input_dir", type=str, required=False, default=None, help="Directory containing test image files (e.g. imagesTs). If given, these are added to the 'testing' split. Also forces train_split=1.0 unless --val_input_dir is given.")
    parser.add_argument("--test_label_dir", type=str, required=False, default=None, help="Directory containing test label files (e.g. labelsTs). Defaults to test_input_dir.")
    parser.add_argument("--val_input_dir", type=str, required=False, default=None, help="Directory containing validation image files. If given, validation subjects come from this directory instead of splitting from input_dir.")
    parser.add_argument("--val_label_dir", type=str, required=False, default=None, help="Directory containing validation label files. Defaults to val_input_dir.")
    parser.add_argument("--subject_prefix", type=str, required=False, default="subj", help="Prefix for subject IDs (default: 'subj').")
    parser.add_argument("--start_index", type=int, required=False, default=0, help="Starting index for subject IDs (default: 0).")
    parser.add_argument("--file_extension", type=str, required=False, default=".nii.gz", help="File extension to look for (default: .nii.gz).")
    parser.add_argument("--atlas_space", type=str, required=False, default="sri24", choices=["sri24", "mni152"], help="Atlas space to use. BraTS data uses 'sri24'. MNI152-registered data uses 'mni152'. (default: sri24)")
    parser.add_argument("--atlas_dir", type=str, required=False, default=None, help="Explicit directory containing atlas masks. Overrides --atlas_space. Defaults to the copy installed with the package.")
    parser.add_argument("--enhancing_label", type=int, required=False, default=3, help="Label value for enhancing tumor (default: 3).")
    parser.add_argument("--nonenhancing_label", type=int, required=False, default=1, help="Label value for non-enhancing tumor (default: 1).")
    parser.add_argument("--oedema_label", type=int, required=False, default=2, help="Label value for oedema (default: 2).")
    parser.add_argument("--z_dim", type=int, required=False, default=-1, help="Dimension of the z-axis (default: -1).")
    parser.add_argument("--cf", type=int, required=False, default=1, help="Connectivity factor (default: 1).")
    parser.add_argument("--t_ependymal", type=int, required=False, default=5000, help="Threshold for ependymal (default: 5000).")
    parser.add_argument("--t_wm", type=int, required=False, default=100, help="Threshold for white matter (default: 100).")
    parser.add_argument("--resolution", type=int, required=False, default=1, help="Image resolution (default: 1).")
    parser.add_argument("--midline_thresh", type=int, required=False, default=5, help="Midline threshold (default: 5).")
    parser.add_argument("--enh_quality_thresh", type=int, required=False, default=15, help="Enhancing quality threshold (default: 15).")
    parser.add_argument("--cyst_thresh", type=int, required=False, default=50, help="Cyst threshold (default: 50).")
    parser.add_argument("--cortical_thresh", type=int, required=False, default=1000, help="Cortical threshold (default: 1000).")
    parser.add_argument("--focus_thresh", type=int, required=False, default=30000, help="Focus threshold (default: 30000).")
    parser.add_argument("--num_components_bin_thresh", type=int, required=False, default=10, help="Number of components binary threshold (default: 10).")
    parser.add_argument("--num_components_cet_thresh", type=int, required=False, default=15, help="Number of components CET threshold (default: 15).")
    parser.add_argument("--train_split", type=float, required=False, default=0.8, help="Proportion of data to use for training (default: 0.8). Use 1.0 to put all subjects in training (no validation split).")
    parser.add_argument("--shuffle_prompt_order", action='store_true', help="Whether to shuffle the order of prompts (default: False).")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--verbose", action='store_true', help="Whether to print outputs sometimes (default: False).")
    parser.add_argument("-j", "--num_workers", type=int, required=False, default=0,
                        help="Number of parallel workers for prompt generation. "
                             "0 = automatic (all CPUs), 1 = serial (default: 0).")

    return parser.parse_args()


def _discover_subjects(directory, file_extension):
    """Return sorted list of unique stem names (minus extension) in directory."""
    ext = file_extension
    stems = set()
    for f in directory.iterdir():
        if not f.is_file() or not f.name.endswith(ext):
            continue
        # Skip macOS resource fork files (._*)
        if f.name.startswith("._"):
            continue
        # Handle double extensions like .nii.gz
        name = f.name
        if name.endswith(".nii.gz"):
            stem = name[: -len(".nii.gz")]
        else:
            stem = f.stem
        stems.add(stem)
    return sorted(stems)


# Top-level function so it can be pickled by multiprocessing.
def _process_one(
    subject: str,
    subject_id: str,
    image_path: str,
    label_path: str,
    atlas_dir: str,
    kwargs: dict,
) -> Optional[dict]:
    """Process a single subject and return its entry dict (or *None* on skip)."""
    if not Path(image_path).exists():
        print(f"Warning: Image file {image_path} does not exist. Skipping {subject}.")
        return None
    if not Path(label_path).exists():
        print(f"Warning: Label file {label_path} does not exist. Skipping {subject}.")
        return None

    prompt = compose_radiology_prompts(
        image_path=image_path,
        label_path=label_path,
        atlas_dir=atlas_dir,
        **kwargs,
    )
    return {
        "image": image_path,
        "label": label_path,
        "subject_id": subject_id,
        "impression": prompt["short"],
        "findings": prompt["long"],
    }


def _process_subjects(subjects, input_dir, label_dir, atlas_dir, args, rng, start_idx):
    """Run prompt composer on a list of subjects, returning (entries, next_idx).

    When ``args.num_workers != 1`` the subjects are processed in parallel
    using a :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    # Pre-build per-subject arguments so indices are deterministic.
    jobs: list[tuple[str, str, str, str]] = []  # (subject, subject_id, image, label)
    for i, subject in enumerate(subjects):
        sid = f"{args.subject_prefix}{start_idx + i}"
        img = str(input_dir / f"{subject}{args.file_extension}")
        lbl = str(label_dir / f"{subject}{args.file_extension}")
        jobs.append((subject, sid, img, lbl))

    # Shared keyword arguments forwarded to compose_radiology_prompts.
    prompt_kwargs = dict(
        enhancing_label=args.enhancing_label,
        nonenhancing_label=args.nonenhancing_label,
        edema_label=args.oedema_label,
        z_dim=args.z_dim,
        cf=args.cf,
        t_ependymal=args.t_ependymal,
        t_wm=args.t_wm,
        resolution=args.resolution,
        midline_thresh=args.midline_thresh,
        enh_quality_thresh=args.enh_quality_thresh,
        cyst_thresh=args.cyst_thresh,
        cortical_thresh=args.cortical_thresh,
        focus_thresh=args.focus_thresh,
        num_components_bin_thresh=args.num_components_bin_thresh,
        num_components_cet_thresh=args.num_components_cet_thresh,
        shuffle_order=args.shuffle_prompt_order,
        seed=args.seed,
        verbose=args.verbose,
    )

    n_workers = args.num_workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1

    entries: list[Optional[dict]] = [None] * len(jobs)

    if n_workers == 1:
        # ── serial path (simpler, easier to debug) ──────────────────────
        for pos, (subj, sid, img, lbl) in enumerate(tqdm(jobs, desc="Processing subjects")):
            entries[pos] = _process_one(subj, sid, img, lbl, str(atlas_dir), prompt_kwargs)
    else:
        # ── parallel path ────────────────────────────────────────────────
        atlas_str = str(atlas_dir)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_pos = {
                pool.submit(_process_one, subj, sid, img, lbl, atlas_str, prompt_kwargs): pos
                for pos, (subj, sid, img, lbl) in enumerate(jobs)
            }
            with tqdm(total=len(jobs), desc=f"Processing subjects ({n_workers} workers)") as pbar:
                for future in as_completed(future_to_pos):
                    pos = future_to_pos[future]
                    try:
                        entries[pos] = future.result()
                    except Exception as exc:
                        subj = jobs[pos][0]
                        print(f"Warning: {subj} failed: {exc}")
                    pbar.update(1)

    # Filter out skipped / failed subjects, preserving order.
    entries = [e for e in entries if e is not None]
    next_idx = start_idx + len(entries)
    return entries, next_idx


def main(args=None):
    if args is None:
        args = parse_args()
    rng = check_random_state(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    label_dir = Path(args.label_dir) if args.label_dir else input_dir
    if args.atlas_dir:
        atlas_dir = Path(args.atlas_dir)
    else:
        atlas_dir = Path(_ATLAS_SPACES.get(args.atlas_space, _ATLAS_DIR_DEFAULT))
    print(f"Using atlas space: {args.atlas_space}  →  {atlas_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover training subjects ----
    subjects = _discover_subjects(input_dir, args.file_extension)
    if not subjects:
        raise FileNotFoundError(
            f"No files with extension '{args.file_extension}' found in {input_dir}"
        )

    # ---- Resolve explicit val / test directories ----
    val_input_dir = Path(args.val_input_dir) if args.val_input_dir else None
    val_label_dir = Path(args.val_label_dir) if args.val_label_dir else val_input_dir
    test_input_dir = Path(args.test_input_dir) if args.test_input_dir else None
    test_label_dir = Path(args.test_label_dir) if args.test_label_dir else test_input_dir

    # When separate directories define the splits, override train_split so
    # all subjects discovered in input_dir go to training.
    if test_input_dir or val_input_dir:
        if args.train_split < 1.0:
            print(
                f"Note: --test_input_dir / --val_input_dir provided; "
                f"overriding train_split from {args.train_split} to 1.0 "
                f"(all input_dir subjects go to training)."
            )
            args.train_split = 1.0

    # ---- Train / validation split ----
    if args.train_split >= 1.0:
        subjects_tr = subjects
        subjects_val = []
    else:
        subjects_tr, subjects_val = train_test_split(
            subjects, train_size=args.train_split, random_state=rng
        )

    print(f"Found {len(subjects)} subjects in input_dir: {len(subjects_tr)} training, {len(subjects_val)} validation")

    # ---- Process training subjects ----
    train_entries, next_idx = _process_subjects(
        subjects_tr, input_dir, label_dir, atlas_dir, args, rng, args.start_index
    )

    # ---- Process validation subjects (explicit dir or split) ----
    if val_input_dir:
        val_subjects = _discover_subjects(val_input_dir, args.file_extension)
        print(f"Found {len(val_subjects)} validation subjects in val_input_dir")
        val_entries, next_idx = _process_subjects(
            val_subjects, val_input_dir, val_label_dir, atlas_dir, args, rng, next_idx
        )
    else:
        val_entries, next_idx = _process_subjects(
            subjects_val, input_dir, label_dir, atlas_dir, args, rng, next_idx
        )

    # ---- Process test subjects (optional, for Decathlon-style datasets) ----
    test_entries = []
    if test_input_dir:
        test_subjects = _discover_subjects(test_input_dir, args.file_extension)
        print(f"Found {len(test_subjects)} test subjects in test_input_dir")
        test_entries, next_idx = _process_subjects(
            test_subjects, test_input_dir, test_label_dir, atlas_dir, args, rng, next_idx
        )

    datalist = {
        "training": train_entries,
        "validation": val_entries,
        "testing": test_entries,
        "n_subjs": len(train_entries) + len(val_entries) + len(test_entries),
        "n_training": len(train_entries),
        "n_validation": len(val_entries),
        "n_testing": len(test_entries),
    }

    # Save datalist to JSON
    out_path = output_dir / "datalist.json"
    with open(out_path, "w") as f:
        json.dump(datalist, f, indent=4)
    print(f"Datalist saved to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)