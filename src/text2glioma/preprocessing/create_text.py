import argparse
from importlib.resources import files as _pkg_files
from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm
import numpy as np

from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split

from .utils import compose_radiology_prompts

# Resolve the atlas_masks directory shipped inside the installed package.
_ATLAS_DIR_DEFAULT = str(Path(_pkg_files("text2glioma.preprocessing").joinpath("atlas_masks")))

def parse_args():
    parser = argparse.ArgumentParser(description="Create datalist and prompts.", help="Create datalist and prompts. Assumes flat directory structure with images in input_dir and labels are in label_dir or input_dir if label_dir is not given.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing image files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output prompts.")
    parser.add_argument("--label_dir", type=str, required=False, default=None, help="Directory containing label files (optional, defaults to input_dir if not given).")
    parser.add_argument("--subject_prefix", type=str, required=False, default="subj", help="Prefix for subject IDs (default: 'subj'). The subject ID will be formed as {subject_prefix}{index}.")
    parser.add_argument("--start_index", type=int, required=False, default=0, help="Starting index for subject IDs (default: 0).")
    parser.add_argument("--file_extension", type=str, required=False, default=".nii.gz", help="File extension to look for (default is .nii.gz).")
    parser.add_argument("--atlas_dir", type=str, required=False, default=_ATLAS_DIR_DEFAULT, help="Directory containing atlas masks. Defaults to the copy installed with the package.")
    parser.add_argument("--enhancing_label", type=int, required=False, default=3, help="Label value for enhancing tumor (default: 3).")
    parser.add_argument("--nonenhancing_label", type=int, required=False, default=1, help="Label value for non-enhancing tumor (default: 1).")
    parser.add_argument("--oedema_label", type=int, required=False, default=2, help="Label value for edema (default: 2).")
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
    parser.add_argument("--train_split", type=float, required=False, default=0.8, help="Proportion of data to use for training (default: 0.8).")
    parser.add_argument("--shuffle_prompt_order", action='store_true', help="Whether to shuffle the order of prompts (default: False).")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--verbose", action='store_true', help="Whether to print outputs sometimes (default: False).")

    return parser.parse_args()

def main(args):
    rng = check_random_state(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    label_dir = Path(args.label_dir) if args.label_dir else input_dir
    atlas_dir = Path(args.atlas_dir) if args.atlas_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(set([f.stem for f in input_dir.glob(f"*{args.subject_prefix}") if f.is_file()]))
    subjects_tr, subjects_val = train_test_split(subjects, train_size=args.train_split, random_state=rng)

    datalist = {
        "training": [],
        "validation": [],
        "testing": [],
        "n_subjs": len(subjects),
        "n_training": len(subjects_tr),
        "n_validation": len(subjects_val),
    }
    ### ----- Prompt composer ----- ###
    for i, subject in enumerate(tqdm(subjects, desc="Processing subjects")):
        subject_id = f"{args.subject_prefix}{i + args.start_index}"
        image_path = input_dir / f"{subject}{args.file_extension}"
        label_path = label_dir / f"{subject}{args.file_extension}"
        
        if not image_path.exists():
            print(f"Warning: Image file {image_path} does not exist. Skipping subject {subject}.")
            continue
        if not label_path.exists():
            print(f"Warning: Label file {label_path} does not exist. Skipping subject {subject}.")
            continue
        
        prompt = compose_radiology_prompts(
            image_path=image_path,
            label_path=label_path,
            atlas_dir=atlas_dir,
            enhancing_label=args.enhancing_label,
            nonenhancing_label=args.nonenhancing_label,
            edema_label=args.edema_label,
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
        
        entry = {
            "image": str(image_path),
            "label": str(label_path),
            "subject_id": subject_id,
            "impression": prompt["short"],
            "findings": prompt["long"],
        }
        
        if subject in subjects_tr:
            datalist["training"].append(entry)
        else:
            datalist["validation"].append(entry)
    
    # Save datalist to JSON
    with open(output_dir / "datalist.json", "w") as f:
        json.dump(datalist, f, indent=4)
    print(f"Datalist saved to {output_dir / 'datalist.json'}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    