import sys, re, shutil, subprocess
from pathlib import Path
from datetime import date
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser

import pydicom

# helpers
DATE_RE = re.compile(r'(?P<mm>\d{2})-(?P<dd>\d{2})-(?P<yyyy>\d{4})')
NEGATIVE_KEYWORDS = ("FLAIR", "DWI", "DIFF", "ADC", "GRE", "SWI", "T1", "LOC", "LOCALIZER", "MRA", "ANGIO")
POS_T2_TSE_HINTS = ("T2", "TSE", "FSE", "TURBO", "FAST")  # robust synonyms
DATASETS = ["ivygap", "ucsf_pdgm", "upenn_gbm", "brats_gli_2024"]

def parse_args():
    parser = ArgumentParser(description="Convert IVYGAP T2 TSE DICOMs to NIfTI format")
    parser.add_argument("source_dir", type=str, help="Source directory containing subject folders")
    parser.add_argument("output_dir", type=str, help="Output directory for converted NIfTI files")
    parser.add_argument("--dataset", required=True, default="ivygap", metavar=DATASETS)

    return parser.parse_args()

def _parse_date_from_name(name: str) -> Optional[date]:
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return date(int(m["yyyy"]), int(m['mm']), int(m['dd']))
    except ValueError:
        return None
    
def _is_dcm(path: Path) -> bool:
    # Heuristic: extension .dcm OR any file that isn't a dir; we will guard by try/except pydicom
    return path.is_file() and (path.suffix.lower() == ".dcm" or path.suffix == "")

def _find_first(series_dir: Path) -> Optional[Path]:
    for p in sorted(series_dir.iterdir()):
        if p.is_file() and _is_dcm(p):
            return p
        
    return None

def _load_dicom_tags(fp: Path):
    try:
        ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        return ds
    except Exception:
        return None
    
def _series_matches_t2_tse(ds) -> bool:
    """
    Accept if:
      - SeriesDescription has 'T2' and ('TSE' or 'FSE' or synonym),
        OR SequenceName contains 'tse'/'fse'
      - AND it does not contain common negatives (FLAIR/DWI/ADC/GRE/T1/etc.)
    """
    def get(tag, default=""):
        # tag can be keyword string for pydicom __getattr__
        try:
            return str(getattr(ds, tag))
        except Exception:
            return default

    sd = (get("SeriesDescription") or "").upper()
    sn = (get("SequenceName") or "").upper()
    prot = (get("ProtocolName") or "").upper()

    haystacks = [sd, sn, prot]

    # Must contain T2 somewhere in Series/Protocol (or SequenceName includes tse/fse and Protocol has T2)
    has_t2 = any("T2" in h for h in haystacks)
    has_tse_syn = any(any(k in h for k in ("TSE","FSE","TURBO","FAST")) for h in haystacks)
    seqname_tse = ("TSE" in sn) or ("FSE" in sn)

    # remove negatives
    has_negative = any(any(neg in h for neg in NEGATIVE_KEYWORDS) for h in haystacks)

    return ((has_t2 and (has_tse_syn or seqname_tse)) and not has_negative)

def _collect_series(root_exam_dir: Path) -> List[Path]:
    return [d for d in root_exam_dir.iterdir() if d.is_dir()]

def _pick_subject_dirs(root: Path) -> List[Path]:
    # subjects are immediate subdirs that are not 'work'
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.lower() != "work":
            out.append(d)
    return out

def _dcm2niix_convert(series_dir: Path, out_dir: Path, out_stem: str) -> Optional[Path]:
    cmd = [
        "dcm2niix",
        "-b", "y",     # write BIDS sidecar
        "-z", "y",     # gzip NIfTI
        "-f", out_stem,
        "-o", str(out_dir),
        str(series_dir)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # find the created nifti
        niftis = list(out_dir.glob(out_stem + "*.nii.gz"))
        return niftis[0] if niftis else None
    except subprocess.CalledProcessError as e:
        print(f"[dcm2niix] failed for {series_dir}: {e}")
        return None
    
def _have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _convert_series(series_dir: Path, out_dir: Path, out_stem: str) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _have_cmd("dcm2niix"):
        return _dcm2niix_convert(series_dir, out_dir, out_stem)
    else:
        raise RuntimeError("need to install dcm2niix!")

def _find_earliest_exam(exam_parent: Path) -> Optional[Tuple[Path, date]]:
    # exam_parent is subject folder (e.g., W1)
    candidates: List[Tuple[Path, date]] = []
    for d in exam_parent.iterdir():
        if d.is_dir():
            dt = _parse_date_from_name(d.name)
            if dt:
                candidates.append((d, dt))
    if not candidates:
        return None
    # earliest by date
    return sorted(candidates, key=lambda x: x[1])[0]

def ivygap_converter(source_dir, output_dir, verbose=False):
    """
    Convert ivygap source T2 TSE dicoms to niftis with the required format for
    the classification models

    Usage:
        convert_data source_dir --ivygap
    
    SOURCE_DIR should contain subject folders like W1, W2, ... and a bunch of date-named
    exam folders ie.
    SOURCE_DIR/
      W1/
        01-28-1997-NA-OUTSIDE MR - HEAD D94 -...
        10-25-1996-NA-MR BRAIN WITHOUT AND WITH CONTRAST D-1-...
      W2/
        ...

    Outputs are stored:
      OUTPUT_DIR/<subject>/<subject>_T2.nii.gz

    """
    root = Path(source_dir)
    work_dir = Path(output_dir)

    if not root.is_dir():
        print(f"source directory not found: {root}")
        sys.exit(1)
    
    work_dir.mkdir(exist_ok=True, parents=True)

    subjects = _pick_subject_dirs(root)
    if not subjects:
        print("no subject folders found.")
        sys.exit(0)

    subject_to_exam: Dict[str, Tuple[Path, date]] = {}

    for subj_dir in subjects:
        res = _find_earliest_exam(subj_dir)
        if res:
            subject_to_exam[subj_dir.name] = res
        else:
            print(f"No exams found for subject: {subj_dir.name}")
            continue
    
    for subj, (exam_dir, dt) in subject_to_exam.items():
        series_dirs = _collect_series(exam_dir)
        matches: List[Tuple[Path, str]] = []

        for sdir in series_dirs:
            fp = _find_first(sdir)
            if not fp:
                continue
            ds = _load_dicom_tags(fp)
            if ds is None:
                continue
            if _series_matches_t2_tse(ds):
                series_desc = getattr(ds, "SeriesDescription", sdir.name)
                matches.append((sdir, series_desc))

        if not matches:
            print(f"[{subj}] No T2 TSE/FSE series found in {exam_dir.name}")
            continue

        out_dir = work_dir / subj
        for sdir, sdesc in matches:
            safe_desc = re.sub(r'[^A-Za-z0-9_.-]+', '_', sdesc)[:80]
            out_stem = f"{subj}_T2TSE_{dt.strftime('%Y%m%d')}__{safe_desc}"
            print(f"[{subj}] Converting: {sdir.name}  ({sdesc})  -> {out_stem}.nii.gz")
            nii = _convert_series(sdir, out_dir, out_stem)
            if nii:
                print(f"  -> OK: {nii}")
            else:
                print(f"  -> FAILED: {sdir}")

def main():
    args = parse_args()
    
    if "ivygap" in args.dataset:
        ivygap_converter(args.source_dir, args.output_dir)
    #elif: others, not currently functional
    else:
        raise ValueError(f"Unrecognized dataset spec: {args.dataset}")
    
if __name__ == "__main__":
    main()
    