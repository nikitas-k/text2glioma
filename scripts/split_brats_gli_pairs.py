"""Subject-level stratified split of the BraTS-GLI longitudinal pairs.

Input  : datalist_brats_gli_2025_pairs.json   (output of build_brats_gli_pairs.py)
Output : datalist_brats_gli_2025_pairs_split.json

Why subject-level
-----------------
Pairs are not independent: 142/586 subjects contribute >=2 pairs (one subject
contributes 9). Splitting on pairs would leak the same patient's anatomy across
folds and inflate validation/test fidelity.

Stratification key
------------------
Each subject is assigned a single stratum tag derived from its most clinically
informative pair, by priority:

    pre->post  >  response  >  progression  >  stable

So a subject that has any pre->post pair is tagged 'pre_post' regardless of
trajectory; otherwise the most informative trajectory wins.  This ensures the
rarest buckets (pre->post at 3.8%, response at 10.6%) are distributed across
train/val/test rather than dumped into one fold.

Split
-----
80 / 10 / 10 of subjects (configurable). Within each stratum we shuffle with a
fixed seed and round-robin into the three folds, so even strata with only a
handful of subjects still appear in val and test (no fold loses a class).

Oversampling (training fold only)
---------------------------------
The raw BraTS-GLI cohort has severe treatment-direction skew (post->post 83%,
pre->post 4%) and trajectory skew (stable 65%, response 11%). At training time
we can't compose `WeightedRandomSampler` with `DistributedSampler`, so balance
is baked into the datalist itself: rare strata are duplicated up to
`--oversample_cap` times so the most-common stratum is at most
`--oversample_max_ratio` times the rarest after duplication. Applied **only
to the training fold** — val/test are kept at natural frequency so metrics
aren't inflated by duplicates.

Self-audit
----------
Output includes per-fold pair counts, trajectory_counts, treatment_direction_counts,
the per-stratum subject distribution, and pre/post oversampling counts so the
split is reproducible and auditable from the JSON alone.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DEFAULT = ROOT / "datalist_brats_gli_2025_pairs.json"
DST_DEFAULT = ROOT / "datalist_brats_gli_2025_pairs_split.json"

# Priority high -> low. First match wins.
STRATA_PRIORITY = ("pre_post", "response", "progression", "stable")


def _stratum_for_subject(pairs: list[dict]) -> str:
    """Pick the most-informative bucket among this subject's pairs."""
    has_pre_post = any(
        p["treatment_status_a"] == "pre" and p["treatment_status_b"] == "post"
        for p in pairs
    )
    if has_pre_post:
        return "pre_post"
    trajs = {p["trajectory"] for p in pairs}
    for s in ("response", "progression", "stable"):
        if s in trajs:
            return s
    # Defensive: shouldn't reach here if input is well-formed
    return "stable"


def _round_robin(items: list, fractions: tuple[float, float, float]) -> tuple[list, list, list]:
    """Deterministic 3-way split by fractions, ensuring every non-empty input
    contributes at least one item to each fold whenever the count allows.

    For N items and fractions (ft, fv, fte):
      - first ceil(N * ft) -> train
      - next  ceil(N * fv) -> val
      - rest               -> test
    For very small N (<=2), prioritise train and put any leftover in val.
    """
    n = len(items)
    if n == 0:
        return [], [], []
    if n == 1:
        return [items[0]], [], []
    if n == 2:
        return [items[0]], [items[1]], []
    ft, fv, _ = fractions
    n_train = max(1, round(n * ft))
    n_val = max(1, round(n * fv))
    # Guarantee at least 1 in test if N>=3
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def _pair_stratum(pair: dict, mode: str) -> str:
    """Per-pair stratum key for oversampling. Mirrors compute_balanced_weights
    in src/text2glioma/preprocessing/inpainting_dataset.py."""
    ta, tb, traj = pair["treatment_status_a"], pair["treatment_status_b"], pair["trajectory"]
    if mode == "direction":
        return f"{ta}->{tb}"
    if mode == "trajectory":
        return traj
    if mode == "joint":
        return f"{ta}->{tb}/{traj}"
    raise ValueError(f"Unsupported oversample mode {mode!r}")


def _oversample_training(
    train_pairs: list[dict],
    mode: str,
    max_ratio: float,
    cap: int,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Duplicate rare strata in the training fold.

    Algorithm:
      1. Count pairs per stratum.
      2. Target count for every stratum = max(count_max / max_ratio, count_self).
         The largest stratum is unchanged; smaller strata are duplicated up to
         that target.
      3. Per-stratum multiplier is capped at `cap` so a stratum with 5 pairs
         doesn't get duplicated 100x.
      4. Pairs are duplicated via copy.deepcopy (no shared MetaDict refs).

    Returns (oversampled_list, per_stratum_audit).
    """
    counts = collections.Counter(_pair_stratum(p, mode) for p in train_pairs)
    if not counts:
        return list(train_pairs), {}
    max_count = max(counts.values())
    target = max(1.0, max_count / max_ratio)

    by_stratum: dict[str, list[dict]] = collections.defaultdict(list)
    for p in train_pairs:
        by_stratum[_pair_stratum(p, mode)].append(p)

    audit: dict[str, dict[str, int]] = {}
    out: list[dict] = []
    for stratum, members in by_stratum.items():
        n = len(members)
        # Desired multiplier rounded up; clamp to [1, cap]
        raw_mult = max(1.0, target / max(n, 1))
        mult = min(cap, int(round(raw_mult)))
        # Cheap deepcopy so transforms can't accidentally share state.
        dupes = [copy.deepcopy(p) for _ in range(mult) for p in members]
        out.extend(dupes)
        audit[stratum] = {
            "original":   n,
            "multiplier": mult,
            "final":      n * mult,
        }
    return out, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--dst", type=Path, default=DST_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--val_frac", type=float, default=0.10)
    # test_frac is implicit = 1 - train - val
    p.add_argument(
        "--oversample_mode",
        choices=["none", "direction", "trajectory", "joint"],
        default="none",
        help=("Stratum key for training-fold oversampling. 'direction' targets "
              "the treatment_a->treatment_b imbalance (pre->post 4%% -> ~33%%); "
              "'trajectory' targets response/stable/progression; 'joint' uses "
              "the cross-product. 'none' (default) leaves training fold at "
              "natural frequency."),
    )
    p.add_argument(
        "--oversample_max_ratio", type=float, default=3.0,
        help=("After oversampling, max ratio between the largest and smallest "
              "stratum. Lower = more aggressive balancing."),
    )
    p.add_argument(
        "--oversample_cap", type=int, default=20,
        help=("Per-stratum duplication cap (safety brake for tiny strata)."),
    )
    args = p.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        raise SystemExit(f"train+val ({args.train_frac + args.val_frac}) leaves no room for test")
    fractions = (args.train_frac, args.val_frac, test_frac)

    src = json.loads(args.src.read_text())
    pairs = src["pairs"]
    if not pairs:
        raise SystemExit(f"{args.src}: 'pairs' is empty")

    # Group pairs by subject
    by_subject: dict[str, list[dict]] = collections.defaultdict(list)
    for pair in pairs:
        by_subject[pair["subject_id"]].append(pair)

    # Assign each subject to a stratum
    subject_stratum: dict[str, str] = {
        sid: _stratum_for_subject(plist) for sid, plist in by_subject.items()
    }

    # Bucket subjects by stratum, shuffle deterministically, then split per-stratum
    rng = random.Random(args.seed)
    fold_subjects: dict[str, list[str]] = {"training": [], "validation": [], "testing": []}
    stratum_split_audit: dict[str, dict[str, int]] = {}

    for stratum in STRATA_PRIORITY:
        members = [sid for sid, s in subject_stratum.items() if s == stratum]
        rng.shuffle(members)
        tr, va, te = _round_robin(members, fractions)
        fold_subjects["training"].extend(tr)
        fold_subjects["validation"].extend(va)
        fold_subjects["testing"].extend(te)
        stratum_split_audit[stratum] = {
            "total":      len(members),
            "training":   len(tr),
            "validation": len(va),
            "testing":    len(te),
        }

    # Materialise per-fold pair lists
    folds: dict[str, list[dict]] = {"training": [], "validation": [], "testing": []}
    for fold, sids in fold_subjects.items():
        sid_set = set(sids)
        folds[fold] = [pair for pair in pairs if pair["subject_id"] in sid_set]

    # Oversample rare strata in the TRAINING fold only. Val/test stay at
    # natural frequency so metrics aren't inflated by duplicates.
    n_training_pairs_pre_oversample = len(folds["training"])
    oversample_audit: dict[str, dict[str, int]] = {}
    if args.oversample_mode != "none":
        folds["training"], oversample_audit = _oversample_training(
            folds["training"],
            mode=args.oversample_mode,
            max_ratio=args.oversample_max_ratio,
            cap=args.oversample_cap,
        )
        # Shuffle deterministically so duplicates don't all land consecutively
        rng.shuffle(folds["training"])

    # Audit: per-fold trajectory and treatment-direction breakdowns
    fold_audit: dict[str, dict] = {}
    for fold, fpairs in folds.items():
        traj = dict(collections.Counter(p["trajectory"] for p in fpairs))
        direc = dict(collections.Counter(
            f"{p['treatment_status_a']}->{p['treatment_status_b']}" for p in fpairs
        ))
        fold_audit[fold] = {
            "n_subjects": len(fold_subjects[fold]),
            "n_pairs":    len(fpairs),
            "trajectory_counts":          traj,
            "treatment_direction_counts": direc,
        }

    # Cross-fold leakage sanity check
    sets = {k: set(v) for k, v in fold_subjects.items()}
    leak_tv = sets["training"] & sets["validation"]
    leak_tt = sets["training"] & sets["testing"]
    leak_vt = sets["validation"] & sets["testing"]
    if leak_tv or leak_tt or leak_vt:
        raise RuntimeError(
            f"Subject leakage between folds: "
            f"train&val={len(leak_tv)} train&test={len(leak_tt)} val&test={len(leak_vt)}"
        )

    out = {
        "training":   folds["training"],
        "validation": folds["validation"],
        "testing":    folds["testing"],
        "training_subjects":   sorted(fold_subjects["training"]),
        "validation_subjects": sorted(fold_subjects["validation"]),
        "testing_subjects":    sorted(fold_subjects["testing"]),
        "n_training_pairs":    len(folds["training"]),
        "n_validation_pairs":  len(folds["validation"]),
        "n_testing_pairs":     len(folds["testing"]),
        "n_training_subjects":   len(fold_subjects["training"]),
        "n_validation_subjects": len(fold_subjects["validation"]),
        "n_testing_subjects":    len(fold_subjects["testing"]),
        "fold_audit":             fold_audit,
        "stratum_split_audit":    stratum_split_audit,
        "oversample": {
            "mode":                              args.oversample_mode,
            "max_ratio":                         args.oversample_max_ratio,
            "cap":                               args.oversample_cap,
            "n_training_pairs_pre_oversample":   n_training_pairs_pre_oversample,
            "n_training_pairs_post_oversample":  len(folds["training"]),
            "per_stratum":                       oversample_audit,
        },
        "_provenance": {
            "src":         str(args.src.name),
            "seed":        args.seed,
            "fractions":   {"train": args.train_frac, "val": args.val_frac, "test": test_frac},
            "strata_priority":   list(STRATA_PRIORITY),
            "pair_mode_input":   src.get("pair_mode"),
            "thresholds_input":  src.get("_thresholds"),
        },
    }
    args.dst.write_text(json.dumps(out, indent=2))

    print(f"Wrote {args.dst}")
    print(f"  subjects: train {out['n_training_subjects']:>4d}  "
          f"val {out['n_validation_subjects']:>4d}  "
          f"test {out['n_testing_subjects']:>4d}")
    print(f"  pairs:    train {out['n_training_pairs']:>4d}  "
          f"val {out['n_validation_pairs']:>4d}  "
          f"test {out['n_testing_pairs']:>4d}")
    print()
    print(f"  {'stratum':<12s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s}")
    for stratum, row in stratum_split_audit.items():
        print(f"  {stratum:<12s} {row['total']:>6d} {row['training']:>6d} "
              f"{row['validation']:>6d} {row['testing']:>6d}")
    print()
    for fold in ("training", "validation", "testing"):
        a = fold_audit[fold]
        print(f"  [{fold:<10s}]  trajectory={a['trajectory_counts']}  "
              f"direction={a['treatment_direction_counts']}")
    if args.oversample_mode != "none":
        print()
        print(f"  Oversample mode={args.oversample_mode!r}  max_ratio={args.oversample_max_ratio}  cap={args.oversample_cap}")
        print(f"  Training pairs: {n_training_pairs_pre_oversample} -> {len(folds['training'])}")
        for stratum, row in sorted(oversample_audit.items(), key=lambda kv: -kv[1]['final']):
            print(f"    {stratum:<24s} orig={row['original']:>4d}  x{row['multiplier']:<3d}  -> {row['final']:>5d}")


if __name__ == "__main__":
    main()
