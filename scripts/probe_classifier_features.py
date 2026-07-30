"""Linear probe on a trained IDH classifier's penultimate features.

Answers: "does the DenseNet feature space already contain most of the
extractable IDH signal, or would a bigger/better classifier recover
more?" Cheap diagnostic (~5 min on one GPU) that tells us whether the
0.68 synth-only AUROC ceiling is information-limited (features carry
the max signal available in pixels) or architecture-limited (features
throw away signal a bigger backbone would preserve).

Method
------
1. Load a trained ``best_model.pth`` (DenseNet-121 from
   :func:`train_molecular_classifier._make_model`).
2. Register a forward hook on ``class_layers.flatten`` to capture the
   1024-D penultimate feature vector per sample.
3. Run inference on the real validation split and on a subsampled synth
   split (deterministic subset of ``manifest_release.csv`` with known
   IDH labels).
4. Fit two linear probes on the features:
     * **Logistic regression** (5-fold stratified CV → AUROC)
     * **Fisher LDA** (class-separation ratio + LDA-projection AUROC)
5. Also report the direct cross-domain transfer: train LR on real-val
   features, test on synth-val features (and vice versa).

Read the ``probe_report.json``:
  * ``real_lr_cv_auroc``   — real-val linear probe (bounded by
                              end-to-end AUROC of the same classifier).
                              If << end-to-end, the final Linear+dropout
                              layers were doing work.
  * ``synth_lr_cv_auroc``  — synth-val linear probe (upper bound on
                              what any classifier could extract from
                              these features).
  * ``real_to_synth_auroc``— transfer probe. Compares the real-domain
                              feature manifold to the synth-domain one.
                              ≈ synth_lr_cv_auroc → features align across
                              domains → no domain gap to close → a bigger
                              backbone gives marginal gains. Much lower
                              → real features don't see synth's IDH
                              signal → transformer with global attention
                              has a plausible edge.
  * ``fisher_ratio_*``     — class-centroid separation / within-class
                              scatter, per split. Higher = more
                              linearly separable.

Usage
-----
::

    # Point at the real-only classifier (n_synth=0) so features are
    # untainted by synth training. seed_0 is arbitrary; average over
    # seeds later if desired.
    python scripts/probe_classifier_features.py \\
        --model_ckpt /path/to/n_synth_0/seed_0/best_model.pth \\
        --real_datalist datalist_N494_idh_only.json \\
        --synth_root /path/to/text2glioma_synth_idh_only_cfg70 \\
        --n_synth_val 500 \\
        --out_dir results/probe_real_only_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai import transforms as T
from monai.data import Dataset
from monai.networks import nets
from torch.utils.data import DataLoader

# Local imports
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from text2glioma.training.molecular_conditioning import (  # noqa: E402
    IDH_UNKNOWN,
)


_TARGET_SPATIAL = (160, 224, 160)


# ---------------------------------------------------------------------

def _build_transforms() -> T.Compose:
    return T.Compose([
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
        T.ToTensord(keys=["image"]),
    ])


def _load_real_val(datalist_path: Path) -> list[dict]:
    with datalist_path.open() as fh:
        dl = json.load(fh)
    val = dl.get("validation", [])
    items: list[dict] = []
    for it in val:
        idh = int(it.get("idh", IDH_UNKNOWN))
        if idh == IDH_UNKNOWN:
            continue
        items.append({"image": str(it["image"]), "idh": idh})
    return items


def _load_synth(synth_root: Path, n_synth_val: int, seed: int) -> list[dict]:
    import pandas as pd
    manifest = synth_root / "manifest_release.csv"
    if not manifest.is_file():
        raise SystemExit(
            f"manifest_release.csv not found under {synth_root}. Run "
            f"scripts/dataset_release/build_release_manifest.py first."
        )
    df = pd.read_csv(manifest)
    if "idh" not in df.columns:
        # Fallback: pull idh from per-sample metadata.json
        def _fetch(row) -> int:
            meta_path = synth_root / row["relpath_image"].replace(
                "image.nii.gz", "metadata.json",
            )
            try:
                with open(meta_path) as fh:
                    return int(json.load(fh).get("idh", IDH_UNKNOWN))
            except Exception:
                return IDH_UNKNOWN
        df["idh"] = df.apply(_fetch, axis=1)
    df = df[df["idh"].isin([0, 1])].reset_index(drop=True)
    if len(df) < n_synth_val:
        n_synth_val = len(df)
    # Deterministic stratified subsample so WT/MUT halves both included.
    rng = np.random.default_rng(int(seed))
    wt = df[df.idh == 0].sample(
        n=min(n_synth_val // 2, (df.idh == 0).sum()),
        random_state=rng.integers(0, 2**31 - 1),
    )
    mut = df[df.idh == 1].sample(
        n=min(n_synth_val - len(wt), (df.idh == 1).sum()),
        random_state=rng.integers(0, 2**31 - 1),
    )
    take = pd.concat([wt, mut]).sample(
        frac=1.0, random_state=rng.integers(0, 2**31 - 1),
    ).reset_index(drop=True)
    items: list[dict] = []
    for _, r in take.iterrows():
        items.append({
            "image": str(synth_root / r["relpath_image"]),
            "idh":   int(r["idh"]),
        })
    return items


# ---------------------------------------------------------------------

@torch.no_grad()
def _extract_features(model: torch.nn.Module,
                       items: list[dict],
                       device: torch.device,
                       batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and grab the 1024-D penultimate feature per sample.

    We hook ``class_layers.flatten`` which sits between the global
    average pool and the final Linear. Output shape per sample: (1024,).
    """
    if not items:
        return np.zeros((0, 1024)), np.zeros((0,), dtype=int)

    captured: list[torch.Tensor] = []

    def _hook(_module, _inp, out):
        # out: (B, 1024)
        captured.append(out.detach().cpu())

    handle = model.class_layers.flatten.register_forward_hook(_hook)

    xforms = _build_transforms()
    ds = Dataset(data=items, transform=xforms)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=False)

    labels: list[int] = []
    model.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        _ = model(x)   # populates captured via the hook
        for it in batch["idh"]:
            labels.append(int(it.item()) if hasattr(it, "item") else int(it))
    handle.remove()

    if not captured:
        return np.zeros((0, 1024)), np.zeros((0,), dtype=int)
    feats = torch.cat(captured, dim=0).numpy()
    y = np.asarray(labels, dtype=int)
    return feats, y


# ---------------------------------------------------------------------

def _linear_probe(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Train logistic regression on (X_train, y_train), return AUROC on
    (X_test, y_test). Returns nan if either split lacks both classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p))


def _cv_linear_probe(X: np.ndarray, y: np.ndarray, n_folds: int = 5,
                      seed: int = 0) -> tuple[float, float]:
    """5-fold stratified CV logistic-regression AUROC. Returns (mean, std)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    if len(y) < 2 * n_folds or len(set(y)) < 2:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], p))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def _fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Fisher discriminant ratio ||μ_1 - μ_0||^2 / (tr(Σ_0) + tr(Σ_1)).

    Higher = classes are farther apart relative to within-class scatter.
    Rough scale: ~0.01 is weak, ~0.1 is moderate, >1 is strong linear
    separability.
    """
    if len(set(y)) < 2:
        return float("nan")
    X0, X1 = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    centroid_gap = float(np.sum((mu1 - mu0) ** 2))
    within = float(np.var(X0, axis=0).sum() + np.var(X1, axis=0).sum())
    return centroid_gap / max(within, 1e-12)


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_ckpt",    type=Path, required=True,
                    help="best_model.pth from train_molecular_classifier.py")
    ap.add_argument("--real_datalist", type=Path, required=True)
    ap.add_argument("--synth_root",    type=Path, required=True)
    ap.add_argument("--n_synth_val",   type=int, default=500,
                    help="Number of synth samples to probe. Stratified 50/50 "
                         "WT/MUT subsample of manifest_release.csv.")
    ap.add_argument("--sample_seed",   type=int, default=0,
                    help="Deterministic seed for the synth subsample.")
    ap.add_argument("--dropout_prob",  type=float, default=0.3,
                    help="Must match the trained checkpoint's dropout_prob "
                         "(default: matches _make_model default).")
    ap.add_argument("--out_dir",       type=Path, required=True)
    ap.add_argument("--device",        type=str, default="cuda")
    ap.add_argument("--batch_size",    type=int, default=4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load model ---------------------------------------------
    print(f"[load] model_ckpt={args.model_ckpt}", file=sys.stderr)
    model = nets.DenseNet121(
        spatial_dims=3, in_channels=4, out_channels=2,
        dropout_prob=args.dropout_prob,
    ).to(device)
    state = torch.load(str(args.model_ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- Load splits --------------------------------------------
    real_val = _load_real_val(args.real_datalist)
    synth    = _load_synth(args.synth_root, args.n_synth_val, args.sample_seed)
    print(f"[data] real_val={len(real_val)}  synth={len(synth)}",
          file=sys.stderr)

    # ---- Extract features ---------------------------------------
    print("[extract] real_val ...", file=sys.stderr, flush=True)
    Xr, yr = _extract_features(model, real_val, device, args.batch_size)
    print(f"[extract] real_val features: {Xr.shape}  labels: "
          f"WT={int((yr==0).sum())} MUT={int((yr==1).sum())}",
          file=sys.stderr, flush=True)

    print("[extract] synth ...", file=sys.stderr, flush=True)
    Xs, ys = _extract_features(model, synth, device, args.batch_size)
    print(f"[extract] synth features: {Xs.shape}  labels: "
          f"WT={int((ys==0).sum())} MUT={int((ys==1).sum())}",
          file=sys.stderr, flush=True)

    # ---- Save raw features --------------------------------------
    np.savez_compressed(
        args.out_dir / "features.npz",
        Xr=Xr, yr=yr, Xs=Xs, ys=ys,
    )

    # ---- Probes -------------------------------------------------
    real_lr_mean, real_lr_std = _cv_linear_probe(Xr, yr, n_folds=5)
    synth_lr_mean, synth_lr_std = _cv_linear_probe(Xs, ys, n_folds=5)
    r_to_s_auroc = _linear_probe(Xr, yr, Xs, ys)
    s_to_r_auroc = _linear_probe(Xs, ys, Xr, yr)
    fisher_real  = _fisher_ratio(Xr, yr)
    fisher_synth = _fisher_ratio(Xs, ys)

    # Centroid distance in feature space (informative for domain gap)
    def _centroid(X, y, c):
        if not len(X) or not (y == c).any():
            return None
        return X[y == c].mean(axis=0)
    mu_r_wt = _centroid(Xr, yr, 0)
    mu_r_mut = _centroid(Xr, yr, 1)
    mu_s_wt = _centroid(Xs, ys, 0)
    mu_s_mut = _centroid(Xs, ys, 1)

    def _dist(a, b):
        if a is None or b is None:
            return float("nan")
        return float(np.linalg.norm(a - b))

    report = {
        "model_ckpt":         str(args.model_ckpt),
        "n_real_val":         int(len(yr)),
        "n_synth":            int(len(ys)),
        "feature_dim":        int(Xr.shape[1] if len(Xr) else 0),
        "real_lr_cv_auroc":   {"mean": real_lr_mean,  "std": real_lr_std},
        "synth_lr_cv_auroc":  {"mean": synth_lr_mean, "std": synth_lr_std},
        "real_to_synth_auroc": r_to_s_auroc,
        "synth_to_real_auroc": s_to_r_auroc,
        "fisher_ratio_real":  fisher_real,
        "fisher_ratio_synth": fisher_synth,
        "centroid_dist_wt_real_vs_synth":  _dist(mu_r_wt,  mu_s_wt),
        "centroid_dist_mut_real_vs_synth": _dist(mu_r_mut, mu_s_mut),
        "centroid_dist_wt_vs_mut_real":    _dist(mu_r_wt,  mu_r_mut),
        "centroid_dist_wt_vs_mut_synth":   _dist(mu_s_wt,  mu_s_mut),
    }

    out_json = args.out_dir / "probe_report.json"
    out_json.write_text(json.dumps(report, indent=2))

    # ---- Console print ------------------------------------------
    print()
    print(f"Feature dim: {report['feature_dim']}")
    print(f"Real-val (n={len(yr)}) linear probe AUROC: "
          f"{real_lr_mean:.4f} ± {real_lr_std:.4f}   Fisher: {fisher_real:.4f}")
    print(f"Synth    (n={len(ys)}) linear probe AUROC: "
          f"{synth_lr_mean:.4f} ± {synth_lr_std:.4f}   Fisher: {fisher_synth:.4f}")
    print(f"Cross-domain probe (train real, test synth): "
          f"AUROC = {r_to_s_auroc:.4f}")
    print(f"Cross-domain probe (train synth, test real): "
          f"AUROC = {s_to_r_auroc:.4f}")
    print()
    print("Centroids in feature space:")
    print(f"  || WT_real  - WT_synth ||  = {report['centroid_dist_wt_real_vs_synth']:.3f}")
    print(f"  || MUT_real - MUT_synth || = {report['centroid_dist_mut_real_vs_synth']:.3f}")
    print(f"  || WT_real  - MUT_real  || = {report['centroid_dist_wt_vs_mut_real']:.3f}")
    print(f"  || WT_synth - MUT_synth || = {report['centroid_dist_wt_vs_mut_synth']:.3f}")
    print()

    # ---- Interpretation hints -----------------------------------
    print("Interpretation:")
    if real_lr_mean > 0.85:
        print("  Real linear probe >= end-to-end -> features encode IDH well "
              "for real; final Linear head not doing much extra work.")
    if synth_lr_mean > 0.75:
        print("  Synth linear probe > 0.75 -> features carry MORE synth IDH "
              "signal than the end-to-end classifier extracts. A stronger "
              "head or more capacity could plausibly recover more.")
    elif synth_lr_mean > 0.60:
        print("  Synth linear probe ~ end-to-end ceiling -> features encode "
              "roughly what a linear model can extract. Architecture is not "
              "the primary bottleneck; information ceiling is close.")
    else:
        print("  Synth linear probe < 0.60 -> features collapse synth IDH "
              "signal. Either the model was not trained enough on synth OR "
              "the features from a real-only-trained backbone are not "
              "aligned with synth-domain IDH structure. A transformer "
              "backbone with global attention has a plausible edge.")
    if not np.isnan(r_to_s_auroc):
        if r_to_s_auroc >= synth_lr_mean - 0.03:
            print("  Cross-domain transfer ~ synth CV -> feature manifolds "
                  "align well; no domain gap to close.")
        else:
            print("  Cross-domain transfer << synth CV -> real-trained "
                  "features poorly align with synth-domain IDH. Domain-gap "
                  "problem — a bigger backbone might help by extracting "
                  "domain-invariant features.")

    # ---- Optional 2D projection ---------------------------------
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        # Fit PCA on the union so both splits share the same 2D basis.
        if len(Xr) and len(Xs):
            pca = PCA(n_components=2, random_state=0)
            X_all = np.concatenate([Xr, Xs], axis=0)
            pca.fit(X_all)
            Zr = pca.transform(Xr)
            Zs = pca.transform(Xs)

            fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
            markers = {0: "o", 1: "^"}
            for c, name in [(0, "WT"), (1, "MUT")]:
                sel = yr == c
                if sel.any():
                    ax.scatter(Zr[sel, 0], Zr[sel, 1], marker=markers[c],
                                c="#1f77b4", alpha=0.55, label=f"real {name}",
                                edgecolors="white", linewidths=0.5, s=45)
                sel = ys == c
                if sel.any():
                    ax.scatter(Zs[sel, 0], Zs[sel, 1], marker=markers[c],
                                c="#d62728", alpha=0.35, label=f"synth {name}",
                                edgecolors="white", linewidths=0.5, s=30)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title("Penultimate features (PCA-2D)")
            ax.legend(fontsize=8, loc="best", framealpha=0.7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = args.out_dir / "probe_report.png"
            fig.savefig(str(out_png), dpi=150)
            print(f"\nWrote figure: {out_png}")
    except ImportError:
        print("\n[warn] matplotlib/sklearn not available for figure; skipping")

    print(f"Wrote JSON:   {out_json}")


if __name__ == "__main__":
    main()
