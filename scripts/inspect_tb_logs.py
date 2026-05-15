#!/usr/bin/env python3
"""Inspect TensorBoard logs and extract reconstruction images for visual fidelity assessment."""
from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LOG_DIR = "/Users/nk233/mhf/projects/text2glioma/runs/logs_2025_02_24/logs"
OUTDIR = Path("/Users/nk233/text2glioma/audit_figures/tb_recons")


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        d = os.path.join(LOG_DIR, split)
        if not os.path.isdir(d):
            print(f"  {split}/ not found, skipping")
            continue

        files = sorted(os.listdir(d))
        print(f"\n=== {split} ({len(files)} file(s)) ===")

        for f in files:
            path = os.path.join(d, f)
            ea = EventAccumulator(path, size_guidance={"images": 100, "scalars": 10000})
            ea.Reload()

            tags_s = ea.Tags().get("scalars", [])
            tags_i = ea.Tags().get("images", [])
            print(f"  Scalar tags ({len(tags_s)}): {tags_s}")
            print(f"  Image tags  ({len(tags_i)}): {tags_i}")

            # Print scalar summaries
            for t in tags_s:
                events = ea.Scalars(t)
                if len(events) == 0:
                    continue
                vals = [e.value for e in events]
                steps = [e.step for e in events]
                print(f"    {t}: {len(events)} events, "
                      f"steps [{steps[0]}..{steps[-1]}], "
                      f"last={vals[-1]:.6f}, "
                      f"min={min(vals):.6f}, max={max(vals):.6f}")

            # Extract and save images
            for t in tags_i:
                events = ea.Images(t)
                if len(events) == 0:
                    continue
                print(f"    {t}: {len(events)} images, "
                      f"steps [{events[0].step}..{events[-1].step}]")

                # Save first, middle, last images
                indices = [0, len(events)//2, -1]
                if len(events) <= 3:
                    indices = list(range(len(events)))

                for idx in indices:
                    ev = events[idx]
                    img = Image.open(BytesIO(ev.encoded_image_string))
                    tag_clean = t.replace("/", "_").replace(" ", "_")
                    fname = f"{split}_{tag_clean}_step{ev.step}.png"
                    out_path = OUTDIR / fname
                    img.save(out_path)
                    print(f"      Saved: {out_path.name} "
                          f"(step {ev.step}, {img.size})")

    # --- Create comparison grid of saved images ---
    saved = sorted(OUTDIR.glob("*.png"))
    if not saved:
        print("\nNo images extracted.")
        return

    print(f"\n  Extracted {len(saved)} images to {OUTDIR}/")

    # Group by tag
    from collections import defaultdict
    groups = defaultdict(list)
    for p in saved:
        # split_tag_stepN.png -> group by split_tag
        parts = p.stem.rsplit("_step", 1)
        if len(parts) == 2:
            groups[parts[0]].append(p)

    # Create a comparison figure per group
    for group_name, paths in sorted(groups.items()):
        paths.sort()
        n = len(paths)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)
        fig.suptitle(group_name.replace("_", " "), fontsize=14, fontweight="bold")
        for i, p in enumerate(paths):
            img = np.array(Image.open(p))
            ax = axes[0, i]
            ax.imshow(img)
            step = p.stem.rsplit("_step", 1)[-1]
            ax.set_title(f"Step {step}", fontsize=11)
            ax.axis("off")
        plt.tight_layout()
        grid_path = OUTDIR / f"grid_{group_name}.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grid: {grid_path.name}")

    print(f"\nAll outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
