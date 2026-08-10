"""nnU-Net v2 trainer that keeps REAL:SYNTH batch exposure balanced at 50:50.

The default ``nnUNetTrainer`` samples cases uniformly over ``imagesTr/``. When
synthetic augmentation is added (Dataset511..514 with 500..10000 SYNTH_* cases
alongside 1187 REAL_* cases), synth dominates the training batch mix -- at
n_synth=10000 the model sees a real case only ~11% of the time. This trainer
overrides case selection so REAL_* and SYNTH_* are each drawn with 50%
aggregate probability regardless of the dose, isolating the effect of
"synth added to training pool" from "synth dominates batch mix".

Registered with nnU-Net via::

    nnUNetv2_train -tr nnUNetTrainerBalancedSynth ...

For the real-only baseline (Dataset510, no SYNTH_*) the sampler collapses to
uniform, so this trainer is equivalent to nnUNetTrainer there.

Implementation note: patching ``get_indices`` at class-level relies on the
Linux ``fork`` start method used by nnU-Net's batchgenerators workers, which
inherits the patched class from the parent process.
"""

from __future__ import annotations

import numpy as np

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# 2D dataloader is renamed/relocated across nnunetv2 versions; patch it only if
# it's importable, since this trainer targets 3d_fullres and never touches 2D.
try:
    from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D as _DL2D
except ImportError:
    try:
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader as _DL2D
    except ImportError:
        _DL2D = None


def _balanced_get_indices(self):
    keys = self.list_of_keys
    probs = getattr(self, "_balanced_probs", None)
    if probs is None:
        real = np.fromiter((k.startswith("REAL_") for k in keys), dtype=bool, count=len(keys))
        synth = np.fromiter((k.startswith("SYNTH_") for k in keys), dtype=bool, count=len(keys))
        n_real = int(real.sum())
        n_synth = int(synth.sum())
        if n_real > 0 and n_synth > 0:
            w = np.zeros(len(keys), dtype=np.float64)
            w[real] = 0.5 / n_real
            w[synth] = 0.5 / n_synth
            probs = w
        else:
            probs = False  # sentinel: uniform, do not recompute
        self._balanced_probs = probs
    if probs is False:
        idx = np.random.choice(len(keys), size=self.batch_size, replace=True)
    else:
        idx = np.random.choice(len(keys), size=self.batch_size, replace=True, p=probs)
    return [keys[i] for i in idx]


nnUNetDataLoader3D.get_indices = _balanced_get_indices
if _DL2D is not None:
    _DL2D.get_indices = _balanced_get_indices


class nnUNetTrainerBalancedSynth(nnUNetTrainer):
    """Real-oversampling trainer that keeps REAL:SYNTH batch exposure 50:50."""

    def on_train_start(self):
        super().on_train_start()
        try:
            keys = list(self.dataloader_train.generator._data.keys())  # type: ignore[attr-defined]
        except AttributeError:
            keys = list(getattr(self.dataloader_train, "_data", {}).keys())
        n_real = sum(1 for k in keys if k.startswith("REAL_"))
        n_synth = sum(1 for k in keys if k.startswith("SYNTH_"))
        self.print_to_log_file(
            f"[BalancedSynth] train pool: n_real={n_real}  n_synth={n_synth}  "
            f"target REAL:SYNTH exposure = 50:50 per batch"
        )
