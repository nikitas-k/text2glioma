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

import importlib
import pkgutil

import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _get_case_keys(dl) -> list[str]:
    """Recover the case-ID list from a dataloader across nnunetv2 versions."""
    for attr in ("list_of_keys", "identifiers", "case_identifiers"):
        v = getattr(dl, attr, None)
        if v is not None:
            return list(v)
    data = getattr(dl, "_data", None) or getattr(dl, "data", None)
    if data is not None:
        if hasattr(data, "identifiers"):
            return list(data.identifiers)
        if hasattr(data, "keys"):
            return list(data.keys())
        try:
            return list(data)
        except TypeError:
            pass
    raise AttributeError(
        f"could not recover case-ID list from {type(dl).__name__}; "
        f"tried list_of_keys, identifiers, case_identifiers, _data, data"
    )


def _balanced_get_indices(self):
    probs = getattr(self, "_balanced_probs", None)
    keys = getattr(self, "_balanced_keys", None)
    if probs is None or keys is None:
        keys = _get_case_keys(self)
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
            probs = False
        self._balanced_probs = probs
        self._balanced_keys = keys
    if probs is False:
        idx = np.random.choice(len(keys), size=self.batch_size, replace=True)
    else:
        idx = np.random.choice(len(keys), size=self.batch_size, replace=True, p=probs)
    return [keys[i] for i in idx]


def _patch_dataloaders() -> list[str]:
    """Locate and patch every nnU-Net dataloader class regardless of layout.

    nnunetv2 split its dataloader into ``data_loader_2d`` / ``data_loader_3d``
    in older versions and merged them into a single ``data_loader`` (with
    ``nnUNetDataLoader``) in newer versions. Walk the package and patch any
    class whose name starts with ``nnUNetDataLoader``.
    """
    import nnunetv2.training.dataloading as pkg

    patched: list[str] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"nnunetv2.training.dataloading.{info.name}")
        except ImportError:
            continue
        for attr in dir(mod):
            if not attr.startswith("nnUNetDataLoader"):
                continue
            obj = getattr(mod, attr)
            if isinstance(obj, type):
                obj.get_indices = _balanced_get_indices
                patched.append(f"{mod.__name__}.{attr}")
    return patched


_PATCHED = _patch_dataloaders()
if not _PATCHED:
    raise RuntimeError(
        "nnUNetTrainerBalancedSynth: no nnU-Net dataloader classes found under "
        "nnunetv2.training.dataloading; balanced sampler cannot be installed."
    )


class nnUNetTrainerBalancedSynth(nnUNetTrainer):
    """Real-oversampling trainer that keeps REAL:SYNTH batch exposure 50:50."""

    def on_train_start(self):
        super().on_train_start()
        dl = self.dataloader_train
        raw = getattr(dl, "generator", dl)
        try:
            keys = _get_case_keys(raw)
        except AttributeError:
            keys = []
        n_real = sum(1 for k in keys if k.startswith("REAL_"))
        n_synth = sum(1 for k in keys if k.startswith("SYNTH_"))
        self.print_to_log_file(f"[BalancedSynth] patched dataloaders: {_PATCHED}")
        self.print_to_log_file(
            f"[BalancedSynth] train pool: n_real={n_real}  n_synth={n_synth}  "
            f"target REAL:SYNTH exposure = 50:50 per batch"
        )
