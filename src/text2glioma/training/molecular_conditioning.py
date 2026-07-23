"""Learnable class-conditioning for molecular status (IDH, MGMT).

Rationale
---------
RadBERT was pretrained on radiology reports which describe tumour
appearance, not molecular pathology. Appending "IDH-mutant" / "IDH-
wildtype" to a text prompt is unreliable: RadBERT often embeds these
tokens to nearly-collinear points in its 768-dim output space, so the
LDM sees essentially the same conditioning signal in either case and
cannot discriminate. See Ho & Salimans 2022 (§3, "Classifier-Free
Diffusion Guidance") for the canonical treatment of this class of
problem — the fix is a dedicated learnable class-conditioning branch
that runs in parallel to the (frozen) text encoder.

This module implements that branch. Two ``nn.Embedding(3, hidden_dim)``
tables — one for IDH status, one for MGMT status — emit two pseudo-
tokens per sample that are **concatenated** onto the RadBERT token
sequence before cross-attention. The tables are randomly initialised,
so all three states within each modality start with orthogonal
representations and the LDM is free to learn discriminative embeddings
during fine-tuning.

The "unknown" state (index 2) doubles as the classifier-free-guidance
null: subjects with unavailable molecular status get embedded there
during training, giving the model an "average glioma" prior that is
also the target of the CFG uncond pass at inference.

Typical use::

    mol_head = MolecularClassConditioning(hidden_dim=768,
                                          dropout_to_unknown_p=0.2)
    mol_tokens = mol_head(idh_class=batch["idh"], mgmt_class=batch["mgmt"])
    # mol_tokens: (B, 2, hidden_dim)

    text_embeds = radbert(...)      # (B, 128, hidden_dim)
    context = torch.cat([text_embeds, mol_tokens], dim=1)   # (B, 130, hidden_dim)
    noise_pred = unet(x=..., timesteps=..., context=context)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ── Class constants ────────────────────────────────────────────────────

# IDH mutation status
IDH_WILDTYPE = 0
IDH_MUTANT   = 1
IDH_UNKNOWN  = 2

# MGMT promoter methylation status
MGMT_UNMETHYLATED = 0
MGMT_METHYLATED   = 1
MGMT_UNKNOWN      = 2

# String-to-integer mapping for datalist ingestion.
IDH_STR_TO_INT = {
    "wildtype": IDH_WILDTYPE, "wt": IDH_WILDTYPE, "0": IDH_WILDTYPE,
    "mutant":   IDH_MUTANT,   "mut": IDH_MUTANT,  "1": IDH_MUTANT,
    "unknown":  IDH_UNKNOWN,  "unk": IDH_UNKNOWN, "":  IDH_UNKNOWN, "nan": IDH_UNKNOWN,
}
MGMT_STR_TO_INT = {
    "unmethylated": MGMT_UNMETHYLATED, "unmeth": MGMT_UNMETHYLATED, "0": MGMT_UNMETHYLATED,
    "methylated":   MGMT_METHYLATED,   "meth":   MGMT_METHYLATED,   "1": MGMT_METHYLATED,
    "unknown":      MGMT_UNKNOWN,      "unk":    MGMT_UNKNOWN,      "":  MGMT_UNKNOWN, "nan": MGMT_UNKNOWN,
}

# Number of embedding tokens contributed per sample (IDH + MGMT).
NUM_MOLECULAR_TOKENS = 2


@dataclass(frozen=True)
class MolecularStatus:
    """Convenience container for the molecular class of one sample."""
    idh: int   # 0=wt, 1=mut, 2=unk
    mgmt: int  # 0=unm, 1=met, 2=unk

    def as_tensor(self, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.as_tensor(self.idh,  dtype=torch.long)
        m = torch.as_tensor(self.mgmt, dtype=torch.long)
        if device is not None:
            i, m = i.to(device), m.to(device)
        return i, m

    @classmethod
    def null(cls) -> "MolecularStatus":
        return cls(idh=IDH_UNKNOWN, mgmt=MGMT_UNKNOWN)


class MolecularClassConditioning(nn.Module):
    """Learnable class-conditioning head for one or more molecular fields.

    Emits ``(B, len(fields), hidden_dim)`` pseudo-tokens that get appended
    to the frozen RadBERT text embedding sequence and fed into the LDM's
    cross-attention. Each field (``"idh"``, ``"mgmt"``) has its own
    randomly-initialised ``nn.Embedding(3, hidden_dim)`` table so the
    three states (wildtype/mutant/unknown or unmethylated/methylated/
    unknown) start with pairwise-orthogonal representations and can
    learn discriminative geometry from scratch during fine-tuning.

    Parameters
    ----------
    hidden_dim
        Dimensionality of the token embeddings — must match the frozen
        text-encoder output width (768 for RadBERT-base).
    dropout_to_unknown_p
        Probability, at training time, of replacing an example's
        molecular class with ``UNKNOWN``. Doubles as the CFG null
        direction — the sampler pushes away from the unconditional
        pass built from all-UNKNOWN tokens.
    init_std
        Standard deviation for the Gaussian initialiser applied to
        every embedding table. 0.02 matches transformer conventions.
    fields
        Which molecular fields to include as separate learnable
        pseudo-tokens. Defaults to ``("idh", "mgmt")`` for backward
        compatibility with v1 checkpoints; pass ``("idh",)`` to build
        an IDH-only head with half the parameters.
    n_idh_classes, n_mgmt_classes
        Number of classes per field. Always 3 (WT/MUT/UNK,
        unm/met/UNK) in this codebase but exposed for future extension.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout_to_unknown_p: float = 0.2,
        init_std: float = 0.02,
        n_idh_classes: int = 3,
        n_mgmt_classes: int = 3,
        fields: tuple[str, ...] = ("idh", "mgmt"),
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout_to_unknown_p <= 1.0):
            raise ValueError(
                f"dropout_to_unknown_p must be in [0, 1], got {dropout_to_unknown_p}"
            )
        if n_idh_classes < 1 or n_mgmt_classes < 1:
            raise ValueError("need at least one class per molecular field")

        # Normalise + validate fields.
        fields = tuple(fields)
        if len(fields) == 0:
            raise ValueError("fields must be non-empty; got ()")
        allowed = {"idh", "mgmt"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown fields {bad!r}; allowed: {allowed}")

        self.hidden_dim = hidden_dim
        self.n_idh_classes = n_idh_classes
        self.n_mgmt_classes = n_mgmt_classes
        self.dropout_to_unknown_p = float(dropout_to_unknown_p)
        self.fields: tuple[str, ...] = fields

        # Create ONLY the requested fields as flat attributes so state_dict
        # keys stay stable: v1 (IDH+MGMT) checkpoints load into
        # ``MolecularClassConditioning(fields=("idh","mgmt"))`` and IDH-only
        # checkpoints into ``fields=("idh",)`` with no key remapping.
        if "idh" in fields:
            self.idh_embedding = nn.Embedding(n_idh_classes, hidden_dim)
            nn.init.normal_(self.idh_embedding.weight, mean=0.0, std=init_std)
        if "mgmt" in fields:
            self.mgmt_embedding = nn.Embedding(n_mgmt_classes, hidden_dim)
            nn.init.normal_(self.mgmt_embedding.weight, mean=0.0, std=init_std)

    # ------------------------------------------------------------------
    # Internals for field access
    # ------------------------------------------------------------------

    def _embedding_for(self, field: str) -> nn.Embedding:
        return getattr(self, f"{field}_embedding")

    def _unknown_index_for(self, field: str) -> int:
        # Unknown class is always the last index by convention.
        n = self.n_idh_classes if field == "idh" else self.n_mgmt_classes
        return n - 1

    def _n_classes_for(self, field: str) -> int:
        return self.n_idh_classes if field == "idh" else self.n_mgmt_classes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idh_class: Optional[torch.Tensor] = None,
        mgmt_class: Optional[torch.Tensor] = None,
        force_unknown: bool = False,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Emit the ``(B, len(self.fields), hidden_dim)`` token sequence.

        Accepts both positional ``mol_head(idh, mgmt)`` (v1 API) and
        keyword-based ``mol_head(idh=idh, mgmt=mgmt)`` calls. Fields
        listed in ``self.fields`` are required; any additional kwargs
        are silently ignored so existing training loops that pass both
        IDH and MGMT keep working when a head has only IDH configured.
        """
        # Assemble kwargs from both positional and named args.
        kwargs: dict[str, torch.Tensor] = {}
        if idh_class is not None:
            kwargs["idh"] = idh_class
        if mgmt_class is not None:
            kwargs["mgmt"] = mgmt_class
        for k, v in extra_kwargs.items():
            if k in ("idh", "mgmt"):
                kwargs[k] = v

        # Validate all required fields are present.
        for f in self.fields:
            if f not in kwargs:
                raise ValueError(
                    f"required field {f!r} missing from forward(); "
                    f"got kwargs {list(kwargs.keys())}, fields {self.fields}"
                )

        # Cast + shape-check each field independently.
        cleaned: dict[str, torch.Tensor] = {}
        ref_shape: Optional[torch.Size] = None
        for f in self.fields:
            t = kwargs[f]
            if t.dtype != torch.long:
                t = t.long()
            if t.ndim != 1:
                raise ValueError(f"expected 1D {f}_class, got shape {tuple(t.shape)}")
            if ref_shape is None:
                ref_shape = t.shape
            elif t.shape != ref_shape:
                raise ValueError(
                    f"class tensors must all have the same shape; "
                    f"got {ref_shape} then {tuple(t.shape)} for {f}"
                )
            n_cls = self._n_classes_for(f)
            if int(t.max()) >= n_cls or int(t.min()) < 0:
                raise ValueError(
                    f"{f}_class out of range [0, {n_cls}); "
                    f"got min={int(t.min())} max={int(t.max())}"
                )
            cleaned[f] = t

        # Apply overrides (force_unknown) or dropout to each field.
        for f in self.fields:
            unk_idx = self._unknown_index_for(f)
            if force_unknown:
                cleaned[f] = torch.full_like(cleaned[f], unk_idx)
            elif self.training and self.dropout_to_unknown_p > 0.0:
                drop = torch.rand_like(cleaned[f], dtype=torch.float32) \
                       < self.dropout_to_unknown_p
                cleaned[f] = torch.where(
                    drop, torch.full_like(cleaned[f], unk_idx), cleaned[f]
                )

        # Look up each field's embedding and stack along a NEW sequence axis.
        tokens = [self._embedding_for(f)(cleaned[f]) for f in self.fields]
        return torch.stack(tokens, dim=1)  # (B, len(fields), D)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def null_tokens(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the CFG-null token sequence ``(B, len(self.fields), D)``.

        Every field is set to its UNKNOWN class index. Equivalent to
        ``forward(force_unknown=True, ...)`` but avoids constructing the
        input tensors.
        """
        tokens: list[torch.Tensor] = []
        for f in self.fields:
            unk_idx = self._unknown_index_for(f)
            cls = torch.full((batch_size,), unk_idx, dtype=torch.long, device=device)
            tokens.append(self._embedding_for(f)(cls).to(dtype=dtype))
        return torch.stack(tokens, dim=1)

    def num_tokens(self) -> int:
        """Number of pseudo-tokens contributed to the cross-attn sequence."""
        return len(self.fields)


# ── Datalist ingestion helper ─────────────────────────────────────────

def parse_status_string(value, mapping: dict[str, int], default: int) -> int:
    """Parse a molecular status value from datalist JSON into a class int.

    Accepts strings (case-insensitive), integers, or NaN. Falls back to
    the UNKNOWN class for unparseable / null inputs so the pipeline is
    robust to sparse molecular annotation.
    """
    if value is None:
        return default
    # NaN check for float labels ingested from CSV.
    if isinstance(value, float) and value != value:
        return default
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    key = str(value).strip().lower()
    return mapping.get(key, default)


def parse_idh(value) -> int:
    return parse_status_string(value, IDH_STR_TO_INT, default=IDH_UNKNOWN)


def parse_mgmt(value) -> int:
    return parse_status_string(value, MGMT_STR_TO_INT, default=MGMT_UNKNOWN)
