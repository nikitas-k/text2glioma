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
    """Two-slot learnable class conditioner for IDH + MGMT status.

    Parameters
    ----------
    hidden_dim
        Dimensionality of the token embeddings — must match the frozen
        text-encoder output width (768 for RadBERT-base).
    dropout_to_unknown_p
        Probability, at training time, of replacing an example's
        molecular class with ``UNKNOWN``. This drives classifier-free
        guidance: the "null" conditioning direction the sampler pushes
        away from at inference is the same distribution the model sees
        for unknown-status subjects.
    init_std
        Standard deviation for the Gaussian initialiser applied to both
        embedding tables. 0.02 matches transformer conventions and
        keeps the pseudo-tokens on a scale comparable to RadBERT
        outputs.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout_to_unknown_p: float = 0.2,
        init_std: float = 0.02,
        n_idh_classes: int = 3,
        n_mgmt_classes: int = 3,
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout_to_unknown_p <= 1.0):
            raise ValueError(
                f"dropout_to_unknown_p must be in [0, 1], got {dropout_to_unknown_p}"
            )
        if n_idh_classes < 1 or n_mgmt_classes < 1:
            raise ValueError("need at least one class per molecular field")

        self.hidden_dim = hidden_dim
        self.n_idh_classes = n_idh_classes
        self.n_mgmt_classes = n_mgmt_classes
        self.dropout_to_unknown_p = float(dropout_to_unknown_p)

        self.idh_embedding  = nn.Embedding(n_idh_classes,  hidden_dim)
        self.mgmt_embedding = nn.Embedding(n_mgmt_classes, hidden_dim)

        nn.init.normal_(self.idh_embedding.weight,  mean=0.0, std=init_std)
        nn.init.normal_(self.mgmt_embedding.weight, mean=0.0, std=init_std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idh_class:  torch.Tensor,
        mgmt_class: torch.Tensor,
        force_unknown: bool = False,
    ) -> torch.Tensor:
        """Emit the (B, 2, hidden_dim) conditioning token sequence.

        Parameters
        ----------
        idh_class, mgmt_class
            Long tensors of shape ``(B,)`` with class indices in
            ``[0, n_classes)``. Any values outside range trigger a
            ``ValueError`` — validate upstream.
        force_unknown
            If ``True``, override both class inputs with the UNKNOWN
            index. Used at inference to build the CFG null branch;
            never set at training time (rely on
            ``dropout_to_unknown_p`` instead).
        """
        idh, mgmt = self._validate_and_align(idh_class, mgmt_class)

        if force_unknown:
            idh  = torch.full_like(idh,  IDH_UNKNOWN)
            mgmt = torch.full_like(mgmt, MGMT_UNKNOWN)
        elif self.training and self.dropout_to_unknown_p > 0.0:
            idh, mgmt = self._apply_dropout(idh, mgmt)

        idh_emb  = self.idh_embedding(idh)    # (B, D)
        mgmt_emb = self.mgmt_embedding(mgmt)  # (B, D)
        # Stack along a NEW sequence axis so the two pseudo-tokens sit
        # in a well-defined order (IDH first, MGMT second).
        tokens = torch.stack([idh_emb, mgmt_emb], dim=1)  # (B, 2, D)
        return tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def null_tokens(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the (B, 2, hidden_dim) unconditional-CFG null tokens.

        Equivalent to calling ``forward(...)`` with all-UNKNOWN classes
        but avoids constructing intermediate tensors. Use this to build
        the unconditional-branch context at inference.
        """
        idh  = torch.full((batch_size,), IDH_UNKNOWN,  dtype=torch.long, device=device)
        mgmt = torch.full((batch_size,), MGMT_UNKNOWN, dtype=torch.long, device=device)
        idh_emb  = self.idh_embedding(idh).to(dtype=dtype)
        mgmt_emb = self.mgmt_embedding(mgmt).to(dtype=dtype)
        return torch.stack([idh_emb, mgmt_emb], dim=1)

    def num_tokens(self) -> int:
        """Number of pseudo-tokens contributed to the cross-attn sequence."""
        return NUM_MOLECULAR_TOKENS

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_and_align(
        self, idh: torch.Tensor, mgmt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idh.dtype != torch.long:
            idh = idh.long()
        if mgmt.dtype != torch.long:
            mgmt = mgmt.long()
        if idh.shape != mgmt.shape:
            raise ValueError(
                f"idh and mgmt must have the same shape; got "
                f"idh={tuple(idh.shape)}, mgmt={tuple(mgmt.shape)}"
            )
        if idh.ndim != 1:
            raise ValueError(f"expected 1D class tensor, got shape {tuple(idh.shape)}")
        if int(idh.max()) >= self.n_idh_classes or int(idh.min()) < 0:
            raise ValueError(
                f"idh_class out of range [0, {self.n_idh_classes}); "
                f"got min={int(idh.min())} max={int(idh.max())}"
            )
        if int(mgmt.max()) >= self.n_mgmt_classes or int(mgmt.min()) < 0:
            raise ValueError(
                f"mgmt_class out of range [0, {self.n_mgmt_classes}); "
                f"got min={int(mgmt.min())} max={int(mgmt.max())}"
            )
        return idh, mgmt

    def _apply_dropout(
        self, idh: torch.Tensor, mgmt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently replace each field with UNKNOWN with prob p.

        Independent per-field dropout (rather than joint) means the
        model sees samples where only one of IDH/MGMT is null. This
        supports CFG on either axis independently at inference — you
        can guide on IDH while leaving MGMT at its true class, for
        example.
        """
        p = self.dropout_to_unknown_p
        drop_idh  = torch.rand_like(idh,  dtype=torch.float32) < p
        drop_mgmt = torch.rand_like(mgmt, dtype=torch.float32) < p
        idh  = torch.where(drop_idh,  torch.full_like(idh,  IDH_UNKNOWN),  idh)
        mgmt = torch.where(drop_mgmt, torch.full_like(mgmt, MGMT_UNKNOWN), mgmt)
        return idh, mgmt


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
