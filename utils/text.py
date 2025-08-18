"""Utility functions for text tokenization and encoding.

This module exposes helpers around the Hugging Face tokenizers and text
encoders used within this project.  The tokenizer is instantiated once at
import time and stored in module level global ``TOKENIZER`` to avoid repeated
construction which can be expensive.

Functions
---------
* :func:`encode_text` – encode raw strings into embedding tensors.
* :func:`prepare_conditioning` – obtain conditional and unconditional text
  embeddings with caching for the unconditional branch.
* :func:`tokenize_function` – simple wrapper suitable for mapping over a
  Hugging Face ``datasets`` corpus.

In addition, the module may be executed as a script to preprocess a text
corpus into a tokenized dataset and persist it to disk ahead of training.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedModel

# -----------------------------------------------------------------------------
# Global resources
# -----------------------------------------------------------------------------
# The tokenizer is created once when the module is imported.  Downstream code
# can simply import ``TOKENIZER`` instead of instantiating their own copy.
# ``distilbert-base-uncased`` is a light‑weight default; callers can override
# by setting the ``HF_TOKENIZER`` environment variable before import.
import os

MODEL_NAME = os.environ.get("HF_TOKENIZER", "distilbert-base-uncased")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Cache for unconditional embeddings keyed by ``(batch_size, device)``.  The
# cache avoids recomputing embeddings for empty prompts which are constant for a
# given batch size and device placement.
_UNCOND_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


# -----------------------------------------------------------------------------
# Text encoding helpers
# -----------------------------------------------------------------------------

def encode_text(
    texts: List[str],
    text_encoder: PreTrainedModel,
    device: torch.device | str,
    max_length: int = 77,
) -> torch.Tensor:
    """Tokenize *texts* and return hidden states from ``text_encoder``.

    Parameters
    ----------
    texts:
        Sequence of input strings.
    text_encoder:
        Hugging Face ``PreTrainedModel`` producing ``last_hidden_state``.
    device:
        Device on which the computation should run.
    max_length:
        Maximum token length; inputs are padded / truncated accordingly.
    """

    tokenized = TOKENIZER(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        outputs = text_encoder(**tokenized)
    return outputs.last_hidden_state


def prepare_conditioning(
    prompts: List[str],
    text_encoder: PreTrainedModel,
    device: torch.device | str,
    max_length: int = 77,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return conditional and unconditional embeddings for *prompts*.

    Unconditional embeddings are cached per ``(batch_size, device)`` pair to
    avoid repeated computation of the empty prompts during training.
    """

    batch_size = len(prompts)
    conditional = encode_text(prompts, text_encoder, device, max_length)
    cache_key = (batch_size, str(device))

    if cache_key not in _UNCOND_CACHE:
        empty_prompts = [""] * batch_size
        _UNCOND_CACHE[cache_key] = encode_text(
            empty_prompts, text_encoder, device, max_length
        )

    unconditional = _UNCOND_CACHE[cache_key]
    return conditional, unconditional


def tokenize_function(examples: Dict[str, List[str]], max_length: int = 77) -> Dict[str, List[int]]:
    """Tokenization mapping function suitable for ``datasets.Dataset.map``.

    The function expects a dictionary containing a ``"text"`` field.
    """

    return TOKENIZER(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


# -----------------------------------------------------------------------------
# CLI for preprocessing a corpus
# -----------------------------------------------------------------------------


def preprocess_corpus(input_path: str, output_path: str, max_length: int = 77) -> None:
    """Pre-tokenize a text corpus and save the resulting dataset to ``output_path``.

    Parameters
    ----------
    input_path:
        Path to a text file containing one document per line.
    output_path:
        Directory where the processed dataset will be stored using
        ``datasets.Dataset.save_to_disk``.
    max_length:
        Maximum number of tokens per example.
    """

    from datasets import load_dataset

    dataset = load_dataset("text", data_files={"train": input_path})["train"]
    tokenized = dataset.map(lambda x: tokenize_function(x, max_length=max_length))
    tokenized.save_to_disk(output_path)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-tokenize a text corpus")
    parser.add_argument("input", help="Path to text file with one document per line")
    parser.add_argument("output", help="Directory to store the tokenized dataset")
    parser.add_argument(
        "--max-length",
        type=int,
        default=77,
        help="Maximum number of tokens per example",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    args = _build_argparser().parse_args()
    preprocess_corpus(args.input, args.output, max_length=args.max_length)
