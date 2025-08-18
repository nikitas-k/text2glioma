"""Command-line interface for text prompt generation."""

from __future__ import annotations

import argparse

from .prompt import generate_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a textual tumour description from a label file"
    )
    parser.add_argument("label", help="Path to the NIfTI label file")
    args = parser.parse_args()

    prompt = generate_prompt(args.label)
    print(prompt)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
