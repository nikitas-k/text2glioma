from __future__ import annotations

from pathlib import Path
import csv
from typing import Dict, Mapping


def load_brats_metadata(csv_path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load BraTS metadata from ``csv_path``.

    The CSV file is expected to contain a ``DataID`` column that uniquely
    identifies each subject.  Two fields are extracted when present:

    ``MGMT status`` and ``Dx``.  The latter is parsed to derive ``idh_status``
    when the substring ``"IDH-wildtype"`` or ``"IDH-mutant"`` is present.

    Parameters
    ----------
    csv_path:
        Path to the metadata CSV file.

    Returns
    -------
    dict
        Mapping of ``DataID`` to a dictionary containing ``mgmt_status`` and
        ``idh_status`` entries when available.
    """

    mapping: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_id = (row.get("DataID") or row.get("ID") or "").strip()
            if not data_id:
                continue
            info: Dict[str, str] = {}
            mgmt = row.get("MGMT status") or row.get("MGMT_status")
            if mgmt and mgmt.strip() and mgmt.strip().lower() != "nan":
                info["mgmt_status"] = mgmt.strip()
            dx = row.get("Dx") or ""
            dx_lower = dx.lower()
            if "idh-wildtype" in dx_lower:
                info["idh_status"] = "wildtype"
            elif "idh-mutant" in dx_lower:
                info["idh_status"] = "mutant"
            if info:
                mapping[data_id] = info
    return mapping


def extract_brats_id(filename: str | Path) -> str:
    """Return the BraTS identifier inferred from ``filename``.

    The function tries common BraTS naming conventions and falls back to the
    file stem when no parent directory is available.
    """

    path = Path(filename)
    parent = path.parent.name
    if parent:
        return parent
    stem = path.stem.split("_")[0]
    return stem


__all__ = ["load_brats_metadata", "extract_brats_id"]
