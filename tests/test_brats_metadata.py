import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataloaders.metadata import load_brats_metadata, extract_brats_id


def test_load_brats_metadata(tmp_path):
    csv_content = (
        "DataID,Dx,MGMT status\n"
        "id1,\"Astrocytoma, IDH-mutant\",positive\n"
        "id2,\"Glioblastoma, IDH-wildtype\",negative\n"
    )
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(csv_content)
    meta = load_brats_metadata(csv_path)
    assert meta["id1"]["mgmt_status"] == "positive"
    assert meta["id1"]["idh_status"] == "mutant"
    assert meta["id2"]["idh_status"] == "wildtype"


def test_extract_brats_id():
    assert extract_brats_id("/root/id1/id1_seg.nii.gz") == "id1"
    assert extract_brats_id("id2_0000.nii.gz") == "id2"

try:
    from dataloaders.monai_brats_dataset import PromptFromLabeld
except Exception:  # pragma: no cover - missing optional deps
    PromptFromLabeld = None


@pytest.mark.skipif(PromptFromLabeld is None, reason="monai not installed")
def test_prompt_from_labeld_uses_metadata(monkeypatch):
    calls = {}

    def fake_generate_prompt(filename, *, mgmt_status=None, idh_status=None):
        calls["args"] = (mgmt_status, idh_status)
        return "text"

    monkeypatch.setattr(
        "dataloaders.monai_brats_dataset.generate_prompt", fake_generate_prompt
    )
    meta = {"id1": {"mgmt_status": "positive", "idh_status": "wildtype"}}
    t = PromptFromLabeld(keys="label", metadata=meta)
    out = t({"label": None, "label_meta_dict": {"filename_or_obj": "id1_seg.nii.gz"}})
    assert out["text"] == "text"
    assert calls["args"] == ("positive", "wildtype")
