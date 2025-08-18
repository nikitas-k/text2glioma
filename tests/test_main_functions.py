import sys
from pathlib import Path
import pytest

# Attempt to import modules; fall back to None if dependencies are missing
try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src import cli, localisation_cli
except Exception:  # pragma: no cover - import fallback
    cli = localisation_cli = None


@pytest.mark.skipif(cli is None, reason="missing dependencies for cli")
def test_cli_main(monkeypatch, capsys):
    outputs = {}

    def fake_generate_prompt(label):
        outputs['label'] = label
        return 'generated text'

    monkeypatch.setattr(cli, 'generate_prompt', fake_generate_prompt)
    monkeypatch.setattr(sys, 'argv', ['text2glioma', 'label.nii'])

    cli.main()
    assert outputs['label'] == 'label.nii'
    captured = capsys.readouterr()
    assert captured.out.strip() == 'generated text'


@pytest.mark.skipif(localisation_cli is None, reason="missing dependencies for localisation_cli")
def test_localisation_cli_main(monkeypatch, tmp_path):
    healthy = tmp_path / 'healthy'
    diseased = tmp_path / 'diseased'
    mask = tmp_path / 'mask'
    healthy.mkdir()
    diseased.mkdir()

    calls = {}

    def fake_collect(h, d, m, t):
        calls['collect'] = (h, d, m, t)

    def fake_train(d, m, e):
        calls['train'] = (d, m, e)

    monkeypatch.setattr(localisation_cli, 'collect_masks', fake_collect)
    monkeypatch.setattr(localisation_cli, 'train_segmentation', fake_train)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'prog',
            str(healthy),
            str(diseased),
            str(mask),
            '--threshold',
            '0.5',
            '--epochs',
            '2',
        ],
    )

    localisation_cli.main()

    assert calls['collect'] == (healthy, diseased, mask, 0.5)
    assert calls['train'] == (diseased, mask, 2)
