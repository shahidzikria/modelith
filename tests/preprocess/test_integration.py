"""A Modelith model linked to a preprocessing pipeline trains on real files through preprocess.py / dataset.py.

Run `npx tsx tests/preprocess/gen-integration.ts` first. Needs torch (CPU is fine).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent / "generated" / "integration"
CASES = sorted(p.name for p in ROOT.iterdir()) if ROOT.exists() else []
if not CASES:
    pytest.skip("run `npx tsx tests/preprocess/gen-integration.ts` first", allow_module_level=True)


def make_data(dst: Path, task: str, rng):
    for split in ("train", "val"):
        for k in range(4):
            img = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
            if task == "classification":
                cls = ("cat", "dog")[k % 2]
                (dst / split / cls).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst / split / cls / f"{k}.png"), img)
            else:
                (dst / split / "images").mkdir(parents=True, exist_ok=True)
                (dst / split / "masks").mkdir(parents=True, exist_ok=True)
                mask = np.zeros((90, 120), np.uint8)
                mask[20:60, 30:90] = 1
                cv2.imwrite(str(dst / split / "images" / f"{k}.png"), img)
                cv2.imwrite(str(dst / split / "masks" / f"{k}.png"), mask)


@pytest.mark.parametrize("name", CASES)
def test_train_two_steps_on_real_files(name, tmp_path):
    src = ROOT / name
    work = tmp_path / name
    shutil.copytree(src, work)
    case = json.loads((work / "case.json").read_text())
    train_py = (work / "train.py").read_text()
    assert "import dataset as D" in train_py and "D.ImageFolderDataset" in train_py or "D.SegmentationDataset" in train_py
    make_data(work / "data", case["task"], np.random.default_rng(0))
    proc = subprocess.run(
        [sys.executable, "train.py", "--epochs", "1", "--max-steps", "2", "--batch-size", "2", "--no-export"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert "using synthetic data" not in proc.stdout, "should have loaded data/ through dataset.py"


@pytest.mark.parametrize("name", CASES)
def test_falls_back_to_synthetic_without_data(name, tmp_path):
    work = tmp_path / name
    shutil.copytree(ROOT / name, work)
    proc = subprocess.run([sys.executable, "train.py", "--epochs", "1", "--max-steps", "1", "--batch-size", "2", "--no-export"], cwd=work, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "using synthetic data" in proc.stdout
