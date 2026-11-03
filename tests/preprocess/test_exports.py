"""Export tests: every preset's generated files run, and the CLI output matches the browser preview."""

import subprocess
import sys

import cv2
import numpy as np
import pytest

from conftest import PARITY, load_module, parity_cases

PRESETS = [c for c in parity_cases() if c.get("preset")]


def write_input(case, folder):
    d = PARITY / case["name"]
    img = np.load(d / "input.npy")
    if img.dtype not in (np.uint8, np.uint16):
        pytest.skip("CLI reads image files; float inputs are covered by test_parity")
    folder.mkdir()
    cv2.imwrite(str(folder / "sample.png"), img)
    return img


@pytest.mark.parametrize("case", PRESETS, ids=[c["name"] for c in PRESETS])
def test_cli_matches_preview(case, tmp_path):
    d = PARITY / case["name"]
    write_input(case, tmp_path / "in")
    args = [sys.executable, str(d / "preprocess.py"), str(tmp_path / "in"), str(tmp_path / "out"), "--npy", "--save-steps", "--seed", str(case["seed"])]
    if case["train"]:
        args.append("--train")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = np.load(tmp_path / "out" / "sample.npy")
    assert any((tmp_path / "out" / "sample_steps").iterdir()), "--save-steps wrote nothing"
    if case["serverOnly"] or case.get("error"):
        return
    ref = np.load(d / "browser.npy")
    if case["batch"]:
        out = out[0]
    if case["layout"] == "CHW":
        out = out[0] if out.shape[0] == 1 else np.moveaxis(out, 0, -1)
    assert out.shape == ref.shape
    diff = np.abs(out.astype(np.float64) - ref)
    assert float((diff > max(case["tolerance"], 1e-6)).mean()) <= 0.002, f"max diff {diff.max()}"


@pytest.mark.parametrize("case", [c for c in PRESETS if c["name"].endswith("__inference")], ids=lambda c: c["name"])
def test_albumentations_transforms_run(case):
    pytest.importorskip("albumentations")
    d = PARITY / case["name"]
    sys.path.insert(0, str(d))
    try:
        sys.modules.pop("preprocess", None)
        mod = load_module(d / "transforms_albumentations.py", f"alb_{case['name']}")
        img = np.load(d / "input.npy")
        for train in (False, True):
            out = mod.build_transforms(train=train)(image=img, bboxes=[[10, 10, 40, 40]], labels=[1])["image"]
            assert np.isfinite(np.asarray(out, np.float64)).all()
    finally:
        sys.path.remove(str(d))
        sys.modules.pop("preprocess", None)


@pytest.mark.parametrize("case", [c for c in PRESETS if c["name"].endswith("__inference")], ids=lambda c: c["name"])
def test_torchvision_transforms_and_dataset_run(case, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    d = PARITY / case["name"]
    sys.path.insert(0, str(d))
    try:
        sys.modules.pop("preprocess", None)
        tv = load_module(d / "transforms_torchvision.py", f"tv_{case['name']}")
        img = np.load(d / "input.npy")
        for train in (False, True):
            out = tv.build_transforms(train=train)(tv.to_image(img))
            assert torch.isfinite(out.float()).all()
        # dataset.py: a two-class image folder through ImageFolderDataset.
        ds_mod = load_module(d / "dataset.py", f"ds_{case['name']}")
        for cls in ("cat", "dog"):
            (tmp_path / cls).mkdir()
            cv2.imwrite(str(tmp_path / cls / "a.png"), img)
        ds = ds_mod.ImageFolderDataset(tmp_path, train=True)
        x, y = ds[1]
        assert x.ndim == 3 and x.dtype == torch.float32 and y == 1
    finally:
        sys.path.remove(str(d))
        sys.modules.pop("preprocess", None)
