"""Browser engine (OpenCV.js) vs the exported preprocess.py (cv2), for every step case and every preset.

Deterministic uint8 steps must match exactly; interpolating / float steps within the step's stated tolerance.
Random steps use the same seed on both sides (the browser ports NumPy's PCG64), so they are compared too.
"""

import json

import numpy as np
import pytest

from conftest import PARITY, load_module, parity_cases

CASES = parity_cases()


def to_hwc(out: np.ndarray, layout: str, batch: bool) -> np.ndarray:
    if batch:
        out = out[0]
    if layout == "CHW":
        out = out[0] if out.shape[0] == 1 else np.moveaxis(out, 0, -1)
    return out


def run_python(case: dict):
    d = PARITY / case["name"]
    mod = load_module(d / "preprocess.py", f"pp_{case['name']}")
    img = np.load(d / "input.npy")
    kwargs = {}
    if case["targets"]:
        from images_py import BOXES, KEYPOINTS

        kwargs = {"mask": np.load(d / "mask.npy"), "boxes": BOXES, "keypoints": KEYPOINTS}
    rng = np.random.Generator(np.random.PCG64(case["seed"]))
    return mod, mod.preprocess(img, train=case["train"], rng=rng, **kwargs)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_parity(case):
    assert not case.get("error"), f"browser engine failed: {case['error']}"
    mod, (out, t) = run_python(case)
    if case["serverOnly"]:
        # No browser implementation: the exported code is the reference; check it runs and is well-formed.
        assert isinstance(out, np.ndarray) and out.size > 0
        return
    d = PARITY / case["name"]
    ref = np.load(d / "browser.npy")
    got = to_hwc(out, case["layout"], case["batch"])
    assert got.shape == ref.shape, f"shape: python {got.shape} vs browser {ref.shape}"
    if case["dtype"] == "float16":
        assert got.dtype == np.float16
    else:
        assert got.dtype == ref.dtype, f"dtype: python {got.dtype} vs browser {ref.dtype}"
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    tol = case["tolerance"]
    if got.dtype == np.float16:
        tol = max(tol, 2e-3 * max(1.0, float(np.abs(ref).max())))
    if tol == 0:
        bad = int((diff > 0).sum())
        assert bad == 0, f"{bad} pixel(s) differ, max abs diff {diff.max()}"
    else:
        # Interpolation / float rounding: a tiny fraction of pixels may differ by more than the tolerance at edges.
        frac = float((diff > tol).mean())
        assert frac <= 0.002, f"{frac:.2%} of values differ by more than {tol} (max {diff.max():.4g})"

    if case["targets"]:
        bt = json.loads((d / "browser_targets.json").read_text())
        if (d / "browser_mask.npy").exists():
            bm = np.load(d / "browser_mask.npy")
            assert t.mask.shape == bm.shape
            assert (t.mask != bm).mean() <= 0.001, "mask differs"
        np.testing.assert_allclose(t.boxes.reshape(-1, 5) if len(t.boxes) else np.zeros((0, 5)), np.array(bt["boxes"]).reshape(-1, 5), atol=1e-6)
        np.testing.assert_allclose(t.keypoints.reshape(-1, 2) if len(t.keypoints) else np.zeros((0, 2)), np.array(bt["keypoints"]).reshape(-1, 2), atol=1e-6)
