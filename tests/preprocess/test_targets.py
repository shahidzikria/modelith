"""Target round trips: transform a synthetic mask / box set with each geometric step, undo it with
postprocess_mask / postprocess_boxes / postprocess_keypoints, and compare with the original (IoU >= 0.99).

Only the part of the image that survives the step (e.g. inside a crop) can come back, so masks are compared
inside that "valid" region. Box round trips are exact for axis-aligned steps; rotations / warps enlarge boxes
(enclosing box of rotated corners), so for those only the mask and keypoints are checked.
"""

import cv2
import numpy as np
import pytest

from conftest import PARITY, load_module, parity_cases

CASES = [c for c in parity_cases() if c.get("geometry") in ("axis", "warp") and not c["serverOnly"] and c["readFlag"] in ("COLOR", "GRAYSCALE", "UNCHANGED")]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 1.0


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_round_trip(case):
    d = PARITY / case["name"]
    mod = load_module(d / "preprocess.py", f"rt_{case['name']}")
    img = np.load(d / "input.npy")
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    mask = (((xx - 40) ** 2 + (yy - 44) ** 2) < 22**2).astype(np.uint8)
    mask[20:70, 76:115] = 2
    boxes = np.array([[18.0, 22, 62, 66, 1], [76, 21, 115, 70, 2], [5, 60, 30, 90, 3]])
    kps = np.array([[40.0, 44], [95.5, 45.25], [12, 80]])

    def run(m, b=None, k=None):
        return mod.preprocess(img, mask=m, boxes=b, keypoints=k, train=case["train"], rng=np.random.Generator(np.random.PCG64(case["seed"])))

    _, t = run(mask, boxes, kps)
    _, tv = run(np.ones((h, w), np.uint8))
    valid = mod.postprocess_mask(tv.mask, tv.history) > 0
    back = mod.postprocess_mask(t.mask, t.history)
    assert back.shape == mask.shape
    # Downscaling a label mask loses boundary pixels for good (nearest neighbour); compare outside a band of
    # ~1/scale pixels around the label edges, so the test checks the geometry rather than resampling loss.
    scale = float(np.prod([np.sqrt(abs(np.linalg.det(r["H"][:2, :2]))) for r in t.history])) if t.history else 1.0
    r = int(np.ceil(1 / min(scale, 1.0)))
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    for label in (1, 2):
        m = (mask == label).astype(np.uint8)
        core = valid & ~((cv2.dilate(m, kernel) > 0) & ~(cv2.erode(m, kernel) > 0))
        a, b = (mask == label) & core, (back == label) & core
        if a.sum() > 50:
            assert iou(a, b) >= 0.99, f"mask label {label}: IoU {iou(a, b):.4f}"

    # Keypoints that stayed inside the image map back to where they were.
    kp_back = mod.postprocess_keypoints(t.keypoints, t.history)
    inside = [i for i, (x, y) in enumerate(kps) if valid[int(y), int(x)]]
    assert np.allclose(kp_back[inside, :2], kps[inside], atol=1e-6)

    if case["geometry"] == "axis":
        back_boxes = mod.postprocess_boxes(t.boxes, t.history)
        for bb in back_boxes:
            orig = boxes[boxes[:, 4] == bb[4]][0]
            # Compare with the original box clipped to the region that survived the step.
            vy, vx = np.nonzero(valid)
            clip = [max(orig[0], vx.min()), max(orig[1], vy.min()), min(orig[2], vx.max() + 1), min(orig[3], vy.max() + 1)]
            assert box_iou(bb[:4], clip) >= 0.99, f"box {bb} vs {clip}"
