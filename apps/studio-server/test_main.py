"""Studio server tests. Uses the preprocess.py files written by tests/preprocess/gen-parity.ts."""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
PARITY = Path(__file__).resolve().parents[2] / "tests" / "preprocess" / "generated" / "parity"


def wire(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": base64.b64encode(a.tobytes()).decode()}


def unwire(w: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(w["data"]), dtype=w["dtype"]).reshape(w["shape"])


def case(name: str) -> Path:
    d = PARITY / name
    if not d.exists():
        pytest.skip("run `npx tsx tests/preprocess/gen-parity.ts` first")
    return d


def test_health():
    r = client.get("/health").json()
    assert r["ok"] and r["opencv"].startswith("4.")


def test_run_matches_browser_preview():
    d = case("preset_imagenet_cls__inference")
    img = np.load(d / "input.npy")
    r = client.post("/preprocess/run", json={"code": (d / "preprocess.py").read_text(), "image": wire(img), "start": 0, "train": False, "seed": 0}).json()
    assert r["ok"], r.get("error")
    assert [s["name"] for s in r["steps"]] == ["01_resize", "02_crop", "03_convert_color", "04_standardize", "05_to_chw"]
    final = unwire(r["steps"][-1]["image"])
    assert final.shape == (3, 224, 224) and final.dtype == np.float32
    ref = np.load(d / "browser.npy")
    assert np.abs(np.moveaxis(final, 0, -1) - ref).max() < 1e-4


def test_run_from_middle_with_targets_and_rng_state():
    d = case("preset_imagenet_train__train")
    img = np.load(d / "input.npy")
    mask = np.load(d / "mask.npy")
    rng = np.random.Generator(np.random.PCG64(5))
    rng.random(3)  # the browser hands over its state after some draws
    st = rng.bit_generator.state["state"]
    body = {
        "code": (d / "preprocess.py").read_text(),
        "image": wire(img),
        "mask": wire(mask),
        "boxes": [[10, 10, 50, 40, 1]],
        "keypoints": [[20, 20]],
        "start": 3,
        "train": True,
        "seed": 0,
        "rng_state": {"state": str(st["state"]), "inc": str(st["inc"])},
        "timing_runs": 2,
    }
    r = client.post("/preprocess/run", json=body).json()
    assert r["ok"], r.get("error")
    assert r["steps"][0]["index"] >= 3
    assert all(s["ms"] >= 0 for s in r["steps"]) and r["total_ms"] >= 0
    assert r["steps"][0]["mask"] is not None


def test_bad_code_is_rejected():
    assert client.post("/preprocess/run", json={"code": "print(1)", "image": wire(np.zeros((2, 2), np.uint8))}).status_code == 400


def test_error_is_reported():
    code = "def run_steps(img, t, rng, train=False, start=0, on_step=None):\n    raise RuntimeError('boom')\n"
    code += "class Targets:\n    def __init__(self, *a): pass\n"
    r = client.post("/preprocess/run", json={"code": code, "image": wire(np.zeros((4, 4), np.uint8))}).json()
    assert not r["ok"] and "boom" in r["error"]


def test_decode_16bit_png():
    img = (np.arange(64 * 48, dtype=np.uint16) * 13).reshape(48, 64)
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/preprocess/decode", files={"file": ("x.png", buf.tobytes(), "image/png")}, data={"flag": "UNCHANGED"})
    out = unwire(r.json()["image"])
    assert out.dtype == np.uint16 and np.array_equal(out, img)
