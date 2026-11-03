"""Modelith Preprocessing Studio server: the Python reference for pipeline previews.

POST /preprocess/run     run the generated preprocess.py from step `start` onward on an image (+ mask / boxes /
                         keypoints, + the browser's RNG state) and return every intermediate step
POST /preprocess/decode  decode an uploaded file exactly like cv2.imdecode (16-bit PNG / TIFF, alpha, EXIF)
GET  /health

Like the verify service, /preprocess/run executes submitted code (in a subprocess with a timeout, inside a
throwaway directory). Run it only on your own machine or in the provided container, never exposed publicly.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

TIMEOUT_S = int(os.environ.get("STUDIO_TIMEOUT", "120"))
MAX_CODE = 2_000_000
MAX_IMAGE = 200_000_000

app = FastAPI(title="Modelith Preprocessing Studio server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("STUDIO_ORIGINS", "http://localhost:5173,http://localhost:4173").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)


class Wire(BaseModel):
    shape: list[int]
    dtype: str = Field(pattern="^(uint8|uint16|float16|float32|float64|int32)$")
    data: str


class RngState(BaseModel):
    state: str
    inc: str


class RunRequest(BaseModel):
    code: str
    image: Wire
    mask: Wire | None = None
    boxes: list[list[float]] = []
    keypoints: list[list[float]] = []
    start: int = 0
    train: bool = False
    seed: int = 0
    rng_state: RngState | None = None
    timing_runs: int = Field(0, ge=0, le=20)


def from_wire(w: Wire) -> np.ndarray:
    raw = base64.b64decode(w.data)
    arr = np.frombuffer(raw, dtype=np.dtype(w.dtype).newbyteorder("<"))
    if arr.size != int(np.prod(w.shape)):
        raise HTTPException(400, f"array size {arr.size} does not match shape {w.shape}")
    return arr.reshape(w.shape).astype(w.dtype)


# Executed in a subprocess next to the submitted preprocess.py.
RUNNER = r"""
import base64, json, sys, time
sys.path.insert(0, ".")
import numpy as np
import preprocess as P

req = json.load(open("request.json"))
img = np.load("image.npy")
mask = np.load("mask.npy") if req["has_mask"] else None


def make_rng():
    rng = np.random.Generator(np.random.PCG64(req["seed"]))
    if req["rng_state"]:
        rng.bit_generator.state = {"bit_generator": "PCG64", "state": {"state": int(req["rng_state"]["state"]), "inc": int(req["rng_state"]["inc"])}, "has_uint32": 0, "uinteger": 0}
    return rng


def wire(a):
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": base64.b64encode(a.astype(a.dtype.newbyteorder("<")).tobytes()).decode()}


def run(collect):
    t = P.Targets((img.shape[1], img.shape[0]), None if mask is None else mask.copy(), req["boxes"] or None, req["keypoints"] or None)
    rng = make_rng()
    out = []
    last = [time.perf_counter()]

    def on_step(i, name, space, x):
        now = time.perf_counter()
        ms = (now - last[0]) * 1000
        if collect:
            out.append({"index": i, "name": name, "ms": ms, "image": wire(x), "boxes": t.boxes[:, :5].tolist() if len(t.boxes) else [], "keypoints": t.keypoints.tolist(), "mask": None if t.mask is None else wire(t.mask)})
        else:
            out.append({"index": i, "name": name, "ms": ms})
        last[0] = time.perf_counter()

    P.run_steps(img.copy(), t, rng, train=req["train"], start=req["start"], on_step=on_step)
    return out


steps = run(True)
for _ in range(req["timing_runs"]):
    again = run(False)
    for s, a in zip(steps, again):
        s.setdefault("runs", []).append(a["ms"])
for s in steps:
    if "runs" in s:
        s["ms"] = float(np.median(s.pop("runs")))
json.dump({"ok": True, "steps": steps, "total_ms": sum(s["ms"] for s in steps)}, open("result.json", "w"))
"""


@app.get("/health")
def health() -> dict:
    return {"ok": True, "opencv": cv2.__version__, "contrib": hasattr(cv2, "xphoto")}


@app.post("/preprocess/run")
def run(req: RunRequest) -> dict:
    if len(req.code) > MAX_CODE or len(req.image.data) > MAX_IMAGE:
        raise HTTPException(413, "request too large")
    if "def run_steps(" not in req.code:
        raise HTTPException(400, "code is not a Modelith preprocess.py")
    img = from_wire(req.image)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "preprocess.py").write_text(req.code, encoding="utf-8")
        np.save(Path(tmp, "image.npy"), img)
        if req.mask is not None:
            np.save(Path(tmp, "mask.npy"), from_wire(req.mask))
        Path(tmp, "request.json").write_text(
            json.dumps(
                {
                    "has_mask": req.mask is not None,
                    "boxes": req.boxes,
                    "keypoints": req.keypoints,
                    "start": req.start,
                    "train": req.train,
                    "seed": req.seed,
                    "rng_state": req.rng_state.model_dump() if req.rng_state else None,
                    "timing_runs": req.timing_runs,
                }
            )
        )
        Path(tmp, "runner.py").write_text(RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "runner.py"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME")},
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {TIMEOUT_S}s"}
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-4000:]}
        return json.loads(Path(tmp, "result.json").read_text())


FLAGS = {
    "COLOR": cv2.IMREAD_COLOR,
    "GRAYSCALE": cv2.IMREAD_GRAYSCALE,
    "UNCHANGED": cv2.IMREAD_UNCHANGED,
    "ANYDEPTH": cv2.IMREAD_ANYDEPTH,
}


@app.post("/preprocess/decode")
async def decode(file: UploadFile = File(...), flag: str = Form("UNCHANGED")) -> dict:
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, FLAGS.get(flag, cv2.IMREAD_UNCHANGED))
    if img is None:
        raise HTTPException(400, f"cannot decode {file.filename}")
    if img.dtype not in (np.uint8, np.uint16, np.float32):
        img = img.astype(np.float32)
    a = np.ascontiguousarray(img)
    return {"image": {"shape": list(a.shape), "dtype": str(a.dtype), "data": base64.b64encode(a.tobytes()).decode()}}
