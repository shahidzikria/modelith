"""Modelith verify service: instantiate generated code, run a dummy forward pass, return real shapes.

POST /verify  {"framework": "pytorch", "files": {"model.py": "..."}}
→ {"ok": true, "params": 11689512, "outputs": {"classifier": [2, 1000]}, "seconds": 1.9}

Generated code is executed in a subprocess with a timeout inside a throwaway directory. Run this
service only inside the provided container (no network, read-only image, resource limits).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

TIMEOUT_S = int(os.environ.get("VERIFY_TIMEOUT", "120"))
MAX_BYTES = 2_000_000

app = FastAPI(title="Modelith verify")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VERIFY_ORIGINS", "http://localhost:5173,http://localhost:4173").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)

TORCH_CHECK = """
import json, sys
sys.path.insert(0, ".")
import torch
import model
net = model.build_model().eval()
with torch.no_grad():
    out = net(**model.example_inputs(2))
def shapes(v):
    if isinstance(v, torch.Tensor):
        return list(v.shape)
    if isinstance(v, dict):
        return {k: shapes(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [shapes(x) for x in v]
    return type(v).__name__
print(json.dumps({"params": sum(p.numel() for p in net.parameters()), "outputs": shapes(out)}))
"""

KERAS_CHECK = """
import json, sys
sys.path.insert(0, ".")
import model
net = model.build_model()
out = net(model.example_inputs(2), training=False)
if not isinstance(out, dict):
    out = {"output": out}
print(json.dumps({"params": int(net.count_params()), "outputs": {k: list(v.shape) for k, v in out.items()}}))
"""


class VerifyRequest(BaseModel):
    framework: str = Field(pattern="^(pytorch|tensorflow)$")
    files: dict[str, str]


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/verify")
def verify(req: VerifyRequest) -> dict:
    if "model.py" not in req.files:
        raise HTTPException(400, "model.py is required")
    if sum(len(v) for v in req.files.values()) > MAX_BYTES:
        raise HTTPException(413, "files too large")
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in req.files.items():
            if Path(name).name != name or not name.endswith((".py", ".yaml", ".json", ".txt")):
                raise HTTPException(400, f"invalid file name: {name}")
            Path(tmp, name).write_text(content, encoding="utf-8")
        script = TORCH_CHECK if req.framework == "pytorch" else KERAS_CHECK
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME")} | {"HF_HUB_OFFLINE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {TIMEOUT_S}s"}
        seconds = round(time.time() - start, 2)
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-4000:], "seconds": seconds}
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"ok": True, **result, "seconds": seconds}
