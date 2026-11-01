"""Import a generated model.py (cwd), run a dummy forward pass and print shapes + params as JSON."""

import json
import sys

sys.path.insert(0, ".")

import torch  # noqa: E402

import model  # noqa: E402

torch.manual_seed(0)
net = model.build_model().eval()
with torch.no_grad():
    out = net(**model.example_inputs(2))
shapes = {}
if isinstance(out, torch.Tensor):
    shapes["__single__"] = list(out.shape)
elif isinstance(out, dict):
    shapes = {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
print(json.dumps({"params": sum(p.numel() for p in net.parameters()), "shapes": shapes}))
