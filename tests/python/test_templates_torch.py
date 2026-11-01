"""Execute every generated PyTorch template: smoke test, shape/param check, 2 training steps, export.

Run `npx tsx tests/generate-fixtures.ts --no-pretrained` first.
"""

import json
import subprocess
import tempfile
import sys
from pathlib import Path

import pytest

from conftest import load_expected, template_dirs

DIRS = template_dirs("pytorch")
HERE = Path(__file__).resolve().parent


def run(args: list[str], cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        pytest.fail(f"{' '.join(args)} failed in {cwd.name}:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}")
    return proc


@pytest.mark.parametrize("d", DIRS, ids=[d.parent.name for d in DIRS])
def test_model_matches_ui(d: Path) -> None:
    expected = load_expected(d)
    run(["model.py"], d)
    info = json.loads(run([str(HERE / "check_model.py")], d).stdout.strip().splitlines()[-1])
    # Parameter count predicted by the UI must match the real model (exact for primitives, ±3% for formulas/tables).
    real, predicted = info["params"], expected["params"]
    assert abs(real - predicted) <= max(16, 0.03 * real), f"params: UI {predicted:,} vs real {real:,}"
    if expected["wrapped"]:
        return
    heads = expected["heads"]
    shapes = info["shapes"]
    for name, head in heads.items():
        if head["task"] in ("detection", "instance", "contrastive"):
            continue
        got = shapes["__single__"] if "__single__" in shapes else shapes[name]
        assert got == [2, *head["dims"]], f"{name}: UI predicted {[2, *head['dims']]}, real {got}"


@pytest.mark.parametrize("d", DIRS, ids=[d.parent.name for d in DIRS])
def test_train_two_steps_and_export(d: Path) -> None:
    # Checkpoints of large models are hundreds of MB: always delete them afterwards.
    with tempfile.TemporaryDirectory() as tmp:
        check_training(d, Path(tmp) / "ckpt")


def check_training(d: Path, out: Path) -> None:
    proc = run(["train.py", "--epochs", "1", "--max-steps", "2", "--batch-size", "2", "--out", str(out)], d)
    assert (out / "last.pt").exists(), proc.stdout
    expected = load_expected(d)
    tasks = {h["task"] for h in expected["heads"].values()}
    if not expected["wrapped"] and not tasks & {"detection", "instance"}:
        assert (out / "model.onnx").exists(), f"ONNX export failed:\n{proc.stdout[-3000:]}"
