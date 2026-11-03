import importlib.util
import json
import sys
from pathlib import Path

import pytest

GENERATED = Path(__file__).resolve().parent / "generated"
PARITY = GENERATED / "parity"


def load_module(path: Path, name: str):
    """Import a generated preprocess.py under a unique module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parity_cases() -> list[dict]:
    manifest = PARITY / "manifest.json"
    if not manifest.exists():
        pytest.skip("run `npx tsx tests/preprocess/gen-parity.ts` first", allow_module_level=True)
    return json.loads(manifest.read_text())
