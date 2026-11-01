import json
from pathlib import Path

GENERATED = Path(__file__).resolve().parent.parent / "generated"


def template_dirs(framework: str) -> list[Path]:
    return sorted(p.parent for p in GENERATED.glob(f"*/{framework}/expected.json"))


def load_expected(d: Path) -> dict:
    return json.loads((d / "expected.json").read_text())
