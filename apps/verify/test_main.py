from pathlib import Path

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "generated" / "simple-cnn" / "pytorch" / "model.py"


def test_verify_simple_cnn() -> None:
    r = client.post("/verify", json={"framework": "pytorch", "files": {"model.py": FIXTURE.read_text()}})
    body = r.json()
    assert body["ok"], body
    assert body["outputs"] == [2, 10]
    assert body["params"] > 0


def test_rejects_path_traversal() -> None:
    r = client.post("/verify", json={"framework": "pytorch", "files": {"model.py": "", "../x.py": ""}})
    assert r.status_code == 400
