"""`gcloud run deploy --source .` uploads app/backend minus .gcloudignore.
gcloud isn't installed in CI, so this applies the same rules (directory
patterns end in "/", the rest are globs on the path or file name) to the
real tree. The authoritative check is still `gcloud meta list-files-for-upload`."""
from __future__ import annotations

import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [p.strip() for p in (ROOT / ".gcloudignore").read_text(encoding="utf-8").splitlines()
            if p.strip() and not p.startswith("#")]


def _ignored(rel: str) -> bool:
    parts = rel.split("/")
    for p in PATTERNS:
        if p.endswith("/"):
            d = p.rstrip("/")
            if d in parts[:-1] or any(fnmatch.fnmatch("/".join(parts[:i]), d) for i in range(1, len(parts))):
                return True
        elif fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(parts[-1], p):
            return True
    return False


def _uploaded() -> list[str]:
    return [f.relative_to(ROOT).as_posix() for f in ROOT.rglob("*") if f.is_file()
            and not _ignored(f.relative_to(ROOT).as_posix())]


def test_secrets_and_local_material_not_uploaded():
    for name in [".env", ".env.local", ".venv/x.py", ".models/m.onnx", "tests/test_x.py",
                 "logs/a.log", "src/__pycache__/a.pyc", ".ruff_cache/x", "latency_bench.py"]:
        assert _ignored(name), name
    up = _uploaded()
    assert not [f for f in up if f.split("/")[-1].startswith(".env")]
    assert not [f for f in up if f.startswith(("tests/", ".venv/", ".models/", "logs/"))]


def test_runtime_files_are_uploaded():
    up = set(_uploaded())
    for need in ["Dockerfile", "requirements.txt", "src/main.py", "storage/chunks.jsonl",
                 "storage/vectors.npy", "storage/bm25.json", "storage/faq_cache.json"]:
        assert need in up, need
