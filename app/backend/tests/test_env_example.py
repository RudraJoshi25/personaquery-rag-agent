import re
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
# Matches both direct os.getenv("X", ...) calls and calls through the
# _bool("X", ...) helper in core/config.py, which itself wraps os.getenv -
# the helper's own indirection would otherwise hide those vars from this check.
GETENV_RE = re.compile(r"(?:os\.getenv|_bool)\(\s*[\"']([A-Z0-9_]+)[\"']")


def _used_env_vars() -> set[str]:
    used = set()
    for p in (BACKEND_ROOT / "src").rglob("*.py"):
        used |= set(GETENV_RE.findall(p.read_text(encoding="utf-8")))
    return used


def test_env_example_exists():
    assert (BACKEND_ROOT / ".env.example").exists()


def test_env_example_documents_all_used_vars():
    documented = set()
    for line in (BACKEND_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            documented.add(line.split("=", 1)[0].strip())
    missing = _used_env_vars() - documented
    assert not missing, f"Undocumented env vars: {sorted(missing)}"
