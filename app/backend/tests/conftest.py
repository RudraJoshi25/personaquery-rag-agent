import pytest


@pytest.fixture(autouse=True)
def _no_model_preload(monkeypatch):
    # API tests stub out retrieval; don't load ~600MB of models at startup.
    monkeypatch.setenv("PRELOAD_MODELS", "false")
