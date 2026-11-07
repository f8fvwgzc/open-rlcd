"""Shared test setup. Every test runs offline: no network, no GPU, no model downloads.

The process-wide RLCD_HOME points at a throwaway directory so nothing ever touches a real artifacts dir.
Put fixtures specific to one area in that area's test module, not here.
"""
import os
import tempfile

os.environ["RLCD_HOME"] = tempfile.mkdtemp(prefix="rlcd-test-home-")
os.environ.pop("RLCD_API_KEY", None)
os.environ.pop("RLCD_ALLOW_REMOTE", None)
os.environ.pop("RLCD_PROJECT", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RLCD_TEST_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="needs network (set RLCD_TEST_NETWORK=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh RLCD_HOME for one test."""
    monkeypatch.setenv("RLCD_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
