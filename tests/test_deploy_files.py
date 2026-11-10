"""The deploy files agree with the service and with docs/deployment.md. No Docker: the files are only read."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from open_rlcd.serve.api import DEFAULT_MAX_BODY_BYTES

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "docs" / "deployment.md"

# Caddy reads `request_body max_size` with go-humanize's ParseBytes: "MB" is 10^6 bytes, "MiB" is 2^20
_UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "kib": 2**10, "mib": 2**20, "gib": 2**30}


def _caddy_bytes(text: str) -> int:
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]*)", text.strip())
    assert m, text
    return int(m.group(1)) * _UNITS[m.group(2).lower()]


def test_the_proxy_refuses_exactly_the_bodies_the_service_refuses():
    caddy = (REPO / "deploy" / "Caddyfile").read_text(encoding="utf-8")
    sizes = re.findall(r"^\s*max_size\s+(\S+)\s*$", caddy, re.M)
    assert sizes == ["16MiB"]
    assert _caddy_bytes(sizes[0]) == DEFAULT_MAX_BODY_BYTES == 16_777_216
    # what it said before: 16,000,000 bytes, so the proxy refused bodies the service accepts
    assert _caddy_bytes("16MB") < DEFAULT_MAX_BODY_BYTES


def test_the_guide_puts_the_project_overlay_into_every_compose_command():
    """A container recreated without deploy/compose.project.yaml has no project and serves on the code defaults
    (rollout full, ...), so the guide sets COMPOSE_FILE in .env instead of a `-f` that one command can miss."""
    overlay = yaml.safe_load((REPO / "deploy" / "compose.project.yaml").read_text(encoding="utf-8"))
    service = overlay["services"]["rlcd"]
    assert service["environment"]["RLCD_PROJECT"] == "/project"
    assert any(str(v).endswith(":/project:ro") for v in service["volumes"])
    guide = GUIDE.read_text(encoding="utf-8")
    assert re.findall(r"^COMPOSE_FILE=(\S+)$", guide, re.M) == ["compose.yaml:deploy/compose.project.yaml"]
    assert all((REPO / f).is_file() for f in ("compose.yaml", "deploy/compose.project.yaml"))
    # an image set in front of one command is gone for the next one: RLCD_IMAGE belongs in .env too
    assert not re.search(r"RLCD_IMAGE=\S+ docker compose", guide)
