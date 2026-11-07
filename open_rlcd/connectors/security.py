"""Host allow-listing for connectors and judges: loopback and unix sockets only, unless remote is allowed.

Why: a project config copied from a teammate, or a URL pasted into a request, must not make this process send
documents, queries or embeddings to some other machine. Remote hosts need an explicit opt-in: the environment
variable `RLCD_ALLOW_REMOTE=1`, `security.allow_remote: true` in rlcd.yaml, or `allow_remote: true` in a pack's
config.

What this does not do: it does not resolve DNS (a name that happens to resolve to 127.0.0.1 is still refused;
only `localhost` and literal loopback addresses pass), and it is not a firewall: code that opens sockets without
calling `check_host` is not covered.
"""
from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import parse_qs, urlparse

LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})
HINT = "set RLCD_ALLOW_REMOTE=1 or security.allow_remote: true in rlcd.yaml to allow it"


def _env_allows() -> bool:
    return os.environ.get("RLCD_ALLOW_REMOTE") == "1"


def is_loopback(host: str) -> bool:
    h = host.strip().strip("[]").lower().rstrip(".")
    if h in LOOPBACK_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def _host_of(target: str) -> str:
    """The host a URL / DSN / `host:port` / socket path points at; "unix" for unix sockets."""
    s = target.strip()
    if s.startswith(("/", "./", "../", "~")) or s.endswith(".sock"):
        return "unix"
    if "://" in s:
        u = urlparse(s)
        if "unix" in u.scheme.lower():
            return "unix"
        if u.hostname:
            return u.hostname
        q = parse_qs(u.query).get("host")        # libpq style: postgresql:///db?host=/var/run/postgresql
        if q and q[0]:
            return "unix" if q[0].startswith("/") else q[0]
        return "unix" if u.scheme.lower().startswith("postgres") else ""
    if s.lower().startswith("unix:"):
        return "unix"
    if s.startswith("[") or s.count(":") <= 1:   # host, host:port, [v6]:port
        return urlparse("//" + s).hostname or ""
    return s                                    # bare IPv6 literal


def check_host(url_or_host: str, allow_remote: bool = False) -> str:
    """Return the host `url_or_host` points at if it is loopback / a unix socket, or if remote hosts are allowed
    (`allow_remote` or env `RLCD_ALLOW_REMOTE=1`); otherwise raise ValueError naming the opt-in."""
    if not isinstance(url_or_host, str) or not url_or_host.strip():
        raise ValueError("empty host")
    host = _host_of(url_or_host)
    if not host:
        raise ValueError(f"cannot tell which host {url_or_host!r} points at")
    if host == "unix" or is_loopback(host):
        return host
    if allow_remote or _env_allows():
        return host
    raise ValueError(f"refusing non-loopback host {host!r}: connectors and judges only talk to localhost by default; "
                     f"{HINT}")


def remote_allowed(project_or_config: Any = None) -> bool:
    """True when remote hosts are allowed by the environment, a `Project` (`allow_remote`), or a config dict
    (`allow_remote`, `security.allow_remote` or `pack_config.allow_remote`)."""
    if _env_allows():
        return True
    if project_or_config is None:
        return False
    if isinstance(project_or_config, dict):
        d = project_or_config
        sec = d.get("security") if isinstance(d.get("security"), dict) else {}
        pc = d.get("pack_config") if isinstance(d.get("pack_config"), dict) else {}
        return bool(d.get("allow_remote") or sec.get("allow_remote") or pc.get("allow_remote"))
    allowed = bool(getattr(project_or_config, "allow_remote", False))
    pack_config = getattr(project_or_config, "pack_config", None)
    if isinstance(pack_config, dict) and pack_config.get("allow_remote"):
        allowed = True
    return allowed
