"""Redact secrets from what open-rlcd writes to its logs: logged states, receipt audit fields and warnings,
feedback / outcome / confirmation annotations.

A state is evidence for a decision; a credential inside it is not evidence of anything and must not end up in an
append-only log that is never rotated. Before a state is logged, every string in it is scanned for well-known
secret shapes and each match is replaced by `[REDACTED:<kind>]`:

    private_key   PEM private key blocks
    aws_key       AWS access key ids (AKIA..., ASIA...)
    github_token  ghp_ / gho_ / ghu_ / ghs_ / ghr_ tokens and github_pat_ fine-grained tokens
    slack_token   xoxb- / xoxp- / xoxa- / xoxr- / xoxs- tokens
    api_key       sk-... style keys (OpenAI, Anthropic sk-ant-..., Stripe sk_live_ / rk_live_)
    jwt           three base64url segments starting with eyJ
    bearer        "Bearer <token>" in text
    url_password  the password of user:password@host in a URL, up to the last `@` before a `/` or a space; in a value
                  declared to be one URL (`url`: an rlcd.yaml key such as the trading pack's quant_url), the whole user
                  info, user name included, up to its last `@`, whatever it holds
    assignment    NAME = value where NAME contains api_key / secret / token / password / access_key ... (the value only;
                  also DB_PASSWORD=..., "client_secret": "...", export API_TOKEN=...)
    secret_field  a string stored under a key that names a secret ({"password": "..."}), replaced whole

Optional (`serve.redact: [secrets, pii]`): `pii` also masks e-mail addresses and card numbers that pass the Luhn
check. Redaction never touches keys, numbers or structure, so a redacted state still trains (feedback rows are
re-featurised at the next train) and still replays; the digest in a receipt is taken from the state *as sent*, so
a caller can still prove which state a decision judged.

This is a safety net with known blind spots (a bare high-entropy string with no recognisable shape is kept): the
first defence is a state builder that never puts secrets into a state (the guide: "secrets not needed to judge").
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

MAX_DEPTH = 64

_SECRETS: list[tuple[str, re.Pattern[str]]] = [
    # a whole block, or a block cut before its END line (a chunk boundary, a truncated paste): the header and the
    # base64 material after it
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----(?:.*?-----END (?:[A-Z0-9]+ )*PRIVATE "
                               r"KEY-----|[A-Za-z0-9+/=\s:,\-]*)", re.S)),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,200}\b")),
    ("api_key", re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,200}|(?:sk|rk)_live_[A-Za-z0-9]{16,200})\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("bearer", re.compile(r"(?i)(?<=\bbearer )[A-Za-z0-9\-._~+/]{16,}=*")),
    # the password runs to the last `@` before a `/` or a space, so an unencoded `@` in it is not left behind
    ("url_password", re.compile(r"(?<=://)([^\s:/@]{1,64}):([^\s/]{1,128})(?=@)")),
    # NAME = value, NAME: value, "NAME": "value", export NAME=value, where NAME contains a secret word anywhere
    # (DB_PASSWORD, AWS_SECRET_ACCESS_KEY, client_secret, my_api_key, API_TOKEN): only the value is replaced
    # (name parts are bounded and the suffix possessive: constant work per start position, no catastrophic
    # backtracking on adversarial input; quoted values may contain spaces)
    ("assignment", re.compile(r"(?i)(?<![A-Za-z0-9])[\"']?[A-Za-z0-9_\-.]{0,64}?"
                              r"(?:api[_-]?key|secret|token|passw(?:or)?d|pwd|access[_-]?key|private[_-]?key)"
                              r"[A-Za-z0-9_\-]{0,64}+[\"']?(\s{0,8}[:=]\s{0,8})"
                              r"(\"[^\"\n]{6,200}\"|'[^'\n]{6,200}'|[^\s\"',;}]{6,200})")),
    ("token_shapes", re.compile(r"\b(?:hf_[A-Za-z0-9]{30,64}|AIza[0-9A-Za-z_\-]{35}|glpat-[A-Za-z0-9_\-]{20,64})\b")),
    ("basic_auth", re.compile(r"(?i)(?<=\bbasic )[A-Za-z0-9+/]{16,}={0,2}")),
]
# a value stored under a key that names a secret is a secret, whatever it looks like (the most common shape in a
# JSON state): the key, lower-cased without separators, ends with one of these
_SECRET_KEYS = ("password", "passwd", "pwd", "passphrase", "secret", "token", "apikey", "accesskey", "privatekey",
                "secretkey", "clientsecret", "credential", "credentials", "authorization", "cookie", "sessionid")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# the start of a value that is one URL (`url`): leading whitespace, a scheme and its slashes (or backslashes), or `//`
_URL_START = re.compile(r"\s*(?:[A-Za-z][A-Za-z0-9+.\-]{0,31}:[/\\]+|//)")


def _userinfo(text: str) -> str | None:
    """`text`, one URL (`url`), with its whole user info redacted, however malformed: everything from the start of the
    URL (`_URL_START`; without a scheme and its slashes, the first character that is not whitespace) to its last `@`,
    so a user name, a password or a token alone is redacted whole, whatever `@`, space, `/`, `?`, `#` or line break it
    holds, and whatever follows the host. None when there is no `@`, or nothing before it."""
    start = _URL_START.match(text)
    begin = start.end() if start else len(text) - len(text.lstrip())
    userinfo, _, rest = text[begin:].rpartition("@")
    if not userinfo:
        return None
    return f"{text[:begin]}[REDACTED:url_password]@{rest}"


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# the shapes that are secrets whatever surrounds them (output screening uses only these: a heuristic such as
# `assignment` also matches documentation like "Password: minimum twelve characters")
SHAPES = ("private_key", "aws_key", "github_token", "slack_token", "api_key", "jwt", "bearer", "url_password",
          "token_shapes", "basic_auth")


def redact_text(text: str, kinds: Iterable[str] = ("secrets",), skip: Iterable[str] = (),
                url: bool = False) -> tuple[str, list[str]]:
    """(text with every match replaced, the kinds found). `skip`: secret kinds not to look for. `url`: `text` is one
    URL whose user info may carry credentials (an rlcd.yaml key declared so, for `rlcd config show` and
    `rlcd config check`), so its whole user info is redacted (`_userinfo`); in free text a URL's password ends at a
    `/` or a space, and its user name is kept."""
    kinds = set(kinds)
    skip = set(skip)
    found: list[str] = []
    out = text
    if "secrets" in kinds:
        for kind, pat in _SECRETS:
            if kind in skip:
                continue
            if kind == "url_password":
                new = _userinfo(out) if url else None
                new = new or pat.sub(lambda m: f"{m.group(1)}:[REDACTED:url_password]", out)
            elif kind == "assignment":
                new = pat.sub(lambda m: m.group(0)[: m.start(2) - m.start(0)] + "[REDACTED:assignment]", out)
            else:
                new = pat.sub(f"[REDACTED:{kind}]", out)
            if new != out:
                found.append(kind)
                out = new
    if "pii" in kinds:
        new = _EMAIL.sub("[REDACTED:email]", out)
        if new != out:
            found.append("email")
            out = new

        def card(m: re.Match) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            return "[REDACTED:card]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

        new = _CARD.sub(card, out)
        if new != out:
            found.append("card")
            out = new
    return out, found


def _secret_key(key: Any) -> bool:
    k = re.sub(r"[^a-z0-9]", "", str(key).lower())
    return bool(k) and k.endswith(_SECRET_KEYS)


def redact(obj: Any, kinds: Iterable[str] = ("secrets",), url: bool = False) -> tuple[Any, list[str]]:
    """A copy of a JSON-like value with every string redacted, and the sorted kinds that were found. Structure,
    keys and non-string values are unchanged, except that (with `secrets`) a string stored under a key that names a
    secret (password, api_key, client_secret, token, ...) is replaced whole; nesting deeper than MAX_DEPTH is
    replaced by a marker. `url`: each string is one URL (`redact_text`)."""
    kinds = tuple(kinds)
    by_key = "secrets" in kinds
    found: set[str] = set()

    def walk(v: Any, depth: int, under: bool = False) -> Any:
        if depth > MAX_DEPTH:
            found.add("too_deep")
            return "[REDACTED:too_deep]"
        if isinstance(v, str):
            if under and v:                    # a string anywhere below a key that names a secret
                found.add("secret_field")
                return "[REDACTED:secret_field]"
            t, f = redact_text(v, kinds, url=url)
            found.update(f)
            return t
        if isinstance(v, dict):
            return {k: walk(x, depth + 1, under or (by_key and _secret_key(k))) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x, depth + 1, under) for x in v]
        return v

    return walk(obj, 0), sorted(found)


def kinds_from_config(value: Any) -> tuple[str, ...]:
    """serve.redact: true (default) -> secrets; false -> nothing; a list -> those kinds (secrets, pii)."""
    if value is None or value is True:
        return ("secrets",)
    if value is False:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"serve.redact must be true, false or a list of kinds (secrets, pii), got {value!r}")
    kinds = tuple(str(v) for v in value)
    unknown = sorted(set(kinds) - {"secrets", "pii"})
    if unknown:
        raise ValueError(f"serve.redact: unknown kinds {unknown}; use secrets and/or pii")
    return kinds
