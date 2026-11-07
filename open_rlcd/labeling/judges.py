"""LLM judges: weak labels for LabelItems, one call per batch of items covering every head of their view.

    claude-cli[:model]           the user's own Claude Code subscription through the `claude` executable
    anthropic[:model]            the Anthropic API, key from ANTHROPIC_API_KEY (default model claude-haiku-5-5)
    ollama:<model>               a local Ollama server (http://127.0.0.1:11434)
    openai:<base_url>|<model>    any OpenAI-compatible chat-completions server (llama.cpp, vLLM, LM Studio, ...)

Every judge returns (answers, problems): answers carry canonical labels validated against the HeadSpec;
problems are human-readable reasons for anything skipped. A problem that will not go away by itself (missing
executable or key, rejected key, unreachable server, timeout) raises JudgeError instead, so a bootstrap run stops
early rather than failing the same way on every batch; it resumes where it stopped.

Security: HTTP judges refuse non-loopback hosts unless remote hosts are allowed (allow_remote=True,
RLCD_ALLOW_REMOTE=1, or security.allow_remote / pack_config.allow_remote in rlcd.yaml), never follow redirects,
and read secrets only from environment variables. The CLI judge runs without a shell (the prompt goes to stdin),
with every Claude Code tool and MCP server disabled, in an empty temporary directory (see ClaudeCliJudge). The
prompt tells every judge that item text is untrusted data, never an instruction, and neutralises the prompt's
own tags inside it (prompts.quote).

What this does not do: make the judge right. Judge labels only train base models; calibrators and holdout metrics
use trusted (human / ground-truth / synthetic) rows only.
"""
from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from ..core.head import HeadSpec
from .prompts import JudgeAnswer, build_prompt, label_index, parse_confidence, parse_response
from .store import LabelItem

__all__ = ["DEFAULT_ANTHROPIC_MODEL", "AnthropicJudge", "ClaudeCliJudge", "Judge", "JudgeAnswer", "JudgeError",
           "OllamaJudge", "OpenAICompatJudge", "ScriptedJudge", "check_host", "make_judge"]

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"   # cheap bulk labelling; claude-sonnet-5-5 / claude-opus-5-5 for hard data
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300.0
# models that reject `output_config.effort` (pass effort=None for others that do)
_NO_EFFORT_PREFIXES = ("claude-3", "claude-haiku-4", "claude-sonnet-4-5", "claude-sonnet-4-0", "claude-sonnet-4-2",
                       "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-2")


class JudgeError(RuntimeError):
    """A judge cannot work at all right now (configuration, credentials, connectivity, timeout)."""


@runtime_checkable
class Judge(Protocol):
    id: str

    def label(self, items: Sequence[LabelItem], heads: Sequence[HeadSpec]) -> tuple[list[JudgeAnswer], list[str]]:
        ...


# ----------------------------------------------------------------------------- host policy

def _env_allows_remote() -> bool:
    return os.environ.get("RLCD_ALLOW_REMOTE") == "1"


def _is_loopback(host: str | None) -> bool:
    """Literal loopback addresses and `localhost` only; names are not resolved (no DNS rebinding games)."""
    if not host:
        return False
    h = host.strip("[]").lower().rstrip(".")
    if h == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def _local_check(url: str, allow_remote: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise JudgeError(f"judge URL must be http(s)://host[:port][/path], got {url!r}")
    if not allow_remote and not _is_loopback(parts.hostname):
        raise JudgeError(f"refusing non-loopback judge host {parts.hostname!r}: item texts would leave this machine. "
                         f"Allow it explicitly with RLCD_ALLOW_REMOTE=1 or `security: {{allow_remote: true}}` in "
                         f"rlcd.yaml.")


def check_host(url: str, allow_remote: bool = False) -> None:
    """Raise JudgeError unless `url` is a loopback http(s) URL or remote hosts are allowed.

    Uses open_rlcd.connectors.security.check_host when that module exists (one policy for connectors and judges);
    the local check here is the equivalent fallback."""
    allow = bool(allow_remote) or _env_allows_remote()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise JudgeError(f"judge URL must be http(s)://host[:port][/path], got {url!r}")
    try:
        from ..connectors.security import check_host as shared  # noqa: PLC0415
    except ImportError:
        shared = None
    if shared is not None:
        try:
            shared(url, allow_remote=allow)
            return
        except TypeError:
            pass  # a different signature: fall back to the local check
        except Exception as e:  # the shared policy refused the host
            raise JudgeError(str(e)) from e
    _local_check(url, allow)


def _allow_remote(project: Any) -> bool:
    if project is None:
        return _env_allows_remote()
    try:
        from ..connectors.security import remote_allowed  # noqa: PLC0415
    except ImportError:
        remote_allowed = None
    if remote_allowed is not None:
        return bool(remote_allowed(project))
    pc = getattr(project, "pack_config", {}) or {}
    return bool(getattr(project, "allow_remote", False)) or bool(pc.get("allow_remote")) or _env_allows_remote()


# ----------------------------------------------------------------------------- shared prompt-judge plumbing

class _PromptJudge:
    """Builds the prompt, asks the model (subclasses implement `complete`), parses the answer."""

    id = "judge"

    def complete(self, system: str, user: str, problems: list[str]) -> str | None:
        raise NotImplementedError

    def label(self, items: Sequence[LabelItem], heads: Sequence[HeadSpec]) -> tuple[list[JudgeAnswer], list[str]]:
        items, heads = list(items), list(heads)
        if not items or not heads:
            return [], []
        system, user = build_prompt(items, heads)
        problems: list[str] = []
        text = self.complete(system, user, problems)
        if text is None:
            return [], problems
        return parse_response(text, items, heads, problems), problems


def _post_json(url: str, body: dict, headers: dict, timeout: float, transport: Any, what: str, hint: str) -> dict:
    import httpx  # noqa: PLC0415

    loopback = _is_loopback(urlsplit(url).hostname)
    try:
        # no redirects: a redirect could send item texts to a host the policy never checked;
        # no proxy env for loopback hosts: a proxy would route local traffic elsewhere
        with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False,
                          trust_env=not loopback) as client:
            r = client.post(url, json=body, headers=headers)
    except httpx.TimeoutException as e:
        raise JudgeError(f"{what}: no answer from {url} within {timeout:g}s; lower --batch or raise the timeout") from e
    except httpx.HTTPError as e:
        raise JudgeError(f"{what}: cannot reach {url} ({type(e).__name__}: {e}). {hint}") from e
    if r.status_code >= 400:
        raise JudgeError(f"{what}: HTTP {r.status_code} from {url}: {r.text[:300]}")
    try:
        data = r.json()
    except ValueError as e:
        raise JudgeError(f"{what}: {url} did not return JSON") from e
    if not isinstance(data, dict):
        raise JudgeError(f"{what}: unexpected response shape from {url}")
    return data


# ----------------------------------------------------------------------------- Claude Code CLI

def parse_cli_envelope(stdout: str) -> tuple[str, dict]:
    """(result text, envelope) from `claude -p --output-format json`. Accepts a single JSON object, a JSON array of
    messages or JSON lines (the last `type: result` wins), and falls back to treating stdout as the text."""
    s = (stdout or "").strip()
    env: Any = None
    try:
        env = json.loads(s)
    except json.JSONDecodeError:
        for line in reversed(s.splitlines()):
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and cand.get("type") == "result":
                env = cand
                break
    if isinstance(env, list):
        env = next((e for e in reversed(env) if isinstance(e, dict) and e.get("type") == "result"), None)
    if not isinstance(env, dict) or not ({"result", "is_error", "subtype"} & set(env)):
        return s, {}
    return str(env.get("result") or ""), env


class ClaudeCliJudge(_PromptJudge):
    """Labels through the `claude` executable (Claude Code) in print mode, on the user's own subscription.

    Runs `claude -p --output-format json --tools "" --strict-mcp-config --no-session-persistence [--model M]`
    with the prompt on stdin (no shell), in a fresh empty temporary directory that is deleted afterwards.
    The item texts it labels are untrusted (corpus documents, query logs), so the agent is locked down: no
    built-in tools at all (`--tools ""`: no shell, no file reads or writes, no web), no MCP servers
    (`--strict-mcp-config` without `--mcp-config`), no saved session, and a working directory with nothing in it
    instead of the project. open-rlcd's own secrets (RLCD_API_KEY, OPENAI_API_KEY, database passwords; see
    `SCRUBBED_ENV`) are removed from the child's environment; everything Claude Code needs to log in is kept.
    Note that Claude Code decides itself which credentials it uses (if ANTHROPIC_API_KEY is set it may bill that
    key rather than the subscription). Item texts are sent to Anthropic by Claude Code, as with the API judge."""

    LOCKDOWN = ("--tools", "", "--strict-mcp-config", "--no-session-persistence")
    SCRUBBED_ENV = ("RLCD_API_KEY", "OPENAI_API_KEY", "PGPASSWORD", "NEO4J_PASSWORD", "QDRANT_API_KEY")

    def __init__(self, model: str | None = None, executable: str = "claude", timeout: float = DEFAULT_TIMEOUT,
                 extra_args: Sequence[str] = ()):
        self.model, self.executable, self.timeout = model or None, executable, float(timeout)
        self.extra_args = tuple(str(a) for a in extra_args)
        self.id = f"claude-cli:{self.model}" if self.model else "claude-cli"
        self.usage: dict[str, Any] = {"calls": 0, "cost_usd": 0.0}

    def _missing(self) -> JudgeError:
        return JudgeError(f"the Claude Code executable {self.executable!r} was not found: install Claude Code "
                          f"(https://claude.com/claude-code) and log in, or use --judge anthropic (ANTHROPIC_API_KEY) "
                          f"or a local model (--judge ollama:<model>)")

    def command(self) -> list[str]:
        exe = shutil.which(self.executable)
        if exe is None:
            raise self._missing()
        exe = os.path.abspath(exe)   # the child runs in another directory: a relative path would not resolve
        return [exe, "-p", "--output-format", "json", *self.LOCKDOWN,
                *(["--model", self.model] if self.model else []), *self.extra_args]

    def child_env(self) -> dict[str, str]:
        """The environment the `claude` process gets: ours minus open-rlcd's own secrets (SCRUBBED_ENV)."""
        return {k: v for k, v in os.environ.items() if k not in self.SCRUBBED_ENV}

    def complete(self, system: str, user: str, problems: list[str]) -> str | None:
        cmd = self.command()
        try:
            with tempfile.TemporaryDirectory(prefix="rlcd-judge-") as cwd:   # empty: nothing to read or change
                proc = subprocess.run(cmd, input=f"{system}\n\n{user}", capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=self.timeout, check=False,
                                      cwd=cwd, env=self.child_env())
        except FileNotFoundError as e:
            raise self._missing() from e
        except subprocess.TimeoutExpired as e:
            raise JudgeError(f"claude did not answer within {self.timeout:g}s; lower --batch or raise the "
                             f"timeout (the run resumes where it stopped)") from e
        self.usage["calls"] += 1
        text, env = parse_cli_envelope(proc.stdout)
        if env.get("total_cost_usd") is not None:
            with contextlib.suppress(TypeError, ValueError):
                self.usage["cost_usd"] += float(env["total_cost_usd"])
        if env.get("is_error"):
            raise JudgeError(f"claude reported an error ({env.get('subtype') or 'error'}): "
                             f"{(text or proc.stderr or '').strip()[:300]}")
        if proc.returncode != 0:
            raise JudgeError(f"claude exited with status {proc.returncode}: "
                             f"{(proc.stderr or proc.stdout or '').strip()[-300:]}")
        return text


# ----------------------------------------------------------------------------- Anthropic API

def _anthropic_module() -> Any:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as e:
        raise JudgeError("the anthropic package is not installed: `uv sync --extra labeling` "
                         "(or `pip install 'open-rlcd[labeling]'`), or use --judge claude-cli") from e
    return anthropic


class AnthropicJudge(_PromptJudge):
    """Labels through the Anthropic Messages API.

    * the key comes from ANTHROPIC_API_KEY only (never from rlcd.yaml);
    * the system prompt (task + heads, identical for every batch) carries `cache_control: ephemeral`, so batches
      after the first read it from the prompt cache (prompts shorter than the model's minimum cacheable length are
      simply not cached);
    * rate-limit (429), overloaded (529) and other retryable errors are retried by the SDK with exponential backoff
      and jitter (`max_retries`); what still fails after that raises JudgeError and the run can be resumed;
    * no temperature is sent: current models and API versions reject sampling parameters. Repeatability comes from
      the strict output format and label validation, not from temperature 0;
    * `effort` (default "low": labelling is classification) is sent as `output_config.effort` on models that take
      it; pass effort=None to omit it;
    * a `refusal` stop reason skips the batch (its items stay unlabelled) and is reported as a problem.
    """

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 16000, effort: str | None = "low",
                 max_retries: int = 4, timeout: float = DEFAULT_TIMEOUT, client: Any = None):
        self.model, self.max_tokens = model or DEFAULT_ANTHROPIC_MODEL, int(max_tokens)
        self.effort = effort if effort and not self.model.startswith(_NO_EFFORT_PREFIXES) else None
        self.id = f"anthropic:{self.model}"
        if client is None:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise JudgeError("ANTHROPIC_API_KEY is not set: export it (keys are read from the environment only), "
                                 "or use --judge claude-cli to label with your Claude Code subscription")
            client = _anthropic_module().Anthropic(api_key=key, max_retries=int(max_retries), timeout=float(timeout))
        self.client = client
        self.usage: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    def request(self, system: str, user: str) -> dict:
        kw: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        return kw

    def complete(self, system: str, user: str, problems: list[str]) -> str | None:
        try:
            resp = self.client.messages.create(**self.request(system, user))
        except Exception as e:
            err = self._error(e)
            if err is e:
                raise
            raise err from e
        self.usage["calls"] += 1
        u = getattr(resp, "usage", None)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = getattr(u, k, None) if u is not None else None
            if isinstance(v, int):
                self.usage[k] += v
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details is not None else None
            problems.append(f"{self.model} declined this batch (refusal{', ' + str(cat) if cat else ''}); its items "
                            f"stay unlabelled")
            return None
        text = "".join(getattr(b, "text", "") for b in (getattr(resp, "content", None) or [])
                       if getattr(b, "type", None) == "text")
        if stop == "max_tokens":
            problems.append(f"the answer was cut at max_tokens={self.max_tokens}; labels after the cut are lost "
                            f"(lower --batch or raise max_tokens)")
        return text

    def _error(self, e: Exception) -> Exception:
        try:
            anthropic = _anthropic_module()
        except JudgeError:
            return e
        if not isinstance(e, anthropic.APIError):
            return e
        msg = getattr(e, "message", None) or str(e)
        if isinstance(e, anthropic.AuthenticationError):
            return JudgeError(f"the Anthropic API rejected ANTHROPIC_API_KEY: {msg}")
        retryable = tuple(c for c in (getattr(anthropic, n, None) for n in
                                      ("RateLimitError", "OverloadedError", "InternalServerError",
                                       "ServiceUnavailableError", "APIConnectionError")) if isinstance(c, type))
        if retryable and isinstance(e, retryable):
            return JudgeError(f"Anthropic API still failing after retries ({type(e).__name__}: {msg}); run the "
                              f"bootstrap again later, it resumes where it stopped")
        return JudgeError(f"Anthropic API error ({type(e).__name__}): {msg}")


# ----------------------------------------------------------------------------- local / OpenAI-compatible servers

class OllamaJudge(_PromptJudge):
    """Labels through a local Ollama server (POST /api/chat, JSON mode, temperature 0, no streaming)."""

    def __init__(self, model: str, base_url: str = DEFAULT_OLLAMA_URL, allow_remote: bool = False,
                 timeout: float = DEFAULT_TIMEOUT, transport: Any = None):
        if not model:
            raise ValueError("OllamaJudge needs a model name, e.g. ollama:qwen2.5:7b-instruct")
        check_host(base_url, allow_remote)
        self.model, self.base_url, self.timeout, self.transport = model, base_url.rstrip("/"), float(timeout), transport
        self.id = f"ollama:{model}"

    def complete(self, system: str, user: str, problems: list[str]) -> str | None:
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                "format": "json", "stream": False, "options": {"temperature": 0}}
        data = _post_json(f"{self.base_url}/api/chat", body, {}, self.timeout, self.transport, "ollama",
                          f"Is Ollama running (`ollama serve`) with the model pulled (`ollama pull {self.model}`)?")
        text = (data.get("message") or {}).get("content") if isinstance(data.get("message"), dict) else None
        if not isinstance(text, str):
            problems.append("ollama answered without message.content")
            return None
        return text


def _chat_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return f"{b}/chat/completions" if b.endswith("/v1") else f"{b}/v1/chat/completions"


class OpenAICompatJudge(_PromptJudge):
    """Labels through any OpenAI-compatible server (POST {base}/v1/chat/completions, JSON object mode).

    `Authorization: Bearer $OPENAI_API_KEY` is sent when that variable is set (`api_key_env` names another one);
    mind that the key then goes to whatever server `base_url` points at."""

    def __init__(self, base_url: str, model: str, allow_remote: bool = False, timeout: float = DEFAULT_TIMEOUT,
                 transport: Any = None, api_key_env: str = "OPENAI_API_KEY"):
        if not model:
            raise ValueError("OpenAICompatJudge needs a model name")
        check_host(base_url, allow_remote)
        self.url, self.model, self.timeout, self.transport = _chat_url(base_url), model, float(timeout), transport
        self.api_key_env = api_key_env
        self.id = f"openai:{model}"

    def complete(self, system: str, user: str, problems: list[str]) -> str | None:
        key = os.environ.get(self.api_key_env, "").strip()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "temperature": 0}
        data = _post_json(self.url, body, headers, self.timeout, self.transport, "openai-compatible judge",
                          "Is the server running and the URL right?")
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        if not isinstance(text, str):
            problems.append("the server answered without choices[0].message.content")
            return None
        return text


# ----------------------------------------------------------------------------- scripted (tests, dry runs)

class ScriptedJudge:
    """Deterministic judge: `fn(item, spec)` returns (label, confidence), a bare label, or None (no answer).

    Labels are validated like a model's; it never calls anything external. `calls` and `batches` record what it
    was asked (for tests and dry runs)."""

    def __init__(self, fn: Callable[[LabelItem, HeadSpec], Any], id: str = "scripted"):  # noqa: A002
        self.fn, self.id = fn, id
        self.calls = 0
        self.batches: list[tuple[list[str], list[str]]] = []

    def label(self, items: Sequence[LabelItem], heads: Sequence[HeadSpec]) -> tuple[list[JudgeAnswer], list[str]]:
        self.calls += 1
        self.batches.append(([str(i.id) for i in items], [s.name for s in heads]))
        answers: list[JudgeAnswer] = []
        problems: list[str] = []
        for it in items:
            for spec in heads:
                res = self.fn(it, spec)
                if res is None:
                    problems.append(f"item {it.id}, {spec.name}: no scripted label")
                    continue
                label, conf = res if isinstance(res, tuple) else (res, None)
                try:
                    idx = label_index(spec, label)
                except (ValueError, TypeError):
                    problems.append(f"item {it.id}, {spec.name}: {label!r} is not an allowed label")
                    continue
                answers.append(JudgeAnswer(str(it.id), spec.name, spec.labels[idx], parse_confidence(conf),
                                           "scripted"))
        return answers, problems


# ----------------------------------------------------------------------------- factory

JUDGE_FORMS = ("claude-cli", "claude-cli:<model>", "anthropic", "anthropic:<model>", "ollama:<model>",
               "openai:<base_url>|<model>")


def make_judge(spec: str | None = None, project: Any = None, transport: Any = None) -> Judge:
    """A judge from a spec string (see JUDGE_FORMS). With no spec, `labeling.judge` / `labeling.model` from
    rlcd.yaml (default claude-cli). `labeling.base_url` sets the Ollama URL and `labeling.timeout` the timeout.
    Remote hosts are allowed by RLCD_ALLOW_REMOTE=1, security.allow_remote or pack_config.allow_remote."""
    lab: dict = project.section("labeling") if project is not None and hasattr(project, "section") else {}
    from_config = not spec
    text = str(spec or lab.get("judge") or "claude-cli").strip()
    kind, _, arg = text.partition(":")
    kind, arg = kind.strip().lower(), arg.strip()
    if from_config and not arg and lab.get("model"):
        arg = str(lab["model"])
    timeout = float(lab.get("timeout") or DEFAULT_TIMEOUT)
    allow = _allow_remote(project)
    if kind == "claude-cli":
        return ClaudeCliJudge(model=arg or None, executable=str(lab.get("executable") or "claude"), timeout=timeout)
    if kind == "anthropic":
        return AnthropicJudge(model=arg or DEFAULT_ANTHROPIC_MODEL, timeout=timeout)
    if kind == "ollama":
        if not arg:
            raise ValueError("ollama needs a model: --judge ollama:<model>")
        return OllamaJudge(arg, base_url=str(lab.get("base_url") or DEFAULT_OLLAMA_URL), allow_remote=allow,
                           timeout=timeout, transport=transport)
    if kind == "openai":
        base, sep, model = arg.rpartition("|")
        if not sep or not base.strip() or not model.strip():
            raise ValueError("openai needs a base URL and a model: --judge 'openai:http://127.0.0.1:8080/v1|<model>'")
        return OpenAICompatJudge(base.strip(), model.strip(), allow_remote=allow, timeout=timeout,
                                 transport=transport)
    raise ValueError(f"unknown judge {text!r}; use one of {list(JUDGE_FORMS)}")
