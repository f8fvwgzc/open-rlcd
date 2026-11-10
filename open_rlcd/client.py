"""A small Python client for the open-rlcd HTTP API: typed questions in, typed answers out, each failure a state.

    from open_rlcd.client import Choice, Client, Noul, Score

    with Client("http://127.0.0.1:8095", fail_closed="abstain") as rlcd:
        r = rlcd.system_one({"query": "Our checkout is down"}, {
            "kind": Choice(head="response_type", allowed=["direct_answer", "escalate_human"]),
            "multi_hop": Noul(head="needs_graph")})
        kind = r.answers["kind"]            # MissingAnswer when the service did not answer it: never a default
        if kind.band() == "high":           # "low" whenever the head is untrained or its holdout showed no skill
            ...
        d = rlcd.decide(state, stakes="medium", capabilities=["vector"])
        if d.allow:                         # act only; confirm -> d.ask; anything else, and every fallback -> deny
            rlcd.check_receipt(d.decision_id, state=state_about_to_act_on).require_current()
            ...                             # then your own gates: identity, permission, user confirmation

Why: an application asks a decision model narrow typed questions while its own code keeps authority (the field
guide "How to use Jev with LLMs", II.D-G, IV, VII.A-E, VIII.I, X.A-B). So this client:

  * builds questions with Noul / Choice / Score; `to_dict()` emits only what is set (`type` always). The head
    defines what a question means: `instructions` are sent, and ignored by the server;
  * reads every response as data (SystemOneResponse, Ranking, Decision, Evaluation, ReceiptCheck) and refuses a
    body that lacks what the contract promises (MalformedResponse) instead of filling the gap;
  * makes each failure its own exception: AuthError (401), NotFound (404), ValidationError (400 / 413 / 422, with
    the server's `kind`, e.g. missing_evidence), Conflict (409), RateLimited (429), Unavailable (503), ServerError
    (other 5xx), TransportError and its Timeout (no response), MalformedResponse, MissingAnswer, NotAllowed;
  * retries only what may succeed unchanged: transport failures and responses whose body says `retryable: true`
    (a `Retry-After` header is honoured, capped by `max_backoff`), at most `max_retries` extra attempts, with
    exponential backoff and jitter. Never a 2xx, never a response marked not retryable (a 422 needs another
    request, not another attempt). Asking again over the same state does not resolve semantic uncertainty: change
    the evidence or the question, or escalate (VII.E). These retries repeat one logical call and are counted as
    transport retries (`transport_retries` on every result and error, `Client.transport_retries` in total), apart
    from the workflow retries your code makes; post both with outcome(retries={"transport": .., "workflow": ..});
  * fail_closed="abstain" (or "confirm"): a decide / evaluate the service could not answer (no response, a 429 or
    5xx after the retries, a malformed body) returns a synthetic Decision / Evaluation with `fallback=True` whose
    action escalates (or asks a person) and whose verdict rejects (or sends to review), with the error in its
    `reason`: it never acts and never ships (X.B). A request or configuration error (401, 404, 409, 422) still
    raises: it needs a fix, not a fallback. system_one never falls back (an answer is never invented), and neither
    do rank, check_receipt or the views: when a receipt cannot be checked, the exception is the "no".

Decision.allow / ask / deny read the server's overall `action` and nothing else: act -> allow, confirm -> ask,
anything else (escalate, hold, a fallback) -> deny. `allow` makes a proposal eligible for your next gate; it never
authorises a side effect by itself (X.A). Right before the side effect, check_receipt says whether the decision
still holds for the state you are about to act on; `digest(state)` (re-exported from core/digest.py: the server's
own canonical digest) can be sent instead of the state.

What it does not do: call an LLM, cache answers, apply thresholds of its own (Answer.band is a reading aid for
/v1/systemone answers; /v1/decide also applies the holdout verdict and the Bernoulli gate, which band cannot), or
replay a request whose effect is unknown: a feedback row without a decision_id is retried only when the request
never reached the server (a second copy would be a second training row). A fallback is never sent to the server
(it has no decision_id and no receipt): count it in your own metrics (`Client.fallbacks` counts them here). The
client is synchronous (httpx.Client); there is no async variant.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import httpx

from . import VERSION
from .core.digest import digest, plain

__all__ = [
    "Answer", "Answers", "AuthError", "Choice", "Client", "Conflict", "Decision", "Evaluation", "MalformedResponse",
    "MissingAnswer", "NotAllowed", "NotFound", "Noul", "RLCDError", "RankedCandidate", "Ranking", "RateLimited",
    "ReceiptCheck", "Score", "ServerError", "SystemOneResponse", "Timeout", "TransportError", "Unavailable",
    "ValidationError", "digest",
]

log = logging.getLogger("open_rlcd.client")

DEFAULT_URL = "http://127.0.0.1:8095"
FAIL_CLOSED_MODES = ("abstain", "confirm")
Band = Literal["high", "medium", "low"]
_UNSET: Any = object()      # "not given", where None is a value the caller may mean (a state can be null)


# ----------------------------------------------------------------------------- errors

class RLCDError(Exception):
    """Base of every error for a call that produced no usable result.

    `status`: the HTTP status (None when there was no response); `kind`: the server's error kind; `body`: its JSON
    error body; `retryable`: whether the same request may succeed later unchanged; `retry_after`: the server's
    Retry-After in seconds; `transport_retries`: how often this client already repeated the request."""

    def __init__(self, message: str, *, status: int | None = None, kind: str | None = None, body: dict | None = None,
                 retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.status, self.kind, self.body, self.retryable = status, kind, body, retryable
        self.retry_after = retry_after
        self.transport_retries = 0

    def __str__(self) -> str:
        return self.message


class AuthError(RLCDError):
    """401: a missing or wrong API key (or an operator route called without RLCD_ADMIN_KEY)."""


class NotFound(RLCDError):
    """404: an unknown receipt (`kind: unknown_decision`), training job or route."""


class ValidationError(RLCDError):
    """400 / 413 / 422: the request cannot be answered as sent. `kind` says why (validation, missing_evidence,
    unknown_head, unknown_model, battery_version_mismatch, unsupported, ...): send a different request."""


class Conflict(RLCDError):
    """409: a training job is running (`busy`, retryable) or the kill switch is locked (`automation_locked`)."""


class RateLimited(RLCDError):
    """429: too many requests; retried while the server marks it retryable."""


class ServerError(RLCDError):
    """5xx: the service failed. A 500 is not retried (the same request would fail the same way)."""


class Unavailable(ServerError):
    """503: a connector, a store or the encoder is unreachable; retried while the server marks it retryable."""


class TransportError(RLCDError):
    """No HTTP response: connection refused or reset, DNS failure, protocol error."""


class Timeout(TransportError):
    """No response within the timeout."""


class MalformedResponse(RLCDError):
    """A response that is not JSON, or lacks what the contract promises: nothing is filled in."""


class MissingAnswer(RLCDError, KeyError):
    """A question id (or decision head) without an answer. Also a KeyError, so `in` and `.get()` behave as on a
    mapping; it is never replaced by a default answer."""


class NotAllowed(RLCDError):
    """require_allow / require_ship / require_current on a result that does not allow the side effect. `result` is
    the Decision, Evaluation or ReceiptCheck that said no."""

    def __init__(self, message: str, result: Any = None):
        super().__init__(message, kind="not_allowed")
        self.result = result


_BY_STATUS: dict[int, type[RLCDError]] = {400: ValidationError, 401: AuthError, 403: AuthError, 404: NotFound,
                                          409: Conflict, 413: ValidationError, 422: ValidationError,
                                          429: RateLimited, 503: Unavailable}
# a decide / evaluate that fails with one of these "could not be answered": fail_closed may stand in for it
_UNANSWERED = (TransportError, ServerError, RateLimited, MalformedResponse)
# transport failures that may succeed when repeated, and those where the request certainly never reached the server
_RETRYABLE_TRANSPORT = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError)
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)


# ----------------------------------------------------------------------------- question builders

@dataclass(frozen=True)
class _Question:
    TYPE: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        """The question as the wire contract takes it: `type`, then only the fields that are set."""
        out: dict[str, Any] = {"type": self.TYPE}
        for f in dataclass_fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = plain(v)
        return out


@dataclass(frozen=True)
class Noul(_Question):
    """Is a statement true? Answered with P(yes). `head`: the trained head that answers (default: the question
    id); `criteria`: optional {"true": ..., "false": ...}; `path` / `inputs`: where in the shared state the head's
    inputs live (a path that does not resolve is a 422 missing_evidence, never a guess)."""

    TYPE: ClassVar[str] = "noul"
    head: str | None = None
    instructions: str | None = None
    criteria: Mapping[str, str] | None = None
    path: str | None = None
    inputs: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Choice(_Question):
    """Which option? `criteria` {option: description} must name exactly the head's options when given;
    `allowed`: the options your deterministic code kept (the answer reports the mass the head put on the removed
    ones as `excluded_mass`); `option`: the option /v1/rank sorts by."""

    TYPE: ClassVar[str] = "choice"
    head: str | None = None
    instructions: str | None = None
    criteria: Mapping[str, str] | None = None
    allowed: Sequence[str] | None = None
    option: str | None = None
    path: str | None = None
    inputs: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Score(_Question):
    """Which level of an ordered rubric? Levels are indexed from 0 in the head's order; `criteria` (the level
    descriptions, in that order) is optional and only relabels them."""

    TYPE: ClassVar[str] = "score"
    head: str | None = None
    instructions: str | None = None
    criteria: Sequence[str] | None = None
    path: str | None = None
    inputs: Mapping[str, str] | None = None


QuestionLike = Noul | Choice | Score | Mapping[str, Any]


def _question(q: Any) -> dict:
    if isinstance(q, _Question):
        return q.to_dict()
    if isinstance(q, Mapping):
        return dict(q)
    raise TypeError(f"a question is a Noul, Choice or Score (or a dict in the wire format), got {type(q).__name__}")


# ----------------------------------------------------------------------------- reading responses

def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _prob(v: Any) -> float | None:
    f = _num(v)
    return f if f is not None and 0.0 <= f <= 1.0 else None


def _probs(v: Any) -> dict[str, float] | None:
    if not isinstance(v, Mapping) or not v:
        return None
    out = {str(k): _prob(p) for k, p in v.items()}
    return None if any(p is None for p in out.values()) else out  # type: ignore[return-value]


def _object(body: Any, what: str) -> dict:
    if not isinstance(body, dict):
        raise MalformedResponse(f"{what} is {type(body).__name__}, not a JSON object")
    return body


def _required_str(body: dict, key: str, what: str) -> str:
    v = body.get(key)
    if not isinstance(v, str) or not v:
        raise MalformedResponse(f"{what} has no {key!r} (got {v!r}): it is not read as anything else")
    return v


def _optional_str(body: dict, key: str, what: str) -> str | None:
    v = body.get(key)
    if v is not None and not isinstance(v, str):
        raise MalformedResponse(f"{what}: {key!r} is {type(v).__name__}, not a string")
    return v


def _optional_obj(body: dict, key: str, what: str) -> dict | None:
    v = body.get(key)
    if v is not None and not isinstance(v, dict):
        raise MalformedResponse(f"{what}: {key!r} is {type(v).__name__}, not an object")
    return v


def _str_list(body: dict, key: str, what: str) -> list[str]:
    v = body.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise MalformedResponse(f"{what}: {key!r} is {type(v).__name__}, not a list")
    return [str(x) for x in v]


def _check_answer(raw: Any, where: str) -> dict:
    """An answer object as /v1/systemone, /v1/rank and decide heads return it, or MalformedResponse."""
    if not isinstance(raw, Mapping):
        raise MalformedResponse(f"{where} is {type(raw).__name__}, not an answer object")
    a = dict(raw)
    t = a.get("type")
    if t == "noul":
        if _prob(a.get("noul")) is None:
            raise MalformedResponse(f"{where}: a noul answer needs `noul`, a probability in [0, 1] (got "
                                    f"{a.get('noul')!r})")
    elif t in ("choice", "score"):
        probs = _probs(a.get("probabilities"))
        if probs is None:
            raise MalformedResponse(f"{where}: a {t} answer needs `probabilities`, an object of probabilities")
        v = a.get(t)
        if t == "choice" and not (isinstance(v, str) and v in probs):
            raise MalformedResponse(f"{where}: `choice` {v!r} is not one of the answered options {sorted(probs)}")
        if t == "score" and not (isinstance(v, int) and not isinstance(v, bool) and str(v) in probs):
            raise MalformedResponse(f"{where}: `score` {v!r} is not one of the answered levels {sorted(probs)}")
    else:
        raise MalformedResponse(f"{where}: unknown answer type {t!r} (noul, choice or score)")
    for k in ("calibrated", "skill"):
        if a.get(k) is not None and not isinstance(a[k], bool):
            raise MalformedResponse(f"{where}: `{k}` is {a[k]!r}, not a boolean")
    if a.get("confidence") is not None and _num(a["confidence"]) is None:
        raise MalformedResponse(f"{where}: `confidence` is {a['confidence']!r}, not a number")
    return a


class Answer:
    """One typed answer, read as data (`raw` is the answer object as the server sent it).

    `type` noul | choice | score; `noul` P(yes) / `choice` the chosen option / `score` the chosen level (each None
    for the other types); `probabilities` (choice: the answered options, with `allowed` not renormalised; score:
    "0".."n-1"; noul: None); `confidence` (the shape of the distribution, not whether it is right); `calibrated`
    (False: the head is untrained, the answer is the uninformed prior); `skill` (False: the holdout showed no skill
    over the base rate); `head`; `allowed` / `excluded_mass` (choice with allowed options); `legend` (score)."""

    __slots__ = ("raw",)

    def __init__(self, raw: Mapping[str, Any], *, where: str = "answer"):
        self.raw: dict[str, Any] = _check_answer(raw, where)

    @property
    def type(self) -> str:
        return self.raw["type"]

    @property
    def noul(self) -> float | None:
        return float(self.raw["noul"]) if self.type == "noul" else None

    @property
    def choice(self) -> str | None:
        return self.raw["choice"] if self.type == "choice" else None

    @property
    def score(self) -> int | None:
        return int(self.raw["score"]) if self.type == "score" else None

    @property
    def probabilities(self) -> dict[str, float] | None:
        p = self.raw.get("probabilities")
        return {str(k): float(v) for k, v in p.items()} if isinstance(p, Mapping) else None

    @property
    def confidence(self) -> float | None:
        return _num(self.raw.get("confidence"))

    @property
    def calibrated(self) -> bool | None:
        v = self.raw.get("calibrated")
        return v if isinstance(v, bool) else None

    @property
    def skill(self) -> bool | None:
        v = self.raw.get("skill")
        return v if isinstance(v, bool) else None

    @property
    def head(self) -> str | None:
        v = self.raw.get("head")
        return v if isinstance(v, str) else None

    @property
    def allowed(self) -> list[str] | None:
        v = self.raw.get("allowed")
        return [str(x) for x in v] if isinstance(v, list) else None

    @property
    def excluded_mass(self) -> float | None:
        return _num(self.raw.get("excluded_mass"))

    @property
    def legend(self) -> dict[str, str] | None:
        v = self.raw.get("legend")
        return {str(k): str(d) for k, d in v.items()} if isinstance(v, Mapping) else None

    @property
    def value(self) -> str | int | bool:
        """What the answer commits to: the chosen option, the chosen level, or (noul) True when P(yes) >= 0.5."""
        if self.type == "noul":
            return float(self.raw["noul"]) >= 0.5
        return self.raw[self.type]

    @property
    def p(self) -> float:
        """The probability of `value`: noul max(P(yes), P(no)); choice the chosen option's (with `allowed`: not
        renormalised, so mass the head put on removed options lowers it); score the chosen level's."""
        if self.type == "noul":
            y = float(self.raw["noul"])
            return max(y, 1.0 - y)
        return float(self.raw["probabilities"][str(self.raw[self.type])])

    def band(self, act: float = 0.85, confirm: float = 0.65) -> Band:
        """How strongly the answer commits to `value`, in the three operating bands (guide VII.B): "high" when
        `p >= act` (eligible for an automatic path), "medium" when `p >= confirm` (confirm, gather state or review),
        else "low" (stop, clarify or fall back).

        Always "low" when the head is untrained (`calibrated` false), its holdout showed no skill (`skill` false),
        or the answer does not say: such probabilities are a prior or a base rate, not evidence about this state
        (do not gate on confidence alone). `p` is the calibrated probability of the answer, not `confidence`
        (a statement about the distribution's shape). The defaults are the policy's medium-stakes thresholds:
        choose your own per question and risk class from labelled data. A band is a reading aid, not a decision:
        /v1/decide also applies the holdout verdict and the Bernoulli track record."""
        if not 0.0 <= confirm <= act <= 1.0:
            raise ValueError(f"band thresholds need 0 <= confirm <= act <= 1, got confirm={confirm}, act={act}")
        if self.calibrated is not True or self.skill is not True:
            return "low"
        p = self.p
        if p >= act:
            return "high"
        return "medium" if p >= confirm else "low"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Answer) and other.raw == self.raw

    __hash__ = None  # type: ignore[assignment]   # mutable raw: not hashable

    def __repr__(self) -> str:
        return (f"Answer(type={self.type!r}, value={self.value!r}, p={self.p:.6g}, head={self.head!r}, "
                f"calibrated={self.calibrated!r}, skill={self.skill!r})")


class Answers(Mapping[str, Answer]):
    """{question id: Answer}. A question id without an answer raises MissingAnswer (a KeyError): there is no
    default answer, and a missing one is a failed call your code handles as such."""

    def __init__(self, answers: Mapping[str, Answer]):
        self._answers = dict(answers)

    def __getitem__(self, qid: str) -> Answer:
        try:
            return self._answers[qid]
        except KeyError:
            raise MissingAnswer(f"question {qid!r} has no answer (answered: {sorted(self._answers)}); there is no "
                                f"default answer: handle it as a failed call", kind="missing_answer") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._answers)

    def __len__(self) -> int:
        return len(self._answers)

    def __repr__(self) -> str:
        return f"Answers({self._answers!r})"


@dataclass(frozen=True)
class SystemOneResponse:
    """POST /v1/systemone. `missing`: question ids that were asked but not answered (reading one of them from
    `answers` raises MissingAnswer). `battery` {name, version, digest} when a battery was asked; `receipt` (the
    compact receipt) when one was recorded."""

    answers: Answers
    model: str
    warnings: list[str]
    usage: dict
    battery: dict | None
    receipt: dict | None
    raw: dict = field(repr=False)
    transport_retries: int = 0
    missing: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, body: Any, *, asked: Sequence[str] = (), transport_retries: int = 0) -> SystemOneResponse:
        what = "the /v1/systemone response"
        b = _object(body, what)
        model = _required_str(b, "model", what)
        raw_answers = b.get("answers")
        if not isinstance(raw_answers, dict):
            raise MalformedResponse(f"{what} has no `answers` object")
        answers = {str(q): Answer(a, where=f"answers.{q}") for q, a in raw_answers.items()}
        return cls(answers=Answers(answers), model=model, warnings=_str_list(b, "warnings", what),
                   usage=_optional_obj(b, "usage", what) or {}, battery=_optional_obj(b, "battery", what),
                   receipt=_optional_obj(b, "receipt", what), raw=b, transport_retries=transport_retries,
                   missing=[q for q in asked if q not in answers])


@dataclass(frozen=True)
class RankedCandidate:
    id: str
    probability: float | None      # None: the head does not apply to this candidate (ranked last)
    rank: int
    answer: Answer


@dataclass(frozen=True)
class Ranking:
    """POST /v1/rank: candidates sorted by the positive probability (P(yes), or the probability of `option`)."""

    candidates: list[RankedCandidate]
    model: str
    head: str | None
    option: str | None
    warnings: list[str]
    usage: dict
    receipt: dict | None
    raw: dict = field(repr=False)
    transport_retries: int = 0

    @classmethod
    def parse(cls, body: Any, *, transport_retries: int = 0) -> Ranking:
        what = "the /v1/rank response"
        b = _object(body, what)
        rows = b.get("candidates")
        if not isinstance(rows, list):
            raise MalformedResponse(f"{what} has no `candidates` list")
        out = []
        for i, c in enumerate(rows):
            w = f"candidates.{i}"
            c = _object(c, w)
            cid, rank, prob = c.get("id"), c.get("rank"), c.get("probability")
            if not isinstance(cid, str) or not isinstance(rank, int) or isinstance(rank, bool):
                raise MalformedResponse(f"{w} needs a string `id` and an integer `rank`")
            if prob is not None and _prob(prob) is None:
                raise MalformedResponse(f"{w}: `probability` {prob!r} is not a probability")
            out.append(RankedCandidate(id=cid, probability=_prob(prob), rank=rank,
                                       answer=Answer(c.get("answer"), where=f"{w}.answer")))
        return cls(candidates=out, model=_required_str(b, "model", what), head=_optional_str(b, "head", what),
                   option=_optional_str(b, "option", what), warnings=_str_list(b, "warnings", what),
                   usage=_optional_obj(b, "usage", what) or {}, receipt=_optional_obj(b, "receipt", what), raw=b,
                   transport_retries=transport_retries)


@dataclass(frozen=True)
class Decision:
    """POST /v1/decide. `allow` / `ask` / `deny` come from `action` and nothing else: act -> allow, confirm -> ask,
    anything else (the abstain action, whatever a pack calls it, and every fallback) -> deny.

    `fallback=True`: the service could not answer and the client's fail_closed policy stands in: no decision_id,
    no receipt, `reason` names the error (`error` is the exception)."""

    decision_id: str | None
    action: str
    tier: str | None
    plan: dict | None
    heads: dict
    rollout: dict | None
    receipt: dict | None
    warnings: list[str]
    raw: dict = field(repr=False)
    fallback: bool = False
    model: str | None = None
    reason: str | None = None
    error: RLCDError | None = field(default=None, repr=False)
    transport_retries: int = 0

    @property
    def allow(self) -> bool:
        return self.action == "act"

    @property
    def ask(self) -> bool:
        return self.action == "confirm"

    @property
    def deny(self) -> bool:
        return not (self.allow or self.ask)

    def require_allow(self) -> Decision:
        """This decision, or NotAllowed unless its action is act. Eligibility for your next gate, not authority."""
        if not self.allow:
            src = f"fallback ({self.reason})" if self.fallback else f"decision {self.decision_id}"
            raise NotAllowed(f"{src}: action is {self.action!r}, not 'act'", result=self)
        return self

    def answer(self, head: str) -> Answer:
        """The typed answer of one head that took part (MissingAnswer when it did not)."""
        h = self.heads.get(head)
        if not isinstance(h, Mapping) or h.get("answer") is None:
            raise MissingAnswer(f"decision {self.decision_id} has no answer for head {head!r} (heads: "
                                f"{sorted(self.heads)})", kind="missing_answer")
        return Answer(h["answer"], where=f"heads.{head}.answer")

    @classmethod
    def parse(cls, body: Any, *, transport_retries: int = 0) -> Decision:
        what = "the /v1/decide response"
        b = _object(body, what)
        return cls(decision_id=_required_str(b, "decision_id", what), action=_required_str(b, "action", what),
                   tier=_optional_str(b, "tier", what), plan=_optional_obj(b, "plan", what),
                   heads=_optional_obj(b, "heads", what) or {}, rollout=_optional_obj(b, "rollout", what),
                   receipt=_optional_obj(b, "receipt", what), warnings=_str_list(b, "warnings", what), raw=b,
                   model=_optional_str(b, "model", what), transport_retries=transport_retries)


@dataclass(frozen=True)
class Evaluation:
    """POST /v1/evaluate: `verdict` ship | review | reject, `next` (ship | human_review | revise | retry_generation |
    escalate_to_human), `decided_by` (checks | artifact | citations | output_scan | revision | heads; "fallback" for a
    fallback),
    `revise` (what failed, for the generator's next attempt). `ship` needs verdict ship AND action act."""

    decision_id: str | None
    verdict: str
    next: str | None
    action: str
    decided_by: str | None
    revise: dict | None
    receipt: dict | None
    warnings: list[str]
    raw: dict = field(repr=False)
    fallback: bool = False
    heads: dict = field(default_factory=dict)
    rollout: dict | None = None
    stakes: str | None = None
    model: str | None = None
    reason: str | None = None
    error: RLCDError | None = field(default=None, repr=False)
    transport_retries: int = 0

    @property
    def ship(self) -> bool:
        return self.verdict == "ship" and self.action == "act"

    def require_ship(self) -> Evaluation:
        """This evaluation, or NotAllowed unless it ships."""
        if not self.ship:
            src = f"fallback ({self.reason})" if self.fallback else f"evaluation {self.decision_id}"
            raise NotAllowed(f"{src}: verdict {self.verdict!r} (action {self.action!r}), next {self.next!r}",
                             result=self)
        return self

    @classmethod
    def parse(cls, body: Any, *, transport_retries: int = 0) -> Evaluation:
        what = "the /v1/evaluate response"
        b = _object(body, what)
        return cls(decision_id=_required_str(b, "decision_id", what), verdict=_required_str(b, "verdict", what),
                   next=_optional_str(b, "next", what), action=_required_str(b, "action", what),
                   decided_by=_optional_str(b, "decided_by", what), revise=_optional_obj(b, "revise", what),
                   receipt=_optional_obj(b, "receipt", what), warnings=_str_list(b, "warnings", what), raw=b,
                   heads=_optional_obj(b, "heads", what) or {}, rollout=_optional_obj(b, "rollout", what),
                   stakes=_optional_str(b, "stakes", what), model=_optional_str(b, "model", what),
                   transport_retries=transport_retries)


@dataclass(frozen=True)
class ReceiptCheck:
    """POST /v1/receipts/{id}/check: does the decision still hold for the state you are about to act on?
    `stale_reasons` says why not (changed state, moved alias, changed policy or battery, automation switched off
    since an act decision, too old, a confirm decision nobody has confirmed yet, or a decision that approves no
    action: an abstained one never holds); `checks` is the per-item comparison."""

    current: bool
    stale_reasons: list[str]
    raw: dict = field(repr=False)
    checks: dict = field(default_factory=dict)
    decision_id: str | None = None
    transport_retries: int = 0

    def require_current(self) -> ReceiptCheck:
        """This check, or NotAllowed (with the stale reasons) unless the decision still holds."""
        if not self.current:
            raise NotAllowed(f"decision {self.decision_id} no longer holds: {'; '.join(self.stale_reasons)}",
                             result=self)
        return self

    @classmethod
    def parse(cls, body: Any, *, transport_retries: int = 0) -> ReceiptCheck:
        what = "the receipt check response"
        b = _object(body, what)
        if not isinstance(b.get("current"), bool):
            raise MalformedResponse(f"{what} has no boolean `current` (got {b.get('current')!r})")
        if not isinstance(b.get("stale_reasons"), list):
            raise MalformedResponse(f"{what} has no `stale_reasons` list")
        return cls(current=b["current"], stale_reasons=_str_list(b, "stale_reasons", what), raw=b,
                   checks=_optional_obj(b, "checks", what) or {}, decision_id=_optional_str(b, "decision_id", what),
                   transport_retries=transport_retries)


def _require(body: dict, what: str, *keys: str) -> dict:
    missing = [k for k in keys if k not in body]
    if missing:
        raise MalformedResponse(f"{what} lacks {missing}")
    return body


# ----------------------------------------------------------------------------- HTTP plumbing

def _encode(body: Any) -> bytes:
    """Canonical-friendly JSON of a request body. Values go through core/digest.py `plain` first (numpy values,
    tuples, sets), so a client-side digest(state) equals the digest the server computes of the state it received."""
    try:
        return json.dumps(plain(body), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except ValueError as e:
        raise ValueError(f"the request cannot be sent as JSON ({e}); a missing value is null, not NaN or "
                         f"infinity") from None


def _snippet(text: str, n: int = 300) -> str:
    s = " ".join((text or "").split())
    return (s[:n] + "...") if len(s) > n else (s or "(empty body)")


def _retry_after(value: str | None) -> float | None:
    """Retry-After in seconds: delay-seconds or an HTTP date; None when absent or unreadable."""
    if not value:
        return None
    try:
        seconds: float | None = float(value)
    except ValueError:
        seconds = None
    if seconds is not None:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _status_error(resp: httpx.Response) -> RLCDError:
    status = resp.status_code
    try:
        data = resp.json()
    except ValueError:
        data = None
    body = data if isinstance(data, dict) else None
    kind = body.get("kind") if body is not None and isinstance(body.get("kind"), str) else None
    text = body["error"] if body is not None and isinstance(body.get("error"), str) else _snippet(resp.text)
    cls = _BY_STATUS.get(status) or (ServerError if status >= 500 else RLCDError)
    return cls(f"HTTP {status}{' ' + kind if kind else ''}: {text}", status=status, kind=kind, body=body,
               retryable=body is not None and body.get("retryable") is True,
               retry_after=_retry_after(resp.headers.get("retry-after")))


def _transport_error(e: httpx.RequestError, url: str) -> RLCDError:
    what = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    if isinstance(e, httpx.DecodingError):
        return MalformedResponse(f"the response from {url} could not be decoded ({what})")
    if isinstance(e, httpx.TimeoutException):
        return Timeout(f"no response from {url} in time ({what})", retryable=True)
    return TransportError(f"no response from {url} ({what})", retryable=isinstance(e, _RETRYABLE_TRANSPORT))


def _json_object(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except ValueError:
        raise MalformedResponse(f"HTTP {resp.status_code} response is not JSON: {_snippet(resp.text)}",
                                status=resp.status_code) from None
    if not isinstance(data, dict):
        raise MalformedResponse(f"HTTP {resp.status_code} response is {type(data).__name__}, not a JSON object",
                                status=resp.status_code)
    return data


def _segment(decision_id: Any) -> str:
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError(f"decision_id must be a non-empty string, got {decision_id!r} (a fallback decision has none: "
                         f"there is no receipt to read or check)")
    return quote(decision_id, safe="")


# ----------------------------------------------------------------------------- the client

class Client:
    """The open-rlcd HTTP API (see the module docstring).

    `base_url`: default the injected `http` client's base_url, else env RLCD_URL, else http://127.0.0.1:8095.
    `api_key`: default env RLCD_API_KEY; sent as `Authorization: Bearer <key>` (operator routes need the admin key
    when the server sets RLCD_ADMIN_KEY). `timeout` (seconds) configures the httpx.Client this creates; an injected
    `http` client (for example fastapi's TestClient) keeps its own settings and is not closed by close().
    `max_retries` extra attempts after a retryable failure, `backoff` * 2**n seconds (with jitter, at most
    `max_backoff`) between them; `sleep` is injectable for tests. `fail_closed`: None (raise), "abstain" or
    "confirm" (see the module docstring)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 10.0,
                 max_retries: int = 2, backoff: float = 0.2, max_backoff: float = 5.0,
                 http: httpx.Client | None = None, fail_closed: Literal["abstain", "confirm"] | None = None,
                 sleep: Callable[[float], Any] = time.sleep):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries must be an integer >= 0, got {max_retries!r}")
        if not (_num(timeout) or 0) > 0 or (_num(backoff) is None or backoff < 0) or (
                _num(max_backoff) is None or max_backoff < 0):
            raise ValueError(f"timeout must be > 0 and backoff / max_backoff >= 0 (seconds), got {timeout!r}, "
                             f"{backoff!r}, {max_backoff!r}")
        if fail_closed not in (None, *FAIL_CLOSED_MODES):
            raise ValueError(f"fail_closed must be None, 'abstain' or 'confirm', got {fail_closed!r}")
        injected = str(http.base_url) if http is not None else ""
        url = (base_url or injected or os.environ.get("RLCD_URL") or DEFAULT_URL).rstrip("/")
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https") or not parsed.host:
            raise ValueError(f"base_url must be an http(s) URL, got {url!r}")
        self.base_url = url
        key = api_key if api_key is not None else os.environ.get("RLCD_API_KEY")
        self._headers = {"Accept": "application/json", "User-Agent": f"open-rlcd-client/{VERSION}"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"
        self._http = http if http is not None else httpx.Client(timeout=timeout)
        self._owns_http = http is None
        self.timeout, self.max_retries, self.backoff, self.max_backoff = timeout, max_retries, backoff, max_backoff
        self.fail_closed = fail_closed
        self._sleep = sleep
        self._rng = random.Random()
        self._lock = threading.Lock()
        self._retries_total = 0
        self._fallbacks_total = 0

    # ------------------------------------------------------------------ lifecycle and counters
    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:   # never the key
        auth = "bearer" if "Authorization" in self._headers else "none"
        return f"Client(base_url={self.base_url!r}, auth={auth}, fail_closed={self.fail_closed!r})"

    @property
    def transport_retries(self) -> int:
        """Requests this client repeated unchanged (transport failures, retryable responses), over all calls."""
        with self._lock:
            return self._retries_total

    @property
    def fallbacks(self) -> int:
        """Synthetic fail_closed decisions and evaluations returned so far."""
        with self._lock:
            return self._fallbacks_total

    # ------------------------------------------------------------------ the request loop
    def _delay(self, retry: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_backoff)
        cap = min(self.max_backoff, self.backoff * (2 ** min(retry, 32)))
        return cap / 2 + self._rng.uniform(0.0, cap / 2)       # equal jitter: [cap / 2, cap]

    def _call(self, method: str, path: str, *, body: Any = _UNSET, params: Mapping[str, Any] | None = None,
              replay_safe: bool = True) -> tuple[dict, int]:
        """One logical call: (JSON object, transport retries made). `replay_safe=False`: a transport failure after
        the request may have reached the server is not repeated (only connect-stage failures are)."""
        url = self.base_url + path
        headers = dict(self._headers)
        content = None
        if body is not _UNSET:
            content = _encode(body)
            headers["Content-Type"] = "application/json"
        query = {k: v for k, v in (params or {}).items() if v is not None} or None
        retries = 0
        while True:
            retry_after: float | None = None
            cause: BaseException | None = None
            try:
                resp = self._http.request(method, url, content=content, headers=headers, params=query)
            except httpx.RequestError as e:
                err, cause = _transport_error(e, url), e
                retry = err.retryable
                if retry and not replay_safe and not isinstance(e, _NOT_SENT):
                    retry = err.retryable = False
                    err.message += ("; not repeated: the request may have reached the server, and a second copy "
                                    "would be stored twice")
            else:
                if 200 <= resp.status_code < 300:
                    try:
                        return _json_object(resp), retries
                    except RLCDError as bad:
                        bad.transport_retries = retries
                        raise
                err = _status_error(resp)
                retry, retry_after = err.retryable, err.retry_after
            if retry and retries < self.max_retries:
                delay = self._delay(retries, retry_after)
                log.info("open-rlcd %s %s: %s; retry %d of %d in %.2fs", method, path, err.message, retries + 1,
                         self.max_retries, delay)
                self._sleep(delay)
                retries += 1
                with self._lock:
                    self._retries_total += 1
                continue
            err.transport_retries = retries
            raise err from cause

    def _fallback(self, what: str, err: RLCDError, then: str) -> str:
        reason = (f"{what} could not be answered ({type(err).__name__}: {err.message}); "
                  f"fail_closed={self.fail_closed}: {then}")
        with self._lock:
            self._fallbacks_total += 1
        log.warning("open-rlcd fallback: %s", reason)
        return reason

    # ------------------------------------------------------------------ answering
    def system_one(self, state: Any, questions: Mapping[str, QuestionLike] | None = None, *,
                   battery: str | None = None, model: str | None = None, strict: bool = False,
                   receipt: bool = False) -> SystemOneResponse:
        """POST /v1/systemone: every question answered on one state, each by its own head. `battery` (name or
        name@version) instead of `questions`; `strict`: refuse a state missing inputs instead of answering it with a
        warning; `receipt`: record the call as a receipt. Never falls back: failures raise."""
        body: dict[str, Any] = {"state": state}
        if questions is not None:
            if not isinstance(questions, Mapping):
                raise TypeError("questions maps a question id to a Noul, Choice or Score (or a wire-format dict)")
            body["questions"] = {str(qid): _question(q) for qid, q in questions.items()}
        if battery is not None:
            body["battery"] = battery
        if model is not None:
            body["model"] = model
        if strict:
            body["strict"] = True
        if receipt:
            body["receipt"] = True
        data, n = self._call("POST", "/v1/systemone", body=body)
        try:
            return SystemOneResponse.parse(data, asked=list(body.get("questions") or ()), transport_retries=n)
        except RLCDError as e:
            e.transport_retries = n
            raise

    def rank(self, state: Any, candidates: Sequence[Mapping[str, Any] | tuple[str, Any]], *,
             question: QuestionLike | None = None, option: str | None = None, model: str | None = None,
             receipt: bool = False) -> Ranking:
        """POST /v1/rank: candidates ({id, state} or (id, state)) sorted by the positive probability; each
        candidate's state is merged over the shared `state`. `question` defaults to the pack's ranking head."""
        rows = [dict(c) if isinstance(c, Mapping) else {"id": c[0], "state": c[1]} for c in candidates]
        body: dict[str, Any] = {"state": state, "candidates": rows}
        if question is not None:
            body["question"] = _question(question)
        if option is not None:
            body["option"] = option
        if model is not None:
            body["model"] = model
        if receipt:
            body["receipt"] = True
        data, n = self._call("POST", "/v1/rank", body=body)
        try:
            return Ranking.parse(data, transport_retries=n)
        except RLCDError as e:
            e.transport_retries = n
            raise

    def decide(self, state: Any, *, stakes: str | None = None, model: str | None = None, **fields: Any) -> Decision:
        """POST /v1/decide. `fields` go to the request as they are (heads, allowed, capabilities, sensitivity,
        latency_budget_ms, strict, explore, return_evidence, baseline, artifact, checks, tags, the pack's own)."""
        body: dict[str, Any] = {"state": state, **fields}
        if stakes is not None:
            body["stakes"] = stakes
        if model is not None:
            body["model"] = model
        n = 0
        try:
            data, n = self._call("POST", "/v1/decide", body=body)
            return Decision.parse(data, transport_retries=n)
        except _UNANSWERED as e:
            e.transport_retries = max(e.transport_retries, n)
            if self.fail_closed is None:
                raise
            action = "escalate" if self.fail_closed == "abstain" else "confirm"
            then = ("escalate: a fallback never acts" if action == "escalate"
                    else "a person confirms: a fallback never acts")
            reason = self._fallback("the decision", e, then)
            return Decision(decision_id=None, action=action, tier=action, plan=None, heads={}, rollout=None,
                            receipt=None, warnings=[reason], raw={}, fallback=True, reason=reason, error=e,
                            transport_retries=e.transport_retries)

    def evaluate(self, state: Any, *, stakes: str | None = None, model: str | None = None,
                 **fields: Any) -> Evaluation:
        """POST /v1/evaluate (a pack with an evaluator, e.g. documents). `fields`: checks, artifact, citations,
        evidence, plan, revision, baseline, tags, strict, explore, the pack's own."""
        body: dict[str, Any] = {"state": state, **fields}
        if stakes is not None:
            body["stakes"] = stakes
        if model is not None:
            body["model"] = model
        n = 0
        try:
            data, n = self._call("POST", "/v1/evaluate", body=body)
            return Evaluation.parse(data, transport_retries=n)
        except _UNANSWERED as e:
            e.transport_retries = max(e.transport_retries, n)
            if self.fail_closed is None:
                raise
            if self.fail_closed == "abstain":
                verdict, nxt, action, then = "reject", "escalate_to_human", "escalate", "reject and escalate"
            else:
                verdict, nxt, action, then = "review", "human_review", "confirm", "a person reviews"
            reason = self._fallback("the evaluation", e, f"{then}: a fallback never ships")
            return Evaluation(decision_id=None, verdict=verdict, next=nxt, action=action, decided_by="fallback",
                              revise=None, receipt=None, warnings=[reason], raw={}, fallback=True, reason=reason,
                              error=e, transport_retries=e.transport_retries)

    # ------------------------------------------------------------------ the reinforcement loop
    def feedback(self, label: Any, *, source: str, decision_id: str | None = None, head: str | None = None,
                 state: Any = None, weight: float | None = None, tags: Sequence[str] | None = None,
                 note: str | None = None, missing_evidence: Sequence[str] | None = None, **fields: Any) -> dict:
        """POST /v1/feedback: the true label of one head's question (a training row at the next train; with a
        `decision_id` also a Bernoulli ledger update, and a later row corrects an earlier one). `fields`: x,
        feature_names, feature_version for a precomputed vector. Without a decision_id the row is not repeated
        after a transport failure that may have reached the server."""
        body: dict[str, Any] = {"label": label, "source": source, **fields}
        for k, v in (("decision_id", decision_id), ("head", head), ("state", state), ("weight", weight),
                     ("tags", tags), ("note", note), ("missing_evidence", missing_evidence)):
            if v is not None:
                body[k] = v
        data, _ = self._call("POST", "/v1/feedback", body=body, replay_safe=decision_id is not None)
        return _require(data, "the /v1/feedback response", "id", "stored")

    def outcome(self, decision_id: str, **fields: Any) -> dict:
        """POST /v1/outcome: what happened downstream (task_success, route_taken, provider, fallback, latency_ms,
        cost, tokens, retries {transport, workflow}, verified_by, note, tags). The last outcome of a decision counts,
        so a repeated request is harmless. Never trains a head."""
        data, _ = self._call("POST", "/v1/outcome", body={"decision_id": decision_id, **fields})
        return _require(data, "the /v1/outcome response", "id", "stored")

    # ------------------------------------------------------------------ receipts
    def receipt(self, decision_id: str) -> dict:
        """GET /v1/receipts/{id}: the decision record as logged (NotFound for an unknown id)."""
        data, _ = self._call("GET", f"/v1/receipts/{_segment(decision_id)}")
        return _require(data, "the receipt", "decision_id")

    def check_receipt(self, decision_id: str, *, state: Any = _UNSET, state_digest: str | None = None,
                      max_age_s: float | None = None) -> ReceiptCheck:
        """POST /v1/receipts/{id}/check with the state you are about to act on, or its `state_digest`
        (digest(state): the state then never leaves your process). `max_age_s`: older decisions are stale."""
        body: dict[str, Any] = {}
        if state is not _UNSET:
            body["state"] = state
        if state_digest is not None:
            body["state_digest"] = state_digest
        if max_age_s is not None:
            body["max_age_s"] = max_age_s
        data, n = self._call("POST", f"/v1/receipts/{_segment(decision_id)}/check", body=body)
        try:
            return ReceiptCheck.parse(data, transport_retries=n)
        except RLCDError as e:
            e.transport_retries = n
            raise

    def confirm_receipt(self, decision_id: str, *, by: str, note: str | None = None,
                        state_digest: str | None = None) -> dict:
        """POST /v1/receipts/{id}/confirm: a person confirmed a confirm-tier decision (a receipt check of a confirm
        decision is not current without it). `state_digest` (digest(state) of what the person reviewed) makes the
        server refuse a confirmation of another state."""
        body: dict[str, Any] = {"by": by}
        if note is not None:
            body["note"] = note
        if state_digest is not None:
            body["state_digest"] = state_digest
        data, _ = self._call("POST", f"/v1/receipts/{_segment(decision_id)}/confirm", body=body)
        return _require(data, "the confirmation", "confirmed", "decision_id")

    # ------------------------------------------------------------------ views and operator controls
    def batteries(self, model: str | None = None) -> dict:
        """GET /v1/batteries: the versioned question sets, with problems (missing or untrained heads)."""
        data, _ = self._call("GET", "/v1/batteries", params={"model": model})
        return _require(data, "the /v1/batteries response", "batteries")

    def heads(self, model: str | None = None) -> dict:
        data, _ = self._call("GET", "/v1/heads", params={"model": model})
        return _require(data, "the /v1/heads response", "heads")

    def models(self) -> dict:
        data, _ = self._call("GET", "/v1/models")
        return _require(data, "the /v1/models response", "data")

    def calibration(self, head: str, model: str | None = None, segment: str | None = None) -> dict:
        """GET /v1/calibration: the trusted-holdout report of a head (ValidationError kind untrained before)."""
        data, _ = self._call("GET", "/v1/calibration", params={"head": head, "model": model, "segment": segment})
        return data

    def bernoulli(self, head: str, model: str | None = None, stakes: str | None = None) -> dict:
        data, _ = self._call("GET", "/v1/bernoulli", params={"head": head, "model": model, "stakes": stakes})
        return data

    def health(self) -> dict:
        data, _ = self._call("GET", "/health")
        return _require(data, "the /health response", "status")

    def automation(self) -> dict:
        """GET /v1/automation: rollout mode, kill switch state and its recent history."""
        data, _ = self._call("GET", "/v1/automation")
        return _require(data, "the /v1/automation response", "automation")

    def set_automation(self, enabled: bool, reason: str | None = None, by: str | None = None) -> dict:
        """POST /v1/automation: the kill switch. Off caps every decision at confirm at once; switching it on
        while the server's RLCD_AUTOMATION=off is a Conflict (`automation_locked`), never retried."""
        body: dict[str, Any] = {"enabled": bool(enabled)}
        if reason is not None:
            body["reason"] = reason
        if by is not None:
            body["by"] = by
        data, _ = self._call("POST", "/v1/automation", body=body)
        return _require(data, "the /v1/automation response", "automation")
