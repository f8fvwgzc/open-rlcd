"""The wire contract: request validation and typed answers.

Follows the reference evaluation contract (`{model, state, questions}` -> `{model, answers, usage}`) with
three question types: noul (probability of yes), choice (one of N options), score (ordered levels).

RLCD extensions, all additive (kept identical to the View Engine RLCD service, so its clients work unchanged):
  * question field `head` binds a question to a trained head (default: the question id);
  * every answer carries `head` and `calibrated`; noul answers also carry `confidence`;
  * `instructions` is accepted and ignored: there is no language model here, the head defines the question.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .confidence import choice_confidence, noul_confidence, score_confidence

QUESTION_TYPES = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
MAX_QUESTIONS = 64
MAX_CANDIDATES = 200
STAKES = ("low", "medium", "high")
STAKES_NAME = r"^[A-Za-z0-9_\-]{1,64}$"   # the shape of a stakes level name; the policy says which levels exist
MAX_FEEDBACK_STATE_BYTES = 2 << 20    # a feedback row is appended to feedback.jsonl: its state is capped
MAX_FEATURES = 100_000                 # longest precomputed vector `x` a feedback row may carry


class ContractError(Exception):
    """A request that cannot be answered as asked. Rendered as HTTP 422 unless `status` says otherwise."""

    def __init__(self, message: str, *, kind: str = "validation", status: int = 422, **details: Any):
        super().__init__(message)
        self.message, self.kind, self.status, self.details = message, kind, status, details

    def body(self) -> dict:
        return {"error": self.message, "kind": self.kind, **self.details}


# ----------------------------------------------------------------------------- request bodies

class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SystemOneRequest(_Body):
    model: str | None = None
    state: Any = None
    questions: dict[str, Any]


class Candidate(_Body):
    id: str
    state: Any = None


class RankRequest(_Body):
    model: str | None = None
    state: Any = None
    candidates: list[Candidate]
    # optional in open-rlcd: defaults to the pack's ranking head (documents: chunk_relevant)
    question: dict[str, Any] | None = None
    option: str | None = None


class DecideRequest(BaseModel):
    """Generic decision request. Packs read their own extra fields (`extra="allow"`)."""
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    state: Any = None
    heads: list[str] | None = None
    # any level the policy defines (low / medium / high by default, plus custom ones); an unknown one is a 422 that
    # lists the configured levels
    stakes: str | None = Field(None, pattern=STAKES_NAME)
    # Thompson sampling: let a draw from the accuracy posterior decide the Bernoulli gate. Off by default;
    # a decision made this way is labelled as exploration.
    explore: bool = False


class EvaluateRequest(BaseModel):
    """Evaluator mode: judge an answer drafted by any LLM. Packs read their own fields."""
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    state: Any = None
    stakes: str | None = Field(None, pattern=STAKES_NAME)
    explore: bool = False


class FeedbackRequest(_Body):
    """An outcome for /v1/feedback. Every row is appended to feedback.jsonl, so its size is capped: the state
    serialises to at most MAX_FEEDBACK_STATE_BYTES, x has at most MAX_FEATURES values."""
    decision_id: str | None = Field(None, max_length=200)
    head: str | None = Field(None, max_length=200)
    state: Any = None
    x: list[float | None] | None = Field(None, max_length=MAX_FEATURES)
    feature_names: list[str] | None = Field(None, max_length=MAX_FEATURES)
    feature_version: str | None = Field(None, max_length=200)
    label: Any = None
    weight: float = Field(1.0, gt=0, le=100)
    source: str = Field(..., min_length=1, max_length=120)

    @model_validator(mode="after")
    def _bounded(self) -> FeedbackRequest:
        import json

        for name in ("state", "label"):
            v = getattr(self, name)
            if v is None:
                continue
            try:
                size = len(json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=str).encode())
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be JSON") from None
            if size > MAX_FEEDBACK_STATE_BYTES:
                raise ValueError(f"{name} is {size} bytes as JSON; at most {MAX_FEEDBACK_STATE_BYTES} are stored "
                                 f"per feedback row (send the decision_id of a logged decision instead)")
        return self


class TrainRequest(_Body):
    heads: list[str] | None = None
    holdout_fraction: float = Field(0.2, ge=0.05, le=0.5)
    n_splits: int = Field(5, ge=2, le=20)
    feedback_weight: float = Field(5.0, gt=0, le=1000)
    calibration: str = Field("auto", pattern="^(auto|sigmoid|isotonic|temperature)$")
    n_estimators: int = Field(300, ge=10, le=3000)
    seed: int = 7


# ----------------------------------------------------------------------------- questions

class Question:
    """A validated question. Structural limits are checked here; head compatibility in `bind`."""

    def __init__(self, qid: str, raw: Any):
        if not isinstance(raw, dict):
            raise ContractError(f"question '{qid}' must be an object", question=qid)
        qtype = raw.get("type")
        if qtype not in QUESTION_TYPES:
            raise ContractError(f"question '{qid}': type must be one of {list(QUESTION_TYPES)}, got {qtype!r}",
                                question=qid)
        instructions = raw.get("instructions", "")
        if instructions is not None and not isinstance(instructions, str):
            raise ContractError(f"question '{qid}': instructions must be a string", question=qid)
        head = raw.get("head", qid)
        if not isinstance(head, str) or not head.strip():
            raise ContractError(f"question '{qid}': head must be a non-empty string", question=qid)
        self.id, self.type, self.head = qid, qtype, head.strip()
        self.instructions = instructions or ""
        self.option: str | None = raw.get("option") if isinstance(raw.get("option"), str) else None
        self.criteria = self._criteria(raw.get("criteria"))

    def _criteria(self, c: Any) -> Any:
        q = self.id
        if self.type == "noul":
            if c is None:
                return None
            if not isinstance(c, dict) or not set(map(str, c)) <= {"true", "false"}:
                raise ContractError(f"question '{q}': noul criteria is an optional object with keys 'true' and "
                                    f"'false'", question=q)
            return {str(k): v for k, v in c.items()}
        if self.type == "choice":
            if c is None:
                return None  # RLCD: the head already fixes the options
            if not isinstance(c, dict) or not c:
                raise ContractError(f"question '{q}': choice criteria must be a non-empty object mapping each "
                                    f"option to its description", question=q)
            if len(c) > MAX_CHOICE_OPTIONS:
                raise ContractError(f"question '{q}': a choice can have at most {MAX_CHOICE_OPTIONS} options, "
                                    f"got {len(c)}", question=q, limit=MAX_CHOICE_OPTIONS)
            return {str(k): v for k, v in c.items()}
        if c is None:
            return None
        if not isinstance(c, list):
            raise ContractError(f"question '{q}': score criteria must be an ordered array of level descriptions",
                                question=q)
        if not MIN_SCORE_LEVELS <= len(c) <= MAX_SCORE_LEVELS:
            raise ContractError(f"question '{q}': a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, "
                                f"got {len(c)}", question=q, limit=[MIN_SCORE_LEVELS, MAX_SCORE_LEVELS])
        return [str(x) for x in c]

    def bind(self, spec: Any) -> None:
        """Check this question against the head it is bound to (`spec`: core.head.HeadSpec)."""
        q = self.id
        if spec.type != self.type:
            raise ContractError(f"question '{q}': head '{spec.name}' answers a {spec.type} question, "
                                f"not a {self.type}", question=q, head=spec.name, head_type=spec.type)
        if self.type == "choice" and self.criteria is not None:
            want, got = set(spec.classes), set(self.criteria)
            if want != got:
                raise ContractError(
                    f"question '{q}': criteria options {sorted(got)} do not match head '{spec.name}' classes "
                    f"{list(spec.classes)}", question=q, head=spec.name, classes=list(spec.classes))
        if self.type == "score" and self.criteria is not None and len(self.criteria) != len(spec.classes):
            raise ContractError(
                f"question '{q}': head '{spec.name}' has {len(spec.classes)} levels, criteria has "
                f"{len(self.criteria)}", question=q, head=spec.name, levels=len(spec.classes))


def parse_questions(raw: Any) -> list[Question]:
    if not isinstance(raw, dict) or not raw:
        raise ContractError("questions must be a non-empty object mapping question id -> question")
    if len(raw) > MAX_QUESTIONS:
        raise ContractError(f"at most {MAX_QUESTIONS} questions per call, got {len(raw)}", limit=MAX_QUESTIONS)
    return [Question(str(qid), q) for qid, q in raw.items()]


# ----------------------------------------------------------------------------- answers

def _r(p: float) -> float:
    return round(float(p), 6)


def noul_answer(p: float, *, head: str, calibrated: bool) -> dict:
    p = min(1.0, max(0.0, float(p)))
    return {"type": "noul", "noul": _r(p), "confidence": _r(noul_confidence(p)), "head": head,
            "calibrated": calibrated}


def choice_answer(options: Sequence[str], probabilities: Sequence[float], *, head: str, calibrated: bool) -> dict:
    ps = [float(p) for p in probabilities]
    best = max(range(len(ps)), key=lambda i: (ps[i], -i))
    return {"type": "choice", "choice": options[best],
            "probabilities": {o: _r(p) for o, p in zip(options, ps, strict=True)},
            "confidence": _r(choice_confidence(ps)), "head": head, "calibrated": calibrated}


def score_answer(levels: Sequence[str], probabilities: Sequence[float], *, head: str, calibrated: bool) -> dict:
    """Levels are indexed from 0 in the order of `criteria`; `legend` maps index -> description."""
    ps = [float(p) for p in probabilities]
    best = max(range(len(ps)), key=lambda i: (ps[i], -i))
    return {"type": "score", "score": best, "legend": {str(i): d for i, d in enumerate(levels)},
            "probabilities": {str(i): _r(p) for i, p in enumerate(ps)},
            "confidence": _r(score_confidence(ps)), "head": head, "calibrated": calibrated}


def build_answer(question: Question | None, spec: Any, probabilities: Sequence[float], calibrated: bool,
                 skill: bool | None = None) -> dict:
    """`probabilities` are in the order of `spec.labels` (for a noul head: [P(false), P(true)]).

    `skill` (open-rlcd extension, additive like `head` and `calibrated`): whether the head's trusted holdout showed
    reliable skill over the base rate. With `skill: false` the probabilities are still calibrated, but they are
    base-rate knowledge: do not gate actions on `confidence` alone."""
    if spec.type == "noul":
        ans = noul_answer(probabilities[-1], head=spec.name, calibrated=calibrated)
    elif spec.type == "choice":
        ans = choice_answer(list(spec.classes), probabilities, head=spec.name, calibrated=calibrated)
    else:
        levels = question.criteria if question is not None and question.criteria is not None else list(spec.legend)
        ans = score_answer(levels, probabilities, head=spec.name, calibrated=calibrated)
    if skill is not None:
        ans["skill"] = bool(skill)
    return ans


def positive_probability(answer: dict, option: str | None) -> float | None:
    """The probability /v1/rank sorts by: the yes-probability of a noul, or one named option of a choice.
    None when the head does not apply to the candidate."""
    if answer["type"] == "noul":
        return None if answer["noul"] is None else float(answer["noul"])
    if answer["type"] == "choice":
        if option is None:
            raise ContractError("ranking by a choice question needs `option`: the option whose probability "
                                "is the sort key", options=list(answer["probabilities"]))
        if option not in answer["probabilities"]:
            raise ContractError(f"option {option!r} is not one of {list(answer['probabilities'])}",
                                options=list(answer["probabilities"]))
        return float(answer["probabilities"][option])
    raise ContractError("ranking needs a noul or a choice question")
