"""Open RLCD — FastAPI app.

    rlcd serve --port 8095                     (or: uvicorn open_rlcd.serve.api:app --port 8095)

Errors are `{"error": "...", "kind": "...", "retryable": bool}`: 401 (only when RLCD_API_KEY is set; with
RLCD_ADMIN_KEY set, POST /v1/train, POST /v1/automation and POST /v1/receipts/{id}/confirm need that key instead —
RLCD_ADMIN_KEY alone protects only those three routes, every other route stays open until RLCD_API_KEY is set,
which is why `rlcd serve` refuses a non-loopback host without RLCD_API_KEY), 404 an
unknown receipt, 413 a request body larger than RLCD_MAX_BODY_BYTES (default 16 MiB), 422 validation / unknown
head / unknown model / feature version mismatch / missing evidence, 409 a training job is already running or the
kill switch is locked by the environment, 503 a connector or the encoder is unreachable (and GET /ready while no
trained model can decide: `not_ready`), 500 `internal`. GET /health (liveness) and GET /ready (readiness) never
need a key.
`retryable` says whether the same request may succeed later unchanged (409 busy, 503): a client retries those
with backoff (a `Retry-After` header gives the delay) and never retries the others — a 422 needs a different
request, not another attempt. A 500 or 503 body never carries exception text (it can hold absolute paths or a
DSN): the details go to the server log, the client gets the exception type and a generic message.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import VERSION
from ..connectors.base import ConnectorError, unavailable
from ..core.config import Project
from ..core.contract import (
    AutomationRequest,
    ConfirmRequest,
    ContractError,
    DecideRequest,
    EvaluateRequest,
    FeedbackRequest,
    OutcomeRequest,
    RankRequest,
    ReceiptCheckRequest,
    SystemOneRequest,
    TrainRequest,
)
from ..core.jobs import Busy
from ..core.service import Service
from ..core.training import TrainError

log = logging.getLogger("open_rlcd")
DEFAULT_MAX_BODY_BYTES = 16 << 20


def max_body_bytes() -> int:
    try:
        return int(os.environ.get("RLCD_MAX_BODY_BYTES") or DEFAULT_MAX_BODY_BYTES)
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES


RETRYABLE = {409: 5, 429: 1, 503: 2}     # status -> Retry-After seconds
ADMIN_PATHS = {"/v1/train", "/v1/automation"}   # write routes that need RLCD_ADMIN_KEY when it is set
OPEN_PATHS = {"/health", "/ready"}              # probes: never need a key, never reveal the server's paths
# a person's confirmation of a confirm decision: a caller holding only the decision key must not confirm its own
_CONFIRM = re.compile(r"/v1/receipts/[^/]+/confirm")


def retryable(status: int, kind: str | None = None) -> bool:
    return status in RETRYABLE and kind != "automation_locked"


def _headers(status: int, kind: str | None = None) -> dict | None:
    return {"Retry-After": str(RETRYABLE[status])} if retryable(status, kind) else None


def route_path(request: Request) -> str:
    """The path the router matches, without the ASGI root_path (`uvicorn --root-path /rlcd`, a proxy prefix):
    `request.url.path` includes the prefix, so comparing it with "/v1/train" would let every prefixed admin call
    through."""
    path = request.scope.get("path") or ""
    root = request.scope.get("root_path") or ""
    if root and path.startswith(root):
        path = path[len(root):] or "/"
    return path


class BodyLimit:
    """Refuse request bodies larger than RLCD_MAX_BODY_BYTES whatever their headers say: the body is read here, up to
    the limit, before the app sees it, and answered 413 as soon as it passes the limit (a Content-Length above the
    limit is refused earlier, from the header, by the auth middleware)."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # every body is counted as it arrives, whatever the headers declare (a Content-Length next to a chunked
        # body, or a length the server did not enforce): what is read is what is limited
        limit = max_body_bytes()
        chunks: list[bytes] = []
        seen = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":      # a disconnect: let the app see it as it would
                break
            body = message.get("body") or b""
            seen += len(body)
            if seen > limit:
                resp = error(413, f"request body larger than the {limit} bytes allowed (RLCD_MAX_BODY_BYTES)",
                             "too_large", limit=limit)
                await resp(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body"):
                break
        done = False

        async def replay() -> dict:
            nonlocal done
            if not done:
                done = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def error(status: int, message: str, kind: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, "kind": kind, "retryable": retryable(status, kind), **extra},
                        status_code=status, headers=_headers(status, kind))


def create_app(service: Service | None = None, project: Project | None = None, home: Path | None = None,
               pack: Any = None) -> FastAPI:
    svc = service or Service(project, home=home, pack=pack)
    app = FastAPI(title="Open RLCD", version=VERSION,
                  description="Typed decisions with calibrated probabilities. A decision model, not a language "
                              "model: it never writes prose.")
    app.state.svc = svc
    app.add_middleware(BodyLimit)

    # ------------------------------------------------------------------ errors and auth
    @app.exception_handler(ContractError)
    async def _contract(_: Request, e: ContractError):
        body = {**e.body(), "retryable": retryable(e.status, e.kind)}
        return JSONResponse(body, status_code=e.status, headers=_headers(e.status, e.kind))

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in d.get("loc", []) if p != "body"), "message": d.get("msg")}
                   for d in e.errors()]
        msg = "; ".join(f"{d['field'] or 'body'}: {d['message']}" for d in details)
        return error(422, f"invalid request: {msg}", "validation", details=details)

    @app.exception_handler(ConnectorError)
    async def _connector(_: Request, e: ConnectorError):
        log.error("connector unavailable: %s", e)
        err = unavailable(f"a configured vector or graph store is unavailable ({type(e).__name__}; details are in "
                          f"the server log)")
        return JSONResponse({**err.body(), "retryable": True}, status_code=err.status,
                            headers=_headers(err.status, err.kind))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        kind = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
        return error(e.status_code, str(e.detail), kind)

    @app.exception_handler(Exception)
    async def _any(_: Request, e: Exception):
        log.exception("unhandled")
        return error(500, f"{type(e).__name__}: internal error (details are in the server log)", "internal")

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        key = os.environ.get("RLCD_API_KEY")
        admin = os.environ.get("RLCD_ADMIN_KEY")
        auth = request.headers.get("authorization", "")
        got = request.headers.get("x-api-key") or (auth[7:] if auth.lower().startswith("bearer ") else "")
        path = route_path(request)
        if admin and request.method != "GET" and (path in ADMIN_PATHS or _CONFIRM.fullmatch(path)):
            # the operator's controls (training, the kill switch) need the admin key when one is set, so a decision
            # caller's key can neither retrain nor switch automation back on
            if not hmac.compare_digest(got.encode(), admin.encode()):
                return error(401, f"{request.url.path} needs the admin key (RLCD_ADMIN_KEY)", "unauthorized")
        elif key and path not in OPEN_PATHS and not (
                hmac.compare_digest(got.encode(), key.encode())
                or (admin and hmac.compare_digest(got.encode(), admin.encode()))):
            return error(401, "missing or wrong API key (send `Authorization: Bearer <key>`)", "unauthorized")
        limit = max_body_bytes()
        try:
            size = int(request.headers.get("content-length") or 0)
        except ValueError:
            return error(400, "invalid Content-Length header", "validation")
        if size > limit:
            return error(413, f"request body of {size} bytes is larger than the {limit} allowed (RLCD_MAX_BODY_BYTES)",
                         "too_large", limit=limit)
        return await call_next(request)

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health():
        h = svc.health()
        if os.environ.get("RLCD_API_KEY"):   # /health is open: with a key set it does not reveal the server's paths
            h.pop("home", None)
        return h

    @app.get("/ready")
    def ready():
        """Readiness: 200 when a trained model can decide, 503 `not_ready` (with `reasons`) until then. /health
        stays the liveness probe: it answers 200 whenever the process is up."""
        r = svc.readiness()
        if r["ready"]:
            return {"status": "ready", **r}
        body = {"error": "not ready to decide: " + "; ".join(r["reasons"]), "kind": "not_ready", "retryable": True,
                "status": "not_ready", **r}
        return JSONResponse(body, status_code=503, headers=_headers(503, "not_ready"))

    @app.get("/v1/models")
    def models():
        return svc.models_view()

    @app.get("/v1/heads")
    def heads(model: str | None = None):
        return svc.heads_view(model)

    # ------------------------------------------------------------------ answering
    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        return svc.systemone(req)

    @app.post("/v1/rank")
    def rank(req: RankRequest):
        return svc.rank(req)

    @app.post("/v1/decide")
    def decide(req: DecideRequest):
        return svc.decide(req)

    @app.post("/v1/evaluate")
    def evaluate(req: EvaluateRequest):
        return svc.evaluate(req)

    # ------------------------------------------------------------------ the reinforcement loop
    @app.post("/v1/feedback")
    def feedback(req: FeedbackRequest):
        return svc.feedback(req)

    @app.post("/v1/outcome")
    def outcome(req: OutcomeRequest):
        return svc.outcome(req)

    @app.post("/v1/train")
    def train(req: TrainRequest | None = None):
        try:
            job = svc.trainer.start(req or TrainRequest())
        except Busy as e:
            return error(409, f"training job {e.job_id} is still running", "busy", job_id=e.job_id)
        except TrainError as e:
            raise ContractError(str(e)) from None
        return JSONResponse({"job_id": job["job_id"], "status": job["status"], "heads": list(job["heads"]),
                             "poll": f"/v1/train/{job['job_id']}"}, status_code=202)

    @app.get("/v1/train/{job_id}")
    def train_status(job_id: str):
        job = svc.trainer.get(job_id)
        if job is None:
            return error(404, f"unknown training job {job_id!r} (jobs are kept in memory until restart)",
                         "unknown_job")
        return job

    # ------------------------------------------------------------------ reports
    @app.get("/v1/calibration")
    def calibration(head: str, model: str | None = None, segment: str | None = None):
        return svc.calibration_view(head, model, segment)

    @app.get("/v1/bernoulli")
    def bernoulli(head: str, model: str | None = None, stakes: str | None = None):
        return svc.bernoulli_view(head, model, stakes)

    # ------------------------------------------------------------------ the control layer
    @app.get("/v1/batteries")
    def batteries(model: str | None = None):
        return svc.batteries_view(model)

    @app.get("/v1/receipts/{decision_id}")
    def receipt(decision_id: str):
        return svc.receipt_view(decision_id)

    @app.post("/v1/receipts/{decision_id}/check")
    def check_receipt(decision_id: str, req: ReceiptCheckRequest):
        kw: dict[str, Any] = {"state_digest": req.state_digest, "max_age_s": req.max_age_s}
        if "state" in req.model_fields_set:
            kw["state"] = req.state
        return svc.check_receipt(decision_id, **kw)

    @app.post("/v1/receipts/{decision_id}/confirm")
    def confirm_receipt(decision_id: str, req: ConfirmRequest):
        return svc.confirm_receipt(decision_id, by=req.by, note=req.note, state_digest=req.state_digest)

    @app.get("/v1/metrics")
    def metrics(since: str | None = None, until: str | None = None, model: str | None = None,
                kind: str | None = None, head: str | None = None):
        return svc.metrics_view(since=since, until=until, model=model, kind=kind, head=head)

    @app.get("/v1/review")
    def review(limit: int = Query(50, ge=1, le=1000), head: str | None = None, kind: str | None = None,
               include_abstain: bool = True, oldest_first: bool = True):
        return svc.review_view(limit=limit, head=head, kind=kind, include_abstain=include_abstain,
                               oldest_first=oldest_first)

    @app.get("/v1/check")
    def check(model: str | None = None):
        from ..core.checklist import run as run_checklist

        return run_checklist(svc, model=model, error_text=False)

    @app.get("/v1/automation")
    def automation():
        return svc.automation_view()

    @app.post("/v1/automation")
    def set_automation(req: AutomationRequest):
        return svc.set_automation(req.enabled, reason=req.reason, by=req.by)

    return app


def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name: str):
    # `uvicorn open_rlcd.serve.api:app` builds the app lazily, from rlcd.yaml / RLCD_HOME of the working directory
    if name == "app":
        return _default_app()
    raise AttributeError(name)
