"""Open RLCD — FastAPI app.

    rlcd serve --port 8095                     (or: uvicorn open_rlcd.serve.api:app --port 8095)

Errors are `{"error": "...", "kind": "..."}`: 401 (only when RLCD_API_KEY is set), 413 a request body larger than
RLCD_MAX_BODY_BYTES (default 16 MiB), 422 validation / unknown head / unknown model / feature version mismatch,
409 a training job is already running, 503 a connector is unreachable (`connector_unavailable`), 500 `internal`.
A 500 or 503 body never carries exception text (it can hold absolute paths or a DSN): the details go to the
server log, the client gets the exception type and a generic message.
"""
from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import VERSION
from ..connectors.base import ConnectorError, unavailable
from ..core.config import Project
from ..core.contract import (
    ContractError,
    DecideRequest,
    EvaluateRequest,
    FeedbackRequest,
    RankRequest,
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


def error(status: int, message: str, kind: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, "kind": kind, **extra}, status_code=status)


def create_app(service: Service | None = None, project: Project | None = None, home: Path | None = None,
               pack: Any = None) -> FastAPI:
    svc = service or Service(project, home=home, pack=pack)
    app = FastAPI(title="Open RLCD", version=VERSION,
                  description="Typed decisions with calibrated probabilities. A decision model, not a language "
                              "model: it never writes prose.")
    app.state.svc = svc

    # ------------------------------------------------------------------ errors and auth
    @app.exception_handler(ContractError)
    async def _contract(_: Request, e: ContractError):
        return JSONResponse(e.body(), status_code=e.status)

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
        return JSONResponse(err.body(), status_code=err.status)

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
        if key and request.url.path != "/health":
            auth = request.headers.get("authorization", "")
            got = request.headers.get("x-api-key") or (auth[7:] if auth.lower().startswith("bearer ") else "")
            if not hmac.compare_digest(got.encode(), key.encode()):
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
        return svc.health()

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

    return app


def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name: str):
    # `uvicorn open_rlcd.serve.api:app` builds the app lazily, from rlcd.yaml / RLCD_HOME of the working directory
    if name == "app":
        return _default_app()
    raise AttributeError(name)
