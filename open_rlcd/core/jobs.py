"""Training jobs: one at a time, in a background thread (or in the foreground for the CLI).

A job asks the project's pack for each head's rows (labelled rows + feedback rows), trains every head
independently (a failure of one head never costs the others), commits the trained heads as a new version
(the others are carried over) and restarts the Bernoulli ledger of each retrained head from its new holdout.
Job state lives in memory; the trained model is on disk.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any

from .contract import TrainRequest
from .training import TrainConfig, TrainError, now_iso, train_head


class Busy(Exception):
    def __init__(self, job_id: str):
        super().__init__(job_id)
        self.job_id = job_id


class Trainer:
    def __init__(self, service: Any):
        self.svc = service
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def running(self) -> dict | None:
        return next((j for j in self.jobs.values() if j["status"] in ("queued", "running")), None)

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def config(self, req: TrainRequest) -> TrainConfig:
        t = self.svc.project.section("train")
        fields = req.model_fields_set
        return TrainConfig(
            holdout_fraction=float(req.holdout_fraction if "holdout_fraction" in fields
                                   else t.get("holdout_fraction", req.holdout_fraction)),
            n_splits=int(req.n_splits if "n_splits" in fields else t.get("n_splits", req.n_splits)),
            ordering=str(t.get("ordering", "auto")), gap=int(t.get("gap", 0)),
            calibration=str(req.calibration if "calibration" in fields else t.get("calibration", "auto")),
            isotonic_min_per_class=int(t.get("isotonic_min_per_class", 3000)),
            conformal_delta=float(t.get("conformal_delta", 0.1)),
            n_estimators=int(req.n_estimators if "n_estimators" in fields else t.get("n_estimators", req.n_estimators)),
            min_trusted=int(t.get("min_trusted", 30)), seed=int(req.seed if "seed" in fields else t.get("seed", 7)),
            policy=self.svc.policy_config())

    def feedback_weight(self, req: TrainRequest) -> float:
        """The up-weight of feedback rows: the request's when it sets one, else rlcd.yaml `train.feedback_weight`
        (default 5), as for the other training fields."""
        if "feedback_weight" in req.model_fields_set:
            return float(req.feedback_weight)
        raw = self.svc.project.section("train").get("feedback_weight", req.feedback_weight)
        try:
            w = float(raw)
        except (TypeError, ValueError):
            raise TrainError(f"train.feedback_weight in rlcd.yaml must be a number, got {raw!r}") from None
        if not 0 < w <= 1000:
            raise TrainError(f"train.feedback_weight in rlcd.yaml must be in (0, 1000], got {w:g}")
        return w

    def plan(self, req: TrainRequest) -> list[str]:
        specs = self.svc.pack.heads(self.svc.project)
        names = req.heads or list(specs)
        unknown = [h for h in names if h not in specs]
        if unknown:
            raise TrainError(f"unknown heads {unknown}; the '{self.svc.pack.name}' pack defines {list(specs)}")
        self.feedback_weight(req)
        return names

    def start(self, req: TrainRequest | None = None, wait: bool = False) -> dict:
        req = req or TrainRequest()
        names = self.plan(req)
        with self._lock:
            running = self.running()
            if running:
                raise Busy(running["job_id"])
            job = {"job_id": uuid.uuid4().hex[:12], "status": "queued", "created_at": now_iso(), "started_at": None,
                   "finished_at": None, "request": req.model_dump(),
                   "progress": {"done": 0, "total": len(names), "message": "queued"},
                   "heads": {n: {"status": "pending"} for n in names}, "model": None, "error": None}
            self.jobs[job["job_id"]] = job
        if wait:
            self._run(job, req, names)
        else:
            self._thread = threading.Thread(target=self._run, args=(job, req, names), daemon=True,
                                            name=f"rlcd-train-{job['job_id']}")
            self._thread.start()
        return job

    def train_heads(self, provider: Any, names: list[str], cfg: TrainConfig, feedback_weight: float, *,
                    specs: dict | None = None, configs: dict[str, TrainConfig] | None = None,
                    on_head: Any = None) -> tuple[dict[str, Any], dict[str, dict], dict[str, Any]]:
        """Build every head's rows with `provider` and train each head independently: a failure of one head never
        costs the others. `specs` / `configs` override the pack's head specs and the shared config per head (a
        weight edit retrains with its base version's specs and splits). `on_head(name, None)` is called before each
        head trains and `on_head(name, status)` after. Returns (trained heads, status per head, datasets)."""
        pack, project = self.svc.pack, self.svc.project
        specs = {**pack.heads(project), **(specs or {})}
        datasets = pack.datasets(project, provider, names, self.svc.logs.feedback(), feedback_weight)
        trained: dict[str, Any] = {}
        status: dict[str, dict] = {}
        for name in names:
            ds = datasets.get(name)
            hcfg = (configs or {}).get(name) or cfg
            if on_head is not None:
                on_head(name, None)
            try:
                # a head's stakes must be a level the policy defines: a typo in rlcd.yaml fails this head at
                # train time instead of every request it would serve
                if specs[name].stakes not in hcfg.policy.stakes:
                    raise TrainError(f"{name}: stakes {specs[name].stakes!r} is not a level the policy defines; "
                                     f"configured: {list(hcfg.policy.stakes)} (fix `heads.{name}.stakes` or "
                                     f"`policy.stakes` in rlcd.yaml)")
                if ds is None or len(ds) == 0:
                    raise TrainError(f"{name}: no labelled rows (label some first: `rlcd label bootstrap` / "
                                     f"`rlcd label audit`)")
                th = train_head(specs[name], ds, hcfg)
                trained[name] = th
                status[name] = {"status": "trained", "metrics": th.metrics, "verdict": th.verdict,
                                "data": th.data, "notes": th.notes}
            except TrainError as e:
                status[name] = {"status": "failed", "error": str(e)}
            except Exception as e:  # keep the other heads going; the trace goes to the job
                status[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-2000:]}
            if on_head is not None:
                on_head(name, status[name])
        return trained, status, datasets

    def _run(self, job: dict, req: TrainRequest, names: list[str]) -> None:
        job["status"], job["started_at"] = "running", now_iso()
        svc = self.svc

        def say(msg: str) -> None:
            job["progress"]["message"] = msg

        def progress(name: str, status: dict | None) -> None:
            if status is None:
                say(f"training {name}")
                return
            job["heads"][name] = status
            job["progress"]["done"] += 1

        trained: dict[str, Any] = {}
        try:
            pack, project = svc.pack, svc.project
            cfg = self.config(req)
            say("building features")
            provider = pack.provider(project, None)
            trained, _, _ = self.train_heads(provider, names, cfg, self.feedback_weight(req), on_head=progress)
            if trained:
                say("saving the new version")
                sources: dict[str, int] = {}
                for th in trained.values():
                    for s, c in (th.data.get("by_source") or {}).items():
                        sources[s] = sources.get(s, 0) + c
                card = {"pack": pack.name, "label_sources": sources,
                        "synthetic_only": bool(sources) and set(sources) <= {"synthetic"},
                        "heads": {n: th.data for n, th in trained.items()}}
                extra = getattr(provider, "data_card", None)
                if callable(extra):
                    card.update(extra())
                pconf = provider.to_config()
                job["model"] = svc.registry.commit(trained, pack=pack.info(provider),
                                                   feature_version=provider.feature_version,
                                                   encoder=pconf.get("encoder") if isinstance(pconf, dict) else None,
                                                   policy=cfg.policy.to_dict(), data_card=card,
                                                   notes=[f"job {job['job_id']}"])
                svc._providers.pop(job["model"], None)
            failed = [k for k, v in job["heads"].items() if v["status"] == "failed"]
            job["status"] = "succeeded" if not failed else ("partial" if trained else "failed")
            if failed:
                job["error"] = "; ".join(f"{k}: {job['heads'][k]['error']}" for k in failed)
            say("done" if not failed else f"done with {len(failed)} failed head(s)")
        except Exception as e:
            job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
            job["trace"] = traceback.format_exc()[-2000:]
        finally:
            job["finished_at"] = now_iso()
