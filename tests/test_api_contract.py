"""The HTTP contract a View Engine `/v1/systemone` client relies on, over the real FastAPI app with the toy pack.

Untrained: every answer is the uninformed prior, flagged calibrated=false, with a warning. Trained: the same
shapes, calibrated=true, and model ids rlcd-0.1.N with the rlcd-latest alias.
"""
import pytest
from fastapi.testclient import TestClient

from open_rlcd.core.config import Project
from open_rlcd.core.contract import TrainRequest
from open_rlcd.core.service import Service
from open_rlcd.serve.api import create_app
from tests.toypack import ToyPack

# The View Engine RLCD answer keys, plus open-rlcd's additive `skill` flag (clients parsing JSON loosely, as the
# View Engine client does, are unaffected; see docs/contract.md).
NOUL_KEYS = {"type", "noul", "confidence", "head", "calibrated", "skill"}
CHOICE_KEYS = {"type", "choice", "probabilities", "confidence", "head", "calibrated", "skill"}
SCORE_KEYS = {"type", "score", "legend", "probabilities", "confidence", "head", "calibrated", "skill"}
ALL = {"good": {"type": "noul"}, "color": {"type": "choice"}, "level": {"type": "score"}}
STATE = {"f": [0.5, -1.0, 0.2, 0.0, 1.0, -0.3]}


def make_client(tmp_path, **pack) -> TestClient:
    svc = Service(Project(root=tmp_path), home=tmp_path, pack=ToyPack(**pack))
    return TestClient(create_app(svc))


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    c = make_client(tmp_path_factory.mktemp("api-contract"))
    job = c.app.state.svc.trainer.start(TrainRequest(), wait=True)
    assert job["status"] == "succeeded", job
    return c


# ----------------------------------------------------------------------------- untrained

def test_untrained_heads_answer_with_the_prior_and_a_warning(client):
    r = client.post("/v1/systemone", json={"model": "rlcd-latest", "state": STATE, "questions": ALL})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"model", "answers", "usage", "warnings"} and body["model"] == "rlcd-0.1.0"
    assert body["usage"]["input_tokens"] == 0 and body["usage"]["output_tokens"] == 0
    assert body["usage"]["questions"] == 3
    a = body["answers"]
    assert a["good"] == {"type": "noul", "noul": 0.5, "confidence": 0.0, "head": "good", "calibrated": False,
                         "skill": False}
    assert set(a["color"]) == CHOICE_KEYS and a["color"]["calibrated"] is False and a["color"]["confidence"] == 0
    assert a["color"]["probabilities"] == {c: pytest.approx(1 / 3, abs=1e-6) for c in ("red", "green", "blue")}
    assert set(a["level"]) == SCORE_KEYS and a["level"]["legend"] == {"0": "none", "1": "low", "2": "mid", "3": "high"}
    assert a["level"]["probabilities"] == {str(i): 0.25 for i in range(4)} and a["level"]["score"] == 0
    assert len(body["warnings"]) == 3 and all("not trained" in w for w in body["warnings"])
    # the untrained service never looks at the state: any state answers with the prior
    assert client.post("/v1/systemone", json={"state": "hi", "questions": {"good": {"type": "noul"}}}).status_code == 200


def test_question_ids_bind_to_heads_through_the_head_field(client):
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {
        "is_it_good": {"type": "noul", "head": "good", "instructions": "ignored: the head defines the question",
                       "criteria": {"true": "yes", "false": "no"}}}})
    assert r.status_code == 200 and r.json()["answers"]["is_it_good"]["head"] == "good"


def test_unknown_head_is_422_and_lists_heads(client):
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"is_spam": {"type": "noul"}}})
    assert r.status_code == 422
    body = r.json()
    assert body["kind"] == "unknown_head" and body["available_heads"] == ["good", "color", "level"]
    assert body["question"] == "is_spam" and body["error"]


def test_choice_limit_255_options(client):
    many = {f"o{i}": "x" for i in range(256)}
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"color": {"type": "choice",
                                                                                 "criteria": many}}})
    assert r.status_code == 422 and "255" in r.json()["error"] and r.json()["kind"] == "validation"


@pytest.mark.parametrize("levels", [1, 11])
def test_score_limit_2_to_10_levels(client, levels):
    q = {"type": "score", "head": "level", "criteria": [f"level {i}" for i in range(levels)]}
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"q": q}})
    assert r.status_code == 422 and "2 to 10" in r.json()["error"]


def test_score_question_on_a_choice_head_is_rejected(client):
    q = {"type": "score", "head": "color", "criteria": ["low", "mid", "high"]}
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"q": q}})
    assert r.status_code == 422 and "answers a choice question" in r.json()["error"]
    assert r.json()["head_type"] == "choice"


def test_criteria_must_match_head_classes(client):
    q = {"type": "choice", "head": "color", "criteria": {"red": "a", "purple": "b"}}
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"q": q}})
    assert r.status_code == 422 and r.json()["classes"] == ["red", "green", "blue"]
    q = {"type": "score", "head": "level", "criteria": ["a", "b", "c"]}
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"q": q}})
    assert r.status_code == 422 and r.json()["levels"] == 4


@pytest.mark.parametrize("body", [{}, {"state": "hi"}, {"state": "hi", "questions": {}},
                                  {"state": "hi", "questions": []},
                                  {"state": "hi", "questions": {"good": {"type": "essay"}}},
                                  {"state": "hi", "questions": {"good": "noul"}},
                                  {"state": "hi", "questions": {f"q{i}": {"type": "noul"} for i in range(65)}}])
def test_malformed_requests_are_422_with_a_body(client, body):
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422 and r.json()["error"] and r.json()["kind"] == "validation"


def test_unparseable_json_is_a_validation_error(client):
    r = client.post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["kind"] == "validation" and r.json()["error"]


def test_unknown_model_is_422(client):
    r = client.post("/v1/systemone", json={"model": "rlcd-9.9.9", "state": STATE, "questions": ALL})
    assert r.status_code == 422
    assert r.json()["kind"] == "unknown_model" and r.json()["available"] == ["rlcd-latest"]
    # the untrained id itself and the alias both resolve
    for model in ("rlcd-0.1.0", "rlcd-latest", None):
        r = client.post("/v1/systemone", json={"model": model, "state": STATE, "questions": ALL})
        assert r.status_code == 200 and r.json()["model"] == "rlcd-0.1.0"


def test_health_models_heads_untrained(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "rlcd-0.1.0" and h["alias"] == "rlcd-latest"
    assert h["heads"] == {n: {"trained": False} for n in ("good", "color", "level")}
    assert h["training"] is None and h["pack"] == "toy"
    m = client.get("/v1/models").json()["data"]
    assert m[0] == {"id": "rlcd-latest", "alias_of": "rlcd-0.1.0"}
    assert m[1]["id"] == "rlcd-0.1.0" and m[1]["trained_at"] is None and len(m) == 2
    body = client.get("/v1/heads").json()
    assert body["model"] == "rlcd-0.1.0"
    heads = {x["name"]: x for x in body["heads"]}
    assert list(heads) == ["good", "color", "level"]
    assert heads["good"]["type"] == "noul" and heads["good"]["decision"] == "gate" and heads["good"]["classes"] == []
    assert heads["color"]["classes"] == ["red", "green", "blue"] and not heads["color"]["trained"]
    assert heads["level"]["legend"] == {"0": "none", "1": "low", "2": "mid", "3": "high"}
    assert heads["level"]["input"] == {"f": "array of 6 numbers"}
    assert all(x["metrics"] is None and x["verdict"] is None and x["feedback_rows"] == 0 for x in heads.values())


def test_untrained_decide_escalates_and_reports_have_nothing_yet(client):
    r = client.post("/v1/decide", json={"state": STATE})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["action"] == "escalate" and d["model"] == "rlcd-0.1.0" and d["decision_id"]
    assert all(h["decision"]["tier"] == "escalate" for h in d["heads"].values())
    assert any("not trained" in w for w in d["warnings"])
    r = client.get("/v1/calibration", params={"head": "good"})
    assert r.status_code == 422 and r.json()["kind"] == "untrained"
    assert client.get("/v1/bernoulli", params={"head": "good"}).json()["kind"] == "untrained"
    assert client.get("/v1/calibration", params={"head": "nope"}).json()["kind"] == "unknown_head"


def test_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("RLCD_API_KEY", "s3cret")
    r = client.get("/v1/heads")
    assert r.status_code == 401 and r.json()["kind"] == "unauthorized" and r.json()["error"]
    assert client.post("/v1/systemone", json={"state": STATE, "questions": ALL}).status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/v1/heads", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/v1/heads", headers={"authorization": "bearer s3cret"}).status_code == 200
    assert client.get("/v1/heads", headers={"x-api-key": "s3cret"}).status_code == 200
    assert client.get("/v1/heads", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/heads", headers={"x-api-key": "s3cret-but-longer"}).status_code == 401


# ----------------------------------------------------------------------------- trained (View Engine compatibility)

def test_trained_answers_keep_the_exact_shapes(trained):
    r = trained.post("/v1/systemone", json={"state": STATE, "questions": ALL})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "rlcd-0.1.1"
    assert set(body) == {"model", "answers", "usage"} | ({"warnings"} if "warnings" in body else set())
    assert body["usage"]["input_tokens"] == body["usage"]["output_tokens"] == 0
    a = body["answers"]
    assert set(a["good"]) == NOUL_KEYS and set(a["color"]) == CHOICE_KEYS and set(a["level"]) == SCORE_KEYS
    assert all(x["calibrated"] is True for x in a.values())
    assert a["good"]["confidence"] == pytest.approx(abs(2 * a["good"]["noul"] - 1), abs=1e-5)
    assert sum(a["color"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert sum(a["level"]["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert a["color"]["choice"] == max(a["color"]["probabilities"], key=a["color"]["probabilities"].get)


def test_trained_model_ids_and_alias(trained):
    for model in (None, "rlcd-latest", "rlcd-0.1.1"):
        r = trained.post("/v1/systemone", json={"model": model, "state": STATE, "questions": ALL})
        assert r.status_code == 200 and r.json()["model"] == "rlcd-0.1.1"
    # rlcd-0.1.0 (the untrained service) is no longer addressable once a version exists
    r = trained.post("/v1/systemone", json={"model": "rlcd-0.1.0", "state": STATE, "questions": ALL})
    assert r.status_code == 422 and r.json()["available"] == ["rlcd-latest", "rlcd-0.1.1"]
    m = trained.get("/v1/models").json()["data"]
    assert m[0] == {"id": "rlcd-latest", "alias_of": "rlcd-0.1.1"} and m[1]["id"] == "rlcd-0.1.1"
    assert m[1]["trained_at"] and set(m[1]["heads"]) == {"good", "color", "level"} and m[1]["pack"] == "toy"
    assert trained.get("/health").json()["heads"]["good"] == {"trained": True}


def test_trained_bad_state_is_a_validation_error(trained):
    r = trained.post("/v1/systemone", json={"state": {"f": [1, 2]}, "questions": ALL})
    assert r.status_code == 422 and r.json()["kind"] == "validation" and "6 numbers" in r.json()["error"]


def test_no_skill_heads_say_so_on_every_answer(tmp_path):
    """A head whose holdout shows no skill still answers with calibrated probabilities, but each answer carries
    `skill: false`, so a client gating on `confidence` can see that it is base-rate knowledge."""
    from fastapi.testclient import TestClient

    from open_rlcd.core.config import Project
    from open_rlcd.core.service import Service
    from open_rlcd.serve.api import create_app
    from tests.toypack import ToyPack

    noise = TestClient(create_app(Service(Project(root=tmp_path), home=tmp_path / "var", pack=ToyPack(signal=0.0))))
    job = noise.post("/v1/train", json={"heads": ["good"]}).json()
    import time
    for _ in range(600):
        if noise.get(f"/v1/train/{job['job_id']}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    r = noise.post("/v1/systemone", json={"state": {"f": [1, 0, 0, 0, 0, 0]},
                                          "questions": {"g": {"type": "noul", "head": "good"}}}).json()
    assert r["answers"]["g"]["calibrated"] is True and r["answers"]["g"]["skill"] is False
    assert any("no skill" in w for w in r["warnings"])


def test_unknown_routes_use_the_error_body(client=None):
    from fastapi.testclient import TestClient

    from open_rlcd.serve.api import create_app
    from tests.toypack import ToyPack

    c = TestClient(create_app(pack=ToyPack()))
    r = c.get("/v1/nope")
    assert r.status_code == 404 and r.json()["kind"] == "not_found" and r.json()["error"]
    r = c.get("/v1/systemone")
    assert r.status_code == 405 and r.json()["kind"] == "method_not_allowed"
