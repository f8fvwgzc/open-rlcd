"""The documents pack: the calibrated router / gatekeeper / evaluator around a RAG system.

Eight heads (pack.yaml): response_type, route, needs_graph on the query; chunk_relevant on (query, chunk) for
/v1/rank; answerable on (query, evidence) as the gate before an LLM writes; grounded, complete and risk_tier on a
drafted answer (evaluator mode). `decide()` composes the first five into a plan, `evaluate()` the last three into
ship / review / reject. The pack never writes prose.

    from open_rlcd.packs.documents import DocumentsPack, write_synth
    write_synth(home, {"kind": "hashing", "dim": 256})        # synthetic corpus + labels, offline
    Service(project).trainer.start(TrainRequest(), wait=True)

Modules: provider.py (features), corpus.py (local corpus: chunks, vectors, graph, BM25), decide.py (decide /
evaluate), tasks.py (label tasks from a query log), world.py + synth.py (synthetic data), settings.py
(pack_config keys).

What it does not do: only the local corpus connectors exist (Qdrant / pgvector / Neo4j are not implemented
here), retrieval for `decide()` runs only against the local corpus or the candidates the caller sends, and the
heads judge relations between texts with features, not by reading them like an LLM would.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from ...core.config import Project, load_project
from ...core.contract import ContractError
from ...core.dataset import Dataset
from ...features.text import EncoderError, EncoderMismatch, make_encoder
from ...labeling.store import LabelItem, LabelStore
from ..base import BasePack, dataset_from_labels
from .provider import VIEWS, DocumentsProvider
from .settings import DEFAULT_ENCODER, config_keys, corpus_settings, encoder_settings

PROVIDER_CONFIG_VERSION = 1


class DocumentsPack(BasePack):
    name = "documents"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")
    abstain = "escalate"
    rank_head = "chunk_relevant"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._defaults: dict | None = None
        self._corpora: dict[tuple[str, str], tuple[int, Any]] = {}

    def defaults(self) -> dict:
        if self._defaults is None:
            self._defaults = super().defaults()
        return self._defaults

    def config_keys(self) -> dict:
        return config_keys()

    # ------------------------------------------------------------------ features
    def provider(self, project: Project, config: dict | None = None) -> DocumentsProvider:
        """Without `config`: the encoder of pack_config.encoder (default ONNX multilingual-e5-small). With a bundle's
        `config`: the exact encoder the bundle was trained with, refused when its sha256 or the feature schema
        differs (the trained weights would be meaningless on other features)."""
        cache = project.pack_config.get("feature_cache")
        opts = {"cache_size": 1024 if cache is None else cache}     # `feature_cache:` with no value is the default
        try:
            if not config:
                return DocumentsProvider(make_encoder(encoder_settings(project.pack_config), project.home), opts)
            if int(config.get("version") or 0) != PROVIDER_CONFIG_VERSION:
                raise ContractError(f"the bundle's documents provider config has version {config.get('version')!r}; "
                                    f"this open-rlcd reads version {PROVIDER_CONFIG_VERSION}", kind="bad_bundle")
            enc_cfg = config.get("encoder")
            if not isinstance(enc_cfg, dict) or not enc_cfg.get("kind"):
                raise ContractError("the bundle's documents provider config has no encoder", kind="bad_bundle")
            enc = make_encoder(enc_cfg, project.home)
            want = enc_cfg.get("sha256")
            if want and enc.sha256 != want:
                raise EncoderMismatch(f"encoder {enc.id} has sha256 {enc.sha256[:16]}..., the bundle was trained "
                                      f"with {str(want)[:16]}...")
            prov = DocumentsProvider(enc, opts)
            got = prov.to_config()["views"]
            for view, n in (config.get("views") or {}).items():
                if got.get(view) != n:
                    raise ContractError(f"feature schema mismatch for view {view!r}: the bundle has {n} features, this "
                                        f"provider builds {got.get(view)}; retrain or use the matching open-rlcd",
                                        kind="feature_schema_mismatch")
            return prov
        except EncoderMismatch as e:
            raise ContractError(f"refusing to serve with a different encoder: {e}. The heads were trained on that "
                                f"encoder's embeddings; restore the original model file or retrain.",
                                kind="encoder_mismatch") from None
        except EncoderError as e:
            raise ContractError(f"the text encoder is not available: {e}", kind="encoder_unavailable",
                                status=503) from None

    def datasets(self, project: Project, provider: Any, heads: list[str], feedback: list[dict] | None = None,
                 feedback_weight: float = 5.0) -> dict[str, Dataset]:
        """Rows per head from the label store (trusted = human / ground_truth / synthetic; judge rows only train)
        plus feedback rows. Heads that share a view share one featurisation (the provider's cache is widened)."""
        specs = self.heads(project)
        store = LabelStore(project.home)
        unknown = [h for h in heads if h not in specs]
        if unknown:
            raise ContractError(f"unknown heads {unknown}; the documents pack defines {list(specs)}")
        labelled = {h: store.labelled(h, specs[h].view) for h in heads}
        reserve = getattr(provider, "reserve_cache", None)
        if callable(reserve):
            reserve(sum(len(v) for v in labelled.values()) + len(feedback or []))
        out = {}
        for h in heads:
            out[h] = dataset_from_labels(specs[h], provider, labelled[h], feedback, feedback_weight,
                                         group_of=_query_group)
        return out

    # ------------------------------------------------------------------ corpus
    def corpus(self, project: Project, provider: Any) -> Any:
        """The project's local corpus opened with the provider's encoder (cached until the next ingest), or None
        when nothing was ingested. A corpus embedded with another encoder raises ContractError (encoder_mismatch)."""
        from .corpus import Corpus

        meta = Path(project.home) / "corpus" / "meta.json"
        if not meta.is_file():
            return None
        key = (str(project.home), provider.encoder.sha256)
        mtime = meta.stat().st_mtime_ns
        with self._lock:
            hit = self._corpora.get(key)
            if hit is not None and hit[0] == mtime:
                return hit[1]
            c = Corpus.open(project.home, provider.encoder, corpus_settings(project.pack_config))
            self._corpora[key] = (mtime, c)
            return c

    def ingest(self, project: Project, args: dict) -> dict:
        """`rlcd ingest [PATH]`: read a folder (txt, md, html, pdf, json, jsonl) into the local corpus with the
        project's encoder. Remote connectors (--qdrant / --neo4j / --csv) are not available in this pack yet."""
        from .corpus import Corpus

        for key in ("qdrant", "neo4j", "csv", "collection"):
            if args.get(key):
                raise ContractError(f"--{key} is not available for the documents pack yet: only local folders can be "
                                    f"ingested", kind="unsupported_connector")
        cfg = project.pack_config
        path = args.get("path") or cfg.get("corpus")
        if not path:
            raise ContractError("nothing to ingest: give a folder (`rlcd ingest ./data`) or set pack_config.corpus")
        chunking = cfg.get("chunking") if isinstance(cfg.get("chunking"), dict) else {}
        provider = self.provider(project)
        corpus = Corpus(project.home, provider.encoder, corpus_settings(cfg))
        out = corpus.ingest_folder(project.resolve_path(path),
                                   max_chars=int(args.get("max_chars") or chunking.get("max_chars") or 1200),
                                   overlap=int(chunking.get("overlap") if chunking.get("overlap") is not None else 150))
        with self._lock:
            self._corpora.pop((str(project.home), provider.encoder.sha256), None)
        return {**out, "corpus": corpus.stats()}

    # ------------------------------------------------------------------ labels and synthetic data
    def label_tasks(self, project: Project, provider: Any = None) -> list[LabelItem]:
        from .tasks import label_tasks

        provider = provider or self.provider(project)
        return label_tasks(project, provider, self.corpus(project, provider))

    def synth(self, seed: int = 7, n: int = 900, home: Path | None = None,
              encoder_config: dict | None = None) -> dict:
        """`write_synth` into `home` (default: the current project's home, i.e. RLCD_HOME / ./var) with `n` queries.
        `encoder_config` must match the project's pack_config.encoder; None = the pack default (ONNX e5-small)."""
        from .synth import write_synth

        home = Path(home) if home is not None else load_project().home
        return write_synth(home, encoder_config, seed=seed, n_queries=n)

    # ------------------------------------------------------------------ decisions
    def decide(self, service: Any, request: dict) -> dict:
        from .decide import decide

        return decide(self, service, request)

    def evaluate(self, service: Any, request: dict) -> dict:
        from .decide import evaluate

        return evaluate(self, service, request)


def write_synth(home: Path, encoder_config: dict | None = None, seed: int = 7, n_queries: int = 900) -> dict:
    """See `synth.write_synth` (imported lazily so that loading the pack stays cheap)."""
    from .synth import write_synth as _write

    return _write(home, encoder_config, seed=seed, n_queries=n_queries)


__all__ = ["DEFAULT_ENCODER", "VIEWS", "DocumentsPack", "DocumentsProvider", "write_synth"]


def _query_group(state: object) -> str | None:
    """The group a feedback state belongs to when its labelled twin used the default query group (tasks.py): the
    same question is one group, so its feedback never trains while its labelled row is scored in the holdout."""
    if isinstance(state, dict) and isinstance(state.get("query"), str):
        return f"query:{hashlib.sha256(state['query'].encode()).hexdigest()[:12]}"
    return None
