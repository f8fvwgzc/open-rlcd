"""The ticket-triage pack: subject + body of a support ticket -> a queue (one of five) and "urgent?" (yes/no).

A request's `state` is `{"subject": "...", "body": "..."}`. Labelled tickets come from a JSON Lines file
(`pack_config.data`), one object per line: `{id, subject, body, created_at, queue, urgent, source?}`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from open_rlcd.core.config_check import Key
from open_rlcd.core.contract import ContractError
from open_rlcd.features import HashedNgrams, TextEmbeddings, text_block
from open_rlcd.labeling.store import LabelItem
from open_rlcd.packs.base import BasePack, dataset_from_labels, epoch

VIEW = "ticket"
SHAPE = ["length_chars", "exclamation_marks"]      # two plain numbers next to the text columns


def text_of(state: dict) -> str:
    return f"{state.get('subject') or ''}\n{state.get('body') or ''}"


class TicketFeatures:
    """One feature row per ticket: the text block's columns, then SHAPE."""

    def __init__(self, text: HashedNgrams | TextEmbeddings):
        self.text = text
        # the block's signature changes with its column names (n-grams) or its encoder: so does the version
        self.feature_version = f"tickets-v1:{text.signature}"

    def names(self, view: str) -> list[str]:
        return [*self.text.names(), *SHAPE]

    def transform(self, view: str, states: list) -> np.ndarray:
        for s in states:
            if not isinstance(s, dict) or not isinstance(s.get("subject") or s.get("body"), str):
                raise ContractError("state must be a ticket: {subject: text, body: text}", field="state")  # -> 422
        texts = [text_of(s) for s in states]
        shape = np.asarray([[len(t), t.count("!") + t.count("！")] for t in texts], np.float32).reshape(len(texts), 2)
        return np.hstack([self.text.transform(texts), shape])

    def groups(self, view: str) -> dict[str, str]:
        return {**self.text.groups("wording"), **dict.fromkeys(SHAPE, "shape")}

    def describe(self, view: str) -> dict:
        return {"subject": "string", "body": "string"}

    def to_config(self) -> dict:
        return {"text": self.text.to_config()}


class TicketsPack(BasePack):
    name = "tickets"
    version = "0.1.0"
    yaml_path = Path(__file__).with_name("pack.yaml")

    # ---------------------------------------------------------------- rlcd.yaml: what `rlcd config check` checks
    def config_keys(self) -> dict:
        return {"data": Key("str", default="data/tickets.jsonl"),
                "text": Key("str", default="ngrams", choices=("ngrams", "embedding")),
                "dim": Key("int", default=2048, min=8),
                "encoder": Key("map")}          # read by TextEmbeddings.for_project when text: embedding

    def check_config(self, project) -> list[dict]:
        path = self.data_path(project)
        if path.is_file():
            return []
        return [{"level": "warn", "path": "pack_config.data",
                 "message": f"{path} does not exist yet: run `python make_tickets.py`, or point it at your tickets"}]

    def data_path(self, project) -> Path:
        return project.resolve_path(project.pack_config.get("data") or "data/tickets.jsonl")

    def rows(self, project) -> list[dict]:
        path = self.data_path(project)
        if not path.is_file():
            raise ContractError(f"no labelled tickets at {path}: run `python make_tickets.py`")
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---------------------------------------------------------------- the pack protocol
    def provider(self, project, config=None) -> TicketFeatures:
        if config:                                   # serving a bundle: rebuild exactly what it was trained with
            return TicketFeatures(text_block(config["text"], home=project.home))
        if project.pack_config.get("text") == "embedding":
            return TicketFeatures(TextEmbeddings.for_project(project, prefix="text"))
        dim = int(project.pack_config.get("dim") or 2048)
        return TicketFeatures(HashedNgrams(dim=dim, prefix="text").fit([text_of(r) for r in self.rows(project)]))

    def datasets(self, project, provider, heads, feedback=None, feedback_weight=5.0):
        rows, specs = self.rows(project), self.heads(project)
        out = {}
        for name in heads:
            labelled = [{"item": LabelItem(id=r["id"], view=VIEW, t=epoch(r.get("created_at")),
                                           state={"subject": r.get("subject"), "body": r.get("body")}),
                         "label": r[name], "source": r.get("source") or "human"}
                        for r in rows if r.get(name) is not None]
            # adds the feedback rows of this head (up-weighted, latest per decision) and says so in a note
            out[name] = dataset_from_labels(specs[name], provider, labelled, feedback, feedback_weight)
        return out

    # ---------------------------------------------------------------- optional
    def synth(self, seed: int = 7, n: int = 3000) -> dict:
        from .synth import write

        return write(Path.cwd() / "data" / "tickets.jsonl", n=n, seed=seed)
