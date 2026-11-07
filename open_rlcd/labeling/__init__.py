"""Labeling: weak labels from an LLM judge, human audit in the terminal, active learning.

Generic over packs: everything works on LabelItems (store.py) and HeadSpecs, so a pack that yields LabelItems from
`label_tasks()` gets bootstrap / audit / next for free.

    store.py      LabelItem, LabelStore (items.jsonl, judge.jsonl, human.jsonl under <home>/labels/)
    prompts.py    build_prompt(items, heads) -> (system, user); parse_response(text, items, heads) -> [JudgeAnswer]
    judges.py     ClaudeCliJudge, AnthropicJudge, OllamaJudge, OpenAICompatJudge, ScriptedJudge, make_judge(spec)
    bootstrap.py  bootstrap(project, pack, provider, judge, ...) -> counts (resumable, never touches human labels)
    audit.py      audit(project, pack, ...) -> agreement + trusted-row counts (interactive, resumable)
    active.py     next_batch(project, service, head, n) -> queue rows (uncertainty + disagreement + diversity)

Names are imported lazily so that `import open_rlcd.labeling.store` (used by packs) stays cheap.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "LabelItem": "store", "LabelStore": "store",
    "JudgeAnswer": "prompts", "build_prompt": "prompts", "parse_response": "prompts",
    "Judge": "judges", "JudgeError": "judges", "ClaudeCliJudge": "judges", "AnthropicJudge": "judges",
    "OllamaJudge": "judges", "OpenAICompatJudge": "judges", "ScriptedJudge": "judges", "make_judge": "judges",
    "next_batch": "active",
}
# `bootstrap` and `audit` are not re-exported: they share their submodule's name, and the package attribute would
# be the function or the module depending on import order. Use `from open_rlcd.labeling.bootstrap import bootstrap`.

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)
