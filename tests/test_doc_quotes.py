"""Messages the guides quote word for word are the messages the code prints or writes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from open_rlcd.cli import main
from open_rlcd.core.config import Project
from open_rlcd.core.contract import TrainRequest
from open_rlcd.core.service import Service
from tests.test_tabular import small_table
from tests.toypack import ToyPack

REPO = Path(__file__).resolve().parent.parent


class TicketsPack(ToyPack):
    """The toy pack under the name of the guides' example pack, so the bundle it trains names pack 'tickets'."""

    name = "tickets"


def quoted(path: Path, marker: str) -> str:
    """The one fenced block of `path` that contains `marker`, with its lines joined by single spaces (the guides
    wrap a long message over several lines)."""
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)
    hits = [b for b in blocks if marker in b]
    assert len(hits) == 1, (path, len(hits))
    return " ".join(hits[0].split())


def test_the_guides_quote_the_bad_bundle_message_rlcd_decide_prints(tmp_path, capsys):
    """docs/writing-a-pack.md ("Share a bundle") and examples/custom-pack/README.md quote what `rlcd decide` prints
    for a bundle trained with the pack 'tickets', in a project whose rlcd.yaml names the tabular pack. The message
    lists the built-in packs (`sorted(BUILTIN)`), so a new built-in pack changes it: the quotes said
    ['documents', 'tabular'] after the trading pack had joined them."""
    home = tmp_path / "home"
    trainer = Service(Project(root=tmp_path, data={}), home=home, pack=TicketsPack(n=300))
    job = trainer.trainer.start(TrainRequest(heads=["good"]), wait=True)
    assert job["model"] == "rlcd-0.1.1", job.get("error")
    other = tmp_path / "other"
    other.mkdir()
    (other / "rlcd.yaml").write_text("name: other\npack: tabular\n", encoding="utf-8")
    capsys.readouterr()
    code = main(["--project", str(other), "--home", str(home), "decide", "--state", json.dumps({"f": [0.0] * 6})])
    printed = [line for line in capsys.readouterr().err.splitlines() if line.startswith("rlcd: ")]
    assert code == 1 and printed, printed
    assert "was trained with pack 'tickets'" in printed[-1], printed
    for path in (REPO / "docs" / "writing-a-pack.md", REPO / "examples" / "custom-pack" / "README.md"):
        assert quoted(path, "was trained with pack 'tickets'") == printed[-1], path


def test_the_guides_quote_the_reason_a_train_gives_for_a_head_it_leaves_out(tmp_path, capsys):
    """A train that leaves a head of rlcd-latest out of the new version gives the reason in a note of that version
    (core/registry.py) and in the line `rlcd train` prints on stderr (cli.py). docs/design.md and
    docs/writing-a-pack.md quote the note; docs/configuration.md (twice), docs/contract.md and docs/control-layer.md
    give the reason in a sentence of their own. Each place is found by the words around the quote. The note and the
    line come from a real train that leaves a head out: the tabular pack, where moving a column to `exclude` changes
    the feature version."""
    root, home = tmp_path / "p", tmp_path / "home"
    root.mkdir()
    small_table(root / "small.csv")
    targets = {"churned": {"label": "churned", "model": "logreg"},
               "next_plan": {"label": "next_plan", "model": "logreg"}}

    def train(*argv: str, **pack_config) -> tuple[dict, str]:
        """`rlcd train ...` on this pack_config: the model.json of the version it saved, and what it printed on
        stderr."""
        cfg = {"pack": "tabular", "pack_config": {"data": "small.csv", "targets": targets, **pack_config}}
        (root / "rlcd.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        capsys.readouterr()
        assert main(["--project", str(root), "--home", str(home), "train", *argv]) == 0
        err = capsys.readouterr().err
        latest = json.loads((home / "registry.json").read_text(encoding="utf-8"))["latest"]
        return json.loads((home / "models" / latest / "model.json").read_text(encoding="utf-8")), err

    first, _ = train()
    meta, err = train("--heads", "churned", exclude=["plan"])     # another feature version: next_plan is left out
    assert meta["not_carried_over"] == ["next_plan"], meta.get("notes")
    notes = [n for n in meta["notes"] if n.startswith("not carried over")]
    m = re.fullmatch(r"not carried over \((.+)\): \['next_plan'\]", notes[0]) if len(notes) == 1 else None
    assert m, notes
    reason = m.group(1)
    lines = [line for line in err.splitlines() if line.startswith("not carried over")]
    start = f"not carried over from {first['version']}: next_plan ({reason});"      # the same reason, on stderr
    assert len(lines) == 1 and lines[0].startswith(start), lines
    # (guide, the words in front of the quote, the words after it)
    places = [("configuration.md", "`rlcd train` on stderr, as ", "; `not_carried_over` in the train job"),
              ("configuration.md", "on stderr, after its table, as ", ", and the train job lists them"),
              ("contract.md", "nor carried them over, as they were ", ". Until a train retrains it"),
              ("control-layer.md", "`rlcd train` names a head it leaves out, as ",
               " ([configuration](configuration.md#"),
              ("design.md", 'names the heads it left out ("not carried over (', '): [...]")'),
              ("writing-a-pack.md", 'notes then say "not carried over (', '): [...]"')]
    for name, before, after in places:
        text = " ".join((REPO / "docs" / name).read_text(encoding="utf-8").split())
        assert before in text, (name, before)
        assert before + reason + after in text, (name, text[text.index(before):][:len(before) + len(reason) + 60])
