"""What `rlcd init --pack tabular` writes besides the keys: the comments in rlcd.yaml and the next steps.

The pack cannot know which columns of a table are outcomes: only the label columns it is told about are kept
out of the features. So the scaffold says it twice, in the file and on the terminal: declare every outcome
under `targets`, and list every other column that is only known after the decision under `exclude`.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_COMMENT = """\
  # --- read this before the first `rlcd train` (docs/tabular-quickstart.md) ---
  # `label` is the one-head shorthand: one outcome column, one head named "outcome".
  # With `features: auto`, EVERY other column becomes a feature. The pack cannot know which of them are only
  # filled in after the decision (a second outcome, a review level, a closing date): a model trained on
  # such a column looks excellent on the holdout and is useless in production.
  #   - list every column that is only known after the decision under `exclude:`
  #   - a table with several outcomes: delete `label:` and declare them all under `targets:`, e.g.
  #
  # targets:
  #   approved:     {label: outcome, positive: approved, stakes: medium}   # 2 values: say which one means true
  #   payout_delay: {label: payout_delay, type: score}                     # ordered levels (or type: choice)
  #
  # Free-text columns go under `text:` (hashed n-gram features; nothing of the text is stored in the bundle).
  # A time column in another notation than ISO 8601: time_format: "%d/%m/%Y %H:%M".
"""


def annotate(yaml_text: str) -> str:
    """rlcd.yaml as `rlcd init` wrote it, with CONFIG_COMMENT at the end of the `pack_config:` block (before the
    next top-level key). Text without a `pack_config:` block is returned unchanged."""
    lines = yaml_text.splitlines(keepends=True)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("pack_config:"))
    except StopIteration:
        return yaml_text
    end = next((i for i in range(start + 1, len(lines)) if lines[i][:1] not in (" ", "#", "\n", "")), len(lines))
    return "".join([*lines[:end], CONFIG_COMMENT, *lines[end:]])


def next_steps(root: Path | str) -> list[str]:
    """The steps after `rlcd init --pack tabular`, in an order that runs as written."""
    return [
        f"cd {root}",
        "copy your table to ./data.csv (or point pack_config.data at it): one row per past case",
        "edit rlcd.yaml: pack_config.label = the outcome column, positive = the value that means true "
        "(several outcomes: declare them all under pack_config.targets); id / group / time columns if you have "
        "them",
        "list every column that is only known after the decision under `exclude:` (a second outcome, a review "
        "level, a closing date): with features: auto every other column is a feature, and such a column leaks "
        "the answer",
        "free-text columns go under `text:`; hash_categories: true if a categorical column could hold personal "
        "data (bundles store its values)",
        "rlcd config check",
        "rlcd train                  # read the notes it prints, then: rlcd report",
        "the whole path on one page: docs/tabular-quickstart.md",
    ]
