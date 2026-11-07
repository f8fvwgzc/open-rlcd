"""MODEL_CARD.md, generated from a bundle's model.json and holdout reports.

The card states what the numbers are (holdout = trusted rows the model never saw), what they are not
(synthetic data is a sanity check, judge labels are only as good as the judge), and every head's verdict in
plain words, including "no skill: this head abstains".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.bundle import Bundle

INTENDED = {
    "documents": ("Routing, gatekeeping and evaluation in front of / behind an LLM over an enterprise document "
                  "store: which response type, which retrieval route, which chunks matter, whether the evidence "
                  "is enough, and whether a drafted answer is grounded, complete and how risky it is."),
    "tabular": ("Calibrated yes/no, one-of-N or ordered-level decisions on rows of a table (churn, fraud, lead "
                "scoring, approvals), with an explicit act / confirm / hold policy."),
    "trading": ("A thin adapter: calibrated outcome probabilities on feature rows served by an external quant "
                "sidecar. Analysis tool, not financial advice."),
}


def categorical_columns(meta: dict) -> dict[str, dict]:
    """{column: {"values": n, "hashed": bool}} for every categorical feature column whose vocabulary is stored in
    the bundle's provider config (the tabular pack's `pack.config.columns`). Read from the bundle itself, so it is
    right for bundles trained before `hash_categories` existed (they store raw values)."""
    cfg = (meta.get("pack") or {}).get("config") or {}
    out: dict[str, dict] = {}
    cols = cfg.get("columns") if isinstance(cfg, dict) else None
    for c in cols if isinstance(cols, list) else []:
        if isinstance(c, dict) and c.get("kind") == "categorical":
            out[str(c.get("name"))] = {"values": len(c.get("vocabulary") or []), "hashed": bool(c.get("hashed"))}
    return out


def categorical_summary(meta: dict) -> str:
    """One line naming the categorical columns stored in a bundle and how their values are stored ('' if none)."""
    cols = categorical_columns(meta)
    if not cols:
        return ""
    parts = [f"{name} ({c['values']} values, {'salted sha256' if c['hashed'] else 'raw'})" for name, c in cols.items()]
    return "categorical columns stored in the bundle: " + ", ".join(parts)


def raw_categorical_warning(meta: dict) -> str:
    """A warning when raw categorical values from the training rows are embedded in the bundle ('' if none)."""
    raw = [name for name, c in categorical_columns(meta).items() if not c["hashed"] and c["values"]]
    if not raw:
        return ""
    return (f"the bundle embeds raw values of categorical column(s) {raw} verbatim, copied from the training rows "
            f"(the most frequent values of each, in model.json and every head's feature names). If any could be "
            f"personal or confidential, set pack_config.hash_categories: true and retrain before sharing.")


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _head_row(name: str, h: Any) -> str:
    """One table row from the hash-checked head.json (the values the service enforces)."""
    e, summ = h.entry, h.summary
    m = summ.get("metrics") or {}
    v = (summ.get("verdict") or {}).get((e.get("spec") or {}).get("stakes") or "medium") or {}
    c = summ.get("conformal") or {}
    return (f"| `{name}` | {(e.get('spec') or {}).get('type')} | {e.get('model')} | "
            f"{(e.get('calibration') or {}).get('method')} | {_fmt(m.get('n_holdout'))} | "
            f"{_fmt(m.get('log_loss'))} / {_fmt(m.get('baseline_log_loss'))} | {_fmt(m.get('brier_skill_score'))} | "
            f"{_fmt(m.get('ece'))} | {_fmt(m.get('has_skill'))} | {v.get('max_tier', '–')} | "
            f"{_fmt(c.get('threshold'))} |")


def _reliability(rep: dict, cls: str | None = None) -> list[str]:
    rel = rep.get("reliability") or {}
    rows = (rel.get("per_class") or {}).get(cls) if cls else rel.get("top_label")
    if not rows:
        return []
    out = ["| bin | mean predicted | observed | rows |", "|---|---|---|---|"]
    for b in rows:
        if b.get("count"):
            out.append(f"| {b['lo']:.1f}–{b['hi']:.1f} | {_fmt(b.get('mean_predicted'))} | "
                       f"{_fmt(b.get('observed_rate'))} | {b['count']} |")
    return out


def model_card(bundle: Bundle) -> str:
    meta = bundle.meta
    pack = (meta.get("pack") or {}).get("name", "?")
    card = meta.get("data_card") or {}
    enc = meta.get("encoder") or {}
    lines = [f"# Model card — {meta['version']} ({pack} pack)", "",
             f"Created {meta.get('created_at')}. Format `{meta.get('format')}`. Feature version "
             f"`{meta.get('feature_version')}`.", ""]
    if card.get("synthetic_only"):
        lines += ["> **Trained on synthetic data only.** These numbers show that the pipeline works and that the "
                  "metrics are computed honestly; they say nothing about accuracy on real traffic.", ""]
    lines += ["## Intended use", "", INTENDED.get(pack, "Calibrated answers to typed questions."), "",
              "## What it is not", "",
              "* Not a language model: it never writes text. It decides what, whether and from which evidence.",
              "* Not magic: heads take microseconds per row, the text encoder milliseconds; run `rlcd bench`.",
              "* Not better than its labels: weak labels from an LLM judge train the base models, but calibration "
              "and every number below use trusted (human-verified, ground-truth or synthetic) rows only.",
              "* A head whose holdout shows no skill over the base rate abstains on every request.", "",
              "## Data", ""]
    if card.get("label_sources"):
        lines.append("Label sources (all heads): " + ", ".join(f"{k} {v}" for k, v in card["label_sources"].items()))
        lines.append("")
    if enc:
        lines += [f"Text encoder: `{enc.get('id')}` (dim {enc.get('dim')}, sha256 `{str(enc.get('sha256'))[:16]}…`).",
                  ""]
    cats = categorical_columns(meta)
    if cats:
        lines.append("Categorical feature columns (their most frequent values are stored in this bundle): " + ", ".join(
            f"`{name}` ({c['values']} values, {'salted SHA-256 hashes' if c['hashed'] else 'raw values'})"
            for name, c in cats.items()) + ".")
        lines.append("")
        warning = raw_categorical_warning(meta)
        if warning:
            lines += [f"> **Raw training values inside.** {warning[0].upper()}{warning[1:]}", ""]
    lines += ["## Heads (trusted holdout)", "",
              "| head | type | model | calibration | holdout rows | log loss / baseline | Brier skill | ECE | skill | "
              "verdict | conformal threshold |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in bundle.names():
        lines.append(_head_row(name, bundle.head(name)))
    lines.append("")
    for name in bundle.names():
        h = bundle.head(name)
        e, rep = h.entry, h.report
        spec = e.get("spec") or {}
        lines += [f"### `{name}`", "", spec.get("description") or "", ""]
        d = e.get("data") or {}
        lines.append(f"Rows {d.get('rows')} (trusted {d.get('trusted')}; by source "
                     f"{json.dumps(d.get('by_source') or {})}), ordering `{d.get('ordering')}`, holdout "
                     f"{d.get('holdout_rows')} trusted rows, groups {d.get('groups')}.")
        lines.append("")
        sk = rep.get("skill") or {}
        lines.append(f"Skill test: {sk.get('method')}; mean Brier gain {_fmt(sk.get('mean_brier_gain'), 4)}, "
                     f"lower 95% {_fmt(sk.get('lower_95'), 4)} -> **{'skill' if sk.get('has_skill') else 'no skill'}**.")
        lines.append("")
        for stakes, v in (h.verdict or {}).items():
            lines.append(f"* {stakes} stakes: **{v.get('max_tier')}** — {v.get('reason')}")
        lines.append("")
        cls = spec.get("classes") or []
        rel = _reliability(rep, "true") if spec.get("type") == "noul" else _reliability(rep)
        if rel:
            what = "P(true)" if spec.get("type") == "noul" else "top-label confidence"
            lines += [f"Reliability ({what}, holdout):", "", *rel, ""]
        conf = rep.get("conformal") or {}
        if conf.get("selective"):
            s, chk = conf["selective"], conf.get("holdout") or {}
            lines.append(f"Conformal promise: error <= {s.get('target_error')} at confidence "
                         f"{1 - float(s.get('delta', 0.1)):g} for P(true) >= {_fmt(s.get('threshold'))} "
                         f"({s.get('reason') or 'dev coverage ' + _fmt(s.get('coverage'))}); holdout: error "
                         f"{_fmt(chk.get('error_rate'))} on {chk.get('n_selected')} rows, coverage "
                         f"{_fmt(chk.get('coverage'))}.")
            lines.append("")
        ja = rep.get("judge_agreement")
        if ja and ja.get("n"):
            lines.append(f"LLM judge vs humans: accuracy {_fmt(ja.get('accuracy'))} (kappa "
                         f"{_fmt(ja.get('cohen_kappa'))}) on {ja['n']} rows labelled by both.")
            lines.append("")
        if e.get("notes"):
            lines += ["Notes:", *[f"* {n}" for n in e["notes"]], ""]
        if cls and spec.get("type") != "noul":
            lines.append(f"Classes: {', '.join(cls)}.")
            lines.append("")
    pol = meta.get("policy") or {}
    if pol.get("stakes"):
        lines += ["## Policy", "", f"Abstain action: `{pol.get('abstain')}`. Thresholds on the calibrated probability "
                  "of the committed class:", "", "| stakes | act | confirm | min accuracy (Bernoulli gate) |",
                  "|---|---|---|---|"]
        for s, lv in pol["stakes"].items():
            lines.append(f"| {s} | {lv['act']} | {lv['confirm']} | {lv['min_accuracy']} |")
        lines.append("")
    lines += ["## Known limits", "",
              "* Calibration holds for data like the training data. A shift in traffic (new products, new document "
              "types, a new language) needs fresh trusted labels and a retrain; `/v1/calibration` will show it.",
              "* The holdout is small when trusted labels are few: read the Wilson intervals, not just the means.",
              "* Feedback rows are biased towards decisions that were acted on; they train, they never calibrate.",
              "* Conformal promises assume exchangeable data (no drift between calibration and use).", ""]
    return "\n".join(lines)


def write_model_card(bundle: Bundle, path: Path | None = None) -> Path:
    path = Path(path) if path else bundle.path / "MODEL_CARD.md"
    path.write_text(model_card(bundle), encoding="utf-8")
    return path
