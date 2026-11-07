"""A self-contained HTML reliability report (inline SVG, no external assets, works offline).

One section per head: the verdict per stakes level, the headline numbers next to the base-rate baseline, and a
reliability diagram: mean predicted probability (x) against the observed rate (y) per bin, dot area = rows.
Calibrated = the dots sit on the diagonal. Bins with few rows wobble; their size shows how much to trust them.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ..core.bundle import Bundle

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#d9d6cf;--card:#ffffff;--accent:#2f6fde;--ok:#1f8a4c;
--warn:#b7791f;--bad:#c2410c;--diag:#9b988f}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a19a;--line:#3a3935;--card:#1e1e1c;
--accent:#7aa7ff;--ok:#4ade80;--warn:#fbbf24;--bad:#fb923c;--diag:#6b6a65}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 8px}p.meta{color:var(--muted);margin:0 0 16px}
.banner{border:1px solid var(--warn);border-radius:8px;padding:10px 14px;margin:12px 0;color:var(--fg)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}.wrap{overflow-x:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.tier{font-weight:700}.act{color:var(--ok)}.confirm{color:var(--warn)}.escalate,.hold{color:var(--bad)}
svg text{fill:var(--muted);font-size:10px}figure{margin:0}figcaption{font-size:12.5px;color:var(--muted)}
code{font-size:12.5px}
"""


def _e(x: Any) -> str:
    return html.escape("–" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x)))


def reliability_svg(rows: list[dict], title: str, size: int = 220) -> str:
    pad, w = 28, size - 36
    pts = [r for r in rows or [] if r.get("count")]
    total = sum(r["count"] for r in pts) or 1
    parts = [f'<svg viewBox="0 0 {size} {size}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<rect x="{pad}" y="8" width="{w}" height="{w}" fill="none" stroke="var(--line)"/>',
             f'<line x1="{pad}" y1="{8 + w}" x2="{pad + w}" y2="8" stroke="var(--diag)" stroke-dasharray="3 3"/>']
    for t in (0, 0.5, 1):
        parts.append(f'<text x="{pad + t * w - 6}" y="{size - 6}">{t:g}</text>')
        parts.append(f'<text x="4" y="{8 + w - t * w + 3}">{t:g}</text>')
    for r in pts:
        x = pad + float(r["mean_predicted"]) * w
        y = 8 + w - float(r["observed_rate"]) * w
        rad = 2.5 + 9 * (r["count"] / total) ** 0.5
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="var(--accent)" fill-opacity="0.75">'
                     f'<title>predicted {r["mean_predicted"]:.3f}, observed {r["observed_rate"]:.3f}, '
                     f'{r["count"]} rows</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def head_section(name: str, h: Any) -> str:
    e, rep = h.entry, h.report
    spec = e.get("spec") or {}
    m = h.summary.get("metrics") or e.get("metrics") or {}
    verdicts = "".join(f'<li>{_e(s)}: <span class="tier {_e(v.get("max_tier"))}">{_e(v.get("max_tier"))}</span> — '
                       f'{_e(v.get("reason"))}</li>' for s, v in (h.verdict or {}).items())
    rel = rep.get("reliability") or {}
    if spec.get("type") == "noul":
        figs = [((rel.get("per_class") or {}).get("true"), "P(true)")]
    else:
        figs = [(rel.get("top_label"), "top-label confidence"),
                *[(rows, f"P({c})") for c, rows in (rel.get("per_class") or {}).items()]]
    fig_html = "".join(f"<figure>{reliability_svg(rows, t)}<figcaption>{_e(t)}</figcaption></figure>"
                       for rows, t in figs if rows)
    sk = rep.get("skill") or {}
    return f"""<section class="card"><h2><code>{_e(name)}</code> <small>({_e(spec.get('type'))},
{_e(e.get('model'))}, {_e((e.get('calibration') or {}).get('method'))})</small></h2>
<p>{_e(spec.get('description'))}</p>
<div class="wrap"><table><tr><th>holdout rows</th><th>log loss</th><th>baseline</th><th>Brier</th><th>baseline</th>
<th>Brier skill</th><th>ECE</th><th>skill (block bootstrap)</th></tr>
<tr><td>{_e(m.get('n_holdout'))}</td><td>{_e(m.get('log_loss'))}</td><td>{_e(m.get('baseline_log_loss'))}</td>
<td>{_e(m.get('brier'))}</td><td>{_e(m.get('baseline_brier'))}</td><td>{_e(m.get('brier_skill_score'))}</td>
<td>{_e(m.get('ece'))}</td><td>{'yes' if sk.get('has_skill') else 'no'} (lower 95% {_e(sk.get('lower_95'))})</td></tr>
</table></div><ul>{verdicts}</ul><div class="grid">{fig_html}</div></section>"""


def report_html(bundle: Bundle) -> str:
    meta = bundle.meta
    card = meta.get("data_card") or {}
    banner = ('<div class="banner"><b>Synthetic data only.</b> These numbers show the pipeline and the honesty of the '
              'metrics, not accuracy on real traffic.</div>' if card.get("synthetic_only") else "")
    sections = "".join(head_section(n, bundle.head(n)) for n in bundle.names())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Calibration report</title>
<style>{CSS}</style></head><body><main><h1>Calibration report — {_e(meta.get('version'))}</h1>
<p class="meta">{_e((meta.get('pack') or {}).get('name'))} pack · created {_e(meta.get('created_at'))} · feature
version <code>{_e(meta.get('feature_version'))}</code> · every number is measured on trusted holdout rows the model
never saw.</p>{banner}{sections}</main></body></html>"""


def write_report(bundle: Bundle, path: Path | None = None) -> Path:
    path = Path(path) if path else bundle.path / "report.html"
    path.write_text(report_html(bundle), encoding="utf-8")
    return path
