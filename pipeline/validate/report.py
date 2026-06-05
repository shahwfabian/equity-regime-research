"""Render validation check results to a self-contained HTML report."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import List

from pipeline.validate.checks import CheckResult

log = logging.getLogger(__name__)

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{{ title }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 40px auto;
         padding: 0 24px; color: #1a1a2e; background: #f8f9fa; line-height: 1.6; }
  h1   { font-size: 1.7em; color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
  h2   { font-size: 1.15em; color: #0f3460; margin-top: 2.2em; }
  .meta { color: #555; font-size: 0.9em; margin-bottom: 1.5em; }
  .summary-bar { display: flex; gap: 16px; margin-bottom: 2em; flex-wrap: wrap; }
  .badge { padding: 10px 22px; border-radius: 6px; font-weight: bold; font-size: 1em; }
  .badge-pass  { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
  .badge-fail  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
  .badge-total { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
  .check-card  { background: white; border-radius: 8px; margin-bottom: 16px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.08); overflow: hidden; }
  .check-header { display: flex; align-items: center; padding: 12px 18px;
                  border-left: 5px solid #aaa; gap: 12px; }
  .check-header.pass { border-left-color: #28a745; background: #f0fff4; }
  .check-header.fail { border-left-color: #dc3545; background: #fff5f5; }
  .status-pill { font-size: 0.78em; font-weight: bold; padding: 2px 10px;
                 border-radius: 12px; white-space: nowrap; }
  .status-pill.pass { background: #28a745; color: white; }
  .status-pill.fail { background: #dc3545; color: white; }
  .check-name  { font-weight: 600; font-size: 1em; flex: 1; }
  .check-body  { padding: 12px 18px 14px; border-top: 1px solid #eee; }
  .detail      { color: #444; font-size: 0.9em; margin-bottom: 8px; }
  .numbers     { font-size: 0.82em; color: #666; font-family: monospace;
                 background: #f4f4f4; padding: 6px 10px; border-radius: 4px;
                 white-space: pre-wrap; word-break: break-all; }
  table.data-tbl { border-collapse: collapse; width: 100%; font-size: 0.82em;
                   margin-top: 8px; }
  table.data-tbl th { background: #0f3460; color: white; padding: 5px 10px; text-align: left; }
  table.data-tbl td { padding: 4px 10px; border-bottom: 1px solid #e5e5e5; }
  table.data-tbl tr:nth-child(even) { background: #f9f9f9; }
  .disclosure  { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
                 padding: 14px 18px; font-size: 0.88em; white-space: pre-wrap; }
  footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 3em; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="meta">
  <strong>Generated:</strong> {{ generated }}&nbsp;|&nbsp;
  <strong>Backend:</strong> {{ backend }}&nbsp;|&nbsp;
  <strong>Window:</strong> {{ window }}
</div>

<div class="summary-bar">
  <div class="badge badge-total">Total checks: {{ total }}</div>
  <div class="badge badge-pass">✓ Pass: {{ n_pass }}</div>
  {% if n_fail > 0 %}
  <div class="badge badge-fail">✗ Fail: {{ n_fail }}</div>
  {% else %}
  <div class="badge badge-pass">✗ Fail: 0</div>
  {% endif %}
</div>

{% for r in results %}
<div class="check-card">
  <div class="check-header {{ 'pass' if r.passed else 'fail' }}">
    <span class="status-pill {{ 'pass' if r.passed else 'fail' }}">
      {{ 'PASS' if r.passed else 'FAIL' }}
    </span>
    <span class="check-name">{{ r.name }}</span>
  </div>
  <div class="check-body">
    {% if 'Survivorship' in r.name %}
    <div class="disclosure">{{ r.detail }}</div>
    {% else %}
    <div class="detail">{{ r.detail }}</div>
    {% endif %}
    {% if r.numbers %}
    <div class="numbers">{{ r.numbers | tojson }}</div>
    {% endif %}
    {% if r.rows is not none %}
    <table class="data-tbl">
      <thead><tr>{% for col in r.rows.columns %}<th>{{ col }}</th>{% endfor %}</tr></thead>
      <tbody>
      {% for _, row in r.rows.iterrows() %}
        <tr>{% for col in r.rows.columns %}<td>{{ row[col] }}</td>{% endfor %}</tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </div>
</div>
{% endfor %}

<footer>equity-regime-research ETL validation &mdash; {{ generated }}</footer>
</body>
</html>
"""


def _tojson_filter(obj) -> str:
    """Safe JSON serialiser for Jinja2 (handles numpy scalars)."""
    import json
    import numpy as np

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return round(float(o), 8)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, dict):
                return {str(k): self.default(v) for k, v in o.items()}
            return super().default(o)

    return json.dumps(obj, cls=_Enc, indent=2)


def render_report(
    results: List[CheckResult],
    cfg,
    out_path: Path | None = None,
) -> Path:
    """
    Render all CheckResult objects to a self-contained HTML report.

    Returns the path to the written file.
    """
    from jinja2 import Environment

    env = Environment()
    env.filters["tojson"] = _tojson_filter

    tmpl = env.from_string(_TEMPLATE)

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = tmpl.render(
        title="ETL Data Validation Report — Equity Regime Research",
        generated=now,
        backend=cfg.store.backend,
        window=f"{cfg.window.start_date} -> {cfg.window.end_date}",
        total=len(results),
        n_pass=n_pass,
        n_fail=n_fail,
        results=results,
    )

    if out_path is None:
        out_path = Path(cfg.validation.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    log.info(
        "[report] Validation report written -> %s  (%d PASS / %d FAIL)",
        out_path, n_pass, n_fail,
    )
    return out_path
