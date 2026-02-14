from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_html_report(
    meta: Mapping[str, Any],
    metrics: Mapping[str, Any],
    chart_paths: Mapping[str, str],
) -> str:
    meta_rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_format_value(value))}</td></tr>"
        for key, value in meta.items()
    )
    metric_rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_format_value(value))}</td></tr>"
        for key, value in metrics.items()
    )
    charts_html = "\n".join(
        (
            f"<div class='chart'>"
            f"<h3>{escape(name)}</h3>"
            f"<img src='{escape(path)}' alt='{escape(name)}' />"
            f"</div>"
        )
        for name, path in chart_paths.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Backtest Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ width: 35%; background: #f7f7f7; }}
    .chart {{ margin-bottom: 18px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Backtest Report</h1>
  <h2>Meta</h2>
  <table>{meta_rows}</table>
  <h2>Metrics</h2>
  <table>{metric_rows}</table>
  <h2>Charts</h2>
  {charts_html if charts_html else "<p>No charts generated.</p>"}
</body>
</html>
"""
