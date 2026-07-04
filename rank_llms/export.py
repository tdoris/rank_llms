"""Export category rankings to JSON and self-contained HTML."""

import json
from html import escape
from typing import Dict, List

import numpy as np


def rankings_to_dict(results: Dict, category: str, promptset: str) -> Dict:
    """
    Convert a generate_rankings() result into a plain, JSON-serializable dict.

    Includes the ranked models with win rate, sample size, and confidence
    interval; the models with no data; and the win-probability matrix.
    """
    rankings = results.get("rankings", [])
    confidence = results.get("confidence", {})
    win_matrix = results["win_matrix"]
    valid_models = [model for model, _ in rankings]

    ranked = []
    for rank, (model, win_rate) in enumerate(rankings, start=1):
        info = confidence.get(model, {})
        interval = info.get("interval")
        ranked.append({
            "rank": rank,
            "model": model,
            "win_rate": round(float(win_rate), 4),
            "comparisons": info.get("comparisons", 0),
            "wins": info.get("wins", 0),
            "ci_low": round(interval[0], 4) if interval else None,
            "ci_high": round(interval[1], 4) if interval else None,
        })

    # Win-probability matrix as nested dict of row -> col -> prob (None on diag/missing)
    matrix = {}
    for row_model in valid_models:
        matrix[row_model] = {}
        for col_model in valid_models:
            if row_model == col_model:
                matrix[row_model][col_model] = None
                continue
            value = win_matrix.loc[row_model, col_model]
            matrix[row_model][col_model] = None if np.isnan(value) else round(float(value), 4)

    return {
        "category": category,
        "promptset": promptset,
        "timestamp": results.get("timestamp"),
        "rankings": ranked,
        "no_data_models": results.get("no_data_models", []),
        "win_matrix": matrix,
    }


def export_json(results: Dict, category: str, promptset: str, output_file: str) -> None:
    """Write the rankings result to a JSON file."""
    data = rankings_to_dict(results, category, promptset)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_html(results: Dict, category: str, promptset: str, output_file: str) -> None:
    """Write a self-contained (no external assets) HTML report of the rankings."""
    data = rankings_to_dict(results, category, promptset)
    ranked = data["rankings"]
    matrix = data["win_matrix"]
    valid_models = list(matrix.keys())

    rows = []
    for entry in ranked:
        if entry["ci_low"] is not None:
            ci = f"{entry['ci_low']:.3f}&ndash;{entry['ci_high']:.3f}"
        else:
            ci = "N/A"
        rows.append(
            "<tr>"
            f"<td>{entry['rank']}</td>"
            f"<td>{escape(entry['model'])}</td>"
            f"<td>{entry['win_rate']:.3f}</td>"
            f"<td>{entry['comparisons']}</td>"
            f"<td>{ci}</td>"
            "</tr>"
        )
    rankings_rows = "\n".join(rows) if rows else (
        f'<tr><td colspan="5">No {escape(category)}-category data found.</td></tr>'
    )

    # Win-probability matrix
    header_cells = "".join(f"<th>{escape(m)}</th>" for m in valid_models)
    matrix_rows = []
    for row_model in valid_models:
        cells = [f"<th>{escape(row_model)}</th>"]
        for col_model in valid_models:
            value = matrix[row_model][col_model]
            cells.append("<td>&ndash;</td>" if value is None else f"<td>{value:.3f}</td>")
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")
    matrix_html = "\n".join(matrix_rows)

    no_data = data["no_data_models"]
    no_data_html = (
        f'<p class="note">No data for: {escape(", ".join(no_data))}</p>' if no_data else ""
    )

    title = f"{escape(category)} Rankings &mdash; {escape(promptset)}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 900px; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  .note {{ color: #a15c00; }}
  .meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated on {escape(str(data.get('timestamp') or ''))} from the {escape(promptset)} promptset.</p>
<h2>Rankings</h2>
<table>
<thead><tr><th>Rank</th><th>Model</th><th>Win Rate</th><th>N</th><th>95% CI</th></tr></thead>
<tbody>
{rankings_rows}
</tbody>
</table>
{no_data_html}
<h2>Win Probability Matrix</h2>
<p class="meta">Probability of row model beating column model.</p>
<table>
<thead><tr><th></th>{header_cells}</tr></thead>
<tbody>
{matrix_html}
</tbody>
</table>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
