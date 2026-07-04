"""Tests for JSON and HTML export of rankings."""

import json

from rank_llms.coding_rank import CodingRankAnalyzer
from rank_llms.export import rankings_to_dict, export_json, export_html


def _results(archive):
    analyzer = CodingRankAnalyzer(test_archive_dir=archive, promptset="coding101")
    return analyzer.generate_rankings(["alpha", "beta", "gamma"])


def test_rankings_to_dict_shape(archive):
    data = rankings_to_dict(_results(archive), "Coding", "coding101")
    assert data["category"] == "Coding"
    assert [r["model"] for r in data["rankings"]] == ["alpha", "beta", "gamma"]
    top = data["rankings"][0]
    assert top["rank"] == 1 and top["model"] == "alpha"
    assert top["comparisons"] == 8 and top["wins"] == 7
    assert 0.0 <= top["ci_low"] <= top["ci_high"] <= 1.0
    # win matrix has an entry per ranked model, diagonal is None
    assert data["win_matrix"]["alpha"]["alpha"] is None
    assert data["win_matrix"]["alpha"]["gamma"] == 1.0


def test_export_json_is_valid(archive, tmp_path):
    out = tmp_path / "r.json"
    export_json(_results(archive), "Coding", "coding101", str(out))
    parsed = json.loads(out.read_text())
    assert parsed["rankings"][0]["model"] == "alpha"


def test_export_html_self_contained(archive, tmp_path):
    out = tmp_path / "r.html"
    export_html(_results(archive), "Coding", "coding101", str(out))
    html = out.read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "alpha" in html and "Win Probability Matrix" in html
    # self-contained: no external asset references
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()
