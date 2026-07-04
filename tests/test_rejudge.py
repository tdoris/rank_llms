"""Tests for re-judging archived comparisons and path loading."""

import json

import rank_llms.compare as compare
from rank_llms.compare import (
    ModelComparison, CategoryResult, ComparisonResult,
    rejudge_comparison_result, load_comparison_result_from_path,
)


def _make_result():
    comps = [
        ModelComparison(model_a="A", model_b="B", prompt="p1", category="Coding",
                        response_a="ra", response_b="rb", response_time_a=1.0,
                        response_time_b=1.0, winner="b"),
        ModelComparison(model_a="A", model_b="B", prompt="p2", category="Coding",
                        response_a="ra", response_b="rb", response_time_a=1.0,
                        response_time_b=1.0, winner="tie"),
        ModelComparison(model_a="A", model_b="B", prompt="p3", category="Reasoning",
                        response_a="ra", response_b="rb", response_time_a=1.0,
                        response_time_b=1.0, winner="a"),
    ]
    return ComparisonResult(
        model_a="A", model_b="B",
        category_results={"Coding": CategoryResult(category="Coding", model_a="A", model_b="B")},
        comparisons=comps,
    )


def test_rejudge_updates_winners_and_recomputes_tallies(monkeypatch):
    verdicts = {
        "p1": {"winner": "a", "reason": "x", "position_bias_detected": False},   # b -> a (changed)
        "p2": {"winner": "a", "reason": "y", "position_bias_detected": True},    # tie -> a (changed, bias)
        "p3": {"winner": "a", "reason": "z", "position_bias_detected": False},   # a -> a (unchanged)
    }

    def fake_eval(client, prompt, ra, rb, ma, mb, category, **kwargs):
        return verdicts[prompt]

    monkeypatch.setattr(compare, "evaluate_comparison", fake_eval)

    result, stats = rejudge_comparison_result(None, _make_result())

    assert stats == {"changed": 2, "bias": 1, "total": 3}
    assert [c.winner for c in result.comparisons] == ["a", "a", "a"]
    assert result.comparisons[1].position_bias_detected is True

    # Tallies rebuilt from the new winners
    coding = result.category_results["Coding"]
    assert (coding.wins_a, coding.wins_b, coding.ties) == (2, 0, 0)
    reasoning = result.category_results["Reasoning"]
    assert (reasoning.wins_a, reasoning.wins_b, reasoning.ties) == (1, 0, 0)


def test_load_from_path_tolerates_missing_timestamp(tmp_path):
    d = {
        "model_a": "A", "model_b": "B", "category_results": {},
        "comparisons": [{
            "model_a": "A", "model_b": "B", "prompt": "p", "category": "Coding",
            "response_a": "x", "response_b": "y",
            "response_time_a": 1.0, "response_time_b": 1.0, "winner": "a",
        }],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    result = load_comparison_result_from_path(p)
    assert result is not None
    assert result.model_a == "A" and len(result.comparisons) == 1


def test_load_from_path_skips_placeholder(tmp_path):
    p = tmp_path / "junk.json"
    p.write_text('{"test": 1}')
    assert load_comparison_result_from_path(p) is None
