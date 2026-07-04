"""Tests for the debiased judge aggregation and verdict parsing."""

import rank_llms.compare as compare
from rank_llms.compare import evaluate_comparison, _parse_verdict


def _script(monkeypatch, mapping):
    """Patch _judge_once to return scripted verdicts keyed by (first, second) label."""
    def fake_judge(client, prompt, first, second, first_label, second_label, category, judge_model):
        return mapping[(first_label, second_label)]
    monkeypatch.setattr(compare, "_judge_once", fake_judge)


def test_consistent_winner_no_bias(monkeypatch):
    # Both orderings agree that model A is better.
    _script(monkeypatch, {
        ("A", "B"): {"winner": "a", "reason": "A better"},   # normal: first=A
        ("B", "A"): {"winner": "b", "reason": "A better"},   # swapped: second=A
    })
    r = evaluate_comparison(None, "q", "ra", "rb", "A", "B", "Coding")
    assert r["winner"] == "a"
    assert r["position_bias_detected"] is False


def test_position_bias_scored_as_tie(monkeypatch):
    # Judge always picks whichever response is shown first -> position bias.
    _script(monkeypatch, {
        ("A", "B"): {"winner": "a", "reason": "first is best"},
        ("B", "A"): {"winner": "a", "reason": "first is best"},
    })
    r = evaluate_comparison(None, "q", "ra", "rb", "A", "B", "Coding")
    assert r["winner"] == "tie"
    assert r["position_bias_detected"] is True
    assert "position bias" in r["reason"].lower()


def test_no_counter_bias_single_call(monkeypatch):
    calls = []

    def fake_judge(client, prompt, first, second, first_label, second_label, category, judge_model):
        calls.append((first_label, second_label))
        return {"winner": "b", "reason": "B better"}

    monkeypatch.setattr(compare, "_judge_once", fake_judge)
    r = evaluate_comparison(None, "q", "ra", "rb", "A", "B", "Coding", counter_position_bias=False)
    assert r["winner"] == "b"
    assert r["position_bias_detected"] is False
    assert len(calls) == 1  # only one ordering when bias-counter is off


def test_samples_multiply_calls(monkeypatch):
    calls = []

    def fake_judge(client, prompt, first, second, first_label, second_label, category, judge_model):
        calls.append((first_label, second_label))
        # normal: 'b' => B; swapped: 'a' => B. So B wins in both orderings.
        return {"winner": "b" if first_label == "A" else "a", "reason": "B better"}

    monkeypatch.setattr(compare, "_judge_once", fake_judge)
    r = evaluate_comparison(None, "q", "ra", "rb", "A", "B", "Coding", samples=3)
    assert r["winner"] == "b"
    assert len(calls) == 6  # 2 orderings x 3 samples


def test_parse_verdict_clean_json():
    assert _parse_verdict('{"winner": "a", "reason": "x"}')["winner"] == "a"


def test_parse_verdict_embedded_json():
    text = 'Sure!\n{"winner": "b", "reason": "clearer"}\nThanks'
    assert _parse_verdict(text)["winner"] == "b"


def test_parse_verdict_invalid_winner_defaults_tie():
    assert _parse_verdict('{"winner": "both"}')["winner"] == "tie"


def test_parse_verdict_unparseable_defaults_tie():
    assert _parse_verdict("no json here")["winner"] == "tie"
