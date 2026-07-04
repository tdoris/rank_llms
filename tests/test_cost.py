"""Tests for the dry-run judge cost estimator."""

from rank_llms.main import estimate_judge_cost


def test_known_model_cost():
    # Opus 4.8 is $5/1M input, $25/1M output.
    cost = estimate_judge_cost("claude-opus-4-8", 1_000_000, 1_000_000)
    assert cost == 5.0 + 25.0


def test_unknown_model_returns_none():
    assert estimate_judge_cost("some-unlisted-model", 1000, 1000) is None


def test_scales_with_tokens():
    small = estimate_judge_cost("claude-opus-4-8", 1000, 300)
    big = estimate_judge_cost("claude-opus-4-8", 100_000, 30_000)
    assert big > small > 0
