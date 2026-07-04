"""
Live integration tests against real Ollama and the real Anthropic API.

These are gated: the Ollama test skips unless a local Ollama server is
reachable, and the Anthropic test skips unless ANTHROPIC_API_KEY is set. Run
them explicitly with, e.g.::

    ANTHROPIC_API_KEY=... python -m pytest tests/test_integration.py -v
"""

import os

import pytest


def _ollama_model():
    """Return the name of an available local Ollama model, or None."""
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        # Prefer a small model if present
        for pref in ("gemma3:latest", "gemma3:12b", "cogito:14b"):
            if pref in models:
                return pref
        return models[0] if models else None
    except Exception:
        return None


OLLAMA_MODEL = _ollama_model()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(OLLAMA_MODEL is None, reason="no local Ollama server/model available")
def test_real_ollama_query():
    from rank_llms.main import query_model

    result = query_model(OLLAMA_MODEL, "Reply with exactly one word: hello")
    assert "response" in result and "response_time" in result
    assert not result["response"].startswith("Error:")
    assert result["response_time"] > 0
    assert len(result["response"].strip()) > 0


@pytest.mark.skipif(not HAS_KEY, reason="ANTHROPIC_API_KEY not set")
def test_real_anthropic_judge_picks_better_response():
    from anthropic import Anthropic
    from rank_llms.compare import evaluate_comparison

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = "Write a Python function that returns the nth Fibonacci number."
    good = (
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    bad = "I don't know how to write that."

    result = evaluate_comparison(
        client, prompt, good, bad, "good_model", "bad_model", "Coding"
    )
    # The clearly-correct implementation should win in both orderings.
    assert result["winner"] == "a"
    assert isinstance(result["position_bias_detected"], bool)


@pytest.mark.skipif(not HAS_KEY, reason="ANTHROPIC_API_KEY not set")
def test_real_anthropic_judge_configurable_model():
    from anthropic import Anthropic
    from rank_llms.compare import evaluate_comparison

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # A cheaper judge, single ordering — exercises --judge-model / no-swap path.
    result = evaluate_comparison(
        client, "Say something.", "A thoughtful, detailed answer.", "meh", "a", "b",
        "General", judge_model="claude-haiku-4-5", counter_position_bias=False,
    )
    assert result["winner"] in ("a", "b", "tie")
