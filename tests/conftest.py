"""Shared fixtures: a self-contained temp test archive of comparison files."""

import json

import pytest


@pytest.fixture
def archive(tmp_path):
    """
    Build a temp ``test_archive`` with a coding101 promptset containing a few
    head-to-head comparison files with known, hand-computable outcomes.

    Coding-category results (winner is 'a' = model_a, 'b' = model_b, 'tie'):
      alpha vs beta:  a a a b   -> alpha 3, beta 1  (n=4)
      alpha vs gamma: a a a a   -> alpha 4, gamma 0 (n=4)
      beta  vs gamma: a a b tie -> beta 2, gamma 1, tie 1 (n=4)

    Pooled: alpha 7/8, beta 3/8, gamma 1/8. One file also carries a
    BugFinding-category result so category filtering can be exercised.
    """
    comp = tmp_path / "coding101" / "comparisons"
    comp.mkdir(parents=True)

    def write(a, b, coding_winners, bug_winners=None):
        comparisons = [{"category": "Coding", "winner": w} for w in coding_winners]
        for w in (bug_winners or []):
            comparisons.append({"category": "BugFinding", "winner": w})
        data = {"model_a": a, "model_b": b, "comparisons": comparisons}
        (comp / f"{a}__vs__{b}.json").write_text(json.dumps(data))

    write("alpha", "beta", ["a", "a", "a", "b"], bug_winners=["b", "b"])
    write("alpha", "gamma", ["a", "a", "a", "a"])
    write("beta", "gamma", ["a", "a", "b", "tie"])

    return str(tmp_path)
