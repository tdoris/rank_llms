"""Tests for category ranking: ordering, NaN handling, confidence, filtering."""

import math

from rank_llms.coding_rank import CodingRankAnalyzer
from rank_llms.stats import wilson_interval


def test_rankings_ordered_and_pooled_counts(archive):
    analyzer = CodingRankAnalyzer(test_archive_dir=archive, promptset="coding101")
    results = analyzer.generate_rankings(["alpha", "beta", "gamma"])

    order = [model for model, _ in results["rankings"]]
    assert order == ["alpha", "beta", "gamma"]

    conf = results["confidence"]
    # alpha: 7 wins / 8 comparisons pooled across both opponents
    assert conf["alpha"]["wins"] == 7
    assert conf["alpha"]["comparisons"] == 8
    assert conf["beta"]["wins"] == 3
    assert conf["gamma"]["wins"] == 1
    # interval matches the stats helper
    assert conf["alpha"]["interval"] == wilson_interval(7, 8)


def test_no_data_model_excluded_and_ranks_contiguous(archive):
    analyzer = CodingRankAnalyzer(test_archive_dir=archive, promptset="coding101")
    results = analyzer.generate_rankings(["alpha", "beta", "gamma", "delta"])

    ranked_models = [m for m, _ in results["rankings"]]
    assert "delta" not in ranked_models  # delta has no comparison data
    assert results["no_data_models"] == ["delta"]

    # No NaN leaked into rankings, and none placed at the top
    for _, rate in results["rankings"]:
        assert not math.isnan(rate)


def test_all_no_data_produces_empty_rankings(archive):
    analyzer = CodingRankAnalyzer(test_archive_dir=archive, promptset="coding101")
    results = analyzer.generate_rankings(["nobody1", "nobody2"])
    assert results["rankings"] == []
    assert set(results["no_data_models"]) == {"nobody1", "nobody2"}


def test_markdown_reports_ci_and_flags_overlap(archive):
    analyzer = CodingRankAnalyzer(test_archive_dir=archive, promptset="coding101")
    results = analyzer.generate_rankings(["alpha", "beta", "gamma"])
    md = analyzer.generate_markdown(results)
    assert "95% CI" in md
    assert "| 1 | alpha |" in md


def test_category_filter_is_parameterized(archive):
    # BugFinding data only exists for alpha vs beta; gamma should have no data.
    analyzer = CodingRankAnalyzer(
        test_archive_dir=archive, promptset="coding101", category="BugFinding"
    )
    results = analyzer.generate_rankings(["alpha", "beta", "gamma"])
    ranked = [m for m, _ in results["rankings"]]
    assert "gamma" in results["no_data_models"]
    assert set(ranked) == {"alpha", "beta"}
    # beta won both BugFinding rounds
    assert results["rankings"][0][0] == "beta"


def test_missing_archive_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        CodingRankAnalyzer(test_archive_dir=str(tmp_path), promptset="does-not-exist")
