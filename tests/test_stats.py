"""Tests for the Wilson interval / overlap helpers."""

import pytest

from rank_llms.stats import wilson_interval, intervals_overlap


def test_wilson_no_data_returns_none():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(5, 0) is None


def test_wilson_known_value():
    # Textbook 95% Wilson interval for 60/100 is approximately (0.502, 0.691)
    low, high = wilson_interval(60, 100)
    assert low == pytest.approx(0.502, abs=0.002)
    assert high == pytest.approx(0.691, abs=0.002)


def test_wilson_bounds_stay_in_unit_interval():
    low, high = wilson_interval(10, 10)  # all wins
    assert 0.0 <= low <= high <= 1.0
    low, high = wilson_interval(0, 10)  # no wins
    assert 0.0 <= low <= high <= 1.0


def test_wilson_wider_with_less_data():
    small = wilson_interval(6, 10)
    large = wilson_interval(60, 100)
    assert (small[1] - small[0]) > (large[1] - large[0])


def test_intervals_overlap():
    assert intervals_overlap((0.4, 0.7), (0.5, 0.8)) is True
    assert intervals_overlap((0.1, 0.3), (0.5, 0.8)) is False
    # touching endpoints count as overlapping
    assert intervals_overlap((0.1, 0.5), (0.5, 0.8)) is True


def test_intervals_overlap_missing_treated_as_overlapping():
    assert intervals_overlap(None, (0.5, 0.8)) is True
    assert intervals_overlap((0.1, 0.3), None) is True
