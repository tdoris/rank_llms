"""Statistical helpers for quantifying ranking confidence."""

import math
from typing import Optional, Tuple


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    The Wilson interval is well behaved for small samples and proportions near
    0 or 1, unlike the naive normal approximation. Returns ``(low, high)`` as
    proportions in [0, 1], or ``None`` when there is no data (``n == 0``).

    Args:
        successes: Number of successes (e.g. wins).
        n: Number of trials (e.g. total comparisons).
        z: Standard-normal critical value (1.96 ≈ 95% confidence).

    Returns:
        Tuple of (lower_bound, upper_bound), or None if n == 0.
    """
    if n <= 0:
        return None

    phat = successes / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def intervals_overlap(
    interval_a: Optional[Tuple[float, float]],
    interval_b: Optional[Tuple[float, float]],
) -> bool:
    """
    Return True if two confidence intervals overlap (or either is missing).

    Overlapping intervals mean the two proportions are not statistically
    distinguishable at the interval's confidence level. A missing interval
    (no data) is treated as overlapping, since no distinction can be drawn.
    """
    if interval_a is None or interval_b is None:
        return True
    low_a, high_a = interval_a
    low_b, high_b = interval_b
    return low_a <= high_b and low_b <= high_a
