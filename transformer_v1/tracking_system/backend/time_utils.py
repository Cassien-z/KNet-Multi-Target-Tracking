"""Time-axis helpers shared by the API service and dependency-light tests."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np


def nearest_time_index(times: np.ndarray, target: float) -> int:
    """Return the closest frame index, preferring the earlier frame on a tie."""
    if times.size == 0:
        raise ValueError("Cannot seek an empty tracking sequence")
    right = bisect_left(times, target)
    if right <= 0:
        return 0
    if right >= times.size:
        return int(times.size - 1)
    left = right - 1
    return left if target - float(times[left]) <= float(times[right]) - target else right
