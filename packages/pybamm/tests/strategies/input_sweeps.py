"""Hypothesis strategies for sweeps over solver input sets."""

from __future__ import annotations

import hypothesis.strategies as st


def decay_rate_sweeps(
    min_sets: int = 2, max_sets: int = 8
) -> st.SearchStrategy[list[float]]:
    """Sweeps of distinct positive decay rates spanning two decades of solve cost."""
    return st.lists(
        st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=min_sets,
        max_size=max_sets,
        unique=True,
    )
