"""Hypothesis strategies for sweeps over solver input sets.

A sweep is what ``num_threads`` parallelises over, so the property these feed is
that solving one concurrently is indistinguishable from solving it in a loop.
"""

from __future__ import annotations

import hypothesis.strategies as st


def decay_rate_sweeps(
    min_sets: int = 2, max_sets: int = 8
) -> st.SearchStrategy[list[float]]:
    """Sweeps of distinct positive decay rates.

    Rates span two decades, so a drawn sweep is heterogeneous in solve cost and
    in whether each set reaches an event, which is where a batch that shared
    scratch or returned completion order would show it.

    Parameters
    ----------
    min_sets : int, optional
        Fewest input sets to draw (default 2, the smallest batch).
    max_sets : int, optional
        Most input sets to draw (default 8).

    Returns
    -------
    :class:`hypothesis.strategies.SearchStrategy`
        Lists of rates, each list free of duplicates.
    """
    return st.lists(
        st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=min_sets,
        max_size=max_sets,
        unique=True,
    )
