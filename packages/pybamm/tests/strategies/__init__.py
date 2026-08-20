"""Hypothesis strategies and shared settings for the property tests.

``serialisation_settings`` is applied as a decorator to the individual
serialisation property/smoke tests rather than registered as a suite-wide
profile, so unrelated property tests (e.g. those under
``tests/unit/test_expression_tree``) keep Hypothesis' default deadlines and
health checks.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

#: Shared Hypothesis settings for the serialisation property tests
# Increase `max_examples` for a larger amount of randomised draws
serialisation_settings = settings(
    max_examples=500,
    deadline=None,
)

#: Shared Hypothesis settings for properties whose example runs a solve.
# Far fewer examples than the serialisation properties: each integrates a sweep twice.
solve_settings = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

__all__ = ["serialisation_settings", "solve_settings"]
