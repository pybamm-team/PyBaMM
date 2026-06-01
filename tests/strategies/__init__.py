"""Hypothesis strategies and shared settings for the serialisation property tests.

``serialisation_settings`` is applied as a decorator to the individual
serialisation property/smoke tests rather than registered as a suite-wide
profile, so unrelated property tests (e.g. those under
``tests/unit/test_expression_tree``) keep Hypothesis' default deadlines and
health checks.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

#: Shared Hypothesis settings for the serialisation property tests. The
#: round-trips can be slow and generation time varies, so deadlines and the
#: ``too_slow`` health check are disabled. ``max_examples`` is overridable via
#: the ``PYBAMM_HYPOTHESIS_EXAMPLES`` environment variable.
serialisation_settings = settings(
    max_examples=int(os.environ.get("PYBAMM_HYPOTHESIS_EXAMPLES", "100")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
