"""
Memory test configuration.

Pre-imports PyBaMM to ensure module-level initialization (model definitions,
CasADi setup, etc.) happens before test collection. This provides consistent
test isolation.
"""

import pybamm  # noqa: F401
