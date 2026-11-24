"""Reusable test helpers for model zoos.

The contract checks are shipped rather than kept in the test tree so that a
third-party package advertising itself through the ``pybamm_zoo_models`` entry
point can hold its own models to the same standard in its own CI.
"""

from __future__ import annotations

from pybamm_model_zoo.testing import contract

__all__ = ["contract"]
