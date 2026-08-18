"""Regression tests for IDAKLU consistent initialization."""

from __future__ import annotations

import pytest

pytest.importorskip("pybamm", reason="PyBaMM not installed")
import pybamm


def test_internal_ic_keeps_converged_state_at_tiny_atol():
    """Internal IC must survive a tiny atol that makes the WRMS step-norm test
    unsatisfiable. Previously aborted at t=0 with IDA_BAD_K when the
    consistent-init Newton's state was discarded for the raw guess."""
    model = pybamm.lithium_ion.DFN()
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-20)
    sim = pybamm.Simulation(model, solver=solver)

    sol = sim.solve([0, 3600])

    assert sol["Voltage [V]"].entries.size > 1
