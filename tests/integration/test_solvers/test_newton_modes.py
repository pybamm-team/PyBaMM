"""Integration tests for Newton IC solver modes (SUBBLOCK vs DECOUPLED_FULL).

Constant-voltage operation forces the solver to compute consistent algebraic
initial conditions at each breakpoint, exercising the Newton IC machinery.
"""

import numpy as np
import pytest

import pybamm


def _solve_cv(model_cls, newton_mode, t_seconds=10):
    """Build a Simulation with a constant-voltage step and solve it."""
    model = model_cls()
    experiment = pybamm.Experiment(
        [f"Hold at 4.0 V for {t_seconds} seconds"],
    )
    solver = pybamm.IDAKLUSolver(
        options={"newton_mode": newton_mode},
    )
    sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
    return sim.solve()


class TestNewtonModes:
    def test_spm_newton_auto(self):
        sol = _solve_cv(pybamm.lithium_ion.SPM, "auto")
        assert sol is not None
        assert len(sol.t) > 1
        np.testing.assert_allclose(
            sol["Voltage [V]"].entries[-1], 4.0, atol=0.01,
        )

    def test_spm_newton_full(self):
        sol = _solve_cv(pybamm.lithium_ion.SPM, "full")
        assert sol is not None
        assert len(sol.t) > 1
        np.testing.assert_allclose(
            sol["Voltage [V]"].entries[-1], 4.0, atol=0.01,
        )

    def test_spm_modes_agree(self):
        sol_auto = _solve_cv(pybamm.lithium_ion.SPM, "auto")
        sol_full = _solve_cv(pybamm.lithium_ion.SPM, "full")
        np.testing.assert_allclose(
            sol_auto["Voltage [V]"].entries,
            sol_full["Voltage [V]"].entries,
            rtol=1e-5,
        )

    def test_dfn_newton_auto(self):
        sol = _solve_cv(pybamm.lithium_ion.DFN, "auto")
        assert sol is not None
        assert len(sol.t) > 1
        np.testing.assert_allclose(
            sol["Voltage [V]"].entries[-1], 4.0, atol=0.01,
        )

    def test_dfn_newton_full(self):
        sol = _solve_cv(pybamm.lithium_ion.DFN, "full")
        assert sol is not None
        assert len(sol.t) > 1
        np.testing.assert_allclose(
            sol["Voltage [V]"].entries[-1], 4.0, atol=0.01,
        )

    def test_dfn_modes_agree(self):
        sol_auto = _solve_cv(pybamm.lithium_ion.DFN, "auto")
        sol_full = _solve_cv(pybamm.lithium_ion.DFN, "full")
        np.testing.assert_allclose(
            sol_auto["Voltage [V]"].entries,
            sol_full["Voltage [V]"].entries,
            rtol=1e-5,
        )
