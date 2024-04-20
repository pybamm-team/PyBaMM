#
# Tests for the Casadi Solver Mode
#
from tests import TestCase
import pybamm
import unittest
import numpy as np


class TestCasadiMode(TestCase):
    def test_casadi_solver_mode(self):
        solvers = [
            pybamm.CasadiSolver(mode="safe", atol=1e-8, rtol=1e-8),
            pybamm.CasadiSolver(mode="fast", atol=1e-8, rtol=1e-8),
            pybamm.CasadiSolver(mode="fast with events", atol=1e-8, rtol=1e-8),
        ]

        # define experiment
        experiment = pybamm.Experiment(
            [
                ("Discharge at 1C until 3.0 V (10 seconds period)",),
            ]
        )

        solutions = []
        for solver in solvers:
            # define model
            model = pybamm.lithium_ion.SPM()

            # solve simulation
            sim = pybamm.Simulation(
                model,
                experiment=experiment,
                solver=solver,
            )

            # append to solutions
            solutions.append(sim.solve())

        # define variables to compare
        output_vars = [
            "Terminal voltage [V]",
            "Positive particle surface concentration [mol.m-3]",
            "Electrolyte concentration [mol.m-3]",
        ]

        # compare solutions
        for var in output_vars:
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[1][var].data, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[2][var].data, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                solutions[1][var].data, solutions[2][var].data, rtol=1e-8, atol=1e-8
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
