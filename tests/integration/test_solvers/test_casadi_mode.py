#
# Tests for the Casadi Solver Modes
#

import pybamm
import numpy as np


class TestCasadiModes:
    def test_casadi_solver_mode(self):
        solvers = [
            pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6),
            pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-6),
            pybamm.CasadiSolver(mode="fast with events", atol=1e-6, rtol=1e-6),
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

        # assert solutions
        for var in output_vars:
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[1][var].data, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[2][var].data, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                solutions[1][var].data, solutions[2][var].data, rtol=1e-6, atol=1e-6
            )
