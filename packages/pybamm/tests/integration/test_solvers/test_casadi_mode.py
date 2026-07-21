import numpy as np

import pybamm


class TestCasadiModes:
    def test_casadi_solver_mode(self):
        solvers = [
            pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6),
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

    def test_casadi_fast_matches_safe_legacy_ode(self):
        # numerical equivalence of "fast" and "safe" on the legacy pure-ODE
        # configuration, solved over a fixed window ("fast" ignores events)
        legacy = {"voltage as a state": "false", "surface form": "false"}
        t_eval = np.linspace(0, 3600, 100)

        solutions = []
        for mode in ["safe", "fast"]:
            model = pybamm.lithium_ion.SPM(legacy)
            solver = pybamm.CasadiSolver(mode=mode, atol=1e-6, rtol=1e-6)
            sim = pybamm.Simulation(model, solver=solver)
            solutions.append(sim.solve(t_eval))

        output_vars = [
            "Terminal voltage [V]",
            "Positive particle surface concentration [mol.m-3]",
            "Electrolyte concentration [mol.m-3]",
        ]
        for var in output_vars:
            np.testing.assert_allclose(
                solutions[0][var].data, solutions[1][var].data, rtol=1e-6, atol=1e-6
            )
