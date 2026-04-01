import numpy as np

import pybamm


class TestNewtonICComputation:
    """Test that the built-in Newton IC solver in IDAKLU produces correct results."""

    def test_idaklu_newton_ic_vs_casadi(self):
        """IDAKLUSolver with built-in Newton IC matches CasADi rootfinder."""
        model = pybamm.lithium_ion.SPM()
        solver_newton = pybamm.IDAKLUSolver()
        sim_newton = pybamm.Simulation(model, solver=solver_newton)
        sol_newton = sim_newton.solve([0, 3600])

        model2 = pybamm.lithium_ion.SPM()
        solver_casadi = pybamm.IDAKLUSolver(root_method="casadi")
        sim_casadi = pybamm.Simulation(model2, solver=solver_casadi)
        sol_casadi = sim_casadi.solve([0, 3600])

        np.testing.assert_allclose(
            sol_newton["Terminal voltage [V]"].entries,
            sol_casadi["Terminal voltage [V]"].entries,
            rtol=1e-4,
        )

    def test_idaklu_newton_ic_experiment(self):
        """Newton IC solver handles experiment re-initialization correctly."""
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment([
            "Discharge at 1C for 600 seconds",
            "Rest for 60 seconds",
        ])
        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve()
        assert sol is not None
        assert len(sol.cycles) >= 1
