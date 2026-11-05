import numpy as np

import pybamm


class TestIDAKLUInfeasibleExperimentFixes:
    """Guards for IDAKLU indexing bug with infeasible experiments."""

    def test_infeasible_experiment_handles_early_termination(self):
        """
        Guards against: PR #4541 - Fix indexing bug with infeasible experiments
        for IDAKLUSolver

        The bug was that when an experiment step terminated early (e.g., due to
        hitting a voltage cutoff before the specified time), the interpolation
        indexing was incorrect.

        Also verifies variable interpolation works after early termination.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        experiment = pybamm.Experiment(["Discharge at 1C for 10 hours or until 2.5 V"])

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        sol = sim.solve()

        assert len(sol.t) > 0

        V = sol["Voltage [V]"].data
        assert not np.any(np.isnan(V))
        assert V[-1] <= 2.55

        t_valid = sol.t[len(sol.t) // 2]
        V_interp = sol["Voltage [V]"](t_valid)
        assert np.isscalar(V_interp) or V_interp.size == 1
        assert not np.isnan(V_interp)

        t_array = np.linspace(sol.t[0], sol.t[-1], 20)
        V_array = sol["Voltage [V]"](t_array)
        assert len(V_array) == 20
        assert not np.any(np.isnan(V_array))


class TestIDAKLUMemorySafetyFixes:
    """Guards for IDAKLU memory safety bug fixes."""

    def test_no_segfault_with_output_variables(self):
        """
        Guards against: PR #4379 - fix segfaults

        The bug was a memory safety issue where `yterm_return = y_val` was
        assigning a pointer instead of copying memory.

        The fix changed this to:
            std::memcpy(yterm_return, y_val, length_of_final_sv_slice * sizeof(realtype*))
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]", "Current [A]"])
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        for _ in range(3):
            sol = sim.solve([0, 600])

            assert len(sol.t) > 0

            V = sol["Voltage [V]"].data
            I = sol["Current [A]"].data

            assert not np.any(np.isnan(V))
            assert not np.any(np.isnan(I))
