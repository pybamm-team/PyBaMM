"""
Regression tests for historical IDAKLUSolver bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

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
        indexing was incorrect, causing errors when accessing solution data.

        The fix ensured proper indexing when the experiment terminates before
        the full time range is completed.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Create an experiment where the step will terminate early
        # due to voltage cutoff (discharge will hit 2.5V before 10 hours)
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 10 hours or until 2.5 V",
            ]
        )

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        # This should not raise an indexing error
        sol = sim.solve()

        # Solution should be valid
        assert len(sol.t) > 0

        # Should be able to access variables without indexing errors
        V = sol["Voltage [V]"].data
        assert not np.any(np.isnan(V))

        # Final voltage should be at or near the cutoff
        assert V[-1] <= 2.55  # Allow small tolerance

    def test_infeasible_multi_step_experiment(self):
        """
        Test multi-step experiment where intermediate steps terminate early.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Multi-step experiment where first step terminates early
        experiment = pybamm.Experiment(
            [
                "Discharge at 2C for 5 hours or until 2.5 V",  # Will hit cutoff
                "Rest for 10 minutes",
                "Charge at 1C for 2 hours or until 4.2 V",
            ]
        )

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        sol = sim.solve()

        # Should complete all feasible steps
        assert len(sol.t) > 0

        # Accessing data should work without errors
        V = sol["Voltage [V]"].data
        I = sol["Current [A]"].data

        assert not np.any(np.isnan(V))
        assert not np.any(np.isnan(I))

    def test_variable_interpolation_after_early_termination(self):
        """
        Test that variable interpolation works correctly after early termination.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        experiment = pybamm.Experiment(
            [
                "Discharge at 1C for 3 hours or until 2.5 V",
            ]
        )

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        sol = sim.solve()

        # Interpolation at valid time points should work
        t_valid = sol.t[len(sol.t) // 2]
        V_interp = sol["Voltage [V]"](t_valid)

        assert np.isscalar(V_interp) or V_interp.size == 1
        assert not np.isnan(V_interp)

        # Interpolation at multiple times should also work
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
        assigning a pointer instead of copying memory. This caused use-after-free
        when the original memory was deallocated.

        The fix changed this to:
            std::memcpy(yterm_return, y_val, length_of_final_sv_slice * sizeof(realtype*))

        This test runs the solver with output_variables to exercise the code path
        where this bug occurred.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]", "Current [A]"])
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=solver,
        )

        # Run multiple times to increase chance of catching memory issues
        for _ in range(3):
            sol = sim.solve([0, 600])

            # Solution should be valid
            assert len(sol.t) > 0

            V = sol["Voltage [V]"].data
            I = sol["Current [A]"].data

            assert not np.any(np.isnan(V))
            assert not np.any(np.isnan(I))

    def test_idaklu_with_events_no_crash(self):
        """
        Verify IDAKLU handles events without crashing.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        # Run until event (voltage cutoff)
        experiment = pybamm.Experiment(
            [
                "Discharge at 1C until 3.0 V",
            ]
        )

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        sol = sim.solve()

        # Should complete without segfault
        assert len(sol.t) > 0
        assert "event" in sol.termination.lower() and "3.0" in sol.termination

    def test_idaklu_parallel_solves(self):
        """
        Test IDAKLU with multiple sequential solves to check memory handling.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        solver = pybamm.IDAKLUSolver()

        solutions = []
        for _ in range(5):
            sim = pybamm.Simulation(
                model,
                parameter_values=param,
                solver=solver,
            )
            sol = sim.solve([0, 300])
            solutions.append(sol)

        # All solutions should be valid
        for sol in solutions:
            assert len(sol.t) > 0
            V = sol["Voltage [V]"].data
            assert not np.any(np.isnan(V))


class TestIDAKLUSolverConsistency:
    """Tests for IDAKLU solver consistency and correctness."""

    def test_idaklu_matches_casadi_for_simple_discharge(self):
        """
        Verify IDAKLU produces results consistent with CasADi solver.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        sim_casadi = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=pybamm.CasadiSolver(),
        )
        sim_idaklu = pybamm.Simulation(
            model,
            parameter_values=param,
            solver=pybamm.IDAKLUSolver(),
        )

        sol_casadi = sim_casadi.solve([0, 600])
        sol_idaklu = sim_idaklu.solve([0, 600])

        # Compare at common time points
        t_compare = np.linspace(0, 600, 50)

        V_casadi = sol_casadi["Voltage [V]"](t_compare)
        V_idaklu = sol_idaklu["Voltage [V]"](t_compare)

        np.testing.assert_allclose(V_casadi, V_idaklu, rtol=1e-3)

    def test_idaklu_output_variables_match_full_output(self):
        """
        Verify output_variables mode produces same results as full output.
        """
        model = pybamm.lithium_ion.SPM()
        param = pybamm.ParameterValues("Chen2020")

        solver_full = pybamm.IDAKLUSolver()
        solver_output = pybamm.IDAKLUSolver(
            output_variables=["Voltage [V]", "Discharge capacity [A.h]"]
        )

        sim_full = pybamm.Simulation(model, parameter_values=param, solver=solver_full)
        sim_output = pybamm.Simulation(
            model, parameter_values=param, solver=solver_output
        )

        sol_full = sim_full.solve([0, 600])
        sol_output = sim_output.solve([0, 600])

        # Compare voltage
        t_compare = np.linspace(0, 600, 50)

        V_full = sol_full["Voltage [V]"](t_compare)
        V_output = sol_output["Voltage [V]"](t_compare)

        np.testing.assert_allclose(V_full, V_output, rtol=1e-4)
