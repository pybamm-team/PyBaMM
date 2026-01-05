import numpy as np
import pytest
from scipy.sparse import eye

import pybamm
from tests import get_discretisation_for_testing, get_mesh_for_testing


class TestCasadiSolver:
    def test_no_sensitivities_error(self):
        model = pybamm.lithium_ion.SPM()
        parameters = model.default_parameter_values
        parameters["Current function [A]"] = "[input]"
        sim = pybamm.Simulation(
            model, solver=pybamm.CasadiSolver(), parameter_values=parameters
        )
        with pytest.raises(
            NotImplementedError,
            match=r"Sensitivity analysis is not implemented",
        ):
            sim.solve(
                [0, 1], inputs={"Current function [A]": 1}, calculate_sensitivities=True
            )

    def test_bad_mode(self):
        with pytest.raises(ValueError, match=r"invalid mode"):
            pybamm.CasadiSolver(mode="bad mode")

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}

        # create discretisation
        disc = pybamm.Discretisation()
        model_disc = disc.process_model(model, inplace=False)
        # Solve
        solver = pybamm.CasadiSolver(
            mode="fast",
            rtol=1e-8,
            atol=1e-8,
            perturb_algebraic_initial_conditions=False,  # added for coverage
        )
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model_disc, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # Safe mode (enforce events that won't be triggered)
        model.events = [pybamm.Event("an event", var + 1)]
        disc.process_model(model)
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # Fast with events
        # with an ODE model this behaves exactly the same as "fast"
        solver = pybamm.CasadiSolver(mode="fast with events", rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

    def test_without_grid(self):
        t_eval = np.linspace(0, 1, 100)

        # ODE model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [pybamm.Event("an event", var + 1)]

        # Safe mode, without grid (enforce events that won't be triggered)
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # DAE model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model.rhs = {var: 0.1 * var}
        model.algebraic = {var2: 1 - var2}
        model.initial_conditions = {var: 1, var2: 1}
        model.events = [pybamm.Event("an event", var + 1)]

        # Safe mode, without grid (enforce events that won't be triggered)
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * t_eval), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[1], np.ones_like(t_eval), rtol=1e-6, atol=1e-5
        )

        # DAE model, errors
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.algebraic = {var2: 1 - var2}
        model.initial_conditions = {var: 1, var2: 1}
        model.events = [pybamm.Event("an event", var + 1)]

        # Safe mode, without grid (enforce events that won't be triggered)
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
        with pytest.raises(pybamm.SolverError, match=r"Maximum number of decreased"):
            solver.solve(model, [0, 10])

    def test_model_solver_python(self):
        # Create model
        pybamm.set_logging_level("ERROR")
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # Solve
        solver = pybamm.CasadiSolver(mode="fast", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        pybamm.set_logging_level("WARNING")

    def test_model_solver_failure(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.algebraic = {var2: var2 - 1}
        model.initial_conditions = {var: 1, var2: 1}
        # add events so that safe mode is used (won't be triggered)
        model.events = [pybamm.Event("10", 10 - var)]

        disc = pybamm.Discretisation()
        model_disc = disc.process_model(model, inplace=False)
        solver = pybamm.CasadiSolver(
            dt_max=1e-3, return_solution_if_failed_early=True, max_step_decrease_count=2
        )
        # Solve with failure at t=2
        # Solution fails early but manages to take some steps so we return it anyway
        # Check that the final solution does indeed stop before t=20
        t_eval = np.linspace(0, 20, 100)
        with pytest.warns(pybamm.SolverWarning):
            solution = solver.solve(model_disc, t_eval)
        assert solution.t[-1] < 20
        # Solve with failure at t=0
        solver = pybamm.CasadiSolver(
            dt_max=1e-3, return_solution_if_failed_early=True, max_step_decrease_count=2
        )
        model.initial_conditions = {var: 0, var2: 1}
        model_disc = disc.process_model(model, inplace=False)
        t_eval = np.linspace(0, 20, 100)
        # This one should fail immediately and throw a `SolverError`
        # since no progress can be made from the first timestep
        with pytest.raises(pybamm.SolverError, match=r"Maximum number of decreased"):
            solver.solve(model, t_eval)

    def test_solver_error(self):
        model = pybamm.lithium_ion.DFN()  # load model
        parameter_values = pybamm.ParameterValues("Chen2020")
        experiment = pybamm.Experiment(
            ["Discharge at 10C for 6 minutes or until 2.5 V"]
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            solver=pybamm.CasadiSolver(mode="fast"),
        )

        with pytest.raises(pybamm.SolverError, match=r"IDA_CONV_FAIL"):
            sim.solve()

    def test_model_solver_events(self):
        # Create model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # Solve using "safe" mode
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y.full()[0, :-1], 1.5)
        np.testing.assert_array_less(solution.y.full()[-1, :-1], 2.5)
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y.full()[:, -1])
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 2 * np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # Solve using "safe" mode with debug off
        pybamm.settings.debug_mode = False
        solver = pybamm.CasadiSolver(mode="safe", rtol=1e-8, atol=1e-8, dt_max=1)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y.full()[0], 1.5)
        np.testing.assert_array_less(solution.y.full()[-1], 2.5 + 1e-10)
        # test the last entry is exactly 2.5
        np.testing.assert_allclose(solution.y[-1, -1], 2.5, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 2 * np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        pybamm.settings.debug_mode = True

        # Try dt_max=0 to enforce using all timesteps
        solver = pybamm.CasadiSolver(dt_max=0, rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y.full()[0], 1.5)
        np.testing.assert_array_less(solution.y.full()[-1], 2.5 + 1e-10)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 2 * np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # Solve using "fast with events" mode
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", 1.5 - var1),
            pybamm.Event("var2 = 2.5", 2.5 - var2),
            pybamm.Event("var1 = 1.5 switch", var1 - 2, pybamm.EventType.SWITCH),
            pybamm.Event("var2 = 2.5 switch", var2 - 3, pybamm.EventType.SWITCH),
        ]

        solver = pybamm.CasadiSolver(mode="fast with events", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y.full()[0, :-1], 1.5)
        np.testing.assert_array_less(solution.y.full()[-1, :-1], 2.5)
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_allclose(
            solution.y_event[:, 0].flatten(), [1.25, 2.5], rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 2 * np.exp(0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

        # Test when an event returns nan
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        model.events = [
            pybamm.Event("event", 1.02 - var),
            pybamm.Event("sqrt event", pybamm.sqrt(1.0199 - var)),
        ]
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y.full()[0], 1.02 + 1e-10)
        np.testing.assert_allclose(solution.y[0, -1], 1.02, rtol=1e-3, atol=1e-2)

    def test_model_step(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)

        # Step once
        dt = 1
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(
            step_sol.y.full()[0], np.exp(0.1 * step_sol.t), rtol=1e-7, atol=1e-6
        )

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.array([0, 1, np.nextafter(1, np.inf), 1.25, 1.5, 1.75, 2])
        )
        np.testing.assert_allclose(
            step_sol_2.y.full()[0], np.exp(0.1 * step_sol_2.t), rtol=1e-7, atol=1e-6
        )

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], step_sol.y.full()[0], rtol=1e-7, atol=1e-6
        )

    def test_model_step_with_input(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        a = pybamm.InputParameter("a")
        model.rhs = {var: a * var}
        model.initial_conditions = {var: 1}
        model.variables = {"a": a}

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)

        # Step with an input
        dt = 0.1
        step_sol = solver.step(None, model, dt, npts=5, inputs={"a": 0.1})
        np.testing.assert_array_equal(step_sol.t, np.linspace(0, dt, 5))
        np.testing.assert_allclose(step_sol.y.full()[0], np.exp(0.1 * step_sol.t))

        # Step again with different inputs
        step_sol_2 = solver.step(step_sol, model, dt, npts=5, inputs={"a": -1})
        np.testing.assert_allclose(
            step_sol_2.t,
            np.array([0, 0.025, 0.05, 0.075, 0.1, 0.1 + 1e-9, 0.125, 0.15, 0.175, 0.2]),
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            step_sol_2["a"].entries,
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, -1, -1, -1, -1, -1]),
        )
        np.testing.assert_allclose(
            step_sol_2.y.full()[0],
            np.concatenate(
                [
                    np.exp(0.1 * step_sol_2.t[:5]),
                    np.exp(0.1 * step_sol_2.t[4])
                    * np.exp(-(step_sol_2.t[5:] - step_sol_2.t[5])),
                ]
            ),
        )

    def test_model_step_events(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
        ]

        # Solve
        step_solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 5
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y.full()[0, :-1], 1.5)
        np.testing.assert_array_less(step_solution.y.full()[-1, :-1], 2.5)
        np.testing.assert_equal(step_solution.t_event[0], step_solution.t[-1])
        np.testing.assert_array_equal(
            step_solution.y_event[:, 0], step_solution.y.full()[:, -1]
        )
        np.testing.assert_allclose(
            step_solution.y.full()[0],
            np.exp(0.1 * step_solution.t),
            rtol=1e-6,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            step_solution.y.full()[-1],
            2 * np.exp(0.1 * step_solution.t),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        model.events = [pybamm.Event("var=0.5", pybamm.min(var - 0.5))]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        assert len(solution.t) < len(t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(-0.1 * solution.t), rtol=1e-04
        )

        # Without grid
        solver = pybamm.CasadiSolver(mode="safe without grid", rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        assert len(solution.t) < len(t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(-0.1 * solution.t), rtol=1e-04
        )
        solution = solver.solve(model, t_eval, inputs={"rate": 1.1})
        assert len(solution.t) < len(t_eval)
        np.testing.assert_allclose(
            solution.y.full()[0], np.exp(-1.1 * solution.t), rtol=1e-04
        )

    def test_model_solver_dae_inputs_in_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: pybamm.InputParameter("rate") * var1}
        model.algebraic = {var2: var1 - var2}
        model.initial_conditions = {
            var1: pybamm.InputParameter("ic 1"),
            var2: pybamm.InputParameter("ic 2"),
        }

        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(
            model, t_eval, inputs={"rate": -1, "ic 1": 0.1, "ic 2": 2}
        )
        np.testing.assert_allclose(
            solution.y.full()[0], 0.1 * np.exp(-solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 0.1 * np.exp(-solution.t), rtol=1e-6, atol=1e-5
        )

        # Solve again with different initial conditions
        solution = solver.solve(
            model, t_eval, inputs={"rate": -0.1, "ic 1": 1, "ic 2": 3}
        )
        np.testing.assert_allclose(
            solution.y.full()[0], 1 * np.exp(-0.1 * solution.t), rtol=1e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            solution.y.full()[-1], 1 * np.exp(-0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

    def test_model_solver_with_non_identity_mass(self):
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", domain="negative electrode")
        var2 = pybamm.Variable("var2", domain="negative electrode")
        model.rhs = {var1: var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        # FV discretisation has identity mass. Manually set the mass matrix to
        # be a diag of 10s here for testing. Note that the algebraic part is all
        # zeros
        mass_matrix = 10 * model.mass_matrix.entries
        model.mass_matrix = pybamm.Matrix(mass_matrix)

        # Note that mass_matrix_inv is just the inverse of the ode block of the
        # mass matrix
        mass_matrix_inv = 0.1 * eye(int(mass_matrix.shape[0] / 2))
        model.mass_matrix_inv = pybamm.Matrix(mass_matrix_inv)

        assert not model.is_standard_form_dae

        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y.full()[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y.full()[-1], 2 * np.exp(0.1 * solution.t))

    def test_dae_solver_algebraic_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}

        solver = pybamm.CasadiSolver()
        t_eval = np.linspace(0, 1)
        with pytest.raises(
            pybamm.SolverError,
            match=r"Cannot use CasadiSolver to solve algebraic model",
        ):
            solver.solve(model, t_eval)

    def test_interpolant_extrapolate(self):
        x = np.linspace(0, 2)
        var = pybamm.Variable("var")
        rhs = pybamm.FunctionParameter("func", {"var": var})

        model = pybamm.BaseModel()
        model.rhs[var] = rhs
        model.initial_conditions[var] = pybamm.Scalar(1)

        # Bug: we need to set the interpolant via parameter values for the extrapolation
        # to be detected
        def func(var):
            return pybamm.Interpolant(x, x, var, interpolator="linear")

        parameter_values = pybamm.ParameterValues({"func": func})
        parameter_values.process_model(model)

        solver = pybamm.CasadiSolver()
        t_eval = [0, 5]

        with pytest.raises(pybamm.SolverError, match=r"interpolation bounds"):
            solver.solve(model, t_eval)

    def test_casadi_safe_no_termination(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -1}
        model.initial_conditions = {v: 1}
        model.events.append(
            pybamm.Event(
                "Triggered event",
                v - 0.5,
                pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            )
        )
        model.events.append(
            pybamm.Event(
                "Ignored event",
                v + 10,
                pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            )
        )
        solver = pybamm.CasadiSolver(mode="safe")
        solver.set_up(model)

        with pytest.raises(pybamm.SolverError, match=r"interpolation bounds"):
            solver.solve(model, t_eval=[0, 1])

    def test_modulo_non_smooth_events(self):
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")

        a = 0.6
        discontinuities = (np.arange(3) + 1) * a

        model.rhs = {var1: pybamm.Modulo(pybamm.t, a)}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 0, var2: 0}
        model.events = [
            pybamm.Event("var1 = 0.55", pybamm.min(0.55 - var1)),
            pybamm.Event("var2 = 1.2", pybamm.min(1.2 - var2)),
        ]
        for discontinuity in discontinuities:
            model.events.append(
                pybamm.Event("nonsmooth rate", pybamm.Scalar(discontinuity))
            )
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        step_solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
        dt = 0.05
        time = 0
        end_time = 3
        step_solution = None
        while time < end_time:
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            time += dt
        np.testing.assert_array_less(step_solution.y[0, :-1], 0.55)
        np.testing.assert_array_less(step_solution.y[-1, :-1], 1.2)
        np.testing.assert_equal(step_solution.t_event[0], step_solution.t[-1])
        np.testing.assert_array_equal(
            step_solution.y_event[:, 0], step_solution.y.full()[:, -1]
        )
        var1_soln = (step_solution.t % a) ** 2 / 2 + a**2 / 2 * (step_solution.t // a)
        var2_soln = 2 * var1_soln
        np.testing.assert_allclose(
            step_solution.y.full()[0], var1_soln, rtol=1e-5, atol=1e-4
        )
        np.testing.assert_allclose(
            step_solution.y.full()[-1], var2_soln, rtol=1e-5, atol=1e-4
        )

    def test_solver_interpolation_warning(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.CasadiSolver()

        # Check for warning with t_interp
        t_eval = np.linspace(0, 1, 10)
        t_interp = t_eval
        with pytest.warns(
            pybamm.SolverWarning,
            match=f"Explicit interpolation times not implemented for {solver.name}",
        ):
            solver.solve(model, t_eval, t_interp=t_interp)

    def test_discontinuous_current(self):
        def car_current(t):
            current = (
                1 * (t >= 0) * (t <= 1000)
                - 0.5 * (1000 < t) * (t <= 2000)
                + 0.5 * (2000 < t)
            )
            return current

        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param["Current function [A]"] = car_current

        sim = pybamm.Simulation(
            model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
        )
        sim.solve([0, 3600])
        current = sim.solution["Current [A]"]
        assert current(0) == 1
        assert current(1500) == -0.5
        assert current(3000) == 0.5
