# Tests for the Scipy Solver class
#
import warnings

import numpy as np
import pytest

import pybamm
from tests import get_mesh_for_testing


class TestScipySolver:
    def test_no_sensitivities_error(self):
        model = pybamm.lithium_ion.SPM()
        parameters = model.default_parameter_values
        parameters["Current function [A]"] = "[input]"
        sim = pybamm.Simulation(
            model, solver=pybamm.ScipySolver(), parameter_values=parameters
        )
        with pytest.raises(
            NotImplementedError,
            match=r"Sensitivity analysis is not implemented",
        ):
            sim.solve(
                [0, 1], inputs={"Current function [A]": 1}, calculate_sensitivities=True
            )

    def test_model_solver_python_and_jax(self):
        if pybamm.has_jax():
            formats = ["python", "jax"]
        else:
            formats = ["python"]

        for convert_to_format in formats:
            # Create model
            model = pybamm.BaseModel()
            model.convert_to_format = convert_to_format
            domain = ["negative electrode", "separator", "positive electrode"]
            var = pybamm.Variable("var", domain=domain)
            model.rhs = {var: 0.1 * var}
            model.initial_conditions = {var: 1}
            # No need to set parameters;
            # can use base discretisation (no spatial operators)

            # create discretisation
            mesh = get_mesh_for_testing()
            spatial_methods = {"macroscale": pybamm.FiniteVolume()}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc.process_model(model)
            # Solve
            # Make sure that passing in extra options works
            solver = pybamm.ScipySolver(
                rtol=1e-8, atol=1e-8, method="RK45", extra_options={"first_step": 1e-4}
            )
            t_eval = np.linspace(0, 1, 80)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_equal(solution.t, t_eval)
            np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

            # Test time
            assert solution.total_time == solution.solve_time + solution.set_up_time
            assert solution.termination == "final time"

    def test_model_solver_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        var = pybamm.Variable("var")
        model.rhs = {var: -pybamm.sqrt(var)}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        # Expect solver to fail when y goes negative
        with pytest.raises(pybamm.SolverError):
            solver.solve(model, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_model_solver_with_event_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -0.1 * var}
        model.initial_conditions = {var: 1}
        # needs to work with multiple events (to avoid bug where only last event is
        # used)
        model.events = [
            pybamm.Event("var=0.5", pybamm.min(var - 0.5)),
            pybamm.Event("var=-0.5", pybamm.min(var + 0.5)),
        ]
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval)
        assert len(solution.t) < len(t_eval)
        np.testing.assert_array_equal(solution.t[:-1], t_eval[: len(solution.t) - 1])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])

        # Test event in solution variables_and_events
        np.testing.assert_allclose(
            solution["Event: var=0.5"].data[-1], 0, rtol=1e-7, atol=1e-6
        )

    def test_model_solver_ode_with_jacobian_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: var1, var2: 1 - var1}
        model.initial_conditions = {var1: 1.0, var2: -1.0}
        model.variables = {"var1": var1, "var2": var2}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # Add user-supplied Jacobian to model
        submesh = mesh[("negative electrode", "separator", "positive electrode")]
        N = submesh.npts

        # construct Jacobian in order of model.rhs
        J = []
        for var in model.rhs.keys():
            if var == var1:
                J.append([np.eye(N), np.zeros((N, N))])
            else:
                J.append([-1.0 * np.eye(N), np.zeros((N, N))])

        J = np.block(J)

        def jacobian(t, y):
            return J

        # Solve
        solver = pybamm.ScipySolver(rtol=1e-9, atol=1e-9)
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)

        T, Y = solution.t, solution.y
        np.testing.assert_allclose(
            model.get_processed_variable("var1").evaluate(T, Y),
            np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model.get_processed_variable("var2").evaluate(T, Y),
            np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_model_step_python(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")

        # Step once
        dt = 1
        step_sol = solver.step(None, model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(
            step_sol.y[0], np.exp(0.1 * step_sol.t), rtol=1e-7, atol=1e-6
        )

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.array([0, 1, np.nextafter(1, np.inf), 1.25, 1.5, 1.75, 2])
        )
        np.testing.assert_allclose(
            step_sol_2.y[0], np.exp(0.1 * step_sol_2.t), rtol=1e-7, atol=1e-6
        )

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0], rtol=1e-7, atol=1e-6)

    def test_step_different_model(self):
        disc = pybamm.Discretisation()

        # Create and discretise model1
        model1 = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model1.rhs = {var: 0.1 * var}
        model1.initial_conditions = {var: 1}
        model1.variables = {"var": var, "mul_var": 2 * var, "var2": var}
        disc.process_model(model1)

        # Create and discretise model2, which is slightly different
        model2 = pybamm.BaseModel()
        var = pybamm.Variable("var")
        var2 = pybamm.Variable("var2")
        model2.rhs = {var: 0.2 * var, var2: -0.5 * var2}
        model2.initial_conditions = {var: 1, var2: 1}
        model2.variables = {"var": var, "mul_var": 3 * var, "var2": var2}
        disc.process_model(model2)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")

        # Step once
        dt = 1
        step_sol1 = solver.step(None, model1, dt)
        np.testing.assert_array_equal(step_sol1.t, [0, dt])
        np.testing.assert_allclose(
            step_sol1.y[0], np.exp(0.1 * step_sol1.t), rtol=1e-7, atol=1e-6
        )

        # Step again, the model has changed so this raises an error
        with pytest.raises(RuntimeError, match=r"already been initialised"):
            solver.step(step_sol1, model2, dt)

    def test_model_solver_with_inputs(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)
        model.events = [pybamm.Event("var=0.5", pybamm.min(var - 0.5))]
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        assert len(solution.t) < len(t_eval)
        np.testing.assert_array_equal(solution.t[:-1], t_eval[: len(solution.t) - 1])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_multiple_inputs_happy_path(self, subtests):
        for convert_to_format in ["python", "casadi"]:
            # Create model
            model = pybamm.BaseModel()
            model.convert_to_format = convert_to_format
            domain = ["negative electrode", "separator", "positive electrode"]
            var = pybamm.Variable("var", domain=domain)
            model.rhs = {var: -pybamm.InputParameter("rate") * var}
            model.initial_conditions = {var: 1}
            # create discretisation
            mesh = get_mesh_for_testing()
            spatial_methods = {"macroscale": pybamm.FiniteVolume()}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            disc.process_model(model)

            solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
            t_eval = np.linspace(0, 10, 100)
            ninputs = 8
            inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(ninputs)]

            solutions = solver.solve(model, t_eval, inputs=inputs_list, nproc=2)
            for i in range(ninputs):
                with subtests.test(i=i):
                    solution = solutions[i]
                    np.testing.assert_array_equal(solution.t, t_eval)
                    np.testing.assert_allclose(
                        solution.y[0], np.exp(-0.01 * (i + 1) * solution.t)
                    )

    def test_model_solver_multiple_inputs_discontinuity_error(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        ninputs = 8
        inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(ninputs)]

        model.events = [
            pybamm.Event(
                "discontinuity",
                pybamm.Scalar(t_eval[-1] / 2),
                event_type=pybamm.EventType.DISCONTINUITY,
            )
        ]
        with pytest.raises(
            pybamm.SolverError,
            match=r"Cannot solve for a list of input parameters"
            " sets with discontinuities",
        ):
            solver.solve(model, t_eval, inputs=inputs_list, nproc=2)

    def test_model_solver_multiple_inputs_initial_conditions_error(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 2 * pybamm.InputParameter("rate")}
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        ninputs = 8
        inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(ninputs)]

        with pytest.raises(
            pybamm.SolverError,
            match=r"Input parameters cannot appear in expression "
            "for initial conditions.",
        ):
            solver.solve(model, t_eval, inputs=inputs_list, nproc=2)

    def test_model_solver_with_event_with_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        for use_jacobian in [True, False]:
            model.use_jacobian = use_jacobian
            model.convert_to_format = "casadi"
            domain = ["negative electrode", "separator", "positive electrode"]
            var = pybamm.Variable("var", domain=domain)
            model.rhs = {var: -0.1 * var}
            model.initial_conditions = {var: 1}
            # needs to work with multiple events (to avoid bug where only last event is
            # used)
            model.events = [
                pybamm.Event("var=0.5", pybamm.min(var - 0.5)),
                pybamm.Event("var=-0.5", pybamm.min(var + 0.5)),
            ]
            # No need to set parameters; can use base discretisation (no spatial
            # operators)

            # create discretisation
            mesh = get_mesh_for_testing()
            spatial_methods = {"macroscale": pybamm.FiniteVolume()}
            disc = pybamm.Discretisation(mesh, spatial_methods)
            model_disc = disc.process_model(model, inplace=False)
            # Solve
            solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
            t_eval = np.linspace(0, 10, 100)
            solution = solver.solve(model_disc, t_eval)
            assert len(solution.t) < len(t_eval)
            np.testing.assert_array_equal(
                solution.t[:-1], t_eval[: len(solution.t) - 1]
            )
            np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_with_inputs_with_casadi(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 1}
        model.events = [pybamm.Event("var=0.5", pybamm.min(var - 0.5))]
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": 0.1})
        assert len(solution.t) < len(t_eval)
        np.testing.assert_array_equal(solution.t[:-1], t_eval[: len(solution.t) - 1])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_inputs_in_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        model.rhs = {var1: pybamm.InputParameter("rate") * var1}
        model.initial_conditions = {var1: pybamm.InputParameter("ic 1")}

        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval, inputs={"rate": -1, "ic 1": 0.1})
        np.testing.assert_allclose(
            solution.y[0], 0.1 * np.exp(-solution.t), rtol=1e-6, atol=1e-5
        )

        # Solve again with different initial conditions
        solution = solver.solve(model, t_eval, inputs={"rate": -0.1, "ic 1": 1})
        np.testing.assert_allclose(
            solution.y[0], 1 * np.exp(-0.1 * solution.t), rtol=1e-6, atol=1e-5
        )

    def test_model_solver_manually_update_initial_conditions(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        model.rhs = {var1: -var1}
        model.initial_conditions = {var1: 1}

        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(
            solution.y[0], 1 * np.exp(-solution.t), rtol=1e-6, atol=1e-5
        )

        # Change initial conditions and solve again
        model.concatenated_initial_conditions = pybamm.NumpyConcatenation(
            pybamm.Vector([[2]])
        )
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(
            solution.y[0], 2 * np.exp(-solution.t), rtol=1e-6, atol=1e-5
        )

    def test_scale_and_reference(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1", scale=2, reference=1)
        model.rhs = {var1: -var1}
        model.initial_conditions = {var1: 3}
        model.variables = {"var1": var1}
        solver = pybamm.ScipySolver()
        t_eval = np.linspace(0, 5, 100)
        solution = solver.solve(model, t_eval)

        # Check that the initial conditions and solution are scaled correctly
        np.testing.assert_allclose(
            model.concatenated_initial_conditions.evaluate(), 1, rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            solution.y[0], (solution["var1"].data - 1) / 2, rtol=1e-15, atol=1e-14
        )
