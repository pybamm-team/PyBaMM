#
# Tests for the Scipy Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing
import warnings
import sys
from platform import system, version


class TestScipySolver(unittest.TestCase):
    def test_model_solver_python_and_jax(self):

        if not (
            system() == "Windows" or (system() == "Darwin" and "ARM64" in version())
        ):
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
            self.assertEqual(
                solution.total_time, solution.solve_time + solution.set_up_time
            )
            self.assertEqual(solution.termination, "final time")

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
        with self.assertRaises(pybamm.SolverError):
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
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t[:-1], t_eval[: len(solution.t) - 1])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])

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
        combined_submesh = mesh.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )
        N = combined_submesh.npts

        # construct jacobian in order of model.rhs
        J = []
        for var in model.rhs.keys():
            if var.id == var1.id:
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
        np.testing.assert_array_almost_equal(
            model.variables["var1"].evaluate(T, Y),
            np.ones((N, T.size)) * np.exp(T[np.newaxis, :]),
        )
        np.testing.assert_array_almost_equal(
            model.variables["var2"].evaluate(T, Y),
            np.ones((N, T.size)) * (T[np.newaxis, :] - np.exp(T[np.newaxis, :])),
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
        np.testing.assert_array_almost_equal(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(step_sol, model, dt, npts=5)
        np.testing.assert_array_equal(
            step_sol_2.t, np.concatenate([np.array([0]), np.linspace(dt, 2 * dt, 5)])
        )
        np.testing.assert_array_almost_equal(
            step_sol_2.y[0], np.exp(0.1 * step_sol_2.t)
        )

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(solution.y[0], step_sol.y[0])

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
        np.testing.assert_array_almost_equal(step_sol1.y[0], np.exp(0.1 * step_sol1.t))

        # Step again, the model has changed
        step_sol2 = solver.step(step_sol1, model2, dt)
        np.testing.assert_array_equal(step_sol2.t, [0, dt, 2 * dt])
        np.testing.assert_array_almost_equal(
            step_sol2.all_ys[0][0], np.exp(0.1 * step_sol1.t)
        )

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
        self.assertLess(len(solution.t), len(t_eval))
        np.testing.assert_array_equal(solution.t[:-1], t_eval[: len(solution.t) - 1])
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))

    def test_model_solver_multiple_inputs_happy_path(self):
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
                with self.subTest(i=i):
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
        with self.assertRaisesRegex(
            pybamm.SolverError,
            (
                "Cannot solve for a list of input parameters"
                " sets with discontinuities"
            ),
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

        with self.assertRaisesRegex(
            pybamm.SolverError,
            ("Input parameters cannot appear in expression " "for initial conditions."),
        ):
            solver.solve(model, t_eval, inputs=inputs_list, nproc=2)

    def test_model_solver_multiple_inputs_jax_format_error(self):
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax"
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: -pybamm.InputParameter("rate") * var}
        model.initial_conditions = {var: 2 * pybamm.InputParameter("rate")}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)
        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8, method="RK45")
        t_eval = np.linspace(0, 10, 100)
        ninputs = 8
        inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(ninputs)]

        with self.assertRaisesRegex(
            pybamm.SolverError,
            (
                "Cannot solve list of inputs with multiprocessing "
                'when model in format "jax".'
            ),
        ):
            solver.solve(model, t_eval, inputs=inputs_list, nproc=2)

    def test_model_solver_with_external(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=domain)
        var2 = pybamm.Variable("var2", domain=domain)
        model.rhs = {var1: -var2}
        model.initial_conditions = {var1: 1}
        model.external_variables = [var2]
        model.variables = {"var2": var2}
        # No need to set parameters; can use base discretisation (no spatial
        # operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        t_eval = np.linspace(0, 10, 100)
        solution = solver.solve(
            model, t_eval, external_variables={"var2": 0.5 * np.ones(100)}
        )
        np.testing.assert_allclose(solution.y[0], 1 - 0.5 * solution.t, rtol=1e-06)

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
            self.assertLess(len(solution.t), len(t_eval))
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
        self.assertLess(len(solution.t), len(t_eval))
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
        np.testing.assert_array_almost_equal(
            solution.y[0], 0.1 * np.exp(-solution.t), decimal=5
        )

        # Solve again with different initial conditions
        solution = solver.solve(model, t_eval, inputs={"rate": -0.1, "ic 1": 1})
        np.testing.assert_array_almost_equal(
            solution.y[0], 1 * np.exp(-0.1 * solution.t), decimal=5
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
        np.testing.assert_array_almost_equal(
            solution.y[0], 1 * np.exp(-solution.t), decimal=5
        )

        # Change initial conditions and solve again
        model.concatenated_initial_conditions = pybamm.NumpyConcatenation(
            pybamm.Vector([[2]])
        )
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], 2 * np.exp(-solution.t), decimal=5
        )


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
