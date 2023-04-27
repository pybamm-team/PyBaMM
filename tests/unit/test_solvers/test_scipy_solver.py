# Tests for the Scipy Solver class
#
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing, get_discretisation_for_testing
import warnings
import sys


class TestScipySolver(unittest.TestCase):
    def test_model_solver_python_and_jax(self):
        if pybamm.have_jax():
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

        # Test event in solution variables_and_events
        np.testing.assert_array_almost_equal(solution["Event: var=0.5"].data[-1], 0)

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

        # Step again, the model has changed so this raises an error
        with self.assertRaisesRegex(RuntimeError, "already been initialised"):
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
        solver = pybamm.ScipySolver(rtol=1e-8, atol=1e-8)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_almost_equal(
            solution.y[0], 2 * np.exp(-solution.t), decimal=5
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
        np.testing.assert_array_almost_equal(
            model.concatenated_initial_conditions.evaluate(), 1
        )
        np.testing.assert_array_almost_equal(
            solution.y[0], (solution["var1"].data - 1) / 2, decimal=14
        )


class TestScipySolverWithSensitivity(unittest.TestCase):
    def test_solve_sensitivity_scalar_var_scalar_input(self):
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        p = pybamm.InputParameter("p")
        model.rhs = {var: p * var}
        model.initial_conditions = {var: 1}
        model.variables = {"var squared": var**2}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(
            model, t_eval, inputs={"p": 0.1}, calculate_sensitivities=True
        )
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(
            solution.sensitivities["p"],
            (solution.t * np.exp(0.1 * solution.t))[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var squared"].data, np.exp(0.1 * solution.t) ** 2
        )
        np.testing.assert_allclose(
            solution["var squared"].sensitivities["p"],
            (2 * np.exp(0.1 * solution.t) * solution.t * np.exp(0.1 * solution.t))[
                :, np.newaxis
            ],
        )

        # More complicated model
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        r = pybamm.InputParameter("r")
        s = pybamm.InputParameter("s")
        model.rhs = {var: p * q}
        model.initial_conditions = {var: r}
        model.variables = {"var times s": var * s}

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(
            model,
            t_eval,
            inputs={"p": 0.1, "q": 2, "r": -1, "s": 0.5},
            calculate_sensitivities=True,
        )
        np.testing.assert_allclose(solution.y[0], -1 + 0.2 * solution.t)
        np.testing.assert_allclose(
            solution.sensitivities["p"],
            (2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution.sensitivities["q"],
            (0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution.sensitivities["r"], 1)
        np.testing.assert_allclose(solution.sensitivities["s"], 0)
        np.testing.assert_allclose(
            solution.sensitivities["all"],
            np.hstack(
                [
                    solution.sensitivities["p"],
                    solution.sensitivities["q"],
                    solution.sensitivities["r"],
                    solution.sensitivities["s"],
                ]
            ),
        )
        np.testing.assert_allclose(
            solution["var times s"].data, 0.5 * (-1 + 0.2 * solution.t)
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["p"],
            0.5 * (2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["q"],
            0.5 * (0.1 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution["var times s"].sensitivities["r"], 0.5)
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["s"],
            (-1 + 0.2 * solution.t)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["all"],
            np.hstack(
                [
                    solution["var times s"].sensitivities["p"],
                    solution["var times s"].sensitivities["q"],
                    solution["var times s"].sensitivities["r"],
                    solution["var times s"].sensitivities["s"],
                ]
            ),
        )

    def test_solve_sensitivity_vector_var_scalar_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()
        param = pybamm.InputParameter("param")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {"var": var}

        # create discretisation
        disc = get_discretisation_for_testing()
        disc.process_model(model)
        n = disc.mesh["negative electrode"].npts

        # Solve - scalar input
        solver = pybamm.ScipySolver()
        t_eval = np.linspace(0, 1)
        solution = solver.solve(
            model, t_eval, inputs={"param": 7}, calculate_sensitivities=True
        )
        np.testing.assert_array_almost_equal(
            solution["var"].data,
            np.tile(2 * np.exp(-7 * t_eval), (n, 1)),
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivities["param"],
            np.repeat(-2 * t_eval * np.exp(-7 * t_eval), n)[:, np.newaxis],
            decimal=4,
        )

        # More complicated model
        # Create model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var", "negative electrode")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        r = pybamm.InputParameter("r")
        s = pybamm.InputParameter("s")
        model.rhs = {var: p * q}
        model.initial_conditions = {var: r}
        model.variables = {"var times s": var * s}

        # Discretise
        disc.process_model(model)

        # Solve
        # Make sure that passing in extra options works
        solver = pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        t_eval = np.linspace(0, 1, 80)
        solution = solver.solve(
            model,
            t_eval,
            inputs={"p": 0.1, "q": 2, "r": -1, "s": 0.5},
            calculate_sensitivities=True,
        )
        np.testing.assert_allclose(solution.y, np.tile(-1 + 0.2 * solution.t, (n, 1)))
        np.testing.assert_allclose(
            solution.sensitivities["p"],
            np.repeat(2 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution.sensitivities["q"],
            np.repeat(0.1 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution.sensitivities["r"], 1)
        np.testing.assert_allclose(solution.sensitivities["s"], 0)
        np.testing.assert_allclose(
            solution.sensitivities["all"],
            np.hstack(
                [
                    solution.sensitivities["p"],
                    solution.sensitivities["q"],
                    solution.sensitivities["r"],
                    solution.sensitivities["s"],
                ]
            ),
        )
        np.testing.assert_allclose(
            solution["var times s"].data, np.tile(0.5 * (-1 + 0.2 * solution.t), (n, 1))
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["p"],
            np.repeat(0.5 * (2 * solution.t), n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["q"],
            np.repeat(0.5 * (0.1 * solution.t), n)[:, np.newaxis],
        )
        np.testing.assert_allclose(solution["var times s"].sensitivities["r"], 0.5)
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["s"],
            np.repeat(-1 + 0.2 * solution.t, n)[:, np.newaxis],
        )
        np.testing.assert_allclose(
            solution["var times s"].sensitivities["all"],
            np.hstack(
                [
                    solution["var times s"].sensitivities["p"],
                    solution["var times s"].sensitivities["q"],
                    solution["var times s"].sensitivities["r"],
                    solution["var times s"].sensitivities["s"],
                ]
            ),
        )

    def test_solve_sensitivity_vector_var_vector_input(self):
        var = pybamm.Variable("var", "negative electrode")
        model = pybamm.BaseModel()

        param = pybamm.InputParameter("param", "negative electrode")
        model.rhs = {var: -param * var}
        model.initial_conditions = {var: 2}
        model.variables = {
            "var": var,
            "integral of var": pybamm.Integral(var, pybamm.standard_spatial_vars.x_n),
        }

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        n = disc.mesh["negative electrode"].npts

        # Solve - constant input
        solver = pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        t_eval = np.linspace(0, 1)
        solution = solver.solve(
            model,
            t_eval,
            inputs={"param": 7 * np.ones(n)},
            calculate_sensitivities=True,
        )
        l_n = mesh["negative electrode"].edges[-1]
        np.testing.assert_array_almost_equal(
            solution["var"].data,
            np.tile(2 * np.exp(-7 * t_eval), (n, 1)),
            decimal=4,
        )

        np.testing.assert_array_almost_equal(
            solution["var"].sensitivities["param"],
            np.vstack([np.eye(n) * -2 * t * np.exp(-7 * t) for t in t_eval]),
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].data,
            2 * np.exp(-7 * t_eval) * l_n,
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].sensitivities["param"],
            np.tile(-2 * t_eval * np.exp(-7 * t_eval) * l_n / n, (n, 1)).T,
        )

        # Solve - linspace input
        solver = pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        t_eval = np.linspace(0, 1)
        p_eval = np.linspace(1, 2, n)
        solution = solver.solve(
            model, t_eval, inputs={"param": p_eval}, calculate_sensitivities=True
        )
        l_n = mesh["negative electrode"].edges[-1]
        np.testing.assert_array_almost_equal(
            solution["var"].data, 2 * np.exp(-p_eval[:, np.newaxis] * t_eval), decimal=4
        )
        np.testing.assert_array_almost_equal(
            solution["var"].sensitivities["param"],
            np.vstack([np.diag(-2 * t * np.exp(-p_eval * t)) for t in t_eval]),
        )

        np.testing.assert_array_almost_equal(
            solution["integral of var"].data,
            np.sum(
                2
                * np.exp(-p_eval[:, np.newaxis] * t_eval)
                * mesh["negative electrode"].d_edges[:, np.newaxis],
                axis=0,
            ),
        )
        np.testing.assert_array_almost_equal(
            solution["integral of var"].sensitivities["param"],
            np.vstack([-2 * t * np.exp(-p_eval * t) * l_n / n for t in t_eval]),
        )


if __name__ == "__main__":
    print("Add -v for more debug output")

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
