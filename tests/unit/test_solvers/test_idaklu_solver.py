#
# Tests for the KLU Solver class
#

from contextlib import redirect_stdout
import io
import unittest

import numpy as np

import pybamm
from tests import get_discretisation_for_testing


@unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
class TestIDAKLUSolver(unittest.TestCase):
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            model.rhs = {u: 0.1 * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}
            model.events = [pybamm.Event("1", 0.2 - u), pybamm.Event("2", v)]

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )

            t_eval = np.linspace(0, 3, 100)
            solution = solver.solve(model, t_eval)

            # test that final time is time of event
            # y = 0.1 t + y0 so y=0.2 when t=2
            np.testing.assert_array_almost_equal(solution.t[-1], 2.0)

            # test that final value is the event value
            np.testing.assert_array_almost_equal(solution.y[0, -1], 0.2)

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(
                solution.y[1, :], np.ones(solution.t.shape)
            )

            # test that y[0] = to true solution
            true_solution = 0.1 * solution.t
            np.testing.assert_array_almost_equal(solution.y[0, :], true_solution)

    def test_model_events(self):
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            # Create model
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            var = pybamm.Variable("var")
            model.rhs = {var: 0.1 * var}
            model.initial_conditions = {var: 1}

            # create discretisation
            disc = pybamm.Discretisation()
            model_disc = disc.process_model(model, inplace=False)
            # Solve
            solver = pybamm.IDAKLUSolver(
                rtol=1e-8,
                atol=1e-8,
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )
            t_eval = np.linspace(0, 1, 100)
            solution = solver.solve(model_disc, t_eval)
            np.testing.assert_array_equal(solution.t, t_eval)
            np.testing.assert_array_almost_equal(
                solution.y[0], np.exp(0.1 * solution.t), decimal=5
            )

            # Check invalid atol type raises an error
            with self.assertRaises(pybamm.SolverError):
                solver._check_atol_type({"key": "value"}, [])

            # enforce events that won't be triggered
            model.events = [pybamm.Event("an event", var + 1)]
            model_disc = disc.process_model(model, inplace=False)
            solver = pybamm.IDAKLUSolver(
                rtol=1e-8,
                atol=1e-8,
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )
            solution = solver.solve(model_disc, t_eval)
            np.testing.assert_array_equal(solution.t, t_eval)
            np.testing.assert_array_almost_equal(
                solution.y[0], np.exp(0.1 * solution.t), decimal=5
            )

            # enforce events that will be triggered
            model.events = [pybamm.Event("an event", 1.01 - var)]
            model_disc = disc.process_model(model, inplace=False)
            solver = pybamm.IDAKLUSolver(
                rtol=1e-8,
                atol=1e-8,
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )
            solution = solver.solve(model_disc, t_eval)
            self.assertLess(len(solution.t), len(t_eval))
            np.testing.assert_array_almost_equal(
                solution.y[0], np.exp(0.1 * solution.t), decimal=5
            )

            # bigger dae model with multiple events
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

            solver = pybamm.IDAKLUSolver(
                rtol=1e-8,
                atol=1e-8,
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )
            t_eval = np.linspace(0, 5, 100)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_less(solution.y[0, :-1], 1.5)
            np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
            np.testing.assert_equal(solution.t_event[0], solution.t[-1])
            np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
            np.testing.assert_array_almost_equal(
                solution.y[0], np.exp(0.1 * solution.t), decimal=5
            )
            np.testing.assert_array_almost_equal(
                solution.y[-1], 2 * np.exp(0.1 * solution.t), decimal=5
            )

    def test_input_params(self):
        # test a mix of scalar and vector input params
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            u1 = pybamm.Variable("u1")
            u2 = pybamm.Variable("u2")
            u3 = pybamm.Variable("u3")
            v = pybamm.Variable("v")
            a = pybamm.InputParameter("a")
            b = pybamm.InputParameter("b", expected_size=2)
            model.rhs = {u1: a * v, u2: pybamm.Index(b, 0), u3: pybamm.Index(b, 1)}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u1: 0, u2: 0, u3: 0, v: 1}

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )

            t_eval = np.linspace(0, 3, 100)
            a_value = 0.1
            b_value = np.array([[0.2], [0.3]])

            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value},
            )

            # test that y[3] remains constant
            np.testing.assert_array_almost_equal(sol.y[3], np.ones(sol.t.shape))

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0], true_solution)

            # test that y[1:3] = to true solution
            true_solution = b_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[1:3], true_solution)

    def test_sensitivities_initial_condition(self):
        for form in ["casadi", "iree"]:
            for output_variables in [[], ["2v"]]:
                if (form == "jax" or form == "iree") and not pybamm.have_jax():
                    continue
                if (form == "iree") and not pybamm.have_iree():
                    continue
                if form == "casadi":
                    root_method = "casadi"
                else:
                    root_method = "lm"
                model = pybamm.BaseModel()
                model.convert_to_format = "jax" if form == "iree" else form
                u = pybamm.Variable("u")
                v = pybamm.Variable("v")
                a = pybamm.InputParameter("a")
                model.rhs = {u: -u}
                model.algebraic = {v: a * u - v}
                model.initial_conditions = {u: 1, v: 1}
                model.variables = {"2v": 2 * v}

                disc = pybamm.Discretisation()
                disc.process_model(model)
                solver = pybamm.IDAKLUSolver(
                    rtol=1e-6,
                    atol=1e-6,
                    root_method=root_method,
                    output_variables=output_variables,
                    options={"jax_evaluator": "iree"} if form == "iree" else {},
                )

                t_eval = np.linspace(0, 3, 100)
                a_value = 0.1

                sol = solver.solve(
                    model, t_eval, inputs={"a": a_value}, calculate_sensitivities=True
                )

                np.testing.assert_array_almost_equal(
                    sol["2v"].sensitivities["a"].full().flatten(),
                    np.exp(-sol.t) * 2,
                    decimal=4,
                )

    def test_ida_roberts_klu_sensitivities(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            a = pybamm.InputParameter("a")
            model.rhs = {u: a * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}
            model.variables = {"2u": 2 * u}

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )

            t_eval = np.linspace(0, 3, 100)
            a_value = 0.1

            # solve first without sensitivities
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value},
            )

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(sol.y[1, :], np.ones(sol.t.shape))

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0, :], true_solution)

            # should be no sensitivities calculated
            with self.assertRaises(KeyError):
                print(sol.sensitivities["a"])

            # now solve with sensitivities (this should cause set_up to be run again)
            sol = solver.solve(
                model, t_eval, inputs={"a": a_value}, calculate_sensitivities=True
            )

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(sol.y[1, :], np.ones(sol.t.shape))

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0, :], true_solution)

            # evaluate the sensitivities using idas
            dyda_ida = sol.sensitivities["a"]

            # evaluate the sensitivities using finite difference
            h = 1e-6
            sol_plus = solver.solve(model, t_eval, inputs={"a": a_value + 0.5 * h})
            sol_neg = solver.solve(model, t_eval, inputs={"a": a_value - 0.5 * h})
            dyda_fd = (sol_plus.y - sol_neg.y) / h
            dyda_fd = dyda_fd.transpose().reshape(-1, 1)

            decimal = (
                2 if form == "iree" else 6
            )  # iree currently operates with single precision
            np.testing.assert_array_almost_equal(dyda_ida, dyda_fd, decimal=decimal)

            # get the sensitivities for the variable
            d2uda = sol["2u"].sensitivities["a"]
            np.testing.assert_array_almost_equal(
                2 * dyda_ida[0:200:2], d2uda, decimal=decimal
            )

    def test_ida_roberts_consistent_initialization(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            model.rhs = {u: 0.1 * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 2}

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )

            # Set up and  model consistently initializate the model
            solver.set_up(model)
            t0 = 0.0
            solver._set_consistent_initialization(model, t0, inputs_dict={})

            # u(t0) = 0, v(t0) = 1
            np.testing.assert_array_almost_equal(model.y0full, [0, 1])
            # u'(t0) = 0.1 * v(t0) = 0.1
            # Since v is algebraic, the initial derivative is set to 0
            np.testing.assert_array_almost_equal(model.ydot0full, [0.1, 0])

    def test_sensitivities_with_events(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["casadi", "python", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            a = pybamm.InputParameter("a")
            b = pybamm.InputParameter("b")
            model.rhs = {u: a * v + b}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}
            model.events = [pybamm.Event("1", 0.2 - u)]

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )

            t_eval = np.linspace(0, 3, 100)
            a_value = 0.1
            b_value = 0.0

            # solve first without sensitivities
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value},
                calculate_sensitivities=True,
            )

            # test that y[1] remains constant
            np.testing.assert_array_almost_equal(sol.y[1, :], np.ones(sol.t.shape))

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0, :], true_solution)

            # evaluate the sensitivities using idas
            dyda_ida = sol.sensitivities["a"]
            dydb_ida = sol.sensitivities["b"]

            # evaluate the sensitivities using finite difference
            h = 1e-6
            sol_plus = solver.solve(
                model, t_eval, inputs={"a": a_value + 0.5 * h, "b": b_value}
            )
            sol_neg = solver.solve(
                model, t_eval, inputs={"a": a_value - 0.5 * h, "b": b_value}
            )
            max_index = min(sol_plus.y.shape[1], sol_neg.y.shape[1]) - 1
            dyda_fd = (sol_plus.y[:, :max_index] - sol_neg.y[:, :max_index]) / h
            dyda_fd = dyda_fd.transpose().reshape(-1, 1)

            decimal = (
                2 if form == "iree" else 6
            )  # iree currently operates with single precision
            np.testing.assert_array_almost_equal(
                dyda_ida[: (2 * max_index), :], dyda_fd, decimal=decimal
            )

            sol_plus = solver.solve(
                model, t_eval, inputs={"a": a_value, "b": b_value + 0.5 * h}
            )
            sol_neg = solver.solve(
                model, t_eval, inputs={"a": a_value, "b": b_value - 0.5 * h}
            )
            max_index = min(sol_plus.y.shape[1], sol_neg.y.shape[1]) - 1
            dydb_fd = (sol_plus.y[:, :max_index] - sol_neg.y[:, :max_index]) / h
            dydb_fd = dydb_fd.transpose().reshape(-1, 1)

            np.testing.assert_array_almost_equal(
                dydb_ida[: (2 * max_index), :], dydb_fd, decimal=decimal
            )

    def test_failures(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        model.use_jacobian = False
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        t_eval = np.linspace(0, 3, 100)
        with self.assertRaisesRegex(pybamm.SolverError, "KLU requires the Jacobian"):
            solver.solve(model, t_eval)

        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        # will give solver error
        t_eval = np.linspace(0, -3, 100)
        with self.assertRaisesRegex(
            pybamm.SolverError, "t_eval must increase monotonically"
        ):
            solver.solve(model, t_eval)

        # try and solve model with numerical issues so the solver fails
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 / u}
        model.initial_conditions = {u: 0}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        t_eval = np.linspace(0, 3, 100)
        with self.assertRaisesRegex(pybamm.SolverError, "idaklu solver failed"):
            solver.solve(model, t_eval)

    def test_dae_solver_algebraic_model(self):
        for form in ["python", "casadi", "jax", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            model = pybamm.BaseModel()
            model.convert_to_format = "jax" if form == "iree" else form
            var = pybamm.Variable("var")
            model.algebraic = {var: var + 1}
            model.initial_conditions = {var: 0}

            disc = pybamm.Discretisation()
            disc.process_model(model)

            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                options={"jax_evaluator": "iree"} if form == "iree" else {},
            )
            t_eval = np.linspace(0, 1)
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_equal(solution.y, -1)

            # change initial_conditions and re-solve (to test if ics_only works)
            model.concatenated_initial_conditions = pybamm.Vector(np.array([[1]]))
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_equal(solution.y, -1)

    def test_banded(self):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "casadi"
        param = model.default_parameter_values
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        t_eval = np.linspace(0, 3600, 100)
        solver = pybamm.IDAKLUSolver()
        soln = solver.solve(model, t_eval)

        options = {
            "jacobian": "banded",
            "linear_solver": "SUNLinSol_Band",
        }
        solver_banded = pybamm.IDAKLUSolver(options=options)
        soln_banded = solver_banded.solve(model, t_eval)

        np.testing.assert_array_almost_equal(soln.y, soln_banded.y, 5)

    def test_setup_options(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1)
        solver = pybamm.IDAKLUSolver()
        soln_base = solver.solve(model, t_eval)

        # test print_stats
        solver = pybamm.IDAKLUSolver(options={"print_stats": True})
        f = io.StringIO()
        with redirect_stdout(f):
            solver.solve(model, t_eval)
        s = f.getvalue()
        self.assertIn("Solver Stats", s)

        solver = pybamm.IDAKLUSolver(options={"print_stats": False})
        f = io.StringIO()
        with redirect_stdout(f):
            solver.solve(model, t_eval)
        s = f.getvalue()
        self.assertEqual(len(s), 0)

        # test everything else
        for jacobian in ["none", "dense", "sparse", "matrix-free", "garbage"]:
            for linear_solver in [
                "SUNLinSol_SPBCGS",
                "SUNLinSol_Dense",
                "SUNLinSol_KLU",
                "SUNLinSol_SPFGMR",
                "SUNLinSol_SPGMR",
                "SUNLinSol_SPTFQMR",
                "garbage",
            ]:
                for precon in ["none", "BBDP"]:
                    options = {
                        "jacobian": jacobian,
                        "linear_solver": linear_solver,
                        "preconditioner": precon,
                        "max_num_steps": 10000,
                    }
                    solver = pybamm.IDAKLUSolver(
                        atol=1e-8,
                        rtol=1e-8,
                        options=options,
                    )
                    if (
                        jacobian == "none"
                        and (linear_solver == "SUNLinSol_Dense")
                        or jacobian == "dense"
                        and (linear_solver == "SUNLinSol_Dense")
                        or jacobian == "sparse"
                        and (
                            linear_solver != "SUNLinSol_Dense"
                            and linear_solver != "garbage"
                        )
                        or jacobian == "matrix-free"
                        and (
                            linear_solver != "SUNLinSol_KLU"
                            and linear_solver != "SUNLinSol_Dense"
                            and linear_solver != "garbage"
                        )
                    ):
                        works = True
                    else:
                        works = False

                    if works:
                        soln = solver.solve(model, t_eval)
                        np.testing.assert_array_almost_equal(soln.y, soln_base.y, 5)
                    else:
                        with self.assertRaises(ValueError):
                            soln = solver.solve(model, t_eval)

    def test_solver_options(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1)
        solver = pybamm.IDAKLUSolver()
        soln_base = solver.solve(model, t_eval)

        options_success = {
            "max_order_bdf": 4,
            "max_num_steps": 490,
            "dt_init": 0.01,
            "dt_max": 1000.9,
            "max_error_test_failures": 11,
            "max_nonlinear_iterations": 5,
            "max_convergence_failures": 11,
            "nonlinear_convergence_coefficient": 1.0,
            "suppress_algebraic_error": True,
            "nonlinear_convergence_coefficient_ic": 0.01,
            "max_num_steps_ic": 6,
            "max_num_jacobians_ic": 5,
            "max_num_iterations_ic": 11,
            "max_linesearch_backtracks_ic": 101,
            "linesearch_off_ic": True,
            "init_all_y_ic": False,
            "linear_solver": "SUNLinSol_KLU",
            "linsol_max_iterations": 6,
            "epsilon_linear_tolerance": 0.06,
            "increment_factor": 0.99,
            "linear_solution_scaling": False,
        }

        # test everything works
        for option in options_success:
            options = {option: options_success[option]}
            solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, options=options)
            soln = solver.solve(model, t_eval)

            np.testing.assert_array_almost_equal(soln.y, soln_base.y, 5)

        options_fail = {
            "max_order_bdf": -1,
            "max_num_steps_ic": -1,
            "max_num_jacobians_ic": -1,
            "max_num_iterations_ic": -1,
            "max_linesearch_backtracks_ic": -1,
            "epsilon_linear_tolerance": -1.0,
            "increment_factor": -1.0,
        }

        # test that the solver throws a warning
        for option in options_fail:
            options = {option: options_fail[option]}
            solver = pybamm.IDAKLUSolver(options=options)

            with self.assertRaises(ValueError):
                solver.solve(model, t_eval)

    def test_with_output_variables(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence
        input_parameters = {}  # Sensitivities dictionary
        t_eval = np.linspace(0, 3600, 100)

        # construct model
        def construct_model():
            model = pybamm.lithium_ion.DFN()
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.update({key: "[input]" for key in input_parameters})
            param.process_model(model)
            param.process_geometry(geometry)
            var_pts = {"x_n": 50, "x_s": 50, "x_p": 50, "r_n": 5, "r_p": 5}
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(
                mesh,
                model.default_spatial_methods,
                remove_independent_variables_from_rhs=True,
            )
            disc.process_model(model)
            return model

        options = {
            "linear_solver": "SUNLinSol_KLU",
            "jacobian": "sparse",
            "num_threads": 4,
        }

        # Use a selection of variables of different types
        output_variables = [
            "Voltage [V]",
            "Time [min]",
            "Current [A]",
            "r_n [m]",
            "x [m]",
            "x_s [m]",
            "Gradient of negative electrolyte potential [V.m-1]",
            "Negative particle flux [mol.m-2.s-1]",
            "Discharge capacity [A.h]",  # ExplicitTimeIntegral
            "Throughput capacity [A.h]",  # ExplicitTimeIntegral
        ]

        # vars that are not in the output_variables list, but are still accessible as
        # they are either model parameters, or do not require access to the state vector
        model_vars = [
            "Time [s]",
            "C-rate",
            "Ambient temperature [K]",
            "Porosity",
        ]

        # A list of variables that are not in the model and cannot be computed
        inaccessible_vars = [
            "Terminal voltage [V]",
            "Negative particle surface stoichiometry",
            "Electrode current density [A.m-2]",
            "Power [W]",
            "Resistance [Ohm]",
        ]

        # Use the full model as comparison (tested separately)
        solver_all = pybamm.IDAKLUSolver(
            atol=1e-8,
            rtol=1e-8,
            options=options,
        )
        sol_all = solver_all.solve(
            construct_model(),
            t_eval,
            inputs=input_parameters,
            calculate_sensitivities=True,
        )

        # Solve for a subset of variables and compare results
        solver = pybamm.IDAKLUSolver(
            atol=1e-8,
            rtol=1e-8,
            options=options,
            output_variables=output_variables,
        )
        sol = solver.solve(
            construct_model(),
            t_eval,
            inputs=input_parameters,
        )

        # Compare output to sol_all
        for varname in [*output_variables, *model_vars]:
            self.assertTrue(np.allclose(sol[varname].data, sol_all[varname].data))

        # Check that the missing variables are not available in the solution
        for varname in inaccessible_vars:
            with self.assertRaises(KeyError):
                sol[varname].data

        # Mock a 1D current collector and initialise (none in the model)
        sol["x_s [m]"].domain = ["current collector"]
        sol["x_s [m]"].initialise_1D()

    def test_with_output_variables_and_sensitivities(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence

        for form in ["casadi", "iree"]:
            if (form == "jax" or form == "iree") and not pybamm.have_jax():
                continue
            if (form == "iree") and not pybamm.have_iree():
                continue
            if form == "casadi":
                root_method = "casadi"
            else:
                root_method = "lm"
            input_parameters = {  # Sensitivities dictionary
                "Current function [A]": 0.222,
                "Separator porosity": 0.3,
            }

            # construct model
            model = pybamm.lithium_ion.DFN()
            model.convert_to_format = "jax" if form == "iree" else form
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.update({key: "[input]" for key in input_parameters})
            param.process_model(model)
            param.process_geometry(geometry)
            var_pts = {"x_n": 50, "x_s": 50, "x_p": 50, "r_n": 5, "r_p": 5}
            mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)

            t_eval = np.linspace(0, 100, 100)

            options = {
                "linear_solver": "SUNLinSol_KLU",
                "jacobian": "sparse",
                "num_threads": 4,
            }
            if form == "iree":
                options["jax_evaluator"] = "iree"

            # Use a selection of variables of different types
            output_variables = [
                "Voltage [V]",
                "Time [min]",
                "x [m]",
                "Negative particle flux [mol.m-2.s-1]",
                "Throughput capacity [A.h]",  # ExplicitTimeIntegral
            ]

            # Use the full model as comparison (tested separately)
            solver_all = pybamm.IDAKLUSolver(
                root_method=root_method,
                atol=1e-8 if form != "iree" else 1e-0,  # iree has reduced precision
                rtol=1e-8 if form != "iree" else 1e-0,  # iree has reduced precision
                options=options,
            )
            sol_all = solver_all.solve(
                model,
                t_eval,
                inputs=input_parameters,
                calculate_sensitivities=True,
            )

            # Solve for a subset of variables and compare results
            solver = pybamm.IDAKLUSolver(
                root_method=root_method,
                atol=1e-8 if form != "iree" else 1e-0,  # iree has reduced precision
                rtol=1e-8 if form != "iree" else 1e-0,  # iree has reduced precision
                options=options,
                output_variables=output_variables,
            )
            sol = solver.solve(
                model,
                t_eval,
                inputs=input_parameters,
                calculate_sensitivities=True,
            )

            # Compare output to sol_all
            tol = 1e-5 if form != "iree" else 1e-2  # iree has reduced precision
            for varname in output_variables:
                np.testing.assert_array_almost_equal(
                    sol[varname].data, sol_all[varname].data, tol
                )

            # Mock a 1D current collector and initialise (none in the model)
            sol["x_s [m]"].domain = ["current collector"]
            sol["x_s [m]"].initialise_1D()

    def test_bad_jax_evaluator(self):
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "jax"
        with self.assertRaises(pybamm.SolverError):
            pybamm.IDAKLUSolver(options={"jax_evaluator": "bad_evaluator"})

    def test_bad_jax_evaluator_output_variables(self):
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "jax"
        with self.assertRaises(pybamm.SolverError):
            pybamm.IDAKLUSolver(
                options={"jax_evaluator": "bad_evaluator"},
                output_variables=["Terminal voltage [V]"],
            )

    def test_with_output_variables_and_event_termination(self):
        model = pybamm.lithium_ion.DFN()
        parameter_values = pybamm.ParameterValues("Chen2020")

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(output_variables=["Terminal voltage [V]"]),
        )
        sol = sim.solve(np.linspace(0, 3600, 1000))
        self.assertEqual(sol.termination, "event: Minimum voltage [V]")

        # create an event that doesn't require the state vector
        eps_p = model.variables["Positive electrode porosity"]
        model.events.append(
            pybamm.Event(
                "Zero positive electrode porosity cut-off",
                pybamm.min(eps_p),
                pybamm.EventType.TERMINATION,
            )
        )

        sim3 = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(output_variables=["Terminal voltage [V]"]),
        )
        sol3 = sim3.solve(np.linspace(0, 3600, 1000))
        self.assertEqual(sol3.termination, "event: Minimum voltage [V]")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True

    unittest.main()
