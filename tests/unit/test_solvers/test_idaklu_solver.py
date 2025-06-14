import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest
from scipy.sparse import eye

import pybamm
from tests import get_discretisation_for_testing


class TestIDAKLUSolver:
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            # Test
            t_eval = [0, 3]
            solution = solver.solve(model, t_eval)

            # test that final time is time of event
            # y = 0.1 t + y0 so y=0.2 when t=2
            np.testing.assert_allclose(solution.t[-1], 2.0, rtol=1e-7, atol=1e-6)

            # test that final value is the event value
            np.testing.assert_allclose(solution.y[0, -1], 0.2, rtol=1e-7, atol=1e-6)

            # test that y[1] remains constant
            np.testing.assert_allclose(
                solution.y[1, :], np.ones(solution.t.shape), rtol=1e-7, atol=1e-6
            )

            # test that y[0] = to true solution
            true_solution = 0.1 * solution.t
            np.testing.assert_allclose(
                solution.y[0, :], true_solution, rtol=1e-7, atol=1e-6
            )

    def test_multiple_inputs(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        rate = pybamm.InputParameter("rate")
        model.rhs = {var: -rate * var}
        model.initial_conditions = {var: 2}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        for num_threads, num_solvers in [
            [1, None],
            [2, None],
            [8, None],
            [8, 1],
            [8, 2],
            [8, 7],
        ]:
            options = {"num_threads": num_threads}
            if num_solvers is not None:
                options["num_solvers"] = num_solvers
            solver = pybamm.IDAKLUSolver(rtol=1e-5, atol=1e-5, options=options)
            t_eval = [0, 1]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 10)
            ninputs = 8
            inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(ninputs)]

            solutions = solver.solve(
                model, t_eval, inputs=inputs_list, t_interp=t_interp
            )

            # check solution
            for inputs, solution in zip(inputs_list, solutions):
                print("checking solution", inputs, solution.all_inputs)
                np.testing.assert_array_equal(solution.t, t_interp)
                np.testing.assert_allclose(
                    solution.y[0],
                    2 * np.exp(-inputs["rate"] * solution.t),
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_model_events(self):
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            t_eval = [0, 1]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 100)

            solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
            np.testing.assert_array_equal(
                solution.t, t_interp, err_msg=f"Failed for form {form}"
            )
            np.testing.assert_allclose(
                solution.y[0],
                np.exp(0.1 * solution.t),
                rtol=1e-6,
                atol=1e-5,
                err_msg=f"Failed for form {form}",
            )

            # Check invalid atol type raises an error
            with pytest.raises(pybamm.SolverError):
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
            solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
            np.testing.assert_array_equal(solution.t, t_interp)
            np.testing.assert_allclose(
                solution.y[0],
                np.exp(0.1 * solution.t),
                rtol=1e-6,
                atol=1e-5,
                err_msg=f"Failed for form {form}",
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
            solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
            assert len(solution.t) < len(t_interp)
            np.testing.assert_allclose(
                solution.y[0],
                np.exp(0.1 * solution.t),
                rtol=1e-6,
                atol=1e-5,
                err_msg=f"Failed for form {form}",
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
            t_eval = np.array([0, 5])
            solution = solver.solve(model, t_eval)
            np.testing.assert_array_less(solution.y[0, :-1], 1.5)
            np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
            np.testing.assert_equal(solution.t_event[0], solution.t[-1])
            np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
            np.testing.assert_allclose(
                solution.y[0],
                np.exp(0.1 * solution.t),
                rtol=1e-6,
                atol=1e-5,
                err_msg=f"Failed for form {form}",
            )
            np.testing.assert_allclose(
                solution.y[-1],
                2 * np.exp(0.1 * solution.t),
                rtol=1e-6,
                atol=1e-5,
                err_msg=f"Failed for form {form}",
            )

    def test_input_params(self):
        # test a mix of scalar and vector input params
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            t_eval = [0, 3]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
            a_value = 0.1
            b_value = np.array([[0.2], [0.3]])

            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value},
                t_interp=t_interp,
            )

            # test that y[3] remains constant
            np.testing.assert_allclose(
                sol.y[3],
                np.ones(sol.t.shape),
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_allclose(
                sol.y[0],
                true_solution,
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # test that y[1:3] = to true solution
            true_solution = b_value * sol.t
            np.testing.assert_allclose(
                sol.y[1:3],
                true_solution,
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

    def test_sensitivities_initial_condition(self):
        for form in ["casadi", "iree"]:
            for output_variables in [[], ["2v"]]:
                if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

                t_eval = [0, 3]
                a_value = 0.1

                sol = solver.solve(
                    model,
                    t_eval,
                    inputs={"a": a_value},
                    calculate_sensitivities=True,
                )

                np.testing.assert_allclose(
                    sol["2v"].sensitivities["a"].flatten(),
                    np.exp(-sol.t) * 2,
                    rtol=1e-5,
                    atol=1e-4,
                    err_msg=f"Failed for form {form}",
                )

    def test_ida_roberts_klu_sensitivities(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            t_eval = [0, 3]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
            a_value = 0.1

            # solve first without sensitivities
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value},
                t_interp=t_interp,
            )

            # test that y[1] remains constant
            np.testing.assert_allclose(
                sol.y[1, :],
                np.ones(sol.t.shape),
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_allclose(
                sol.y[0, :],
                true_solution,
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # should be no sensitivities calculated
            with pytest.raises(KeyError):
                print(sol.sensitivities["a"])

            # now solve with sensitivities (this should cause set_up to be run again)
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value},
                calculate_sensitivities=True,
                t_interp=t_interp,
            )

            # test that y[1] remains constant
            np.testing.assert_allclose(
                sol.y[1, :],
                np.ones(sol.t.shape),
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_allclose(
                sol.y[0, :],
                true_solution,
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # evaluate the sensitivities using idas
            dyda_ida = sol.sensitivities["a"]

            # evaluate the sensitivities using finite difference
            h = 1e-6
            sol_plus = solver.solve(
                model, t_eval, inputs={"a": a_value + 0.5 * h}, t_interp=t_interp
            )
            sol_neg = solver.solve(
                model, t_eval, inputs={"a": a_value - 0.5 * h}, t_interp=t_interp
            )
            dyda_fd = (sol_plus.y - sol_neg.y) / h
            dyda_fd = dyda_fd.transpose().reshape(-1, 1)

            decimal = (
                2 if form == "iree" else 6
            )  # iree currently operates with single precision
            np.testing.assert_allclose(
                dyda_ida,
                dyda_fd,
                err_msg=f"Failed for form {form}",
                rtol=10 ** (-decimal - 1),
                atol=10 ** (-decimal),
            )

            # get the sensitivities for the variable
            d2uda = sol["2u"].sensitivities["a"]
            np.testing.assert_allclose(
                2 * dyda_ida[0:200:2],
                d2uda,
                rtol=10 ** (-decimal - 1),
                atol=10 ** (-decimal),
                err_msg=f"Failed for form {form}",
            )

    def test_ida_roberts_consistent_initialization(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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
            np.testing.assert_allclose(
                model.y0full,
                [0, 1],
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )
            # u'(t0) = 0.1 * v(t0) = 0.1
            # Since v is algebraic, the initial derivative is set to 0
            np.testing.assert_allclose(
                model.ydot0full,
                [0.1, 0],
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

    def test_sensitivities_with_events(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            t_eval = [0, 3]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 100)

            a_value = 0.1
            b_value = 0.0

            # solve first without sensitivities
            sol = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value},
                calculate_sensitivities=True,
                t_interp=t_interp,
            )

            # test that y[1] remains constant
            np.testing.assert_allclose(
                sol.y[1, :],
                np.ones(sol.t.shape),
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_allclose(
                sol.y[0, :],
                true_solution,
                err_msg=f"Failed for form {form}",
                rtol=1e-7,
                atol=1e-6,
            )

            # evaluate the sensitivities using idas
            dyda_ida = sol.sensitivities["a"]
            dydb_ida = sol.sensitivities["b"]

            # evaluate the sensitivities using finite difference
            h = 1e-6
            sol_plus = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value + 0.5 * h, "b": b_value},
                t_interp=t_interp,
            )
            sol_neg = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value - 0.5 * h, "b": b_value},
                t_interp=t_interp,
            )
            max_index = min(sol_plus.y.shape[1], sol_neg.y.shape[1]) - 1
            dyda_fd = (sol_plus.y[:, :max_index] - sol_neg.y[:, :max_index]) / h
            dyda_fd = dyda_fd.transpose().reshape(-1, 1)

            decimal = (
                2 if form == "iree" else 6
            )  # iree currently operates with single precision
            np.testing.assert_allclose(
                dyda_ida[: (2 * max_index), :],
                dyda_fd,
                rtol=10 ** (-decimal - 1),
                atol=10 ** (-decimal),
                err_msg=f"Failed for form {form}",
            )

            sol_plus = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value + 0.5 * h},
                t_interp=t_interp,
            )
            sol_neg = solver.solve(
                model,
                t_eval,
                inputs={"a": a_value, "b": b_value - 0.5 * h},
                t_interp=t_interp,
            )
            max_index = min(sol_plus.y.shape[1], sol_neg.y.shape[1]) - 1
            dydb_fd = (sol_plus.y[:, :max_index] - sol_neg.y[:, :max_index]) / h
            dydb_fd = dydb_fd.transpose().reshape(-1, 1)

            np.testing.assert_allclose(
                dydb_ida[: (2 * max_index), :],
                dydb_fd,
                rtol=10 ** (-decimal - 1),
                atol=10 ** (-decimal),
                err_msg=f"Failed for form {form}",
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

        t_eval = [0, 3]
        with pytest.raises(pybamm.SolverError, match="KLU requires the Jacobian"):
            solver.solve(model, t_eval)

        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -0.1 * u}
        model.initial_conditions = {u: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        # will give solver error
        t_eval = [0, -3]
        with pytest.raises(
            pybamm.SolverError, match="t_eval must increase monotonically"
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

        t_eval = [0, 3]
        with pytest.raises(ValueError):
            solver.solve(model, t_eval)

    def test_dae_solver_algebraic_model(self):
        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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
            t_eval = [0, 1]
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

        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        solver = pybamm.IDAKLUSolver()
        soln = solver.solve(model, t_eval, t_interp=t_interp)

        options = {
            "jacobian": "banded",
            "linear_solver": "SUNLinSol_Band",
        }
        solver_banded = pybamm.IDAKLUSolver(options=options)
        soln_banded = solver_banded.solve(model, t_eval, t_interp=t_interp)

        np.testing.assert_allclose(soln.y, soln_banded.y, rtol=1e-6, atol=1e-5)

    def test_setup_options(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = [0, 1]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        solver = pybamm.IDAKLUSolver()
        soln_base = solver.solve(model, t_eval, t_interp=t_interp)

        # test print_stats
        solver = pybamm.IDAKLUSolver(options={"print_stats": True})
        f = io.StringIO()
        with redirect_stdout(f):
            solver.solve(model, t_eval, t_interp=t_interp)
        s = f.getvalue()
        assert "Solver Stats" in s

        solver = pybamm.IDAKLUSolver(options={"print_stats": False})
        f = io.StringIO()
        with redirect_stdout(f):
            solver.solve(model, t_eval, t_interp=t_interp)
        s = f.getvalue()
        assert len(s) == 0

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
                    works = (
                        (jacobian == "none" and (linear_solver == "SUNLinSol_Dense"))
                        or (
                            jacobian == "dense" and (linear_solver == "SUNLinSol_Dense")
                        )
                        or (
                            jacobian == "sparse"
                            and (
                                linear_solver != "SUNLinSol_Dense"
                                and linear_solver != "garbage"
                            )
                        )
                        or (
                            jacobian == "matrix-free"
                            and (
                                linear_solver != "SUNLinSol_KLU"
                                and linear_solver != "SUNLinSol_Dense"
                                and linear_solver != "garbage"
                            )
                        )
                    )

                    if works:
                        soln = solver.solve(model, t_eval, t_interp=t_interp)
                        np.testing.assert_allclose(
                            soln.y, soln_base.y, rtol=1e-5, atol=1e-4
                        )
                    else:
                        with pytest.raises(ValueError):
                            soln = solver.solve(model, t_eval, t_interp=t_interp)

    def test_solver_options(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = [0, 1]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        solver = pybamm.IDAKLUSolver()
        soln_base = solver.solve(model, t_eval, t_interp=t_interp)

        options_success = {
            "max_order_bdf": 4,
            "max_num_steps": 490,
            "dt_init": 0.01,
            "dt_min": 1e-6,
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
            soln = solver.solve(model, t_eval, t_interp=t_interp)

            np.testing.assert_allclose(soln.y, soln_base.y, rtol=1e-5, atol=1e-4)

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

            with pytest.raises(ValueError):
                solver.solve(model, t_eval)

    def test_with_output_variables(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence
        input_parameters = {}  # Sensitivities dictionary
        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)

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
            t_interp=t_interp,
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
            t_interp=t_interp,
        )

        # Compare output to sol_all
        for varname in [*output_variables, *model_vars]:
            np.testing.assert_allclose(
                sol[varname](t_eval), sol_all[varname](t_eval), rtol=1e-4, atol=1e-3
            )

        # Check that the missing variables are not available in the solution
        for varname in inaccessible_vars:
            with pytest.raises(KeyError):
                sol[varname].data

        # Check Solution is marked
        assert sol.variables_returned is True

        # Mock a 1D current collector and initialise (none in the model)
        sol["x_s [m]"].domain = ["current collector"]
        sol["x_s [m]"].entries

    def test_with_output_variables_and_sensitivities(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence

        for form in ["casadi", "iree"]:
            if (form == "iree") and (not pybamm.has_jax() or not pybamm.has_iree()):
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

            t_eval = [0, 100]
            t_interp = np.linspace(t_eval[0], t_eval[-1], 5)

            options = {
                "linear_solver": "SUNLinSol_KLU",
                "jacobian": "sparse",
                "num_threads": 4,
                "max_num_steps": 1000,
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
                t_interp=t_interp,
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
                t_interp=t_interp,
            )

            # Compare output to sol_all
            tol = 1e-5 if form != "iree" else 1e-2  # iree has reduced precision
            for varname in output_variables:
                np.testing.assert_allclose(
                    sol[varname](t_interp),
                    sol_all[varname](t_interp),
                    rtol=tol,
                    atol=tol,
                    err_msg=f"Failed for {varname} with form {form}",
                )

            # Mock a 1D current collector and initialise (none in the model)
            sol["x_s [m]"].domain = ["current collector"]
            sol["x_s [m]"].entries

    def test_bad_jax_evaluator(self):
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "jax"
        with pytest.raises(pybamm.SolverError):
            pybamm.IDAKLUSolver(options={"jax_evaluator": "bad_evaluator"})

    def test_bad_jax_evaluator_output_variables(self):
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "jax"
        with pytest.raises(pybamm.SolverError):
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
        sol = sim.solve(np.linspace(0, 3600, 2))
        assert sol.termination == "event: Minimum voltage [V]"

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
        sol3 = sim3.solve(np.linspace(0, 3600, 2))
        assert sol3.termination == "event: Minimum voltage [V]"

    def test_simulation_period(self):
        model = pybamm.lithium_ion.DFN()
        parameter_values = pybamm.ParameterValues("Chen2020")
        solver = pybamm.IDAKLUSolver()

        experiment = pybamm.Experiment(
            ["Charge at C/10 for 10 seconds"], period="0.1 seconds"
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
            solver=solver,
        )
        sol = sim.solve()

        np.testing.assert_allclose(sol.t, np.arange(0, 10.1, 0.1), rtol=1e-5, atol=1e-5)

    def test_interpolate_time_step_start_offset(self):
        model = pybamm.lithium_ion.SPM()

        def experiment_setup(period=None):
            return pybamm.Experiment(
                [
                    "Discharge at C/10 for 10 seconds",
                    "Charge at C/10 for 10 seconds",
                ],
                period=period,
            )

        experiment_1s = experiment_setup(period="1 seconds")
        solver = pybamm.IDAKLUSolver()
        sim_1s = pybamm.Simulation(model, experiment=experiment_1s, solver=solver)
        sol_1s = sim_1s.solve()
        np.testing.assert_equal(
            np.nextafter(sol_1s.sub_solutions[0].t[-1], np.inf),
            sol_1s.sub_solutions[1].t[0],
        )

        assert not sol_1s.hermite_interpolation

        experiment = experiment_setup(period=None)
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve(model)

        assert sol.hermite_interpolation

        rtol = solver.rtol
        atol = solver.atol
        np.testing.assert_allclose(
            sol_1s["Voltage [V]"].data,
            sol["Voltage [V]"](sol_1s.t),
            rtol=rtol,
            atol=atol,
        )

    def test_python_idaklu_deprecation_errors(self):
        for form in ["python", "", "jax"]:
            if form == "jax" and not pybamm.has_jax():
                continue

            model = pybamm.BaseModel()
            model.convert_to_format = form
            u = pybamm.Variable("u")
            v = pybamm.Variable("v")
            model.rhs = {u: 0.1 * v}
            model.algebraic = {v: 1 - v}
            model.initial_conditions = {u: 0, v: 1}
            model.events = [pybamm.Event("1", 0.2 - u), pybamm.Event("2", v)]

            disc = pybamm.Discretisation()
            disc.process_model(model)

            t_eval = [0, 3]

            solver = pybamm.IDAKLUSolver(
                root_method="lm",
            )

            if form == "python":
                with pytest.raises(
                    pybamm.SolverError,
                    match="Unsupported option for convert_to_format=python",
                ):
                    with pytest.raises(
                        DeprecationWarning,
                        match="The python-idaklu solver has been deprecated.",
                    ):
                        _ = solver.solve(model, t_eval)
            elif form == "jax":
                with pytest.raises(
                    pybamm.SolverError,
                    match="Unsupported evaluation engine for convert_to_format=jax",
                ):
                    _ = solver.solve(model, t_eval)

    def test_extrapolation_events_with_output_variables(self):
        # Make sure the extrapolation checks work with output variables
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        c = pybamm.Variable("c")
        model.variables = {"v": v, "c": c}
        model.rhs = {v: -1, c: 0}
        model.initial_conditions = {v: 1, c: 2}
        model.events.append(
            pybamm.Event(
                "Triggered event",
                v - 0.5,
                pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
            )
        )
        solver = pybamm.IDAKLUSolver(output_variables=["c"])
        solver.set_up(model)

        with pytest.warns(pybamm.SolverWarning, match="extrapolation occurred for"):
            solver.solve(model, t_eval=[0, 1])

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
        mass_matrix_inv = 0.1 * eye(mass_matrix.shape[0] // 2)
        model.mass_matrix_inv = pybamm.Matrix(mass_matrix_inv)

        assert not model.is_standard_form_dae

        # Solve
        solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        t_eval = [0, 1]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)
        solution = solver.solve(model, t_eval, t_interp=t_interp)
        np.testing.assert_array_equal(solution.t, t_interp)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))
        np.testing.assert_allclose(solution.y[-1], 2 * np.exp(0.1 * solution.t))

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

        # Test with on_extrapolation="error"
        solver = pybamm.IDAKLUSolver(on_extrapolation="error")
        t_eval = [0, 5]

        with pytest.raises(pybamm.SolverError, match="interpolation bounds"):
            solver.solve(model, t_eval)

        # Test with on_extrapolation="warn"
        solver = pybamm.IDAKLUSolver(on_extrapolation="warn")
        t_eval = [0, 5]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve(model, t_eval)
            assert len(w) > 0
            assert "extrapolation occurred" in str(w[0].message)

        # Test with on_extrapolation="ignore"
        solver = pybamm.IDAKLUSolver(on_extrapolation="ignore")
        t_eval = [0, 5]

        # Should not raise an error or warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve(model, t_eval)
            assert len(w) == 0
