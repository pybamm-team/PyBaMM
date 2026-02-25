import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import eye

import pybamm
from tests import get_discretisation_for_testing, no_internet_connection


def _hermite_wrms(sol_base, sol_reduced, atol, rtol) -> list[tuple[int, float]]:
    """
    Compute the integral L2 WRMS error between two Hermite-interpolated solutions
    using Gauss quadrature

    Parameters
    ----------
    sol_base : pybamm.Solution
    sol_reduced : pybamm.Solution
    atol : float
    rtol : float

    Returns
    -------
    list[tuple[int, float]]
        A list of tuples, each containing the segment index and the WRMS error
    """
    n_states = sol_base.all_ys[0].shape[0]
    atol_vec = np.full(n_states, atol)
    wrms_values = []

    def cubic_hermite_spline(sol):
        tb = np.asarray(sol.all_ts[0])
        yb = np.asarray(sol.all_ys[0])
        ypb = np.asarray(sol.all_yps[0])
        return CubicHermiteSpline(tb, yb.T, ypb.T)

    for seg in range(len(sol_base.all_ts)):
        tb = sol_base.all_ts[seg]
        tr = sol_reduced.all_ts[seg]

        if len(tb) < 2 or len(tr) < 2:
            continue
        sub = sol_base.sub_solutions[seg]
        itp_base = cubic_hermite_spline(sub)
        itp_red = cubic_hermite_spline(sol_reduced.sub_solutions[seg])

        t_span = tb[-1] - tb[0]

        def integrand(t, itp_base, itp_red, atol_vec, rtol):
            y_b = itp_base(t)
            y_r = itp_red(t)
            w = 1.0 / (atol_vec + rtol * np.abs(y_b))
            return (w * (y_b - y_r)) ** 2

        t_evals = np.asarray(sub.all_t_evals[0])
        points = t_evals[(t_evals > tb[0]) & (t_evals < tb[-1])]

        integral, _ = quad_vec(
            integrand,
            tb[0],
            tb[-1],
            points=points,
            args=(itp_base, itp_red, atol_vec, rtol),
        )
        wrms = np.sqrt(np.mean(integral) / t_span)
        wrms_values.append((seg, wrms))

    return wrms_values


class TestIDAKLUSolver:
    def test_ida_roberts_klu(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.events = [pybamm.Event("1", 0.2 - u), pybamm.Event("2", v)]

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

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
            for inputs, solution in zip(inputs_list, solutions, strict=False):
                print("checking solution", inputs, solution.all_inputs)
                np.testing.assert_array_equal(solution.t, t_interp)
                np.testing.assert_allclose(
                    solution.y[0],
                    2 * np.exp(-inputs["rate"] * solution.t),
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_model_events(self):
        # Create model
        model = pybamm.BaseModel()
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
        )

        t_eval = [0, 1]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 100)

        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        np.testing.assert_array_equal(
            solution.t,
            t_interp,
        )
        np.testing.assert_allclose(
            solution.y[0],
            np.exp(0.1 * solution.t),
            rtol=1e-6,
            atol=1e-5,
        )

        # Check invalid atol type raises an error
        with pytest.raises(pybamm.SolverError):
            solver._check_atol_type({"key": "value"}, model)

        # enforce events that won't be triggered
        model.events = [pybamm.Event("an event", var + 1)]
        model_disc = disc.process_model(model, inplace=False)
        solver = pybamm.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
        )
        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        np.testing.assert_array_equal(solution.t, t_interp)
        np.testing.assert_allclose(
            solution.y[0],
            np.exp(0.1 * solution.t),
            rtol=1e-6,
            atol=1e-5,
        )

        # enforce events that will be triggered
        model.events = [pybamm.Event("an event", 1.01 - var)]
        model_disc = disc.process_model(model, inplace=False)
        solver = pybamm.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
        )
        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        assert len(solution.t) < len(t_interp)
        np.testing.assert_allclose(
            solution.y[0],
            np.exp(0.1 * solution.t),
            rtol=1e-6,
            atol=1e-5,
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
        )
        np.testing.assert_allclose(
            solution.y[-1],
            2 * np.exp(0.1 * solution.t),
            rtol=1e-6,
            atol=1e-5,
        )

    def test_input_params(self):
        # test a mix of scalar and vector input params
        model = pybamm.BaseModel()
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

        solver = pybamm.IDAKLUSolver()

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
            rtol=1e-7,
            atol=1e-6,
        )

        # test that y[0] = to true solution
        true_solution = a_value * sol.t
        np.testing.assert_allclose(
            sol.y[0],
            true_solution,
            rtol=1e-7,
            atol=1e-6,
        )

        # test that y[1:3] = to true solution
        true_solution = b_value * sol.t
        np.testing.assert_allclose(
            sol.y[1:3],
            true_solution,
            rtol=1e-7,
            atol=1e-6,
        )

    def test_sensitivities_initial_condition(self):
        for output_variables in [[], ["2v"]]:
            model = pybamm.BaseModel()
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
                output_variables=output_variables,
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
            )

    def test_ida_roberts_klu_sensitivities(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        a = pybamm.InputParameter("a")
        model.rhs = {u: a * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.variables = {"2u": 2 * u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

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
            rtol=1e-7,
            atol=1e-6,
        )

        # test that y[0] = to true solution
        true_solution = a_value * sol.t
        np.testing.assert_allclose(
            sol.y[0, :],
            true_solution,
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
            rtol=1e-7,
            atol=1e-6,
        )

        # test that y[0] = to true solution
        true_solution = a_value * sol.t
        np.testing.assert_allclose(
            sol.y[0, :],
            true_solution,
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

        np.testing.assert_allclose(
            dyda_ida,
            dyda_fd,
            rtol=1e-7,
            atol=1e-6,
        )

        # get the sensitivities for the variable
        d2uda = sol["2u"].sensitivities["a"]
        np.testing.assert_allclose(
            2 * dyda_ida[0:200:2].flatten(),
            d2uda,
            rtol=1e-7,
            atol=1e-6,
        )

    def test_ida_roberts_consistent_initialization(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 2}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        # Set up and  model consistently initialize the model
        solver.set_up(model)
        t0 = 0.0
        solver._set_consistent_initialization(model, t0, inputs_list=[{}])

        # u(t0) = 0, v(t0) = 1
        np.testing.assert_allclose(
            model.y0full[0],
            [0, 1],
            rtol=1e-7,
            atol=1e-6,
        )
        # u'(t0) = 0.1 * v(t0) = 0.1
        # Since v is algebraic, the initial derivative is set to 0
        np.testing.assert_allclose(
            model.ydot0full[0],
            [0.1, 0],
            rtol=1e-7,
            atol=1e-6,
        )

    def test_sensitivities_with_events(self):
        # this test implements a python version of the ida Roberts
        # example provided in sundials
        # see sundials ida examples pdf
        model = pybamm.BaseModel()
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

        solver = pybamm.IDAKLUSolver()

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
            rtol=1e-7,
            atol=1e-6,
        )

        # test that y[0] = to true solution
        true_solution = a_value * sol.t
        np.testing.assert_allclose(
            sol.y[0, :],
            true_solution,
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

        np.testing.assert_allclose(
            dyda_ida[: (2 * max_index), :],
            dyda_fd,
            rtol=1e-7,
            atol=1e-6,
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
            rtol=1e-7,
            atol=1e-6,
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
        with pytest.raises(pybamm.SolverError, match=r"KLU requires the Jacobian"):
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
            pybamm.SolverError, match=r"t_eval must increase monotonically"
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
        with pytest.raises(pybamm.SolverError):
            solver.solve(model, t_eval)

    def test_dae_solver_algebraic_model(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
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
            "hermite_reduction_factor": 1.1,
        }

        # test everything works
        for option in options_success:
            options = {option: options_success[option]}
            solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, options=options)
            soln = solver.solve(model, t_eval, t_interp=t_interp)

            # Asserts
            assert all(v == solver.options[k] for k, v in options.items())
            np.testing.assert_allclose(soln.y, soln_base.y, rtol=1e-5, atol=1e-4)

        options_fail = {
            "max_order_bdf": -1,
            "max_num_steps_ic": -1,
            "max_num_jacobians_ic": -1,
            "max_num_iterations_ic": -1,
            "max_linesearch_backtracks_ic": -1,
            "epsilon_linear_tolerance": -1.0,
            "increment_factor": -1.0,
            "hermite_reduction_factor": -1.0,
        }

        # test that the solver throws a warning
        for option in options_fail:
            options = {option: options_fail[option]}
            solver = pybamm.IDAKLUSolver(options=options)

            with pytest.raises((pybamm.SolverError, ValueError)):
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

    def test_with_sparse_output_variables_and_sensitivities(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence
        input_parameters = {  # Sensitivities dictionary
            "Current function [A]": 0.222,
            "Separator porosity": 0.3,
        }

        # construct model
        solver = pybamm.IDAKLUSolver(
            output_variables=["Negative particle flux [mol.m-2.s-1]"],
        )
        model = pybamm.lithium_ion.DFN()
        params = model.default_parameter_values
        params.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
        with pytest.raises(
            pybamm.SolverError,
            match=r"Sensitivity of sparse variables not supported",
        ):
            sim.solve([0, 100], inputs=input_parameters, calculate_sensitivities=True)

    def test_with_output_variables_and_sensitivities(self):
        # Construct a model and solve for all variables, then test
        # the 'output_variables' option for each variable in turn, confirming
        # equivalence
        input_parameters = {  # Sensitivities dictionary
            "Current function [A]": 0.222,
            "Separator porosity": 0.3,
        }

        # construct model
        model = pybamm.lithium_ion.DFN()
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

        # Use a selection of variables of different types
        output_variables = [
            "Voltage [V]",  # 0D
            "x [m]",  # 1D, empty sensitivities
            "Negative electrode potential [V]",  # 1D
            "Negative particle concentration [mol.m-3]",  # 2D
            "Throughput capacity [A.h]",  # ExplicitTimeIntegral
        ]

        # Use the full model as comparison (tested separately)
        solver_all = pybamm.IDAKLUSolver(
            atol=1e-8,
            rtol=1e-8,
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
            atol=1e-8,
            rtol=1e-8,
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
        tol = 1e-5
        for varname in output_variables:
            np.testing.assert_allclose(
                sol[varname](t_interp),
                sol_all[varname](t_interp),
                rtol=tol,
                atol=tol,
            )

            # Test `all` key shape
            assert (
                sol[varname].sensitivities["all"].shape
                == sol_all[varname].sensitivities["all"].shape
            )

        # test each of the sensitivity calculations match
        for varname in output_variables:
            for key in input_parameters:
                np.testing.assert_allclose(
                    sol[varname].sensitivities[key],
                    sol_all[varname].sensitivities[key],
                    rtol=tol,
                    atol=tol,
                    err_msg=f"Failed for '{varname}', sensitivity '{key}'",
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
        model = pybamm.BaseModel()
        model.convert_to_format = "python"
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

        with pytest.raises(
            pybamm.SolverError,
            match=r"Unsupported option for convert_to_format=python",
        ):
            with pytest.raises(
                DeprecationWarning,
                match=r"The python-idaklu solver has been deprecated.",
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

        with pytest.warns(pybamm.SolverWarning, match=r"extrapolation occurred for"):
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

    def test_multiple_initial_conditions_single_variable(self):
        model = pybamm.BaseModel()
        model.convert_to_format = None
        u = pybamm.Variable("u")
        u0 = pybamm.InputParameter("u0")
        model.rhs = {u: -u}
        model.initial_conditions = {u: u0}
        model.variables = {"u": u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver(options={"num_threads": 1})

        n_sims = 3
        initial_condition_inputs = [{"u0": i + 1} for i in range(n_sims)]
        t_eval = np.array([0, 1])
        t_interp = np.linspace(0, 1, 10)

        solutions = solver.solve(
            model,
            t_eval,
            inputs=initial_condition_inputs,
            t_interp=t_interp,
        )

        assert len(solutions) == n_sims
        for i, solution in enumerate(solutions):
            expected_initial_value = i + 1
            np.testing.assert_allclose(solution["u"](0), expected_initial_value)
            np.testing.assert_allclose(
                solution["u"](t_eval),
                expected_initial_value * np.exp(-t_eval),
                rtol=1e-3,
                atol=1e-5,
            )

    def test_single_initial_condition_single_variable(self):
        model = pybamm.BaseModel()
        model.convert_to_format = "casadi"
        u = pybamm.Variable("u")
        u0 = pybamm.InputParameter("u0")
        model.rhs = {u: -u}
        model.initial_conditions = {u: u0}
        model.variables = {"u": u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()

        initial_condition_input = {"u0": 5}
        t_eval = np.array([0, 1])
        t_interp = np.linspace(0, 1, 10)

        solution = solver.solve(
            model, t_eval, inputs=initial_condition_input, t_interp=t_interp
        )

        np.testing.assert_allclose(solution["u"](0), 5)
        np.testing.assert_allclose(
            solution["u"](t_eval), 5 * np.exp(-t_eval), rtol=1e-3, atol=1e-5
        )

    def test_multiple_initial_conditions_multiple_variables(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        u0 = pybamm.InputParameter("u0")
        v0 = pybamm.InputParameter("v0")
        model.rhs = {u: -u, v: -2 * v}
        model.initial_conditions = {u: u0, v: v0}
        model.variables = {"u": u, "v": v}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Use default solver tolerances
        solver = pybamm.IDAKLUSolver()

        initial_conditions = [{"u0": 3, "v0": 4}, {"u0": 5, "v0": 6}]

        t_eval = np.array([0, 1])
        t_interp = np.linspace(0, 1, 10)

        solutions = solver.solve(
            model,
            t_eval,
            inputs=initial_conditions,
            t_interp=t_interp,
        )

        assert len(solutions) == 2

        np.testing.assert_allclose(solutions[0]["u"](0), 3)
        np.testing.assert_allclose(solutions[0]["v"](0), 4)
        np.testing.assert_allclose(
            solutions[0]["u"](t_eval), 3 * np.exp(-t_eval), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            solutions[0]["v"](t_eval), 4 * np.exp(-2 * t_eval), rtol=1e-3, atol=1e-5
        )

        np.testing.assert_allclose(solutions[1]["u"](0), 5)
        np.testing.assert_allclose(solutions[1]["v"](0), 6)
        np.testing.assert_allclose(
            solutions[1]["u"](t_eval), 5 * np.exp(-t_eval), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            solutions[1]["v"](t_eval), 6 * np.exp(-2 * t_eval), rtol=1e-3, atol=1e-5
        )

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

        with pytest.raises(pybamm.SolverError, match=r"interpolation bounds"):
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

    def test_on_failure_option(self):
        input_parameters = {"Positive electrode active material volume fraction": 0.01}
        t_eval = [0, 100]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 10)

        model = pybamm.lithium_ion.DFN()
        model.events = []  # Requires events to be off
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({key: "[input]" for key in input_parameters})
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(
            mesh,
            model.default_spatial_methods,
            remove_independent_variables_from_rhs=True,
        )
        disc.process_model(model)

        # Test default "raise"
        solver = pybamm.IDAKLUSolver()
        with pytest.raises(pybamm.SolverError):
            solver.solve(
                model, t_eval=t_eval, t_interp=t_interp, inputs=input_parameters
            )

        # Test "ignore"
        solver = pybamm.IDAKLUSolver(on_failure="ignore")
        sol = solver.solve(
            model, t_eval=t_eval, t_interp=t_interp, inputs=input_parameters
        )
        assert sol.termination == "failure"

        # Test "warn"
        solver = pybamm.IDAKLUSolver(on_failure="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.solve(
                model, t_eval=t_eval, t_interp=t_interp, inputs=input_parameters
            )
            assert len(w) > 0
            assert "_FAIL" in str(w[0].message)

    def test_no_progress_early_termination(self):
        # SPM at rest
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.update({"Current function [A]": 0})

        t_eval = [0, 10000]

        options_successes = [
            # Case 1: feature disabled because num_steps_no_progress is default (0)
            # even if t_no_progress is huge
            {
                "t_no_progress": 1e10,
                "num_steps_no_progress": 0,
            },
            # Case 2: feature disabled because t_no_progress is default (0.0)
            # even if num_steps_no_progress is positive
            {
                "num_steps_no_progress": 5,
                "t_no_progress": 0.0,
            },
        ]

        for options in options_successes:
            solver = pybamm.IDAKLUSolver(on_failure="ignore", options=options)
            sim = pybamm.Simulation(
                model, parameter_values=parameter_values, solver=solver
            )
            sol = sim.solve(t_eval)
            assert sol.termination == "final time"

        ## Check failure
        options_failures = {
            "num_steps_no_progress": 5,
            "t_no_progress": 1e10,
        }
        solver = pybamm.IDAKLUSolver(on_failure="ignore", options=options_failures)
        sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)
        sol = sim.solve(t_eval)
        assert sol.termination == "failure"

        assert len(sol.t) == options_failures["num_steps_no_progress"]
        assert sol.t[-1] < options_failures["t_no_progress"]

    @pytest.mark.skipif(
        no_internet_connection(),
        reason="Network not available to download files from registry",
    )
    def test_drive_cycle_knot_reduction(self):
        """Test knot reduction with a drive cycle (many t_eval breakpoints).

        Verifies that:
          1. The reduced solution has fewer points than the baseline.
          2. All derivatives are finite (no NaN from LS solve).
          3. The Hermite spline error (integral L2 WRMS) stays below 1.0.
        """
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        data_loader = pybamm.DataLoader()
        drive_cycle = pd.read_csv(
            pybamm.get_parameters_filepath(data_loader.get_data("US06.csv")),
            comment="#",
            skip_blank_lines=True,
            header=None,
        ).to_numpy()
        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )
        param["Current function [A]"] = current_interpolant

        rtol = 1e-4
        atol = 1e-6
        hermite_reduction_factor = 2.0

        # Baseline: no knot reduction
        solver_base = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        sim_base = pybamm.Simulation(model, parameter_values=param, solver=solver_base)
        sol_base = sim_base.solve()

        # Reduced: with knot reduction (and optionally LS refinement)
        solver_red = pybamm.IDAKLUSolver(
            rtol=rtol,
            atol=atol,
            options={"hermite_reduction_factor": hermite_reduction_factor},
        )
        sim_red = pybamm.Simulation(model, parameter_values=param, solver=solver_red)
        sol_red = sim_red.solve()

        # 1. Fewer points
        n_base = sum(len(s) for s in sol_base.all_ts)
        n_red = sum(len(s) for s in sol_red.all_ts)
        assert n_red < n_base, (
            f"Knot reduction should reduce points: {n_red} >= {n_base}"
        )

        # 2. All derivatives must be finite (no NaN from LS)
        for seg in range(len(sol_red.all_ts)):
            yp = np.asarray(sol_red.all_yps[seg])
            assert np.all(np.isfinite(yp)), f"Non-finite derivatives in segment {seg}"

        # 3. Integral L2 WRMS error must be bounded
        for seg, wrms in _hermite_wrms(sol_base, sol_red, atol, rtol):
            assert wrms < 1.0, f"Segment {seg} integral L2 WRMS too large: {wrms:.4e}"

    def test_reduce_solution_errors(self):
        """Test that reduce_solution raises on invalid inputs."""
        model = pybamm.lithium_ion.SPM()
        solver_base = pybamm.IDAKLUSolver(rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, solver=solver_base)
        sol = sim.solve([0, 3600])

        # No Hermite data: disable all_yps
        sol_no_hermite = sol.copy()
        sol_no_hermite._all_yps = None
        with pytest.raises(pybamm.SolverError, match="Hermite interpolation data"):
            solver_base.reduce_solution(sol_no_hermite)

        # Solver had reduction active
        solver_active = pybamm.IDAKLUSolver(
            rtol=1e-4,
            atol=1e-6,
            options={"hermite_reduction_factor": 2.0},
        )
        with pytest.raises(pybamm.SolverError, match=r"hermite_reduction_factor = 1.0"):
            solver_active.reduce_solution(sol)

    def test_reduce_solution_basic(self):
        """Test basic post-hoc reduce_solution: fewer points, finite yps, bounded error."""
        model = pybamm.lithium_ion.SPM()
        rtol = 1e-4
        atol = 1e-6
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        sim = pybamm.Simulation(model, solver=solver)
        sol = sim.solve([0, 3600])

        reduced = solver.reduce_solution(sol, hermite_reduction_factor=2.0)

        # 1. Fewer points
        n_orig = sum(len(s) for s in sol.all_ts)
        n_red = sum(len(s) for s in reduced.all_ts)
        assert n_red < n_orig, (
            f"reduce_solution should reduce points: {n_red} >= {n_orig}"
        )

        # 2. All derivatives finite
        for seg in range(len(reduced.all_ts)):
            yp = np.asarray(reduced.all_yps[seg])
            assert np.all(np.isfinite(yp)), f"Non-finite derivatives in segment {seg}"

        # 3. Bounded WRMS error
        for seg, wrms in _hermite_wrms(sol, reduced, atol, rtol):
            assert wrms < 1.0, f"Segment {seg} integral L2 WRMS too large: {wrms:.4e}"

    def test_reduce_solution_metadata(self):
        """Test that reduce_solution preserves metadata from the original solution."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver(rtol=1e-4, atol=1e-6)
        sim = pybamm.Simulation(model, solver=solver)
        sol = sim.solve([0, 3600])

        reduced = solver.reduce_solution(sol, hermite_reduction_factor=2.0)

        assert reduced.termination == sol.termination
        assert reduced.all_inputs == sol.all_inputs
        assert len(reduced.all_models) == len(sol.all_models)
        for rm, sm in zip(reduced.all_models, sol.all_models, strict=True):
            assert rm is sm
        if sol.t_event is not None:
            np.testing.assert_array_equal(reduced.t_event, sol.t_event)
        if sol.y_event is not None:
            np.testing.assert_array_equal(reduced.y_event, sol.y_event)
        # all_t_evals preserved
        assert len(reduced.all_t_evals) == len(sol.all_t_evals)
        for rte, ste in zip(reduced.all_t_evals, sol.all_t_evals, strict=True):
            np.testing.assert_array_equal(rte, ste)

    def test_reduce_solution_vs_online(self):
        """Compare post-hoc reduce_solution with online knot reduction on a drive cycle.

        Verifies that:
          1. Post-hoc reduction produces similar point counts to online reduction.
          2. Both have finite derivatives.
          3. Both have bounded WRMS error vs the uncompressed baseline.
        """
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values

        time = np.arange(100)
        np.random.seed(0)
        current = 1 + 0.1 * np.random.rand(time.size)
        current_interpolant = pybamm.Interpolant(time, current, pybamm.t)
        param["Current function [A]"] = current_interpolant

        rtol = 1e-4
        atol = 1e-6
        hermite_reduction_factor = 2.0

        # Baseline: no reduction
        solver_base = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        sim_base = pybamm.Simulation(model, parameter_values=param, solver=solver_base)
        sol_base = sim_base.solve()

        # Online reduction
        solver_online = pybamm.IDAKLUSolver(
            rtol=rtol,
            atol=atol,
            options={"hermite_reduction_factor": hermite_reduction_factor},
        )
        sim_online = pybamm.Simulation(
            model, parameter_values=param, solver=solver_online
        )
        sol_online = sim_online.solve()

        # Post-hoc reduction
        sol_posthoc = solver_base.reduce_solution(
            sol_base, hermite_reduction_factor=hermite_reduction_factor
        )

        n_base = sum(len(s) for s in sol_base.all_ts)
        n_online = sum(len(s) for s in sol_online.all_ts)
        n_posthoc = sum(len(s) for s in sol_posthoc.all_ts)

        # Point counts should be equal
        assert n_posthoc == n_online

        # Time arrays should be equal
        np.testing.assert_array_equal(sol_posthoc.t, sol_online.t)

        # Both should reduce points
        assert n_online < n_base

        sols = {
            "online": sol_online,
            "posthoc": sol_posthoc,
        }

        for label, sol_r in sols.items():
            # Both must have finite derivatives
            for seg in range(len(sol_r.all_ts)):
                yp = np.asarray(sol_r.all_yps[seg])
                assert np.all(np.isfinite(yp)), (
                    f"{label}: non-finite derivatives in segment {seg}"
                )

            # WRMS error bounded for both
            for seg, wrms in _hermite_wrms(sol_base, sol_r, atol, rtol):
                assert wrms < 1.0, (
                    f"{label} segment {seg} integral L2 WRMS too large: {wrms:.4e}"
                )
