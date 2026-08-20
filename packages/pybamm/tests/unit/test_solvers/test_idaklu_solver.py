import io
import itertools
import logging
import sys
import warnings
from contextlib import redirect_stdout

import casadi
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import CubicHermiteSpline

import pybamm
from pybamm.solvers.observation import (
    NativeInterpolatingObservation,
    NativeObservation,
)
from pybamm.solvers.variable_observer import NativeObserver
from tests import (
    get_broken_input_model,
    get_discretisation_for_testing,
    no_internet_connection,
)


def _rust_decay_model(with_input=True):
    """``dvar/dt = -rate*var`` (or ``-var``), lowered to the Rust backend."""
    model = pybamm.BaseModel()
    var = pybamm.Variable("var")
    rate = pybamm.InputParameter("rate") if with_input else 1
    model.rhs = {var: -rate * var}
    model.initial_conditions = {var: 2 if with_input else 1}
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


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


@pytest.fixture
def decay_model():
    """Discretised ``du/dt = -a u``, with ``a`` as the only input parameter."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    model.rhs = {u: -pybamm.InputParameter("a") * u}
    model.initial_conditions = {u: 1}
    model.variables = {"u": u}
    pybamm.Discretisation().process_model(model)
    return model


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

    def test_multiple_inputs_rust_solves_in_parallel(self):
        # One evaluator per solver over a shared tape, so a parallel solve must
        # match per-input solves with no cross-input interference.
        model = _rust_decay_model()

        t_eval = [0, 1]
        t_interp = np.linspace(t_eval[0], t_eval[-1], 10)
        inputs_list = [{"rate": 0.01 * (i + 1)} for i in range(4)]

        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10, options={"num_threads": 4})
        batched_solutions = solver.solve(
            model, t_eval, inputs=inputs_list, t_interp=t_interp
        )

        sequential_solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10)
        for inputs, batched in zip(inputs_list, batched_solutions, strict=True):
            sequential = sequential_solver.solve(
                model, t_eval, inputs=inputs, t_interp=t_interp
            )
            np.testing.assert_array_equal(batched.t, sequential.t)
            np.testing.assert_allclose(
                batched.y[0], sequential.y[0], rtol=1e-8, atol=1e-10
            )

    @pytest.mark.parametrize(
        ("num_threads", "num_solvers"), [(4, None), (4, 1), (4, 2), (1, 1)]
    )
    def test_rust_runs_one_solver_per_thread(self, num_threads, num_solvers):
        # Equality is what makes SetupOptions derive one thread per solver, so no
        # solver is given the OpenMP N_Vectors that were 28x slower on DFN x32.
        model = _rust_decay_model()

        options = {"num_threads": num_threads}
        if num_solvers is not None:
            options["num_solvers"] = num_solvers
        solver = pybamm.IDAKLUSolver(rtol=1e-5, atol=1e-5, options=options)
        solver.solve(model, [0, 1], inputs=[{"rate": 0.01 * (i + 1)} for i in range(4)])

        assert solver._options["num_threads"] == num_threads
        assert solver._options["num_solvers"] == num_threads
        assert len(solver._setup["rust_evaluators"]) == num_threads

    def test_rust_evaluator_pool_hands_out_distinct_evaluators(self):
        solver = pybamm.IDAKLUSolver(options={"num_threads": 3})
        solver.set_up(_rust_decay_model(with_input=False))
        pool = solver._setup["rust_evaluators"]

        assert len(pool) == 3
        # set_up's solver group consumed every handout: an address is given out
        # once, so nothing else can be handed a solver's evaluator.
        with pytest.raises(RuntimeError, match=r"already handed to a solver"):
            pool.as_ptr(0)

        fresh = solver._setup["rust_model"].evaluator_pool(3)
        assert len({fresh.as_ptr(i) for i in range(3)}) == 3
        with pytest.raises(IndexError, match=r"out of range for a pool of 3"):
            fresh.as_ptr(3)
        with pytest.raises(RuntimeError, match=r"already handed to a solver"):
            fresh.as_ptr(1)

    @pytest.mark.parametrize("convert_to_format", ["casadi", "rust"])
    def test_a_failing_input_set_is_named(self, convert_to_format):
        model = get_broken_input_model(convert_to_format)
        solver = pybamm.IDAKLUSolver(options={"num_threads": 4})
        inputs_list = [{"k": k} for k in (1.0, 2.0, -1.0, 3.0, 4.0)]
        with pytest.raises(pybamm.SolverError, match=r"input set 2:"):
            solver.solve(model, np.linspace(0, 1, 10), inputs=inputs_list)

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

    def test_vector_input_parameter_rust(self):
        # rust `check_p` must compare packed-input length against total input
        # *width*, not input count, or vector inputs are misrejected.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", expected_size=2)
        model.rhs = {u: -a * u * (pybamm.Index(b, 0) + pybamm.Index(b, 1))}
        model.initial_conditions = {u: 1}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        solver = pybamm.IDAKLUSolver()
        inputs = {"a": 0.5, "b": np.array([[0.2], [0.3]])}
        sol = solver.solve(model, np.linspace(0, 1, 10), inputs=inputs)
        np.testing.assert_allclose(
            sol["u"].data, np.exp(-0.5 * 0.5 * sol.t), rtol=1e-3, atol=1e-5
        )

    def test_calculate_sensitivities_rejects_vector_width_input_parameter(self):
        # Rust JVP/tangent seeds one scalar direction per named parameter; a
        # width>1 parameter would silently under-seed, so this must raise instead.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        b = pybamm.InputParameter("b", expected_size=2)
        model.rhs = {u: -(pybamm.Index(b, 0) + pybamm.Index(b, 1)) * u}
        model.initial_conditions = {u: 1}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        solver = pybamm.IDAKLUSolver()
        with pytest.raises(
            pybamm.SolverError, match=r"vector-width input parameters.*'b'"
        ):
            solver.solve(
                model,
                np.linspace(0, 1, 10),
                inputs={"b": np.array([0.2, 0.3])},
                calculate_sensitivities=["b"],
            )

    def test_scalar_sensitivities_with_wide_input_registered_first(self):
        # Guard only rejects sensitivities FOR a wide parameter; a scalar request
        # with an earlier-registered width-2 input must still seed correctly.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a", expected_size=2)
        c = pybamm.InputParameter("c")
        model.rhs = {u: -c * (pybamm.Index(a, 0) + pybamm.Index(a, 1)) * u}
        model.initial_conditions = {u: 1}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        solver = pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-10)
        sol = solver.solve(
            model,
            np.linspace(0, 1, 11),
            inputs={"a": np.array([0.4, 0.6]), "c": 0.5},
            calculate_sensitivities=["c"],
        )
        # u = exp(-c*(a0+a1)*t) with a0+a1 = 1: du/dc = -t*exp(-0.5*t)
        analytic = -sol.t * np.exp(-0.5 * sol.t)
        np.testing.assert_allclose(
            np.asarray(sol["u"].sensitivities["c"]).ravel(),
            analytic,
            rtol=1e-5,
            atol=1e-8,
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

        # Solve a short interval -- consistent IC is computed in C++
        # by the Newton solver and IDACalcIC during solve()
        t_eval = np.linspace(0, 1, 10)
        sol = solver.solve(model, t_eval)

        # u(t0) = 0, v(t0) = 1 (corrected from v=2 by Newton IC solver)
        np.testing.assert_allclose(
            sol.y[:, 0],
            [0, 1],
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

    def test_rust_klu_requires_jacobian(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -u}
        model.initial_conditions = {u: 1}
        model.use_jacobian = False
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)
        with pytest.raises(pybamm.SolverError, match=r"KLU requires the Jacobian"):
            pybamm.IDAKLUSolver().solve(model, np.array([0.0, 1.0]))

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

    @pytest.mark.parametrize(
        ("num_threads", "n_inputs", "all_on_calling_thread"),
        [
            # One solver, so every solve runs on the GIL-holding thread
            (1, 2, True),
            # Fewer input sets than solvers, so all of them land in the
            # serial remainder loop, which also runs on that thread
            (4, 2, True),
            # Two solves per solver, so a worker thread must buffer its share
            (2, 4, False),
        ],
    )
    def test_diagnostics_emitted_once_per_input_set(
        self, decay_model, num_threads, n_inputs, all_on_calling_thread, caplog, capsys
    ):
        t_eval = np.linspace(0, 1, 3)
        inputs = [{"a": 1.0 + i} for i in range(n_inputs)]
        solver = pybamm.IDAKLUSolver(
            options={"print_stats": True, "num_threads": num_threads}
        )

        # Send the log to stdout too, so it can be ordered against py::print
        handler = logging.StreamHandler(sys.stdout)
        pybamm.logger.addHandler(handler)
        try:
            with caplog.at_level(logging.DEBUG, logger=pybamm.logger.name):
                solver.solve(decay_model, t_eval, t_interp=t_eval, inputs=inputs)
        finally:
            pybamm.logger.removeHandler(handler)

        lines = capsys.readouterr().out.splitlines()
        starts = [i for i, line in enumerate(lines) if line.startswith("Integrating")]
        stats = [i for i, line in enumerate(lines) if line.startswith("Solver Stats:")]
        assert len(starts) == len(stats) == n_inputs
        # Values are printed through py::print, so the tab prefix is preserved
        assert "\tNumber of steps =" in "\n".join(lines)
        if all_on_calling_thread:
            # Buffering would emit every trace before the first statistics block
            assert stats[0] < starts[1]

    def test_debug_log_emitted_when_solve_fails(self, caplog):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        model.rhs = {u: a * u**2}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)

        # Two solves per solver, so one solver buffers on a worker thread and
        # only the flush at the end of the parallel region can emit its trace
        solver = pybamm.IDAKLUSolver(options={"num_threads": 2})
        inputs = [{"a": 1.0 + i} for i in range(4)]
        # u' = a u^2 blows up at t = 1/a, so integrating to t = 5 fails
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(pybamm.SolverError, match="IDA_ERR_FAIL"),
        ):
            solver.solve(model, np.array([0.0, 5.0]), inputs=inputs)

        # A partial solution is still returned, so every solve runs and the two
        # traces beyond the calling thread's own prove its buffer was drained
        starts = [m for m in caplog.messages if m.startswith("Integrating from t =")]
        assert len(starts) == len(inputs)
        assert any(m.startswith("Step ") for m in caplog.messages)

    def test_debug_log_flushed_when_solve_raises(self, caplog):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        # The residual is NaN from t = 0, so the C++ solve throws instead of
        # returning a partial solution, taking the rethrow path out of the group
        model.rhs = {u: a * pybamm.sqrt(-u)}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)
        # The worker-thread flush under test is OpenMP-only; rust clamps to one
        # solver, so every trace would come from the calling thread instead.
        model.convert_to_format = "casadi"

        # Two solves per solver, so one solver buffers on a worker thread
        solver = pybamm.IDAKLUSolver(options={"num_threads": 2})
        inputs = [{"a": 1.0 + i} for i in range(4)]
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(pybamm.SolverError),
        ):
            solver.solve(model, np.array([0.0, 5.0]), inputs=inputs)

        # Every set is attempted, and only the calling thread's traces stream
        # directly, so the rest can only come from the pre-rethrow flush
        starts = [m for m in caplog.messages if m.startswith("Integrating from t =")]
        assert len(starts) == 4

    def test_solve_interrupted_from_debug_logger(self, caplog, monkeypatch):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        _debug_logger = pybamm.logger.debug

        # applies a ctrl-C to keyboard interrupt on the first step of the simulation
        def logger_interrupts_on_first_step(msg, *args, **kwargs):
            _debug_logger(msg, *args, **kwargs)
            if isinstance(msg, str) and msg.startswith("Step "):
                raise KeyboardInterrupt

        monkeypatch.setattr(pybamm.logger, "debug", logger_interrupts_on_first_step)
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(KeyboardInterrupt),
        ):
            sim.solve([0, 3600])

        assert not any(m.startswith("Integration complete") for m in caplog.messages)

    def test_setup_options(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.linspace(0, 1, 3)
        t_interp = t_eval
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

        jacobians = ["none", "dense", "sparse", "matrix-free", "garbage"]
        linear_solvers = [
            "SUNLinSol_SPBCGS",
            "SUNLinSol_Dense",
            "SUNLinSol_KLU",
            "SUNLinSol_SPFGMR",
            "SUNLinSol_SPGMR",
            "SUNLinSol_SPTFQMR",
            "garbage",
        ]
        preconditions = ["none", "BBDP"]

        # Test jacobian/linear_solver/preconditioner combinations
        for jacobian, linear_solver, precon in itertools.product(
            jacobians, linear_solvers, preconditions
        ):
            options = {
                "jacobian": jacobian,
                "linear_solver": linear_solver,
                "preconditioner": precon,
            }
            solver = pybamm.IDAKLUSolver(
                atol=1e-8,
                rtol=1e-8,
                options=options,
            )
            works = (
                (jacobian == "none" and (linear_solver == "SUNLinSol_Dense"))
                or (jacobian == "dense" and (linear_solver == "SUNLinSol_Dense"))
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
                np.testing.assert_allclose(soln.y, soln_base.y, rtol=1e-5, atol=1e-4)
            else:
                with pytest.raises(ValueError):
                    _ = solver.solve(model, t_eval, t_interp=t_interp)

    def test_rust_dense_jacobian_matches_sparse(self):
        # rhs depends only on u, leaving a structural zero (row 0, col v) that
        # exercises the dense scatter path; nonzero ICs force real Newton iterations.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -0.1 * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        model.convert_to_format = "rust"
        disc = pybamm.Discretisation()
        disc.process_model(model)

        t_eval = np.array([0.0, 1.0])
        t_interp = np.linspace(0, 1, 100)
        base = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8).solve(
            model, t_eval, t_interp=t_interp
        )
        for jacobian, linsol in [
            ("dense", "SUNLinSol_Dense"),
            ("none", "SUNLinSol_Dense"),
        ]:
            options = {
                "jacobian": jacobian,
                "linear_solver": linsol,
                "preconditioner": "none",
                "max_num_steps": 10_000,
            }
            solver = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8, options=options)
            soln = solver.solve(model, t_eval, t_interp=t_interp)
            np.testing.assert_allclose(soln.y, base.y, rtol=1e-5, atol=1e-4)

    def test_rust_pure_algebraic_jacobian_diagonal(self):
        # Pure-algebraic model (empty rhs): the empty child's zero-derivative must
        # not widen to length 1, which would shift the diagonal and zero the Jacobian.
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var + 1}
        model.initial_conditions = {var: 0}
        model.convert_to_format = "rust"

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(model, [0, 1])
        np.testing.assert_array_equal(solution.y, -1)

    def test_an_unknown_option_is_rejected(self):
        # Merging a misspelt key through leaves the caller on the default.
        with pytest.raises(pybamm.SolverError, match=r"Unknown IDAKLU solver option"):
            pybamm.IDAKLUSolver(options={"num_thread": 4})

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
        for option, value in options_success.items():
            options = {option: value}
            solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, options=options)
            soln = solver.solve(model, t_eval)
            # Hermite upsample y
            itp = CubicHermiteSpline(soln.t, soln.y, soln.yp, axis=1)
            y_upsampled = itp(t_interp)

            # Asserts
            assert all(v == solver.options[k] for k, v in options.items())
            np.testing.assert_allclose(y_upsampled, soln_base.y, rtol=1e-5, atol=1e-4)

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
        for option, value in options_fail.items():
            options = {option: value}
            with pytest.raises(pybamm.SolverError):
                solver = pybamm.IDAKLUSolver(options=options)
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
        # The nnz-compression this guards against only occurs on the casadi path;
        # the rust equivalent is test_sparse_output_variable_sensitivities_rust_matches_casadi.
        model.convert_to_format = "casadi"
        params = model.default_parameter_values
        params.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
        with pytest.raises(
            pybamm.SolverError,
            match=r"Sensitivity of sparse variables not supported",
        ):
            sim.solve([0, 100], inputs=input_parameters, calculate_sensitivities=True)

    def test_sparse_output_variable_sensitivities_rust_matches_casadi(self):
        # Rust output lengths are always dense (== prod(shape)), so the casadi
        # nnz-compression bug behind the guard above is unreachable on rust.
        input_parameters = {
            "Current function [A]": 0.222,
            "Separator porosity": 0.3,
        }
        var_name = "Negative particle flux [mol.m-2.s-1]"

        # Reference: full-state casadi solve bypasses the output-variables
        # fast path (and its guard) entirely.
        model_ref = pybamm.lithium_ion.DFN()
        model_ref.convert_to_format = "casadi"
        params_ref = model_ref.default_parameter_values
        params_ref.update({"Current function [A]": "[input]"})
        sim_ref = pybamm.Simulation(
            model_ref, solver=pybamm.IDAKLUSolver(), parameter_values=params_ref
        )
        sol_ref = sim_ref.solve(
            [0, 100], inputs=input_parameters, calculate_sensitivities=True
        )

        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = "rust"
        params = model.default_parameter_values
        params.update({"Current function [A]": "[input]"})
        solver = pybamm.IDAKLUSolver(output_variables=[var_name])
        sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
        sol = sim.solve([0, 100], inputs=input_parameters, calculate_sensitivities=True)

        np.testing.assert_allclose(
            sol[var_name].data, sol_ref[var_name].data, rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(sol[var_name].sensitivities["Current function [A]"]),
            np.asarray(sol_ref[var_name].sensitivities["Current function [A]"]),
            rtol=1e-5,
            atol=1e-9,
        )

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

    def test_closest_event_idx_set_after_root_return(self):
        # IDAKLU must populate Solution.closest_event_idx after a root return so
        # BaseSolver.get_termination_reason short-circuits instead of re-walking
        # every TERMINATION event's symbolic expression on the Python side. That
        # slow path generated tens of thousands of small numpy allocations per
        # long event-terminated cycling run.
        cycle = (
            "Discharge at 1C until 3.0 V",
            "Charge at 1C until 4.2 V",
            "Hold at 4.2 V until C/50",
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment([cycle] * 2, period=300),
            solver=pybamm.IDAKLUSolver(output_variables=["Voltage [V]"]),
        )
        sim.solve()

        event_steps = [
            step
            for cycle_sol in sim.solution.cycles
            for step in cycle_sol.steps
            if step.termination.startswith("event:")
        ]
        assert event_steps, "expected at least one event-terminated step"
        # The index must also resolve to the same event name the slow path in
        # BaseSolver.get_termination_reason would have picked.
        for step in event_steps:
            assert step.closest_event_idx is not None, (
                f"event-terminated step {step.termination!r} has "
                f"closest_event_idx=None — BaseSolver will fall back to "
                f"per-step Python event re-evaluation"
            )
            terminate_events = [
                e
                for e in step.all_models[-1].events
                if e.event_type == pybamm.EventType.TERMINATION
            ]
            picked = terminate_events[step.closest_event_idx].name
            assert step.termination == f"event: {picked}", (
                f"closest_event_idx={step.closest_event_idx} resolves to "
                f"{picked!r}, but step.termination is {step.termination!r}"
            )

    def test_pickle_roundtrip_preserves_closest_event_idx(self):
        # The pickle drops _setup; the next solve rebuilds it from the model.
        import pickle

        solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(
                [("Discharge at 1C until 3.0 V", "Charge at 1C until 4.2 V")]
            ),
            solver=solver,
        )
        sim.solve()

        roundtripped = pickle.loads(pickle.dumps(solver))
        sim2 = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=pybamm.Experiment(
                [("Discharge at 1C until 3.0 V", "Charge at 1C until 4.2 V")]
            ),
            solver=roundtripped,
        )
        sim2.solve()

        for step in sim2.solution.cycles[0].steps:
            if step.termination.startswith("event:"):
                assert step.closest_event_idx is not None, (
                    "round-tripped IDAKLUSolver must still set "
                    "closest_event_idx after a root return"
                )

    def test_closest_event_idx_set_after_root_return_rust(self):
        # Rust-path variant of test_closest_event_idx_set_after_root_return;
        # _set_up_rust must bind the compiled model's event tapes for this to pass.
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "rust"
        cycle = (
            "Discharge at 1C until 3.0 V",
            "Charge at 1C until 4.2 V",
            "Hold at 4.2 V until C/50",
        )
        sim = pybamm.Simulation(
            model,
            experiment=pybamm.Experiment([cycle] * 2, period=300),
            solver=pybamm.IDAKLUSolver(output_variables=["Voltage [V]"]),
        )
        sim.solve()

        event_steps = [
            step
            for cycle_sol in sim.solution.cycles
            for step in cycle_sol.steps
            if step.termination.startswith("event:")
        ]
        assert event_steps, "expected at least one event-terminated step"
        for step in event_steps:
            assert step.all_models[-1].convert_to_format == "rust"
            assert step.closest_event_idx is not None, (
                f"event-terminated step {step.termination!r} has "
                f"closest_event_idx=None — BaseSolver will fall back to "
                f"per-step Python event re-evaluation"
            )
            terminate_events = [
                e
                for e in step.all_models[-1].events
                if e.event_type == pybamm.EventType.TERMINATION
            ]
            picked = terminate_events[step.closest_event_idx].name
            assert step.termination == f"event: {picked}", (
                f"closest_event_idx={step.closest_event_idx} resolves to "
                f"{picked!r}, but step.termination is {step.termination!r}"
            )

    def test_rust_fused_events_parity_with_casadi(self):
        # SPMe has >= 2 termination events, so rust evaluates them through the fused
        # tape; a tightened cut-off fires one, and both paths must agree on which.
        def _solve(fmt):
            params = pybamm.ParameterValues("Chen2020")
            params["Lower voltage cut-off [V]"] = 3.5
            model = pybamm.lithium_ion.SPMe()
            model.convert_to_format = fmt
            sim = pybamm.Simulation(
                model, parameter_values=params, solver=pybamm.IDAKLUSolver()
            )
            return sim.solve(np.linspace(0, 3600, 100))

        sol_rust = _solve("rust")
        sol_casadi = _solve("casadi")

        assert sol_rust.termination.startswith("event:"), (
            f"expected an event to fire, got {sol_rust.termination!r}"
        )
        assert sol_rust.termination == sol_casadi.termination
        assert sol_rust.closest_event_idx == sol_casadi.closest_event_idx
        np.testing.assert_allclose(
            float(sol_rust.t_event[0]), float(sol_casadi.t_event[0]), rtol=1e-4
        )

    def test_rust_events_survive_compiled_model_pickle(self):
        # Pickling round-trips through `_rebuild`, which re-runs `build_from_parts`
        # and thus `fuse_events`, so the rebuilt model must evaluate events the same.
        import pickle

        model = pybamm.lithium_ion.SPMe()
        model.convert_to_format = "rust"
        sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())
        sol = sim.solve(np.linspace(0, 3600, 10))

        rust_model = sim._solver._setup["rust_model"]
        assert rust_model.n_events >= 2  # fusion is active

        # A representative evaluation point from the solve.
        y = np.ascontiguousarray(sol.all_ys[0][:, 0], dtype=np.float64)
        p = np.zeros(rust_model.n_inputs, dtype=np.float64)
        t = float(sol.all_ts[0][0])
        before = [np.asarray(fn(t, y, p)).ravel() for fn in rust_model.events]

        restored = pickle.loads(pickle.dumps(rust_model))
        assert restored.n_events == rust_model.n_events
        after = [np.asarray(fn(t, y, p)).ravel() for fn in restored.events]
        for a, b in zip(before, after, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_rust_diffsol_event_termination(self):
        # Diffsol root-finding uses the same fused event tape (RootOp path).
        params = pybamm.ParameterValues("Chen2020")
        params["Lower voltage cut-off [V]"] = 3.5
        model = pybamm.lithium_ion.SPMe()
        model.convert_to_format = "rust"
        sim = pybamm.Simulation(
            model, parameter_values=params, solver=pybamm.DiffsolSolver()
        )
        sol = sim.solve(np.linspace(0, 3600, 100))
        assert sol.termination.startswith("event:"), (
            f"expected diffsol root-finding to terminate on an event, "
            f"got {sol.termination!r}"
        )
        assert float(sol.t[-1]) < 3600.0

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

    def test_idaklu_forces_casadi_format(self):
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

        solver = pybamm.IDAKLUSolver()
        assert model.convert_to_format == "python"
        solver.set_up(model)
        assert model.convert_to_format == "casadi"

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
        model.convert_to_format = "casadi"
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

    def test_hermite_reduction_factor_incompatible(self):
        """Test errors/warnings when hermite_reduction_factor conflicts with other options."""
        # Error at construction: hermite_reduction_factor + output_variables
        with pytest.raises(pybamm.SolverError, match="output_variables"):
            pybamm.IDAKLUSolver(
                options={"hermite_reduction_factor": 2.0},
                output_variables=["Voltage [V]"],
            )

        # Error at construction: hermite_reduction_factor + hermite_interpolation disabled
        with pytest.raises(pybamm.SolverError, match="hermite_interpolation"):
            pybamm.IDAKLUSolver(
                options={
                    "hermite_reduction_factor": 2.0,
                    "hermite_interpolation": False,
                },
            )

        # Warning at solve: hermite_reduction_factor + sensitivities
        model_sens = pybamm.lithium_ion.SPM()
        param = model_sens.default_parameter_values
        param["Current function [A]"] = pybamm.InputParameter("I")
        solver = pybamm.IDAKLUSolver(options={"hermite_reduction_factor": 2.0})
        sim = pybamm.Simulation(model_sens, parameter_values=param, solver=solver)
        with pytest.warns(pybamm.SolverWarning, match="not currently supported"):
            sim.solve([0, 1], inputs={"I": 1.0}, calculate_sensitivities=True)

    def test_hermite_reduction_factor_sensitivities_warning_rust(self):
        # Regression: `_set_up_rust` used to return before the casadi path's
        # hermite_reduction_factor + sensitivities check, skipping the warning.
        model_sens = pybamm.lithium_ion.SPM()
        model_sens.convert_to_format = "rust"
        param = model_sens.default_parameter_values
        param["Current function [A]"] = pybamm.InputParameter("I")
        solver = pybamm.IDAKLUSolver(options={"hermite_reduction_factor": 2.0})
        sim = pybamm.Simulation(model_sens, parameter_values=param, solver=solver)
        with pytest.warns(pybamm.SolverWarning, match="not currently supported"):
            sim.solve([0, 1], inputs={"I": 1.0}, calculate_sensitivities=True)

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

    def test_reduce_solution_keeps_native_observation(self):
        """A reduced solution must not fall back to CasADi observation."""
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8)
        sol = pybamm.Simulation(model, solver=solver).solve([0, 3600])
        assert isinstance(sol.observation, NativeInterpolatingObservation)

        reduced = solver.reduce_solution(sol, hermite_reduction_factor=2.0)

        # Thinning knots leaves the segments and their models alone.
        assert isinstance(reduced.observation, NativeInterpolatingObservation)
        assert reduced.observation.segment_models == sol.observation.segment_models
        assert reduced.observation.compile_cache is sol.observation.compile_cache
        assert isinstance(reduced["Terminal voltage [V]"]._observer, NativeObserver)
        # ... and the reduced spline still reads within its error budget
        t = np.linspace(0, 3600, 51)
        np.testing.assert_allclose(
            reduced["Terminal voltage [V]"](t),
            sol["Terminal voltage [V]"](t),
            rtol=0,
            atol=1e-4,
        )

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

    def test_solution_user_options_forwarded(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        t_eval = [0, 1]

        sol_default = pybamm.IDAKLUSolver().solve(model, t_eval)
        assert sol_default.user_options == {"compile": False}
        assert sol_default.options["compile"] is False

        sol_vm = pybamm.IDAKLUSolver(
            options={"compile": False, "num_threads": 2}
        ).solve(model, t_eval)
        assert sol_vm.user_options == {"compile": False}
        assert "num_threads" not in sol_vm.user_options

    def test_solution_user_options_survive_pickle(self, tmp_path):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -u}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u, "2u": 2 * u}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        sol = pybamm.IDAKLUSolver().solve(model, [0, 1])
        path = tmp_path / "idaklu_sol.pickle"
        sol.save(path)
        loaded = pybamm.load(path)

        assert loaded.user_options == sol.user_options
        assert loaded.options == sol.options
        np.testing.assert_allclose(
            loaded["2u"].entries, sol["2u"].entries, rtol=1e-12, atol=1e-12
        )

    def test_idaklu_dispatch_is_flag_driven(self):
        with pytest.raises(TypeError):
            pybamm.IDAKLUSolver(evaluator="rust")
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        solver = pybamm.IDAKLUSolver()
        solver.set_up(model, inputs=[{"a": 0.5}])
        assert "rust_model" in solver._setup
        assert model.convert_to_format == "rust"

    def test_idaklu_rust_output_sensitivities_set_up_ok(self):
        # output_variables + calculate_sensitivities is supported for
        # convert_to_format="rust", so set_up must not raise for the combination.
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        model.calculate_sensitivities = ["a"]
        solver = pybamm.IDAKLUSolver(output_variables=["u"])
        solver.set_up(model, inputs=[{"a": 0.5}])
        assert model.convert_to_format == "rust"

    def test_idaklu_rust_output_values_ok_without_sens(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        solver = pybamm.IDAKLUSolver(output_variables=["u"])
        solver.set_up(model, inputs=[{"a": 0.5}])
        assert model.convert_to_format == "rust"

    def test_rhs_dot_consistent_init_rust_inputs(self):
        # Regression: _rhs_dot_consistent_initialization must stack inputs for rust,
        # not hand a dict to the RustEvaluator.
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        solver = pybamm.IDAKLUSolver()
        solver.set_up(model, inputs=[{"a": 0.5}])
        y0 = np.asarray(model.y0_list[0]).ravel()
        ydot0 = solver._rhs_dot_consistent_initialization(y0, model, 0.0, {"a": 0.5})
        assert ydot0.shape == y0.shape

    def test_sensitivity_consistent_init_gate_fires_for_rust(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        model.calculate_sensitivities = ["a"]
        solver = pybamm.IDAKLUSolver()
        solver.set_up(model, inputs=[{"a": 0.5}])
        solver._set_consistent_initialization(model, 0.0, [{"a": 0.5}])
        # 1 sens param: y0full should be len_rhs_and_alg * 2
        assert model.y0full[0].shape[0] == model.len_rhs_and_alg * 2


class TestIDAKLUNativeObservation:
    def _solve(self, convert_to_format, calculate_sensitivities=False, t_interp=None):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        if calculate_sensitivities:
            param.update({"Current function [A]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        model.convert_to_format = convert_to_format
        solver = pybamm.IDAKLUSolver()
        t_eval = [0, 3600]
        sol = solver.solve(
            model,
            t_eval,
            t_interp=t_interp,
            inputs={"Current function [A]": 0.68} if calculate_sensitivities else None,
            calculate_sensitivities=calculate_sensitivities,
        )
        return sol, solver

    def test_offgrid_values_match_casadi(self):
        # An off-grid t_interp exercises native Hermite against CasADi's
        # observe_hermite_interp.
        t_interp = np.linspace(0, 3600, 97)
        cas, _ = self._solve("casadi", t_interp=t_interp)
        rust, _ = self._solve("rust", t_interp=t_interp)
        for name in [
            "Terminal voltage [V]",
            "X-averaged negative particle concentration [mol.m-3]",
        ]:
            np.testing.assert_allclose(
                rust[name](t_interp), cas[name](t_interp), rtol=1e-6, atol=1e-6
            )

    def test_native_processed_variable_matches_casadi_direct(self):
        # Build a native-backed ProcessedVariable by hand (routing is Task A4)
        # and assert its native leaves + sensitivities match a CasADi solve.
        t_interp = np.linspace(0, 3600, 41)
        cas, _ = self._solve("casadi", calculate_sensitivities=True, t_interp=t_interp)
        sol, solver = self._solve(
            "rust", calculate_sensitivities=True, t_interp=t_interp
        )

        name = "Terminal voltage [V]"
        vars_pybamm = [m.get_processed_variable_or_event(name) for m in sol.all_models]
        rust_model = solver._setup["rust_model"]
        rust_fns = [sol.observation._leaf(name, vp, rust_model) for vp in vars_pybamm]
        pv = pybamm.process_variable(
            name,
            vars_pybamm,
            NativeObserver(rust_fns, sol.observation),
            sol,
            time_integral=None,
        )

        # raw entries (exercises _observe_raw_native)
        np.testing.assert_allclose(pv.entries, cas[name].entries, rtol=1e-6, atol=1e-6)
        # off-grid query (exercises _observe_hermite_native)
        np.testing.assert_allclose(
            pv(t_interp), cas[name](t_interp), rtol=1e-6, atol=1e-6
        )
        # sensitivities (exercises _initialise_sensitivity_native)
        np.testing.assert_allclose(
            pv.sensitivities["Current function [A]"],
            cas[name].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_flip_compiles_no_casadi_observe(self):
        # An idaklu rust-mode full-state solve observes natively: the
        # ProcessedVariable is Rust-backed with no CasADi observe function.
        rust, _ = self._solve("rust")
        v = rust["Terminal voltage [V]"]
        _ = v.entries  # force observation
        assert isinstance(v._observer, NativeObserver)
        assert not any(isinstance(f, casadi.Function) for f in v._observer.leaves)

    def test_offgrid_values_native_backed(self):
        # The off-grid parity above only exercises the native path if the observed
        # ProcessedVariable is Rust-backed with no CasADi funcs.
        rust, _ = self._solve("rust")
        for name in [
            "Terminal voltage [V]",
            "X-averaged negative particle concentration [mol.m-3]",
        ]:
            v = rust[name]
            _ = v.entries
            assert isinstance(v._observer, NativeObserver)
            assert not any(isinstance(f, casadi.Function) for f in v._observer.leaves)

    def test_sensitivities_match_casadi(self):
        # 0D non-time-integral variable sensitivities via the native chain rule.
        cas, _ = self._solve("casadi", calculate_sensitivities=True)
        rust, _ = self._solve("rust", calculate_sensitivities=True)
        for name in ["Terminal voltage [V]"]:
            assert isinstance(rust[name]._observer, NativeObserver)
            np.testing.assert_allclose(
                rust[name].sensitivities["Current function [A]"],
                cas[name].sensitivities["Current function [A]"],
                rtol=1e-5,
                atol=1e-6,
            )

    def _solve_output_vars(
        self, convert_to_format, output_variables, model=None, inputs=None, var_pts=None
    ):
        if model is None:
            model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        if inputs is None:
            inputs = {"Current function [A]": 0.68}
        param.update({key: "[input]" for key in inputs})
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(
            geometry, model.default_submesh_types, var_pts or model.default_var_pts
        )
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        model.convert_to_format = convert_to_format
        solver = pybamm.IDAKLUSolver(output_variables=output_variables)
        # The two lowerings take different adaptive steps, so without t_interp each
        # solution lands on its own grid and callers compare mismatched times.
        sol = solver.solve(
            model,
            [0, 3600],
            inputs=inputs,
            calculate_sensitivities=True,
            t_interp=np.linspace(0, 3600, 100),
        )
        return sol, solver

    def test_output_variables_sensitivities_match_casadi(self):
        # save_outputs_only path: output_variables + calculate_sensitivities
        # together exercise the native yS projection consumer (B1+B2+B3).
        ov = ["Terminal voltage [V]"]
        cas, _ = self._solve_output_vars("casadi", ov)
        rust, _ = self._solve_output_vars("rust", ov)
        np.testing.assert_allclose(
            rust["Terminal voltage [V]"].sensitivities["Current function [A]"],
            cas["Terminal voltage [V]"].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_output_variables_sensitivities_match_casadi_spme(self):
        # Repeat the single-param output-variable-sensitivity parity check on
        # a second chemistry, guarding against SPM-only coincidences.
        ov = ["Terminal voltage [V]"]
        cas, _ = self._solve_output_vars("casadi", ov, model=pybamm.lithium_ion.SPMe())
        rust, _ = self._solve_output_vars("rust", ov, model=pybamm.lithium_ion.SPMe())
        np.testing.assert_allclose(
            rust["Terminal voltage [V]"].sensitivities["Current function [A]"],
            cas["Terminal voltage [V]"].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_output_variables_sensitivities_match_casadi_dfn(self):
        # DFN is a much heavier solve, so a coarse mesh and one output variable bound
        # the cost while still exercising the native yS projection.
        ov = ["Terminal voltage [V]"]
        var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 5, "r_p": 5}
        cas, _ = self._solve_output_vars(
            "casadi", ov, model=pybamm.lithium_ion.DFN(), var_pts=var_pts
        )
        rust, _ = self._solve_output_vars(
            "rust", ov, model=pybamm.lithium_ion.DFN(), var_pts=var_pts
        )
        np.testing.assert_allclose(
            rust["Terminal voltage [V]"].sensitivities["Current function [A]"],
            cas["Terminal voltage [V]"].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_output_variables_multi_param_sensitivities_match_casadi(self):
        # The two sensitivity params reach solve() reversed from the sorted order that
        # sets yS columns, so keying off insertion order would mislabel one.
        ov = ["Terminal voltage [V]"]
        inputs = {
            "Negative electrode active material volume fraction": 0.6,
            "Current function [A]": 0.68,
        }
        cas, _ = self._solve_output_vars("casadi", ov, inputs=inputs)
        rust, _ = self._solve_output_vars("rust", ov, inputs=inputs)
        for name in inputs:
            np.testing.assert_allclose(
                rust["Terminal voltage [V]"].sensitivities[name],
                cas["Terminal voltage [V]"].sensitivities[name],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"sensitivity mismatch for param '{name}'",
            )

    def test_multi_output_multi_param_sensitivities_match_casadi(self):
        # Two outputs and two parameters: yS is scattered as (output, param), and
        # a transposed write agrees on the diagonal, so it needs both to be > 1.
        ov = ["Terminal voltage [V]", "Discharge capacity [A.h]"]
        inputs = {
            "Negative electrode active material volume fraction": 0.6,
            "Current function [A]": 0.68,
        }
        cas, _ = self._solve_output_vars("casadi", ov, inputs=inputs)
        rust, _ = self._solve_output_vars("rust", ov, inputs=inputs)
        for var in ov:
            for name in inputs:
                np.testing.assert_allclose(
                    rust[var].sensitivities[name],
                    cas[var].sensitivities[name],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"sensitivity mismatch for '{var}', parameter '{name}'",
                )

    def test_output_variables_time_integral_sensitivities_match_casadi(self):
        # "Discharge capacity [A.h]" is a plain ODE state under the default
        # discretisation, so this exercises the sensitivity chain off voltage.
        ov = ["Discharge capacity [A.h]"]
        cas, _ = self._solve_output_vars("casadi", ov)
        rust, _ = self._solve_output_vars("rust", ov)
        np.testing.assert_allclose(
            rust["Discharge capacity [A.h]"].sensitivities["Current function [A]"],
            cas["Discharge capacity [A.h]"].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_output_variables_sensitivities_no_casadi_compiled(self):
        # BaseSolver.set_up skips the computed_var_fcns loop for
        # convert_to_format="rust", so no CasADi var/sens keys are ever compiled.
        ov = ["Terminal voltage [V]"]
        rust, solver = self._solve_output_vars("rust", ov)
        assert solver.computed_var_fcns == {}
        assert solver.computed_dvar_dy_fcns == {}
        assert solver.computed_dvar_dp_fcns == {}
        assert "var_fcns" not in solver._setup
        assert "dvar_dy_idaklu_fcns" not in solver._setup
        assert "dvar_dp_idaklu_fcns" not in solver._setup
        assert rust.variables_returned is True

    @staticmethod
    def _discretise_dfn(convert_to_format):
        model = pybamm.lithium_ion.DFN()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        var_pts = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 10, "r_p": 10}
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        model.convert_to_format = convert_to_format
        return model

    def test_spatial_1d_2d_variables_match_casadi(self):
        # 1D (x) and 2D (r, x) spatial variables exercise the native
        # order="F" reshape/segment layout beyond the 0D cases above.
        t_interp = np.linspace(0, 3600, 51)
        cas = pybamm.IDAKLUSolver().solve(
            self._discretise_dfn("casadi"), [0, 3600], t_interp=t_interp
        )
        rust = pybamm.IDAKLUSolver().solve(
            self._discretise_dfn("rust"), [0, 3600], t_interp=t_interp
        )
        # off-grid query points strictly inside the solved interval
        off = t_interp[:-1] + np.diff(t_interp) / 3
        for name in [
            "Electrolyte concentration [mol.m-3]",  # 1D in x
            "Negative particle concentration [mol.m-3]",  # 2D (r, x)
        ]:
            assert isinstance(rust[name]._observer, NativeObserver)
            np.testing.assert_allclose(
                rust[name](off), cas[name](off), rtol=1e-6, atol=1e-6
            )

    @staticmethod
    def _discretise_spm(convert_to_format, as_input=False):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        if as_input:
            param.update({"Current function [A]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        model.convert_to_format = convert_to_format
        return model

    def test_multi_segment_experiment_matches_casadi(self):
        # A stepped solve builds all_ys with >1 segment, exercising per-segment
        # native Hermite routing and flag propagation through Solution.__add__.
        def stepped(convert_to_format):
            model = self._discretise_spm(convert_to_format)
            solver = pybamm.IDAKLUSolver()
            sol = None
            for _ in range(2):
                sol = solver.step(sol, model, dt=1800)
            return sol

        cas = stepped("casadi")
        rust = stepped("rust")
        assert len(rust.all_ys) == 2
        assert isinstance(rust.observation, NativeInterpolatingObservation)
        off = np.linspace(0, rust.t[-1], 73)[1:-1] + 5.0
        off = off[off < rust.t[-1]]
        for name in ["Terminal voltage [V]"]:
            assert isinstance(rust[name]._observer, NativeObserver)
            np.testing.assert_allclose(
                rust[name](off), cas[name](off), rtol=1e-6, atol=1e-6
            )

    def test_event_termination_matches_casadi(self):
        # A voltage-cutoff termination yields a partial final segment; the
        # native path must still off-grid-interpolate to match CasADi.
        cas = pybamm.IDAKLUSolver().solve(self._discretise_spm("casadi"), [0, 100000])
        rust = pybamm.IDAKLUSolver().solve(self._discretise_spm("rust"), [0, 100000])
        assert rust.termination.startswith("event:")
        name = "Terminal voltage [V]"
        assert isinstance(rust[name]._observer, NativeObserver)
        t_end = min(rust.t[-1], cas.t[-1])
        off = np.linspace(0, t_end, 40)[1:-1] + 1.0
        off = off[off < t_end]
        np.testing.assert_allclose(
            rust[name](off), cas[name](off), rtol=1e-6, atol=1e-6
        )

    def test_hermite_off_still_native_no_casadi(self):
        # hermite_interpolation=False disables yps, but the solve must stay on the
        # native-backed ProcessedVariable, not diffsol's ProcessedVariableComputed.
        solver_kwargs = {"options": {"hermite_interpolation": False}}
        cas = pybamm.IDAKLUSolver(**solver_kwargs).solve(
            self._discretise_spm("casadi"), [0, 3600]
        )
        rust = pybamm.IDAKLUSolver(**solver_kwargs).solve(
            self._discretise_spm("rust"), [0, 3600]
        )
        assert not rust.hermite_interpolation
        v = rust["Terminal voltage [V]"]
        assert isinstance(v._observer, NativeObserver)
        assert not any(isinstance(f, casadi.Function) for f in v._observer.leaves)
        np.testing.assert_allclose(
            v.entries, cas["Terminal voltage [V]"].entries, rtol=1e-6, atol=1e-6
        )

    def test_time_integral_sensitivities_match_casadi(self):
        # An ExplicitTimeIntegral output variable exercises the native postfix
        # (_native_postfix_sensitivities) sensitivity path.
        inputs = {"Current function [A]": 0.68}
        cas = pybamm.IDAKLUSolver().solve(
            self._discretise_spm("casadi", as_input=True),
            [0, 3600],
            inputs=inputs,
            calculate_sensitivities=True,
        )
        rust = pybamm.IDAKLUSolver().solve(
            self._discretise_spm("rust", as_input=True),
            [0, 3600],
            inputs=inputs,
            calculate_sensitivities=True,
        )
        name = "Discharge capacity [A.h]"
        assert isinstance(rust[name]._observer, NativeObserver)
        np.testing.assert_allclose(
            rust[name].sensitivities["Current function [A]"],
            cas[name].sensitivities["Current function [A]"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_output_variables_first_last_state_observe_natively(self):
        # Outputs-only solves must attach the observation context: their first/last
        # states carry full state vectors, so summary variables must match exactly.
        def solve(output_variables):
            model = self._discretise_spm("rust")
            solver = pybamm.IDAKLUSolver(output_variables=output_variables)
            return solver.solve(model, [0, 3600])

        full = solve(None)
        outputs_only = solve(["Voltage [V]"])
        assert isinstance(outputs_only.observation, NativeObservation)
        assert isinstance(outputs_only.last_state.observation, NativeObservation)
        name = "Total lithium in electrolyte [mol]"
        np.testing.assert_array_equal(
            outputs_only.last_state[name].data, full.last_state[name].data
        )

    def test_output_variables_stay_aligned_across_the_batch_window(self):
        # 200 points cross the 128-point batched-evaluation window on the rust
        # FFI; a flush off-by-one would shift every value after the boundary.
        t_interp = np.linspace(0, 3600, 200)
        names = [
            "Voltage [V]",
            "Negative particle surface concentration [mol.m-3]",
        ]

        def solve(output_variables):
            model = self._discretise_spm("rust")
            solver = pybamm.IDAKLUSolver(output_variables=output_variables)
            return solver.solve(model, [0, 3600], t_interp=t_interp)

        full = solve(None)
        outputs_only = solve(names)
        for name in names:
            np.testing.assert_allclose(
                np.asarray(outputs_only[name](t_interp)),
                np.asarray(full[name](t_interp)),
                rtol=1e-7,
                atol=1e-9,
                err_msg=name,
            )

    def test_output_variables_genuine_time_integral_matches_casadi(self):
        # remove_independent_variables_from_rhs=True turns "Discharge capacity [A.h]"
        # into a genuine ExplicitTimeIntegral, evaluated natively then postfixed.
        def solve(convert_to_format):
            model = pybamm.lithium_ion.SPM()
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.update({"Current function [A]": "[input]"})
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(
                mesh,
                model.default_spatial_methods,
                remove_independent_variables_from_rhs=True,
            )
            disc.process_model(model)
            model.convert_to_format = convert_to_format
            solver = pybamm.IDAKLUSolver(output_variables=["Discharge capacity [A.h]"])
            return solver.solve(
                model,
                [0, 3600],
                inputs={"Current function [A]": 0.68},
                calculate_sensitivities=True,
            )

        cas = solve("casadi")
        rust = solve("rust")
        name = "Discharge capacity [A.h]"
        assert rust[name].data.shape == (1,)
        np.testing.assert_allclose(
            rust[name].data, cas[name].data, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            rust[name].sensitivities["Current function [A]"],
            cas[name].sensitivities["Current function [A]"],
            rtol=1e-6,
            atol=1e-8,
        )


class TestIDAKLUSensitivityScales:
    """IDAS ``pbar``, so a tiny parameter's sensitivity column stays solvable."""

    @staticmethod
    def _tiny_parameter_model():
        """``du/dt = -(a / a0) u`` with ``a0 = 1e-14``, so ``du/da ~ 1e14``.

        Analytically ``u = exp(-t)`` at ``a = a0`` and
        ``du/da = -t exp(-t) / a0``, a column no absolute tolerance can hold at
        the default ``pbar = 1``.
        """
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        model.rhs = {u: -(a / 1e-14) * u}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)
        return model

    def test_scales_are_the_parameter_magnitudes(self):
        scales = pybamm.solvers.idaklu_solver._sensitivity_scales(
            {"a": -3.0, "b": 4e-15, "c": 1.0}, ["b", "a"]
        )
        np.testing.assert_allclose(scales, [4e-15, 3.0])

    def test_magnitudes_are_handed_over_unclamped(self):
        # The solver owns the zero/non-finite clamp, so a raw 0.0 reaches it.
        scales = pybamm.solvers.idaklu_solver._sensitivity_scales(
            {"a": 0.0, "b": 2.0}, ["a", "b"]
        )
        np.testing.assert_allclose(scales, [0.0, 2.0])

    def test_a_zero_parameter_still_solves(self):
        # IDAS rejects pbar = 0, so the solver has to clamp it to the unit scale.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        model.rhs = {u: -u + a}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)

        solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        sol = solver.solve(
            model,
            np.linspace(0, 1, 10),
            inputs={"a": 0.0},
            calculate_sensitivities=True,
        )
        # du/da = 1 - exp(-t) regardless of a, so the clamp must not skew it.
        np.testing.assert_allclose(
            np.asarray(sol["u"].sensitivities["a"]).ravel(),
            1.0 - np.exp(-sol.t),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_a_vector_parameter_uses_its_largest_magnitude(self):
        scales = pybamm.solvers.idaklu_solver._sensitivity_scales(
            {"a": np.array([1e-3, -5e-3, 2e-3])}, ["a"]
        )
        np.testing.assert_allclose(scales, [5e-3])

    def test_scales_reach_the_solver_group(self):
        model = self._tiny_parameter_model()
        solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
        solve_kwargs = {"inputs": {"a": 1e-14}, "calculate_sensitivities": True}
        # Solve once so the second solve reuses this group rather than rebuilding
        # it, which would discard the spy.
        solver.solve(model, [0, 1], **solve_kwargs)

        seen = {}
        original = solver._setup["solver"].solve

        def spy(*args, **kwargs):
            seen["pbar"] = np.asarray(args[5])
            return original(*args, **kwargs)

        solver._setup["solver"] = type("Spy", (), {"solve": staticmethod(spy)})()
        solver.solve(model, [0, 1], **solve_kwargs)
        np.testing.assert_allclose(seen["pbar"], [[1e-14]])

    @pytest.mark.parametrize("convert_to_format", ["casadi", "rust"])
    def test_a_tight_dfn_diffusivity_sensitivity_solve_converges(
        self, convert_to_format
    ):
        # D_p is 4e-15, so dy/dD_p reaches ~1e14; unscaled in the corrector's
        # weighted norm that is an IDA_CONV_FAIL.
        model = pybamm.lithium_ion.DFN()
        model.convert_to_format = convert_to_format
        name = "Positive particle diffusivity [m2.s-1]"
        parameter_values = pybamm.ParameterValues("Chen2020")
        nominal = float(parameter_values[name])
        parameter_values[name] = pybamm.InputParameter("D_p")
        simulation = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )
        sol = simulation.solve(
            np.linspace(0, 600, 20),
            inputs={"D_p": nominal},
            calculate_sensitivities=["D_p"],
        )
        gradient = np.asarray(sol["Voltage [V]"].sensitivities["D_p"]).ravel()
        assert np.all(np.isfinite(gradient))
        assert np.abs(gradient).max() > 0.0

    @pytest.mark.parametrize("convert_to_format", ["casadi", "rust"])
    def test_tiny_parameter_sensitivity_matches_the_analytic_column(
        self, convert_to_format
    ):
        # Scaling the weights must not disturb the column itself.
        model = self._tiny_parameter_model()
        model.convert_to_format = convert_to_format
        solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
        sol = solver.solve(
            model,
            np.linspace(0, 3, 20),
            inputs={"a": 1e-14},
            calculate_sensitivities=True,
        )
        expected = -sol.t * np.exp(-sol.t) / 1e-14
        np.testing.assert_allclose(
            np.asarray(sol["u"].sensitivities["a"]).ravel(),
            expected,
            rtol=1e-5,
            atol=1e-5 / 1e-14,
        )


class TestIDAKLUSensitivityScaleOrdering:
    """One scale per parameter, in the solver's column order rather than the
    input dict's insertion order. Both orderings have the right length, so a
    mix-up degrades the weighting silently instead of raising."""

    def test_each_parameter_gets_its_own_magnitude(self):
        scales = pybamm.solvers.idaklu_solver._sensitivity_scales(
            {"b": 1e2, "a": 1e-14}, ["a", "b"]
        )
        np.testing.assert_allclose(scales, [1e-14, 1e2])


class TestIDAKLUModelAtol:
    """``model.atol`` overrides the solver's own tolerance, per state as well as
    uniformly. Only IDAKLU reads it."""

    def _two_state_model(self, atol=None):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        w = pybamm.Variable("w")
        model.rhs = {u: -u, w: -2 * w}
        model.initial_conditions = {u: 1.0, w: 1.0}
        model.variables = {"u": u, "w": w}
        pybamm.Discretisation().process_model(model)
        if atol is not None:
            model.atol = atol
        return model

    def _steps(self, model, atol):
        solution = pybamm.IDAKLUSolver(rtol=1e-6, atol=atol).solve(
            model, np.linspace(0, 1, 5)
        )
        return solution.solver_statistics.number_of_steps

    def test_a_per_state_model_atol_wins_over_the_solvers(self):
        tight = self._steps(self._two_state_model(), 1e-12)
        loose = self._steps(self._two_state_model(), 1e-1)
        assert self._steps(self._two_state_model(np.full(2, 1e-1)), 1e-12) == loose
        assert loose < tight

    def test_a_wrong_width_model_atol_is_rejected(self):
        model = self._two_state_model(np.full(3, 1e-6))
        with pytest.raises(pybamm.SolverError, match=r"shape \(3,\) but \(2,\)"):
            self._steps(model, 1e-6)
