import dataclasses
import io
import itertools
import json
import logging
import os
import re
import subprocess  # nosec B404 - runs this interpreter on a fixed script
import sys
import textwrap
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad_vec
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import csc_matrix

import pybamm
from tests import (
    get_broken_input_model,
    get_discretisation_for_testing,
    no_internet_connection,
)


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

    def test_every_failing_input_set_is_named(self):
        model = get_broken_input_model()
        solver = pybamm.IDAKLUSolver(options={"num_threads": 4})
        inputs_list = [{"k": k} for k in (1.0, -1.0, 2.0, -2.0, 3.0)]
        with pytest.raises(pybamm.SolverError) as error:
            solver.solve(model, np.linspace(0, 1, 10), inputs=inputs_list)
        assert re.findall(r"input set (\d+): ", str(error.value)) == ["1", "3"]

    def test_a_single_input_set_failure_is_named(self):
        model = get_broken_input_model()
        with pytest.raises(pybamm.SolverError, match=r"^input set 0: "):
            pybamm.IDAKLUSolver().solve(model, [0, 1], inputs={"k": -1.0})

    def test_every_input_set_is_solved_by_a_smaller_team(self):
        # The OpenMP runtime reads OMP_THREAD_LIMIT once, so this needs a fresh
        # process; the team then has fewer threads than the solver has solvers.
        script = textwrap.dedent(
            """
            import json

            import numpy as np

            import pybamm

            model = pybamm.BaseModel()
            u = pybamm.Variable("u")
            model.rhs = {u: -pybamm.InputParameter("a") * u}
            model.initial_conditions = {u: 1}
            model.variables = {"u": u}
            pybamm.Discretisation().process_model(model)
            solver = pybamm.IDAKLUSolver(
                rtol=1e-8, atol=1e-10, options={"num_threads": 4}
            )
            inputs = [{"a": 1.0 + i} for i in range(6)]
            solutions = solver.solve(model, [0, 1], inputs=inputs)
            print(json.dumps([float(np.squeeze(s["u"](1.0))) for s in solutions]))
            """
        )
        result = subprocess.run(  # nosec B603 - fixed arguments, no shell
            [sys.executable, "-c", script],
            env={**os.environ, "OMP_THREAD_LIMIT": "2"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        values = json.loads(result.stdout.strip().splitlines()[-1])
        np.testing.assert_allclose(values, np.exp(-(1.0 + np.arange(6))), rtol=1e-6)

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
        ("num_threads", "n_inputs"),
        [
            # One solver, so every solve runs on the GIL-holding thread
            (1, 2),
            # More solvers than input sets, so the team is capped at one thread
            # per set and the worker buffers its set
            (4, 2),
            # More input sets than threads, so each thread solves several
            (2, 4),
        ],
    )
    def test_diagnostics_emitted_once_per_input_set(
        self, decay_model, num_threads, n_inputs, caplog, capsys
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
        # The calling thread streams its first set live, so that set's
        # statistics come before every later trace
        assert stats[0] < starts[1]

    def test_debug_log_emitted_when_solve_fails(self, caplog):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        a = pybamm.InputParameter("a")
        model.rhs = {u: a * u**2}
        model.initial_conditions = {u: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)

        # More input sets than threads, so the worker thread buffers the sets it
        # takes and only the flush after the parallel region can emit them
        solver = pybamm.IDAKLUSolver(options={"num_threads": 2})
        inputs = [{"a": 1.0 + i} for i in range(4)]
        # u' = a u^2 blows up at t = 1/a, so integrating to t = 5 fails
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(pybamm.SolverError, match="IDA_ERR_FAIL"),
        ):
            solver.solve(model, np.array([0.0, 5.0]), inputs=inputs)

        # A partial solution is still returned, so every set is solved and the
        # worker's traces prove its buffer was drained
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

        # More input sets than threads, so the worker thread buffers its sets
        solver = pybamm.IDAKLUSolver(options={"num_threads": 2})
        inputs = [{"a": 1.0 + i} for i in range(4)]
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(pybamm.SolverError),
        ):
            solver.solve(model, np.array([0.0, 5.0]), inputs=inputs)

        # A failing set does not stop the others, and the worker's traces can
        # only come from the flush before the rethrow
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

    @staticmethod
    def _count_sets_started_after_interrupt(
        model, num_threads, n_inputs, caplog, monkeypatch
    ):
        _debug_logger = pybamm.logger.debug
        interrupted = []

        # Interrupts only once, so the flush after the parallel region still
        # emits the traces other threads buffered
        def logger_interrupts_on_first_step(msg, *args, **kwargs):
            _debug_logger(msg, *args, **kwargs)
            if not interrupted and isinstance(msg, str) and msg.startswith("Step "):
                interrupted.append(msg)
                raise KeyboardInterrupt

        monkeypatch.setattr(pybamm.logger, "debug", logger_interrupts_on_first_step)
        solver = pybamm.IDAKLUSolver(options={"num_threads": num_threads})
        inputs = [{"a": 1.0 + i} for i in range(n_inputs)]
        with (
            caplog.at_level(logging.DEBUG, logger=pybamm.logger.name),
            pytest.raises(KeyboardInterrupt),
        ):
            solver.solve(model, np.array([0.0, 5.0]), inputs=inputs)
        return sum(m.startswith("Integrating from t =") for m in caplog.messages)

    def test_solve_interrupted_stops_later_input_sets(
        self, decay_model, caplog, monkeypatch
    ):
        started = self._count_sets_started_after_interrupt(
            decay_model, 1, 5, caplog, monkeypatch
        )
        assert started == 1

    def test_solve_interrupted_stops_every_thread(
        self, decay_model, caplog, monkeypatch
    ):
        # The other thread may finish the set it holds, but takes no more
        n_inputs = 16
        started = self._count_sets_started_after_interrupt(
            decay_model, 2, n_inputs, caplog, monkeypatch
        )
        assert started < n_inputs

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

    def test_get_jacobian_sparsity(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
        solver.set_up(model)
        J = solver.get_jacobian_sparsity()

        assert J.shape == (2, 2)
        assert J.nnz > 0
        assert isinstance(J, csc_matrix)

    def test_get_jacobian_sparsity_not_set_up(self):
        solver = pybamm.IDAKLUSolver()
        with pytest.raises(pybamm.SolverError, match="Solver not set up"):
            solver.get_jacobian_sparsity()

    def test_spy(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 0.1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
        solver.set_up(model)

        import matplotlib

        matplotlib.use("Agg")
        ax = solver.spy(show_plot=False)
        assert ax is not None
        assert "nnz" in ax.get_title()

        import matplotlib.pyplot as plt

        _, existing_axes = plt.subplots(1, 2)
        ax = solver.spy(ax=existing_axes[1], show_plot=False)
        assert ax is existing_axes[1]
        assert "nnz" in ax.get_title()
        plt.close("all")

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


class TestIDAKLUAtol:
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

    @pytest.mark.parametrize(
        "atol",
        [
            1e-3,
            1,
            np.float32(1e-3),
            np.array(1e-3),
            [1e-3, 1e-3],
            (1e-3, 1e-3),
            np.full(2, 1e-3),
            np.full((2, 1), 1e-3),
        ],
    )
    def test_accepted_atol_widens_to_one_value_per_state(self, atol):
        model = self._two_state_model()
        widened = pybamm.IDAKLUSolver()._check_atol_type(atol, model)
        assert widened.dtype == np.float64
        assert widened.shape == (2,)
        np.testing.assert_allclose(widened, np.full(2, float(np.ravel(atol)[0])))

    @pytest.mark.parametrize(
        ("atol", "match"),
        [
            (True, r"must be a float, or a list, tuple or array"),
            ({"u": 1e-3}, r"must be a float, or a list, tuple or array"),
            ("1e-3", r"must be a float, or a list, tuple or array"),
            ([True, False], r"must be real numbers"),
            (["a", "b"], r"must be real numbers"),
            (np.array([1e-3 + 1j, 1e-3]), r"must be real numbers"),
            ([[1e-3], [1e-3, 1e-3]], r"must be a flat list"),
            ([1e-3, 1e-3, 1e-3], r"shape \(3,\) but \(2,\)"),
            (np.full((1, 2), 1e-3), r"shape \(1, 2\) but \(2,\)"),
            (np.full((2, 2), 1e-3), r"shape \(2, 2\) but \(2,\)"),
        ],
    )
    def test_invalid_atol_is_rejected(self, atol, match):
        model = self._two_state_model()
        with pytest.raises(pybamm.SolverError, match=match):
            pybamm.IDAKLUSolver()._check_atol_type(atol, model)

    @pytest.mark.parametrize("sequence", [list, tuple])
    def test_solve_with_a_sequence_atol(self, sequence):
        model = self._two_state_model()
        t_eval = np.linspace(0, 1, 5)
        expected = pybamm.IDAKLUSolver(atol=np.array([1e-3, 1e-9])).solve(model, t_eval)
        solution = pybamm.IDAKLUSolver(atol=sequence([1e-3, 1e-9])).solve(model, t_eval)
        assert solution.solver_statistics == expected.solver_statistics
        np.testing.assert_allclose(solution.y, expected.y, rtol=1e-12, atol=0)

    def test_per_state_atol_survives_a_config_round_trip(self):
        model = self._two_state_model()
        t_eval = np.linspace(0, 1, 5)
        solver = pybamm.IDAKLUSolver(atol=np.array([1e-3, 1e-9]))
        restored = pybamm.BaseSolver.from_config(solver.to_config())
        assert isinstance(restored.atol, list)
        expected = solver.solve(model, t_eval)
        solution = restored.solve(model, t_eval)
        assert solution.solver_statistics == expected.solver_statistics


def _van_der_pol_model():
    """Discretised stiff van der Pol oscillator."""
    model = pybamm.BaseModel()
    x = pybamm.Variable("x")
    y = pybamm.Variable("y")
    model.rhs = {x: y, y: 1000 * (1 - x**2) * y - x}
    model.initial_conditions = {x: 2, y: 0}
    model.variables = {"x": x}
    pybamm.Discretisation().process_model(model)
    return model


class TestIDAKLUSolverStatistics:
    def test_statistics_match_the_printed_statistics(self):
        # A cap of two Newton iterations makes every counter nonzero
        solver = pybamm.IDAKLUSolver(
            options={"print_stats": True, "max_nonlinear_iterations": 2}
        )
        printed = io.StringIO()
        with redirect_stdout(printed):
            solution = solver.solve(_van_der_pol_model(), [0, 3000])

        def printed_count(label):
            return int(re.search(rf"\t{label} = (\d+)", printed.getvalue())[1])

        stats = solution.solver_statistics
        assert stats == pybamm.SolverStatistics(
            number_of_steps=printed_count("Number of steps"),
            number_of_linear_solver_setups=printed_count(
                "Number of linear solver setup calls"
            ),
            number_of_nonlinear_solver_iterations=printed_count(
                "Number of nonlinear iterations performed"
            ),
            number_of_nonlinear_solver_fails=printed_count(
                "Number of nonlinear convergence failures"
            ),
            number_of_error_test_failures=printed_count(
                "Number of error test failures"
            ),
        )
        assert all(count > 0 for count in dataclasses.astuple(stats))
        assert stats.number_of_nonlinear_solver_iterations >= stats.number_of_steps

    def test_statistics_accumulate_across_a_breakpoint(self, decay_model):
        solver = pybamm.IDAKLUSolver()
        inputs = {"a": 50.0}
        first = solver.solve(decay_model, [0, 1], inputs=inputs).solver_statistics
        # The breakpoint at t = 1 reinitialises the integrator, which resets
        # its own counters; the tail after it takes far fewer steps than [0, 1]
        both = solver.solve(
            decay_model, np.array([0.0, 1.0, 2.0]), inputs=inputs
        ).solver_statistics
        assert both.number_of_steps > first.number_of_steps
        assert (
            both.number_of_nonlinear_solver_iterations
            > first.number_of_nonlinear_solver_iterations
        )

    def test_each_input_set_has_its_own_statistics(self, decay_model):
        rates = (0.1, 1.0, 10.0, 100.0)
        solutions = pybamm.IDAKLUSolver(options={"num_threads": 2}).solve(
            decay_model, [0, 1], inputs=[{"a": a} for a in rates]
        )
        alone = [
            pybamm.IDAKLUSolver().solve(decay_model, [0, 1], inputs={"a": a})
            for a in rates
        ]
        assert [s.solver_statistics for s in solutions] == [
            s.solver_statistics for s in alone
        ]
        assert len({s.solver_statistics.number_of_steps for s in solutions}) > 1

    def test_reused_solver_resets_statistics(self, decay_model):
        solver = pybamm.IDAKLUSolver()
        first = solver.solve(decay_model, [0, 1], inputs={"a": 10.0})
        second = solver.solve(decay_model, [0, 1], inputs={"a": 10.0})
        assert second.solver_statistics == first.solver_statistics

    def test_step_with_save_sums_statistics(self, decay_model):
        solver = pybamm.IDAKLUSolver()
        first = solver.step(None, decay_model, 1, inputs={"a": 10.0})
        second = solver.step(first, decay_model, 1, inputs={"a": 10.0}, save=False)
        combined = solver.step(first, decay_model, 1, inputs={"a": 10.0})
        assert combined.solver_statistics == (
            first.solver_statistics + second.solver_statistics
        )

    def test_experiment_solution_sums_its_steps(self):
        experiment = pybamm.Experiment(
            [
                (
                    "Discharge at 1C for 10 minutes",
                    "Rest for 5 minutes",
                    "Charge at 1C until 4.1 V",
                )
            ]
            * 2
        )
        solution = pybamm.Simulation(
            pybamm.lithium_ion.SPM(), experiment=experiment
        ).solve()
        steps = [step for cycle in solution.cycles for step in cycle.steps]
        assert all(step.solver_statistics is not None for step in steps)
        total = sum(
            (step.solver_statistics for step in steps), pybamm.SolverStatistics()
        )
        assert solution.solver_statistics == total
        assert solution.cycles[0].solver_statistics.number_of_steps > 0

    def test_reduced_solution_keeps_statistics(self, decay_model):
        solver = pybamm.IDAKLUSolver(options={"hermite_reduction_factor": 1.0})
        solution = solver.solve(decay_model, [0, 1], inputs={"a": 10.0})
        reduced = solver.reduce_solution(solution)
        assert reduced.solver_statistics == solution.solver_statistics

    @pytest.mark.parametrize(
        ("preconditioner", "uses_preconditioner"), [("none", False), ("BBDP", True)]
    )
    def test_iterative_linear_solver_statistics(
        self, preconditioner, uses_preconditioner
    ):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -pybamm.InputParameter("a") * u}
        model.algebraic = {v: v - u}
        model.initial_conditions = {u: 1, v: 1}
        model.variables = {"u": u}
        pybamm.Discretisation().process_model(model)
        # Output variables with sensitivities fill the model data that a BBD
        # counter read without a BBD preconditioner would alias
        solver = pybamm.IDAKLUSolver(
            output_variables=["u"],
            options={
                "linear_solver": "SUNLinSol_SPBCGS",
                "preconditioner": preconditioner,
                "print_stats": True,
            },
        )
        printed = io.StringIO()
        with redirect_stdout(printed):
            solution = solver.solve(
                model,
                np.linspace(0, 1, 3),
                inputs={"a": 0.1},
                calculate_sensitivities=True,
            )

        evaluations = re.search(
            r"residual function in preconditioner = (-?\d+)", printed.getvalue()
        )
        if uses_preconditioner:
            assert int(evaluations[1]) > 0
        else:
            assert int(evaluations[1]) == 0
        stats = solution.solver_statistics
        assert stats.number_of_steps > 0
        assert stats.number_of_nonlinear_solver_iterations >= stats.number_of_steps
