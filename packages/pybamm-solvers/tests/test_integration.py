"""Integration tests for pybammsolvers pybindings functionality.

These tests verify that the C++ bindings work correctly when called from Python
with realistic data and parameter configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

import pybammsolvers


def is_monotonic_increasing(arr):
    return np.all(np.diff(arr) >= 0)


class TestExponentialDecaySolver:
    """Integration tests using exponential decay model to test solver and Solution."""

    pytestmark = pytest.mark.integration

    def test_exponential_decay_solve(self, exponential_decay_solver):
        """
        Verify the solver can solve exponential decay and return a Solution object.

        Tests that the complete solver pipeline works: setup, solve, and return results.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]

        # Solve the system
        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)

        # Get the first solution object
        sol = solution[0]
        assert isinstance(sol, pybammsolvers.idaklu.solution)
        assert isinstance(sol.y, np.ndarray)
        assert isinstance(sol.y[0], np.floating)

    def test_solution_has_time_array(self, exponential_decay_solver):
        """
        Verify Solution object contains time array with correct values.

        Tests that the Solution.t attribute contains the evaluation times.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # Check time array exists and has correct properties
        assert hasattr(sol, "t")
        assert isinstance(sol.t, np.ndarray)
        assert sol.t.shape[0] == len(t_eval)

        # Verify times are increasing
        assert is_monotonic_increasing(sol.t)

        # Verify times are in expected range
        assert sol.t[0] >= t_eval[0]
        assert sol.t[-1] <= t_eval[-1]

    def test_solution_has_derivative_array(self, exponential_decay_solver):
        """
        Verify Solution object contains state derivative array.

        Tests that the Solution.yp attribute contains derivatives at each time point.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # Check derivative array exists
        assert hasattr(sol, "yp")
        assert isinstance(sol.yp, np.ndarray)

    def test_solution_has_termination_flag(self, exponential_decay_solver):
        """
        Verify Solution object contains termination flag.

        Tests that the Solution.flag attribute indicates successful completion.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # Check flag exists
        assert hasattr(sol, "flag")
        assert isinstance(sol.flag, int)

        # Flag should indicate success
        # IDA_SUCCESS=0, IDA_TSTOP_RETURN=1, IDA_ROOT_RETURN=2 are all success codes
        assert sol.flag in [0, 1, 2], f"Solver failed with flag {sol.flag}"

    def test_solution_accuracy_exponential_decay(self, exponential_decay_solver):
        """
        Verify Solution matches exact solution for exponential decay.

        Tests numerical accuracy by comparing solver output to known analytical solution.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]
        exact_solution = solver_data["model"]["exact_solution"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # Compare numerical solution to exact solution
        y_numerical = sol.y
        y_exact = exact_solution(sol.t)
        np.testing.assert_allclose(y_numerical, y_exact, rtol=1e-5, atol=1e-8)

    def test_solution_initial_conditions(self, exponential_decay_solver):
        """
        Verify Solution respects initial conditions.

        Tests that the first point in the solution matches the provided initial conditions.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]
        model_y0 = solver_data["model"]["y0"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # First state value should match initial condition
        assert sol.t[0] == pytest.approx(t_eval[0])
        np.testing.assert_allclose(sol.y[0], model_y0, rtol=1e-10)

    def test_solution_output_variables(self, exponential_decay_solver):
        """
        Verify Solution contains output variable evaluations.

        Tests that output variables (y_term) are computed correctly.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]
        exact_solution = solver_data["model"]["exact_solution"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]
        y_exact = exact_solution(sol.t)

        # For our model, the output variable is simply the final state slice
        np.testing.assert_allclose(sol.y_term, y_exact[-1], rtol=1e-5)

    def test_solution_dimensions_consistency(self, exponential_decay_solver):
        """
        Verify Solution arrays have consistent dimensions.

        Tests that t, y, and yp arrays have compatible shapes.
        """
        solver_data = exponential_decay_solver
        solver = solver_data["solver"]
        y0 = solver_data["y0"]
        yp0 = solver_data["yp0"]
        inputs = solver_data["inputs"]
        t_eval = solver_data["model"]["t_eval"]

        solution = solver.solve(t_eval, t_eval, y0, yp0, inputs)
        sol = solution[0]

        # All arrays should have compatible dimensions
        n_times = len(sol.t)
        assert len(sol.y) == n_times
        assert len(sol.yp) == 0  # hermite_interpolation == False
