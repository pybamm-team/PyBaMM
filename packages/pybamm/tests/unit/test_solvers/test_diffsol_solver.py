"""General ``DiffsolSolver`` behaviour on small models."""

import multiprocessing
import pickle
import re

import numpy as np
import pytest
from hypothesis import given

import pybamm
from pybamm.solvers.observation import _native_caches
from pybamm.solvers.variable_observer import NativeObserver
from tests.shared import get_broken_input_model as _broken_input_model
from tests.strategies import solve_settings
from tests.strategies.input_sweeps import decay_rate_sweeps


def _decay_model(with_event=False, with_input=False):
    """``dv/dt = -v`` (or ``-k*v``), so every value is analytically known."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    rate = -pybamm.InputParameter("k") * v if with_input else -v
    model.rhs = {v: rate}
    model.initial_conditions = {v: 1.0}
    model.variables = {"v": v, "2v": 2 * v}
    if with_event:
        model.events = [pybamm.Event("v = 0.5", v - 0.5)]
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


class TestDiffsolOutputsOnlyStates:
    """Outputs-only solves must still expose observable first and last states."""

    def _solve_outputs(self, model, t_eval):
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10, output_variables=["2v"])
        return solver.solve(model, t_eval)

    def test_last_state_carries_the_terminal_state(self):
        sol = self._solve_outputs(_decay_model(), np.linspace(0, 1, 5))
        assert sol.termination == "final time"
        assert _native_caches(sol) is not None
        last = sol.last_state
        np.testing.assert_allclose(last["v"].data, np.exp(-1.0), rtol=1e-8)
        np.testing.assert_allclose(last["2v"].data, 2 * np.exp(-1.0), rtol=1e-8)

    def test_first_state_rebuilds_from_initial_conditions(self):
        sol = self._solve_outputs(_decay_model(), np.linspace(0, 1, 5))
        np.testing.assert_allclose(sol.first_state["v"].data, 1.0, rtol=1e-10)

    def test_event_terminated_last_state_is_the_root_state(self):
        sol = self._solve_outputs(_decay_model(with_event=True), np.linspace(0, 2, 9))
        assert sol.termination.startswith("event")
        np.testing.assert_allclose(sol.t[-1], np.log(2.0), rtol=1e-6)
        np.testing.assert_allclose(sol.last_state["v"].data, 0.5, rtol=1e-6)

    def test_last_state_matches_the_full_state_solve(self):
        t_eval = np.linspace(0, 1, 5)
        sol_out = self._solve_outputs(_decay_model(), t_eval)
        sol_full = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10).solve(
            _decay_model(), t_eval
        )
        # Integrator state vs dense interpolation: each errs at the solve tolerance.
        for solution in (sol_out, sol_full):
            np.testing.assert_allclose(
                solution.last_state["v"].data, np.exp(-1.0), rtol=1e-9
            )
        np.testing.assert_allclose(
            sol_out.last_state["v"].data,
            sol_full.last_state["v"].data,
            rtol=1e-8,
        )

    def _solve_outputs_with_sensitivities(self, model, t_eval):
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10, output_variables=["2v"])
        return solver.solve(
            model, t_eval, inputs={"k": 0.7}, calculate_sensitivities=True
        )

    def test_first_and_last_state_carry_state_sensitivities(self):
        # v = exp(-k t), so dv/dk = -t exp(-k t)
        sol = self._solve_outputs_with_sensitivities(
            _decay_model(with_input=True), np.linspace(0, 1, 5)
        )
        np.testing.assert_allclose(
            sol.first_state["v"].sensitivities["k"], 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            sol.last_state["v"].sensitivities["k"], -np.exp(-0.7), rtol=1e-6
        )

    def test_event_terminated_last_state_carries_the_root_sensitivity(self):
        sol = self._solve_outputs_with_sensitivities(
            _decay_model(with_event=True, with_input=True), np.linspace(0, 2, 9)
        )
        assert sol.termination.startswith("event")
        t_root = np.log(2.0) / 0.7
        np.testing.assert_allclose(
            sol.last_state["v"].sensitivities["k"], -t_root * 0.5, rtol=1e-5
        )

    def test_a_sensitivity_solve_continues_from_an_outputs_only_step(self):
        model = _decay_model(with_input=True)
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10, output_variables=["2v"])
        kwargs = {"inputs": {"k": 0.7}, "calculate_sensitivities": True}
        sol = solver.step(None, model, 1.0, t_interp=np.linspace(0, 1, 5), **kwargs)
        sol = solver.step(sol, model, 1.0, t_interp=np.linspace(0, 1, 5), **kwargs)
        # 2v = 2 exp(-k t), so d(2v)/dk = -4 exp(-2 k) at t = 2
        np.testing.assert_allclose(
            sol["2v"].sensitivities["k"][-1], -4 * np.exp(-1.4), rtol=1e-6
        )


class TestDiffsolTInterp:
    def test_t_interp_merges_with_a_multi_point_t_eval(self):
        t_eval = np.array([0.0, 0.5, 1.0])
        t_interp = np.linspace(0.0, 1.0, 21)
        sol = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10).solve(
            _decay_model(), t_eval, t_interp=t_interp
        )
        np.testing.assert_array_equal(sol.t, np.union1d(t_eval, t_interp))
        np.testing.assert_allclose(sol["v"](t_interp), np.exp(-t_interp), rtol=1e-8)

    def test_empty_t_interp_keeps_t_eval(self):
        t_eval = np.linspace(0, 1, 5)
        sol = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10).solve(
            _decay_model(), t_eval, t_interp=np.array([])
        )
        np.testing.assert_array_equal(sol.t, t_eval)

    def test_bare_span_t_eval_densifies_the_output_grid(self):
        # A two-point solution would make every later sol[...](t) read a single chord.
        sol = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10).solve(_decay_model(), [0, 1])
        np.testing.assert_array_equal(sol.t, np.linspace(0.0, 1.0, 100))
        np.testing.assert_allclose(sol["v"](sol.t), np.exp(-sol.t), rtol=1e-8)

    def test_bare_span_off_grid_reads_are_not_a_chord(self):
        sol = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10).solve(_decay_model(), [0, 1])
        t_check = np.array([0.31, 0.5, 0.77])
        np.testing.assert_allclose(
            sol["v"](t_check), np.exp(-t_check), rtol=0, atol=2e-5
        )

    def test_outputs_only_rows_follow_the_merged_grid(self):
        t_eval = np.array([0.0, 1.0])
        t_interp = np.linspace(0.0, 1.0, 33)
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10, output_variables=["2v"])
        sol = solver.solve(_decay_model(), t_eval, t_interp=t_interp)
        np.testing.assert_allclose(
            sol["2v"](t_interp), 2.0 * np.exp(-t_interp), rtol=1e-8
        )


class TestDiffsolHermiteInterpolation:
    def _solve(self, model, t_eval, hermite=True, **solve_kwargs):
        solver = pybamm.DiffsolSolver(
            rtol=1e-10, atol=1e-10, hermite_interpolation=hermite
        )
        return solver.solve(model, t_eval, **solve_kwargs)

    def test_off_grid_reads_beat_the_chord(self):
        t_eval = np.linspace(0, 1, 6)
        t_check = t_eval[:-1] + 0.1  # knot-interval midpoints
        hermite = self._solve(_decay_model(), t_eval)
        linear = self._solve(_decay_model(), t_eval, hermite=False)
        hermite_error = np.max(np.abs(hermite["v"](t_check) - np.exp(-t_check)))
        linear_error = np.max(np.abs(linear["v"](t_check) - np.exp(-t_check)))
        assert hermite_error < 1e-5
        assert linear_error > 1e-3

    def test_yp_matches_the_analytic_derivative(self):
        sol = self._solve(_decay_model(), np.linspace(0, 1, 5))
        assert sol.hermite_interpolation
        # The t0 knot stores the consistent derivative; later knots are ~1e-6 off.
        np.testing.assert_allclose(sol.yp[0], -np.exp(-sol.t), rtol=1e-5)
        np.testing.assert_allclose(sol.yp[0][0], -1.0, rtol=1e-12)

    def test_disabling_hermite_reads_through_the_native_observer(self):
        sol = self._solve(_decay_model(), np.linspace(0, 1, 5), hermite=False)
        assert not sol.hermite_interpolation
        assert sol.all_yps is None
        assert isinstance(sol["v"]._observer, NativeObserver)
        np.testing.assert_allclose(sol["v"].entries, np.exp(-sol.t), rtol=1e-8)

    def test_a_diffsol_solve_leaves_how_an_idaklu_solution_is_read(self):
        model = _decay_model()
        idaklu = pybamm.IDAKLUSolver(options={"hermite_interpolation": False})
        earlier = idaklu.solve(model, np.linspace(0, 1, 5))
        before = earlier["v"]

        self._solve(model, np.linspace(0, 1, 5), hermite=False)
        earlier.update("v")

        assert type(earlier["v"]) is type(before)
        np.testing.assert_array_equal(earlier["v"].entries, before.entries)

    @pytest.mark.parametrize(
        ("points", "stored"), [(4096, True), (4097, False)], ids=["limit", "past"]
    )
    def test_a_dense_output_grid_gives_up_the_derivatives(self, points, stored):
        # Past the limit the chord is already at the integration error floor.
        sol = self._solve(_decay_model(), np.linspace(0, 1, points))
        assert sol.hermite_interpolation is stored
        assert (sol.all_yps is not None) is stored
        np.testing.assert_allclose(sol["v"].data, np.exp(-sol.t), rtol=1e-7)

    def test_a_dense_grid_reads_off_grid_to_the_solver_tolerance(self):
        t_check = np.linspace(0.05, 0.95, 41)
        sol = self._solve(_decay_model(), np.linspace(0, 1, 8193))
        assert not sol.hermite_interpolation
        np.testing.assert_allclose(sol["v"](t_check), np.exp(-t_check), atol=1e-8)

    def test_hermite_observation_is_the_native_processed_variable(self):
        sol = self._solve(_decay_model(), np.linspace(0, 1, 5))
        assert not isinstance(sol["v"], pybamm.ProcessedVariableComputed)
        np.testing.assert_allclose(sol["v"].data, np.exp(-sol.t), rtol=1e-8)

    def test_outputs_only_solves_store_no_derivatives(self):
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10, output_variables=["2v"])
        sol = solver.solve(_decay_model(), np.linspace(0, 1, 5))
        assert sol.all_yps is None
        assert _native_caches(sol) is not None
        assert isinstance(sol.first_state["v"]._observer, NativeObserver)

    def test_the_event_root_column_carries_the_wound_back_slope(self):
        # Coarse grid, so the root at ln 2 lands mid-interval.
        sol = self._solve(_decay_model(with_event=True), np.linspace(0, 2, 5))
        assert sol.termination.startswith("event")
        np.testing.assert_allclose(sol.yp[0, -1], -0.5, rtol=1e-6)
        t_check = np.array([0.6, 0.69])
        np.testing.assert_allclose(
            sol["v"](t_check), np.exp(-t_check), rtol=0, atol=1e-5
        )

    def test_sensitivities_survive_the_native_hermite_path(self):
        sol = self._solve(
            _decay_model(with_input=True),
            np.linspace(0, 1, 9),
            inputs={"k": 0.5},
            calculate_sensitivities=True,
        )
        t = sol.t
        np.testing.assert_allclose(
            sol["v"].sensitivities["k"],
            -t * np.exp(-0.5 * t),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_stepped_solve_keeps_hermite_across_segments(self):
        model = _decay_model()
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10)
        sol = None
        for _ in range(2):
            sol = solver.step(sol, model, dt=0.5, npts=6)
        assert sol.hermite_interpolation
        t_check = np.array([0.25, 0.75])
        np.testing.assert_allclose(
            sol["v"](t_check), np.exp(-t_check), rtol=0, atol=1e-6
        )

    def test_a_time_discontinuity_corner_is_not_smoothed(self):
        # Each side of the corner's ULP bracket must use its own branch's slope.
        solver = pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8)
        sol = solver.solve(_ramp_then_hold_model(), np.linspace(0, 10, 11))
        t_check = np.array([4.75, 5.25])
        np.testing.assert_allclose(
            sol["v"](t_check), np.array([4.75, 5.0]), rtol=0, atol=1e-6
        )


class TestDiffsolStep:
    def test_stepped_solve_stays_observable_across_segments(self):
        model = _decay_model()
        solver = pybamm.DiffsolSolver(rtol=1e-10, atol=1e-10)
        sol = None
        for _ in range(2):
            sol = solver.step(sol, model, dt=0.5, npts=6)

        assert len(sol.sub_solutions) == 2
        assert _native_caches(sol) is not None
        # One on-grid point per segment, so grid interpolation adds no error.
        t_check = np.array([0.2, 0.7])
        np.testing.assert_allclose(sol["v"](t_check), np.exp(-t_check), rtol=1e-6)
        np.testing.assert_allclose(sol.last_state["v"].data, np.exp(-1.0), rtol=1e-6)


class TestDiffsolFailureHandling:
    def test_integration_failure_raises_solver_error(self):
        # dv/dt = v^2 with v(0) = 1 blows up at t = 1.
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: v**2}
        model.initial_conditions = {v: 1.0}
        model.variables = {"v": v}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)

        with pytest.raises(pybamm.SolverError, match=r"diffsol error"):
            pybamm.DiffsolSolver().solve(model, np.linspace(0, 2, 10))

    def test_a_failed_sensitivity_retry_is_a_solver_error(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: pybamm.InputParameter("k") * v**2}
        model.initial_conditions = {v: 1.0}
        model.variables = {"v": v}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)

        with pytest.raises(
            pybamm.SolverError, match=r"^input set 0: sensitivity solve failed"
        ):
            pybamm.DiffsolSolver().solve(
                model,
                np.linspace(0, 2, 10),
                inputs={"k": 1.0},
                calculate_sensitivities=["k"],
            )


def _algebraic_model(z_guess, event=None):
    """``dv/dt = -v`` with ``0 = 2v - z - z^3``, consistent at ``z = 1``."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    z = pybamm.Variable("z")
    model.rhs = {v: -v}
    model.algebraic = {z: 2 * v - z - z**3}
    model.initial_conditions = {v: 1.0, z: z_guess}
    model.variables = {"v": v, "z": z}
    if event is not None:
        name, expression = event
        model.events = [pybamm.Event(name, expression(z))]
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


def _exact_z(t):
    """The real root of ``z^3 + z = 2 exp(-t)``, by Cardano."""
    q = np.exp(-np.asarray(t))
    root = np.sqrt(q**2 + 1 / 27)
    return np.cbrt(q + root) + np.cbrt(q - root)


class TestDiffsolInitialConditions:
    """Events and stored derivatives use the consistent state, not the guess."""

    def test_native_initialisation_is_the_default(self):
        model = _algebraic_model(5.0)
        native = pybamm.DiffsolSolver()
        python = pybamm.DiffsolSolver(calc_ic=False)
        native.solve(model, [0, 1])
        python.solve(_algebraic_model(5.0), [0, 1])

        assert native._internal_initialisation
        assert native.root_method is None
        assert not python._internal_initialisation
        assert isinstance(python.root_method, pybamm.NonlinearSolver)

    @pytest.mark.parametrize("calc_ic", [True, False])
    def test_both_initialisations_reach_the_consistent_state(self, calc_ic):
        t_eval = np.linspace(0, 1, 6)
        solver = pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8, calc_ic=calc_ic)
        sol = solver.solve(_algebraic_model(5.0), t_eval)
        np.testing.assert_allclose(sol["z"](t_eval), _exact_z(t_eval), rtol=1e-6)

    @pytest.mark.parametrize("root_method", ["lm", "hybr"])
    def test_a_generic_root_method_reaches_the_consistent_state(self, root_method):
        t_eval = np.linspace(0, 1, 6)
        solver = pybamm.DiffsolSolver(
            rtol=1e-8, atol=1e-8, calc_ic=False, root_method=root_method
        )
        sol = solver.solve(_algebraic_model(5.0), t_eval)
        np.testing.assert_allclose(sol["z"](t_eval), _exact_z(t_eval), rtol=1e-6)

    @pytest.mark.parametrize("calc_ic", [True, False])
    def test_off_grid_reads_in_the_first_interval_follow_the_solution(self, calc_ic):
        t_eval = np.linspace(0, 1, 6)
        t_check = np.linspace(0.02, 0.18, 9)
        solver = pybamm.DiffsolSolver(calc_ic=calc_ic)
        sol = solver.solve(_algebraic_model(5.0), t_eval)
        np.testing.assert_allclose(sol["z"](t_check), _exact_z(t_check), atol=2e-5)
        # dz/dt = 2 dv/dt / (1 + 3 z^2) = -0.5 at t = 0
        np.testing.assert_allclose(sol.yp[1, 0], -0.5, rtol=1e-6)

    @pytest.mark.parametrize("output_variables", [None, ["z"]])
    def test_a_guess_on_an_event_boundary_does_not_trip_it(self, output_variables):
        # The guess z = 0.5 sits exactly on the event; the consistent z = 1 does not.
        model = _algebraic_model(0.5, event=("z above 0.5", lambda z: z - 0.5))
        solver = pybamm.DiffsolSolver(output_variables=output_variables)
        sol = solver.solve(model, np.linspace(0, 0.1, 3))
        assert sol.termination == "final time"

    @pytest.mark.parametrize("calc_ic", [True, False])
    @pytest.mark.parametrize("output_variables", [None, ["z"]])
    def test_an_event_violated_at_the_consistent_state_raises(
        self, calc_ic, output_variables
    ):
        # The guess z = 0.5 clears the event; the consistent z = 1 violates it.
        model = _algebraic_model(0.5, event=("z below 0.9", lambda z: 0.9 - z))
        solver = pybamm.DiffsolSolver(
            calc_ic=calc_ic, output_variables=output_variables
        )
        with pytest.raises(
            pybamm.SolverError,
            match=re.escape("Events ['z below 0.9'] are non-positive at initial"),
        ):
            solver.solve(model, np.linspace(0, 1, 6))

    @pytest.mark.parametrize("output_variables", [None, ["z"]])
    def test_a_violation_only_diffsol_sees_raises_without_calc_ic(
        self, monkeypatch, output_variables
    ):
        # Stands in for a root solve that clears the event by less than diffsol's
        # own refinement then moves the state, so only the solve can name it.
        model = _algebraic_model(0.5, event=("z below 0.9", lambda z: 0.9 - z))
        solver = pybamm.DiffsolSolver(calc_ic=False, output_variables=output_variables)
        monkeypatch.setattr(
            solver, "_check_event_violation_on_initialisation", lambda *_: None
        )
        with pytest.raises(
            pybamm.SolverError,
            match=re.escape("Events ['z below 0.9'] are non-positive at initial"),
        ):
            solver.solve(model, np.linspace(0, 1, 6))

    def test_outputs_only_first_state_is_the_consistent_state(self):
        solver = pybamm.DiffsolSolver(output_variables=["z"])
        sol = solver.solve(_algebraic_model(5.0), np.linspace(0, 1, 6))
        np.testing.assert_allclose(sol.first_state["z"](0), 1.0, rtol=1e-6)
        np.testing.assert_allclose(sol.first_state["v"](0), 1.0, rtol=1e-12)

    @pytest.mark.parametrize("sensitivities", [False, ["k"]])
    def test_time_dependent_constraint_has_the_analytic_initial_slope(
        self, sensitivities
    ):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        z = pybamm.Variable("z")
        model.rhs = {v: -pybamm.InputParameter("k") * v}
        # dz/dt = 1000 at t0 comes only from the constraint's time derivative
        model.algebraic = {z: z - 1000 * pybamm.t * (1 + 1000 * pybamm.t)}
        model.initial_conditions = {v: 1.0, z: 0.0}
        model.variables = {"v": v, "z": z}

        solution = pybamm.DiffsolSolver(rtol=1e-9, atol=1e-9).solve(
            model,
            np.array([0.0, 0.001, 0.01]),
            inputs={"k": 1.0},
            calculate_sensitivities=sensitivities,
        )

        np.testing.assert_allclose(solution.yp[1, 0], 1000.0, rtol=1e-12)
        np.testing.assert_allclose(solution["z"](0.0005), 0.75, rtol=1e-9, atol=1e-9)


class TestDiffsolPickling:
    def test_a_solved_solver_pickles_and_solves_again(self):
        t_eval = np.linspace(0, 1, 10)
        solver = pybamm.DiffsolSolver()
        first = solver.solve(_decay_model(), t_eval)

        restored = pickle.loads(pickle.dumps(solver))
        second = restored.solve(_decay_model(), t_eval)

        np.testing.assert_array_equal(second.y, first.y)


class TestDiffsolSolverOptions:
    def test_defaults_are_used_when_no_options_are_given(self):
        solver = pybamm.DiffsolSolver()
        assert solver._options == pybamm.DiffsolSolver.DEFAULT_OPTIONS

    def test_an_override_leaves_the_other_defaults_alone(self):
        solver = pybamm.DiffsolSolver(options={"max_error_test_failures": 7})
        assert solver._options["max_error_test_failures"] == 7
        untouched = set(pybamm.DiffsolSolver.DEFAULT_OPTIONS) - {
            "max_error_test_failures"
        }
        for key in untouched:
            assert solver._options[key] == pybamm.DiffsolSolver.DEFAULT_OPTIONS[key]

    def test_defaults_match_the_rust_defaults(self):
        # Catches silent drift in diffsol's OdeSolverOptions defaults on a bump.
        from pybamm.rust import default_solver_options

        assert pybamm.DiffsolSolver._INTEGRATOR_DEFAULTS == default_solver_options()

    def test_an_unknown_option_is_rejected(self):
        # A typo would otherwise be dropped silently, leaving the default.
        with pytest.raises(pybamm.SolverError, match=r"Unknown diffsol solver option"):
            pybamm.DiffsolSolver(options={"max_error_test_failure": 7})

    def test_the_failure_budget_default_is_not_diffsols_own(self):
        # diffsol counts failures cumulatively, so its own 50 caps solve length.
        assert (
            pybamm.DiffsolSolver.DEFAULT_OPTIONS["max_nonlinear_solver_failures"]
            > 10000
        )

    def test_options_reach_the_integrator(self):
        model = _decay_model()
        starved = pybamm.DiffsolSolver(
            options={"max_nonlinear_solver_failures": 0, "min_timestep": 0.5}
        )
        with pytest.raises(pybamm.SolverError, match=r"diffsol error"):
            starved.solve(model, np.linspace(0, 1, 5))

        assert pybamm.DiffsolSolver().solve(model, np.linspace(0, 1, 5)) is not None

    def test_an_out_of_range_option_is_rejected_by_the_solver(self):
        model = _decay_model()
        solver = pybamm.DiffsolSolver(options={"nonlinear_solver_tolerance": -1.0})
        with pytest.raises(ValueError, match=r"nonlinear_solver_tolerance"):
            solver.solve(model, np.linspace(0, 1, 5))


def _solve_both_ways(inputs_list, num_threads, t_eval, with_event=True):
    """The same sweep solved at ``num_threads`` and serially."""
    parallel = pybamm.DiffsolSolver(
        rtol=1e-8, atol=1e-10, options={"num_threads": num_threads}
    ).solve(
        _decay_model(with_event=with_event, with_input=True), t_eval, inputs=inputs_list
    )
    serial = pybamm.DiffsolSolver(rtol=1e-8, atol=1e-10).solve(
        _decay_model(with_event=with_event, with_input=True), t_eval, inputs=inputs_list
    )
    return parallel, serial


def _assert_sweeps_identical(parallel, serial):
    """Bit-identical, not close: the batch fans out over the same entry point."""
    for i, (got, want) in enumerate(zip(parallel, serial, strict=True)):
        np.testing.assert_array_equal(got.t, want.t, err_msg=f"set {i} times")
        np.testing.assert_array_equal(got.y, want.y, err_msg=f"set {i} states")
        assert got.termination == want.termination, f"set {i} termination"
        assert (
            got.solver_statistics.number_of_steps
            == want.solver_statistics.number_of_steps
        ), f"set {i} step count"


class TestDiffsolNumThreads:
    """``num_threads`` solves that many input sets at once, or one set's LU with them."""

    @staticmethod
    def _sweep(n):
        return [{"k": 0.3 * (i + 1)} for i in range(n)]

    @pytest.mark.parametrize("num_threads", [1, 2, 8])
    @pytest.mark.parametrize("with_event", [False, True])
    def test_a_parallel_sweep_is_identical_to_the_serial_one(
        self, num_threads, with_event
    ):
        parallel, serial = _solve_both_ways(
            self._sweep(6), num_threads, np.linspace(0, 3, 40), with_event=with_event
        )
        _assert_sweeps_identical(parallel, serial)

    def test_results_follow_input_order_not_completion_order(self):
        # Descending solve cost, so returning completion order would reverse it.
        t_eval = np.linspace(0, 3, 40)
        inputs_list = [{"k": k} for k in (0.1, 0.5, 2.0, 9.0)]
        solutions = pybamm.DiffsolSolver(
            rtol=1e-10, atol=1e-12, options={"num_threads": 4}
        ).solve(_decay_model(with_input=True), t_eval, inputs=inputs_list)

        for inputs, solution in zip(inputs_list, solutions, strict=True):
            assert solution.all_inputs[0] == inputs
            np.testing.assert_allclose(
                solution["v"](t_eval),
                np.exp(-inputs["k"] * t_eval),
                rtol=1e-6,
                atol=1e-12,
            )

    @pytest.mark.parametrize("num_threads", [1, 4])
    def test_every_failing_set_is_named_whether_batched_or_serial(self, num_threads):
        model = _broken_input_model()
        inputs_list = [{"k": k} for k in (1.0, -1.0, 2.0, -2.0, 3.0)]
        solver = pybamm.DiffsolSolver(options={"num_threads": num_threads})
        with pytest.raises(pybamm.SolverError) as excinfo:
            solver.solve(model, np.linspace(0, 1, 10), inputs=inputs_list)
        message = str(excinfo.value)
        assert re.findall(r"input set (\d+): diffsol error", message) == ["1", "3"]

    def test_a_single_input_set_failure_is_named(self):
        with pytest.raises(pybamm.SolverError, match=r"^input set 0: diffsol error"):
            pybamm.DiffsolSolver().solve(
                _broken_input_model(), np.linspace(0, 1, 10), inputs=[{"k": -1.0}]
            )

    def test_one_pool_is_shared_by_every_solver_of_the_same_width(self):
        from pybamm.rust._core import _pool_ids

        t_eval = np.linspace(0, 1, 10)
        inputs_list = self._sweep(8)
        first = pybamm.DiffsolSolver(options={"num_threads": 8})
        first.solve(_decay_model(with_input=True), t_eval, inputs=inputs_list)
        pool_id = _pool_ids()[8]

        second = pybamm.DiffsolSolver(options={"num_threads": 8})
        second.solve(_decay_model(with_input=True), t_eval, inputs=inputs_list)
        assert _pool_ids()[8] == pool_id

    def test_a_single_set_factorises_on_the_shared_pool_unchanged(self):
        from pybamm.rust._core import _pool_ids

        # A width no other test uses, so the pool is this solve's.
        parallel, serial = _solve_both_ways({"k": 0.6}, 5, np.linspace(0, 3, 40))
        _assert_sweeps_identical([parallel], [serial])
        assert 5 in _pool_ids()

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
    )
    @pytest.mark.filterwarnings("ignore:This process:DeprecationWarning")
    def test_a_forked_child_builds_its_own_pool(self):
        t_eval = np.linspace(0, 1, 10)
        inputs_list = self._sweep(4)

        def solve():
            pybamm.DiffsolSolver(options={"num_threads": 3}).solve(
                _decay_model(with_input=True), t_eval, inputs=inputs_list
            )

        # Solve first, so the child inherits the parent's pool and not its workers
        solve()
        child = multiprocessing.get_context("fork").Process(target=solve)
        child.start()
        child.join(timeout=60)
        if child.is_alive():
            child.kill()
            child.join()
            pytest.fail("the forked child hung on the pool it inherited")
        assert child.exitcode == 0

    def test_the_default_builds_no_pool_at_all(self):
        from pybamm.rust._core import _pool_ids

        pybamm.DiffsolSolver().solve(
            _decay_model(with_input=True), np.linspace(0, 1, 10), inputs=self._sweep(4)
        )
        assert 1 not in _pool_ids()

    def test_pools_are_keyed_on_the_configured_width_not_the_sweep(self):
        from pybamm.rust._core import _pool_ids

        # The cache is process-wide, so only the keys this solver adds are ours.
        before = set(_pool_ids())
        model = _decay_model(with_input=True)
        solver = pybamm.DiffsolSolver(options={"num_threads": 7})
        for n_sets in (2, 3):
            solver.solve(model, np.linspace(0, 1, 10), inputs=self._sweep(n_sets))
        added = set(_pool_ids()) - before
        assert 7 in _pool_ids()
        assert added.isdisjoint({2, 3})

    @pytest.mark.parametrize("num_threads", [0, -1, 2.5, "4", True, None])
    def test_a_num_threads_that_is_not_a_count_is_rejected(self, num_threads):
        with pytest.raises(pybamm.SolverError, match=r"num_threads must be an integer"):
            pybamm.DiffsolSolver(options={"num_threads": num_threads})

    def test_num_threads_does_not_reach_the_integrator_options(self):
        solver = pybamm.DiffsolSolver(options={"num_threads": 4})
        assert "num_threads" not in solver._integrator_options()
        assert set(solver._integrator_options()) == set(
            pybamm.DiffsolSolver._INTEGRATOR_DEFAULTS
        )


@solve_settings
@given(rates=decay_rate_sweeps())
def test_a_batch_is_indistinguishable_from_the_serial_loop(rates):
    parallel, serial = _solve_both_ways(
        [{"k": rate} for rate in rates], 4, np.linspace(0, 3, 40)
    )
    _assert_sweeps_identical(parallel, serial)


class TestDiffsolIntegrationTime:
    def test_each_set_gets_its_own_integration_time(self):
        # Batched wall clocks overlap, so a Python timer would report the batch time.
        inputs_list = [{"k": 0.3 * (i + 1)} for i in range(4)]
        solutions = pybamm.DiffsolSolver(options={"num_threads": 4}).solve(
            _decay_model(with_input=True), np.linspace(0, 2, 30), inputs=inputs_list
        )
        times = [solution.integration_time for solution in solutions]
        assert all(time > 0 for time in times)
        assert len(set(times)) > 1

    def test_the_reported_time_covers_setup_and_integration(self):
        solution = pybamm.DiffsolSolver().solve(_decay_model(), np.linspace(0, 2, 30))
        statistics = solution.solver_statistics
        assert solution.integration_time >= (
            statistics.ic_time_secs + statistics.solver_setup_time_secs
        )


class TestDiffsolSensitivityConfigErrors:
    def test_unknown_sensitivity_parameter_is_a_single_clear_error(self):
        model = _decay_model(with_input=True)
        with pytest.raises(ValueError, match=r"no sensitivity parameters") as excinfo:
            pybamm.DiffsolSolver().solve(
                model,
                np.linspace(0, 1, 5),
                inputs={"k": 1.0},
                calculate_sensitivities=["not_an_input"],
            )
        # A config error must not go through the relaxed retry and be double-reported.
        assert "retry" not in str(excinfo.value)


def _two_scale_model():
    """``du/dt = -u`` and ``dw/dt = -w`` with ``u`` at O(1) and ``w`` at O(1e4)."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    w = pybamm.Variable("w")
    model.rhs = {u: -u, w: -w}
    model.initial_conditions = {u: 1.0, w: 1e4}
    model.variables = {"u": u, "w": w}
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


@pytest.mark.parametrize("solver_class", [pybamm.DiffsolSolver, pybamm.IDAKLUSolver])
class TestPerStateAtol:
    def _steps(self, solver_class, atol):
        solution = solver_class(rtol=1e-6, atol=atol).solve(
            _two_scale_model(), np.linspace(0, 1, 5)
        )
        return solution.solver_statistics.number_of_steps

    def test_a_uniform_array_matches_the_scalar(self, solver_class):
        t_eval = np.linspace(0, 1, 5)
        scalar = solver_class(rtol=1e-8, atol=1e-8).solve(_two_scale_model(), t_eval)
        array = solver_class(rtol=1e-8, atol=np.full(2, 1e-8)).solve(
            _two_scale_model(), t_eval
        )
        np.testing.assert_array_equal(array["u"](t_eval), scalar["u"](t_eval))
        np.testing.assert_array_equal(array["w"](t_eval), scalar["w"](t_eval))

    def test_every_entry_reaches_its_own_state(self, solver_class):
        loose = self._steps(solver_class, np.full(2, 1e-1))
        assert self._steps(solver_class, 1e-1) == loose
        assert self._steps(solver_class, np.array([1e-12, 1e-1])) > loose
        assert self._steps(solver_class, np.array([1e-1, 1e-12])) > loose

    def test_a_per_state_atol_survives_a_dae_solve(self, solver_class):
        # The initial-condition root solver takes a scalar tolerance, not the array.
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        z = pybamm.Variable("z")
        model.rhs = {u: -u}
        model.algebraic = {z: z - 2 * u}
        model.initial_conditions = {u: 1.0, z: 2.0}
        model.variables = {"u": u, "z": z}
        model.convert_to_format = "rust"
        pybamm.Discretisation().process_model(model)

        t_eval = np.linspace(0, 1, 5)
        solution = solver_class(rtol=1e-8, atol=np.array([1e-9, 1e-7])).solve(
            model, t_eval
        )
        np.testing.assert_allclose(
            solution["z"](t_eval), 2 * np.exp(-t_eval), rtol=1e-6
        )

    def test_a_config_round_trip_keeps_the_per_state_atol(self, solver_class):
        # to_config puts the array through JSON, which has no arrays to put back.
        atol = np.array([1e-1, 1e-12])
        restored = pybamm.BaseSolver.from_config(
            solver_class(rtol=1e-6, atol=atol).to_config()
        )
        assert self._steps(solver_class, restored.atol) == self._steps(
            solver_class, atol
        )

    def test_a_wrong_width_atol_is_rejected(self, solver_class):
        solver = solver_class(atol=np.full(3, 1e-6))
        with pytest.raises(pybamm.SolverError, match=r"shape \(3,\) but \(2,\)"):
            solver.solve(_two_scale_model(), np.linspace(0, 1, 5))

    @pytest.mark.parametrize(
        ("bad_atol", "message"),
        [
            ("tight", r"must be a float, or a list, tuple or array"),
            (["tight", "loose"], r"must be real numbers"),
            (np.array(["tight", "loose"]), r"must be real numbers"),
            ([True, 1e-6], r"not booleans"),
        ],
    )
    def test_a_non_numeric_atol_is_rejected(self, solver_class, bad_atol, message):
        solver = solver_class(atol=bad_atol)
        with pytest.raises(pybamm.SolverError, match=message):
            solver.solve(_two_scale_model(), np.linspace(0, 1, 5))


def _ramp_then_hold_model():
    """``dv/dt = 1`` until t = 5 and 0 after, so ``v(t) = min(t, 5)``."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    model.rhs = {v: pybamm.t < 5}
    model.initial_conditions = {v: pybamm.Scalar(0)}
    model.variables = {"v": v}
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


def _algebraic_jump_model(rate=0.1, terminate=False):
    """``dv/dt = -rate v`` with ``z = v`` until t = 0.5 and ``z = -v`` after."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    z = pybamm.Variable("z")
    model.rhs = {v: -rate * v}
    model.algebraic = {z: z - v * (1 - 2 * (pybamm.t > 0.5))}
    model.initial_conditions = {v: 1.0, z: 1.0}
    model.variables = {"v": v, "z": z}
    if terminate:
        model.events = [pybamm.Event("positive z", z)]
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


def _switch_at_the_end_model(switch):
    """``dv/dt = -v/10`` with ``z = v * switch(t)``, for a switch that flips at the end."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    z = pybamm.Variable("z")
    model.rhs = {v: -0.1 * v}
    model.algebraic = {z: z - v * switch(pybamm.t)}
    model.initial_conditions = {v: 1.0, z: 1.0}
    model.variables = {"v": v, "z": z}
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


def _jump_just_after_start_model():
    """``dv/dt = -k v`` with ``z`` stepping from 0 to 1 at t = 1e-8."""
    model = pybamm.BaseModel()
    v = pybamm.Variable("v")
    z = pybamm.Variable("z")
    model.rhs = {v: -pybamm.InputParameter("k") * v}
    model.algebraic = {z: z - (pybamm.t > 1e-8)}
    model.initial_conditions = {v: 1.0, z: 0.0}
    model.variables = {"v": v, "z": z}
    model.convert_to_format = "rust"
    pybamm.Discretisation().process_model(model)
    return model


class TestDiffsolTimeDiscontinuities:
    """A constant time discontinuity reaches diffsol as a stop time, as for IDAKLU."""

    @pytest.mark.parametrize(
        "solver_class", [pybamm.DiffsolSolver, pybamm.IDAKLUSolver]
    )
    def test_a_heaviside_in_time_is_solved_in_one_pass(self, solver_class):
        solver = solver_class(rtol=1e-8, atol=1e-8)
        solution = solver.solve(_ramp_then_hold_model(), np.linspace(0, 10, 21))

        assert len(solution.sub_solutions) == 1
        np.testing.assert_allclose(
            solution["v"](np.array([2.5, 5.0, 7.5])),
            np.array([2.5, 5.0, 5.0]),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "solver_class", [pybamm.DiffsolSolver, pybamm.IDAKLUSolver]
    )
    def test_an_algebraic_jump_restarts_on_the_new_branch(self, solver_class):
        solver = solver_class(rtol=1e-8, atol=1e-8)
        solution = solver.solve(_algebraic_jump_model(), np.linspace(0, 2, 9))

        t_check = np.array([0.4, 0.5, 0.6, 1.0, 2.0])
        v = np.exp(-0.1 * t_check)
        np.testing.assert_allclose(solution["v"](t_check), v, rtol=1e-6)
        np.testing.assert_allclose(
            solution["z"](t_check), np.where(t_check > 0.5, -v, v), rtol=1e-6
        )

    @pytest.mark.parametrize("output_variables", [None, ["z"]])
    @pytest.mark.parametrize("sensitivities", [False, ["k"]])
    def test_an_algebraic_jump_terminates_when_it_exceeds_an_event(
        self, output_variables, sensitivities
    ):
        model = _algebraic_jump_model(rate=pybamm.InputParameter("k"), terminate=True)
        solution = pybamm.DiffsolSolver(output_variables=output_variables).solve(
            model,
            np.linspace(0, 2, 9),
            inputs={"k": 0.1},
            calculate_sensitivities=sensitivities,
        )

        assert solution.termination == "event: positive z"
        assert solution.t[-1] > 0.5
        np.testing.assert_allclose(solution.t[-1], 0.5, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            solution["z"](solution.t[-1]), -np.exp(-0.05), rtol=1e-5
        )
        np.testing.assert_allclose(
            solution.last_state["v"](solution.t[-1]), np.exp(-0.05), rtol=1e-5
        )

    @pytest.mark.parametrize(
        ("switch", "t_end"),
        [
            (lambda t: 1 - 2 * (t >= 0.5), 0.5),
            # A sawtooth drive whose span is a whole number of periods.
            (lambda t: 1 + t % 0.25, 1.0),
        ],
        ids=["heaviside", "sawtooth"],
    )
    def test_a_switch_at_the_final_time_restarts_on_its_far_side(self, switch, t_end):
        # IDAKLU reports the pre-switch value at t_end instead.
        solution = pybamm.DiffsolSolver(rtol=1e-8, atol=1e-8).solve(
            _switch_at_the_end_model(switch), [0, t_end]
        )

        t_check = np.array([0.1, t_end])
        v = np.exp(-0.1 * t_check)
        np.testing.assert_allclose(solution["v"](t_check), v, rtol=1e-6)
        np.testing.assert_allclose(
            solution["z"](t_check), v * [switch(t) for t in t_check], rtol=1e-6
        )

    @pytest.mark.parametrize("sensitivities", [False, ["k"]])
    def test_a_jump_just_after_the_start_stays_out_of_its_slope(self, sensitivities):
        solution = pybamm.DiffsolSolver().solve(
            _jump_just_after_start_model(),
            np.linspace(0, 1, 5),
            inputs={"k": 1.0},
            calculate_sensitivities=sensitivities,
        )

        np.testing.assert_allclose(solution.yp[1, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            solution["z"](np.array([0.0, 2.5e-9, 5e-9, 0.5, 1.0])),
            [0.0, 0.0, 0.0, 1.0, 1.0],
            atol=1e-12,
        )


def _native_prepared_solver(with_sens=False):
    """``dy/dt = -k*y`` natively, with two outputs against its single state."""
    from pybamm.rust import CompiledModel, ExprGraph, PreparedSolver

    g = ExprGraph()
    y = g.state_vector(0, 1)
    rate = g.mul(g.mul(g.scalar(-1.0), g.input_parameter("k")), y)
    model = CompiledModel.from_expr(
        g,
        rate,
        np.ones(1),
        np.arange(2, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        n_inputs=1,
        sens_param_indices=[0] if with_sens else [],
        output_exprs=[g.mul(g.scalar(2.0), y), g.mul(g.scalar(3.0), y)],
        event_exprs=[],
    )
    return PreparedSolver(model, 1e-10, 1e-10)


class TestNativeSolveRequest:
    _T_EVAL = np.linspace(0.0, 1.0, 11)
    _NO_STOPS = np.array([], dtype=np.float64)

    def _solve(self, solver, **flags):
        return solver.solve(
            self._T_EVAL, self._NO_STOPS, np.array([1.0]), np.array([1.0]), **flags
        )

    def test_the_row_space_follows_the_outputs_flag(self):
        solver = _native_prepared_solver()
        states = self._solve(solver)
        outputs = self._solve(solver, outputs=True)

        # One state, two outputs (2y, 3y): the flag swaps the rows, not the columns.
        assert states.y.shape == (1, self._T_EVAL.size)
        assert outputs.y.shape == (2, self._T_EVAL.size)
        np.testing.assert_allclose(outputs.y[0], 2.0 * states.y[0], rtol=1e-8)
        np.testing.assert_allclose(outputs.y[1], 3.0 * states.y[0], rtol=1e-8)

    def test_payloads_not_asked_for_read_as_none(self):
        solver = _native_prepared_solver(with_sens=True)

        assert self._solve(solver).yS is None
        assert self._solve(solver, sensitivities=True).yS is not None
        # yp is the state trajectory's slopes, and this solver does not store them.
        assert self._solve(solver).yp is None

    def test_a_seed_without_sensitivities_is_rejected(self):
        solver = _native_prepared_solver(with_sens=True)

        # Ignoring the seed would return zero sensitivities that look computed.
        with pytest.raises(ValueError, match=r"y0_sens was given but"):
            self._solve(solver, y0_sens=np.array([0.5]))

    def test_a_batch_answers_one_shared_request(self):
        solver = _native_prepared_solver(with_sens=True)
        rates = [0.5, 1.0, 2.0]

        results = solver.solve_batch(
            self._T_EVAL,
            self._NO_STOPS,
            np.ones((len(rates), 1)),
            np.array(rates).reshape(-1, 1),
            1,
            outputs=True,
            sensitivities=True,
        )

        assert len(results) == len(rates)
        for i, (result, rate) in enumerate(zip(results, rates, strict=True)):
            serial = solver.solve(
                self._T_EVAL,
                self._NO_STOPS,
                np.array([1.0]),
                np.array([rate]),
                outputs=True,
                sensitivities=True,
            )
            assert result.yS is not None, f"set {i} dropped its blocks"
            np.testing.assert_array_equal(result.y, serial.y, err_msg=f"set {i}")
            np.testing.assert_array_equal(
                result.yS[0], serial.yS[0], err_msg=f"set {i} sensitivities"
            )
