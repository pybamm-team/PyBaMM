"""Tests for solution observation and variable observers."""

import pickle
from itertools import pairwise
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import scipy

import pybamm
from pybamm.solvers.observation import ObserverCache
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    NativeObserver,
    SegmentSelector,
    SharedOutputLeaf,
    SharedOutputs,
    _sweep_sensitivities,
    as_observer,
    chain_rule_sensitivities,
    pack_sensitivity_dict,
)


@pytest.fixture(scope="module")
def spm_solution():
    # CasADi whatever the default, since these tests read CasADi leaves
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = "casadi"
    return pybamm.Simulation(model).solve([0, 600])


def _split(solution, *boundaries):
    """``solution``'s trajectory as consecutive Solutions, cut at ``boundaries``."""
    model = solution.all_models[0]
    edges = [0, *boundaries, len(solution.t)]
    return [
        pybamm.Solution(
            solution.t[start:end],
            solution.y[:, start:end],
            model,
            {},
            all_yps=solution.all_yps[0][:, start:end],
        )
        for start, end in pairwise(edges)
    ]


def _decay_model(factor):
    """A discretised model whose "w" is ``factor`` times its decaying state."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    model.rhs = {u: -u}
    model.initial_conditions = {u: 1}
    model.variables = {"u": u, "w": factor * u}
    pybamm.Discretisation().process_model(model)
    # CasADi whatever the default, since most of these tests read CasADi leaves
    model.convert_to_format = "casadi"
    return model


class TestObserverCache:
    def test_each_segment_reads_through_its_own_model(self):
        doubled, tripled = _decay_model(2), _decay_model(3)
        first = pybamm.IDAKLUSolver().solve(doubled, [0, 1])
        second = pybamm.IDAKLUSolver().solve(tripled, [1, 2])

        joined = first + second

        np.testing.assert_allclose(joined["w"](0.5), 2 * first["u"](0.5), rtol=1e-6)
        np.testing.assert_allclose(joined["w"](1.5), 3 * second["u"](1.5), rtol=1e-6)
        leaves = joined["w"]._observer.leaves
        assert leaves[0] is first["w"]._observer.leaves[0]
        assert leaves[1] is second["w"]._observer.leaves[0]
        assert leaves[0] is not leaves[1]

    def test_solving_a_model_again_reuses_its_leaves(self):
        model = _decay_model(2)
        solver = pybamm.IDAKLUSolver()
        leaf = solver.solve(model, [0, 1])["w"]._observer.leaves[0]

        assert solver.solve(model, [0, 2])["w"]._observer.leaves[0] is leaf

    def test_derived_solutions_reuse_the_models_leaves(self, spm_solution):
        name = "Voltage [V]"
        leaf = spm_solution[name]._observer.leaves[0]
        first, second = _split(spm_solution, 5)

        for derived in (
            spm_solution.first_state,
            spm_solution.last_state,
            spm_solution.copy(),
            first + second,
            pybamm.Solution.from_sub_solutions([first, second]),
        ):
            assert all(each is leaf for each in derived[name]._observer.leaves)

    def test_a_model_copy_starts_with_its_leaves_and_grows_apart(self):
        model = _decay_model(2)
        leaf = pybamm.IDAKLUSolver().solve(model, [0, 1])["w"]._observer.leaves[0]
        model_copy = model.new_copy()

        solution = pybamm.IDAKLUSolver().solve(model_copy, [0, 1])
        assert solution["w"]._observer.leaves[0] is leaf
        solution["u"]
        assert "u" in ObserverCache.of(model_copy)._casadi_leaves
        assert "u" not in ObserverCache.of(model)._casadi_leaves

    def test_a_model_pickled_before_the_cache_existed_builds_one(self):
        model = _decay_model(2)
        del model._observer_cache

        restored = pickle.loads(pickle.dumps(model))

        assert restored._observer_cache is None
        solution = pybamm.IDAKLUSolver().solve(restored, [0, 1])
        np.testing.assert_allclose(solution["w"].entries, 2 * solution["u"].entries)
        assert "w" in ObserverCache.of(restored)._casadi_leaves

    def test_a_time_integral_is_converted_once_per_model(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -u}
        model.initial_conditions = {u: 1}
        model.variables = {
            "u": u,
            "Q": pybamm.ExplicitTimeIntegral(u, pybamm.Scalar(1)),
            "Q squared": pybamm.ExplicitTimeIntegral(u, pybamm.Scalar(0)) ** 2,
        }
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "casadi"
        solver = pybamm.IDAKLUSolver()
        convert = pybamm.Solution._convert_to_casadi

        with mock.patch.object(
            pybamm.Solution, "_convert_to_casadi", autospec=True, side_effect=convert
        ) as spy:
            for t_end in (1, 2):
                solution = solver.solve(model, [0, t_end])
                integral = scipy.integrate.trapezoid(solution["u"].entries, solution.t)
                np.testing.assert_allclose(solution["Q"].entries, [1 + integral])
                np.testing.assert_allclose(solution["Q squared"].entries, [integral**2])

        # u, Q and Q squared, each once across both solves
        assert spy.call_count == 3


def _rust_decay_model(factor):
    model = _decay_model(factor)
    model.convert_to_format = "rust"
    return model


class TestNativeReading:
    def test_a_rust_solve_binds_its_compiled_model(self):
        model = _rust_decay_model(2)
        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(model, [0, 1])

        assert ObserverCache.of(model).rust_model is solver._setup["rust_model"]
        assert isinstance(solution["w"]._observer, NativeObserver)
        np.testing.assert_allclose(solution["w"].entries, 2 * solution["u"].entries)

    def test_derived_solutions_read_through_the_same_tapes(self):
        solution = pybamm.IDAKLUSolver().solve(_rust_decay_model(2), [0, 1])
        leaf = solution["w"]._observer.leaves[0]

        for derived in (solution.copy(), solution.first_state, solution.last_state):
            assert derived["w"]._observer.leaves == [leaf]

    def test_solutions_of_two_rust_models_join_natively(self):
        first = pybamm.IDAKLUSolver().solve(_rust_decay_model(2), [0, 1])
        second = pybamm.IDAKLUSolver().solve(_rust_decay_model(3), [1, 2])

        joined = first + second

        assert isinstance(joined["w"]._observer, NativeObserver)
        np.testing.assert_allclose(joined["w"](0.5), 2 * first["u"](0.5), rtol=1e-6)
        np.testing.assert_allclose(joined["w"](1.5), 3 * second["u"](1.5), rtol=1e-6)

    def test_a_join_with_a_casadi_segment_reads_through_casadi(self):
        casadi_model = _decay_model(3)
        casadi_model.convert_to_format = "casadi"
        first = pybamm.IDAKLUSolver().solve(_rust_decay_model(2), [0, 1])
        second = pybamm.IDAKLUSolver().solve(casadi_model, [1, 2])

        joined = first + second

        assert isinstance(joined["w"]._observer, CasadiObserver)
        np.testing.assert_allclose(joined["w"](0.5), 2 * first["u"](0.5), rtol=1e-6)
        np.testing.assert_allclose(joined["w"](1.5), 3 * second["u"](1.5), rtol=1e-6)

    def test_a_model_switched_back_to_casadi_reads_through_casadi(self):
        model = _rust_decay_model(2)
        solution = pybamm.IDAKLUSolver().solve(model, [0, 1])

        model.convert_to_format = "casadi"

        assert isinstance(solution["w"]._observer, CasadiObserver)

    def test_setting_up_again_drops_the_previous_graph_s_tapes(self):
        model = _rust_decay_model(2)
        pybamm.IDAKLUSolver().solve(model, [0, 1])["w"]
        cache = ObserverCache.of(model)
        cache.bind_rust_model(cache.rust_model)
        assert "w" in cache._native_leaves

        solver = pybamm.IDAKLUSolver()
        solver.set_up(model)

        assert cache.rust_model is solver._setup["rust_model"]
        assert "w" not in cache._native_leaves

    def test_rebinding_leaves_how_a_solution_without_yps_is_read(self):
        model = _rust_decay_model(2)
        solver = pybamm.IDAKLUSolver(options={"hermite_interpolation": False})
        solution = solver.solve(model, [0, 1])
        assert not solution.hermite_interpolation
        before = solution["w"].entries

        pybamm.IDAKLUSolver().set_up(model)
        solution.update("w")

        assert isinstance(solution["w"]._observer, NativeObserver)
        np.testing.assert_array_equal(solution["w"].entries, before)

    def test_a_model_copy_keeps_the_binding_and_its_tapes(self):
        model = _rust_decay_model(2)
        leaf = pybamm.IDAKLUSolver().solve(model, [0, 1])["w"]._observer.leaves[0]

        cache = ObserverCache.of(model.new_copy())

        assert cache.rust_model is ObserverCache.of(model).rust_model
        assert cache._native_leaves["w"][0] is leaf


class _Variable:
    """The slice of ProcessedVariable an observer is allowed to read."""

    def __init__(self, all_ts, all_ys, all_yps=None):
        self.all_ts = all_ts
        self.all_ys = all_ys
        self.all_yps = all_yps
        self.all_inputs = [{} for _ in all_ts]
        self.t_pts = np.concatenate(all_ts)
        self.time_integral = None

    @property
    def hermite_interpolation(self):
        return self.all_yps is not None

    def _shape(self, t):
        return [len(t)]


class _Leaf:
    """Stand-in for a compiled tape: records its calls, returns 2*y0."""

    def __init__(self):
        self.trajectory_calls = []
        self.hermite_calls = []

    def eval_trajectory(self, ts, ys, inputs):
        self.trajectory_calls.append((ts, ys))
        return np.asfortranarray(2.0 * np.asarray(ys)[:1, :])

    def eval_trajectory_hermite(self, t_query, ts, ys, yps, inputs):
        self.hermite_calls.append(t_query)
        return np.asfortranarray(np.zeros((1, len(t_query))))


class TestNativeObserverStandalone:
    def test_observe_raw_evaluates_every_segment(self):
        leaves = [_Leaf(), _Leaf()]
        observer = NativeObserver(leaves)
        variable = _Variable(
            [np.array([0.0, 1.0]), np.array([2.0])],
            [np.array([[1.0, 2.0]]), np.array([[3.0]])],
        )

        np.testing.assert_allclose(observer.observe_raw(variable), [2.0, 4.0, 6.0])
        assert observer.leaves is leaves

    def test_outputs_only_solves_evaluate_on_shaped_zeros(self):
        leaf = _Leaf()
        observer = NativeObserver([leaf], placeholder_states=[3])
        variable = _Variable([np.array([0.0, 1.0])], [np.zeros((0, 0))])

        np.testing.assert_allclose(observer.observe_raw(variable), [0.0, 0.0])
        _, ys = leaf.trajectory_calls[0]
        assert ys.shape == (3, 2)

    def test_empty_segments_are_skipped(self):
        leaves = [_Leaf(), _Leaf()]
        observer = NativeObserver(leaves)
        variable = _Variable(
            [np.array([]), np.array([0.0, 1.0])],
            [np.zeros((1, 0)), np.array([[1.0, 2.0]])],
        )

        np.testing.assert_allclose(observer.observe_raw(variable), [2.0, 4.0])
        assert leaves[0].trajectory_calls == []

    def test_the_query_times_a_leaf_is_handed_are_the_observer_s_own(self):
        # A shared tape keeps its last call, so a leaf must not see the caller's array.
        leaf = _Leaf()
        observer = NativeObserver([leaf])
        variable = _Variable(
            [np.array([0.0, 1.0])],
            [np.array([[1.0, 2.0]])],
            all_yps=[np.array([[0.0, 0.0]])],
        )

        t = np.array([0.2, 0.7])
        observer.observe_hermite(variable, t)
        t[:] = [0.9, 0.95]
        np.testing.assert_array_equal(leaf.hermite_calls[0], [0.2, 0.7])

    def test_segment_selection_is_computed_once_per_observer(self):
        observer = NativeObserver([_Leaf()])
        variable = _Variable([np.array([0.0, 1.0])], [np.array([[1.0, 2.0]])])

        observer.observe_raw(variable)
        selector = observer._selector
        observer.observe_raw(variable)
        assert observer._selector is selector


class _SharedFunction:
    """Stand-in for a multi-output CompiledFunction: records its calls."""

    output_names = ("a", "b")
    output_lens = (2, 1)
    input_names = ("p",)
    n_inputs = 1

    output_len = 3

    def __init__(self):
        self.calls = []
        self.jvp_calls = []

    def eval_trajectory(self, ts, ys, inputs):
        self.calls.append(("eval_trajectory", ts))
        return np.vstack(
            [
                np.asarray(ys)[:1] * 2.0,
                np.asarray(ys)[:1] * 3.0,
                np.asarray(ys)[:1] * 5.0,
            ]
        )

    def jvp_trajectory(self, ts, ys, inputs, vy_traj, vp=None):
        self.jvp_calls.append((vy_traj, vp))
        seed = 0.0 if vp is None else float(vp[0])
        row = np.asarray(vy_traj)[:1]
        return np.vstack([row * 2.0, row * 3.0, row * 5.0]) + seed


class TestSharedOutputs:
    def _shared(self):
        function = _SharedFunction()
        shared = SharedOutputs(function)
        return (
            function,
            shared,
            SharedOutputLeaf(shared, "a"),
            SharedOutputLeaf(shared, "b"),
        )

    def test_each_leaf_sees_only_its_own_rows(self):
        _, _, leaf_a, leaf_b = self._shared()
        ys = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(
            leaf_a.eval_trajectory(np.array([0.0, 1.0]), ys, {}),
            [[2.0, 4.0], [3.0, 6.0]],
        )
        np.testing.assert_allclose(
            leaf_b.eval_trajectory(np.array([0.0, 1.0]), ys, {}), [[5.0, 10.0]]
        )
        assert leaf_a.output_len == 2
        assert leaf_b.output_len == 1

    def test_one_evaluation_serves_every_leaf_of_a_call(self):
        function, _, leaf_a, leaf_b = self._shared()
        ts, ys = np.array([0.0, 1.0]), np.array([[1.0, 2.0]])
        leaf_a.eval_trajectory(ts, ys, {})
        leaf_b.eval_trajectory(ts, ys, {})
        assert len(function.calls) == 1

    def test_a_different_call_evaluates_afresh(self):
        function, _, leaf_a, leaf_b = self._shared()
        ys = np.array([[1.0, 2.0]])
        leaf_a.eval_trajectory(np.array([0.0, 1.0]), ys, {})
        leaf_b.eval_trajectory(np.array([2.0, 3.0]), ys, {})
        assert len(function.calls) == 2

    def test_equal_arrays_from_different_objects_still_share(self):
        function, _, leaf_a, leaf_b = self._shared()
        leaf_a.eval_trajectory(np.array([0.0, 1.0]), np.array([[1.0, 2.0]]), {})
        leaf_b.eval_trajectory(np.array([0.0, 1.0]), np.array([[1.0, 2.0]]), {})
        assert len(function.calls) == 1

    def test_the_held_result_is_released_once_every_leaf_has_read_it(self):
        _, shared, leaf_a, leaf_b = self._shared()
        ts, ys = np.array([0.0, 1.0]), np.array([[1.0, 2.0]])
        leaf_a.eval_trajectory(ts, ys, {})
        assert shared._result is not None
        leaf_b.eval_trajectory(ts, ys, {})
        assert shared._result is None

    @staticmethod
    def _sweep_arguments():
        """``(ts, ys, dy_dp)`` for a two-time, one-state, two-parameter sweep."""
        return (
            np.array([0.0, 1.0]),
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 0.5], [0.25, 2.0]]),
        )

    def test_one_sweep_per_parameter_serves_the_whole_set(self):
        function, _, leaf_a, leaf_b = self._shared()
        ts, ys, dy_dp = self._sweep_arguments()

        # sensitivity_names is rebuilt per variable: equal, but not the same list.
        chain_rule_sensitivities(leaf_a, ts, ys, {}, dy_dp, ["p", "q"])
        chain_rule_sensitivities(leaf_b, ts, ys, {}, dy_dp, ["p", "q"])
        assert len(function.jvp_calls) == 2

    def test_each_leaf_takes_its_own_rows_of_the_set_s_sweep(self):
        function, _, leaf_a, leaf_b = self._shared()
        ts, ys, dy_dp = self._sweep_arguments()
        names = ["p", "q"]

        blocks = _sweep_sensitivities(function, ts, ys, {}, dy_dp, names).reshape(
            len(ts), function.output_len, len(names)
        )
        np.testing.assert_array_equal(
            chain_rule_sensitivities(leaf_a, ts, ys, {}, dy_dp, names),
            blocks[:, 0:2, :].reshape(2 * len(ts), len(names)),
        )
        np.testing.assert_array_equal(
            chain_rule_sensitivities(leaf_b, ts, ys, {}, dy_dp, names),
            blocks[:, 2:3, :].reshape(len(ts), len(names)),
        )

    def test_pickling_drops_the_held_result(self):
        import pickle

        function = _SharedFunction()
        shared = SharedOutputs(function)
        leaf = SharedOutputLeaf(shared, "a")
        leaf.eval_trajectory(np.array([0.0]), np.array([[1.0]]), {})
        restored = pickle.loads(pickle.dumps(shared))
        assert restored._result is None
        assert restored.span("b") == slice(2, 3)


class TestSegmentSelector:
    def test_full_range_keeps_every_nonempty_segment(self):
        selector = SegmentSelector(
            [np.array([]), np.array([0.0, 1.0]), np.array([2.0])]
        )
        np.testing.assert_array_equal(
            selector.select(np.array([0.0]), full_range=True), [1, 2]
        )

    def test_restricted_range_keeps_only_covering_segments(self):
        selector = SegmentSelector([np.array([0.0, 1.0]), np.array([2.0, 3.0])])
        np.testing.assert_array_equal(
            selector.select(np.array([2.5]), full_range=False), [1]
        )

    def test_extrapolating_past_the_end_keeps_the_last_segment(self):
        selector = SegmentSelector([np.array([0.0, 1.0]), np.array([2.0, 3.0])])
        np.testing.assert_array_equal(
            selector.select(np.array([4.0]), full_range=False), [1]
        )


class TestObserverSegments:
    def test_selection_follows_the_variable_asked_about(self):
        observer = CasadiObserver([None, None])
        early = SimpleNamespace(all_ts=[np.array([0.0, 1.0]), np.array([1.0, 2.0])])
        late = SimpleNamespace(all_ts=[np.array([0.0, 5.0]), np.array([5.0, 9.0])])
        t = np.array([3.0])

        np.testing.assert_array_equal(observer.segments(early, t, False), [1])
        np.testing.assert_array_equal(observer.segments(late, t, False), [0])
        np.testing.assert_array_equal(observer.segments(early, t, False), [1])

    def test_an_observer_shared_across_solutions_reads_each_correctly(
        self, spm_solution
    ):
        name = "Voltage [V]"
        early = pybamm.Solution.from_sub_solutions(_split(spm_solution, 3))
        late = pybamm.Solution.from_sub_solutions(_split(spm_solution, 12))
        t = np.array([spm_solution.t[6]])
        base = [m.get_processed_variable_or_event(name) for m in early.all_models]
        observer = early[name]._observer
        early[name](t)

        shared = pybamm.process_variable(name, base, observer, late)

        np.testing.assert_allclose(shared(t), late[name](t), rtol=1e-12)


class TestObserverCoercion:
    def test_a_bare_casadi_list_becomes_a_casadi_observer(self):
        observer = as_observer([None, None])
        assert isinstance(observer, CasadiObserver)
        assert observer.leaves == [None, None]

    def test_an_observer_is_passed_through(self):
        observer = CasadiObserver([None])
        assert as_observer(observer) is observer

    def test_a_native_observer_is_passed_through(self):
        observer = NativeObserver([_Leaf()])
        assert as_observer(observer) is observer

    def test_process_variable_accepts_either_form(self, spm_solution):
        name = "Terminal voltage [V]"
        base = [
            m.get_processed_variable_or_event(name) for m in spm_solution.all_models
        ]
        leaves = spm_solution[name]._observer.leaves

        direct = pybamm.process_variable(name, base, leaves, spm_solution)
        wrapped = pybamm.process_variable(name, base, as_observer(leaves), spm_solution)
        np.testing.assert_allclose(direct.entries, wrapped.entries)


class TestCasadiObserverCaches:
    def test_casadi_leaves_serialise_once_across_calls(self, spm_solution):
        variable = spm_solution.copy()["Terminal voltage [V]"]

        variable.entries
        serialised = dict(variable._observer._serialised)
        assert serialised
        variable(np.linspace(0, 600, 11))
        variable(np.linspace(0, 600, 7))
        for key, value in variable._observer._serialised.items():
            assert value is serialised[key]

    def test_pickling_drops_the_caches(self, spm_solution):
        variable = spm_solution.copy()["Terminal voltage [V]"]
        t = np.linspace(0, 600, 11)
        expected = variable(t)
        observer = variable._observer
        assert observer._serialised is not None
        assert observer._selector is not None

        restored = pickle.loads(pickle.dumps(observer))

        for cache in ("_selector", "_selector_ts", "_serialised"):
            assert cache not in restored.__dict__
            assert cache in observer.__dict__
        variable._observer = restored
        np.testing.assert_allclose(variable(t), expected)


class TestPackSensitivityDict:
    def test_all_block_plus_one_flat_vector_per_parameter(self):
        sensitivity_matrix = np.arange(6.0).reshape(3, 2)
        packed = pack_sensitivity_dict(sensitivity_matrix, ["a", "b"])

        assert set(packed) == {"all", "a", "b"}
        np.testing.assert_array_equal(packed["all"], sensitivity_matrix)
        np.testing.assert_array_equal(packed["a"], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(packed["b"], [1.0, 3.0, 5.0])
