"""Contract tests for the observation seam: backends and observers standalone."""

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import (
    CASADI_OBSERVATION,
    CasadiObservation,
    NativeComputedObservation,
    NativeInterpolatingObservation,
    NativeObservation,
    join_observations,
)
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    NativeObserver,
    SegmentSelector,
    as_observer,
    pack_sensitivity_dict,
)


class _Model:
    """Stand-in for a CompiledModel: only identity matters to the backend."""

    def __init__(self, tag):
        self.tag = tag


class TestSegmentSlicing:
    def test_the_casadi_backend_is_a_shared_stateless_value(self):
        # It reaches every per-segment artifact through the Solution it is
        # handed, so slicing a segment run cannot change it.
        assert isinstance(CASADI_OBSERVATION, CasadiObservation)
        assert CASADI_OBSERVATION[:1] is CASADI_OBSERVATION
        assert CASADI_OBSERVATION[-1:] is CASADI_OBSERVATION

    def test_native_slices_its_per_segment_models(self):
        models = [_Model(i) for i in range(3)]
        cache = {"tape": object()}
        backend = NativeInterpolatingObservation(models, cache=cache)

        assert backend.n_segments == 3
        assert backend.primary_model is models[0]
        assert backend[:1].segment_models == models[:1]
        assert backend[-1:].segment_models == models[-1:]
        # a slice keeps the concrete backend and shares the cache by identity
        assert isinstance(backend[1:], NativeInterpolatingObservation)
        assert backend[1:].compile_cache is cache
        assert backend[-1:].primary_model is models[-1]

    def test_uniform_gives_every_segment_the_same_model(self):
        model = _Model("a")
        backend = NativeComputedObservation.uniform(model, 3)
        assert backend.segment_models == [model] * 3
        assert isinstance(backend, NativeComputedObservation)

    def test_the_native_base_cannot_be_instantiated(self):
        # Which concrete backend a solver builds is what picks the kind of
        # processed variable its solutions hand back, so there is no default.
        with pytest.raises(TypeError, match="abstract"):
            NativeObservation([_Model("a")])


class TestJoin:
    def test_casadi_only_runs_join_to_the_shared_value(self):
        joined = join_observations([(CASADI_OBSERVATION, 2), (CASADI_OBSERVATION, 3)])
        assert joined is CASADI_OBSERVATION

    def test_native_runs_concatenate_their_models(self):
        left = NativeComputedObservation([_Model("a")])
        right = NativeComputedObservation([_Model("b"), _Model("c")])
        joined = join_observations([(left, 1), (right, 2)])

        assert joined.segment_models == left.segment_models + right.segment_models
        assert isinstance(joined, NativeComputedObservation)

    def test_joining_never_narrows_how_a_solution_can_be_read(self):
        # One interpolating run is enough: the join must not demote the whole
        # span to grid-aligned reads.
        interpolating = NativeInterpolatingObservation([_Model("a")])
        computed = NativeComputedObservation([_Model("b")])

        for runs in (
            [(interpolating, 1), (computed, 1)],
            [(computed, 1), (interpolating, 1)],
        ):
            assert isinstance(join_observations(runs), NativeInterpolatingObservation)

    def test_many_runs_join_in_one_pass(self):
        runs = [(NativeComputedObservation([_Model(i)]), 1) for i in range(50)]
        joined = join_observations(runs)
        assert joined.n_segments == 50
        assert [m.tag for m in joined.segment_models] == list(range(50))

    def test_shared_cache_identity_survives_a_join(self):
        cache = {}
        runs = [
            (NativeComputedObservation([_Model(i)], cache=cache), 1) for i in range(3)
        ]
        assert join_observations(runs).compile_cache is cache

    def test_distinct_caches_merge_with_the_earlier_run_winning(self):
        left = NativeComputedObservation([_Model("a")], cache={"k": "left"})
        right = NativeComputedObservation(
            [_Model("b")], cache={"k": "right", "extra": 1}
        )
        merged = join_observations([(left, 1), (right, 1)]).compile_cache
        assert merged == {"k": "left", "extra": 1}

    def test_a_casadi_run_is_observed_by_the_first_native_model(self):
        native = NativeInterpolatingObservation([_Model("a")])
        joined = join_observations([(CASADI_OBSERVATION, 2), (native, 1)])

        # Every segment gets a concrete model, so there is no fallback rule
        assert joined.segment_models == [native.primary_model] * 3
        assert isinstance(joined, NativeInterpolatingObservation)

        # ... and in the other order
        joined = join_observations([(native, 1), (CASADI_OBSERVATION, 1)])
        assert joined.segment_models == [native.primary_model] * 2


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

    def eval_trajectory(self, ts, ys, inputs):
        self.trajectory_calls.append((ts, ys))
        return np.asfortranarray(2.0 * np.asarray(ys)[:1, :])


class TestNativeObserverStandalone:
    def test_observe_raw_evaluates_every_segment(self):
        leaves = [_Leaf(), _Leaf()]
        observer = NativeObserver(leaves, backend=None)
        variable = _Variable(
            [np.array([0.0, 1.0]), np.array([2.0])],
            [np.array([[1.0, 2.0]]), np.array([[3.0]])],
        )

        np.testing.assert_allclose(observer.observe_raw(variable), [2.0, 4.0, 6.0])
        assert observer.leaves is leaves

    def test_outputs_only_solves_evaluate_on_shaped_zeros(self):
        leaf = _Leaf()
        observer = NativeObserver([leaf], backend=None, placeholder_states=[3])
        variable = _Variable([np.array([0.0, 1.0])], [np.zeros((0, 0))])

        np.testing.assert_allclose(observer.observe_raw(variable), [0.0, 0.0])
        _, ys = leaf.trajectory_calls[0]
        assert ys.shape == (3, 2)

    def test_empty_segments_are_skipped(self):
        leaves = [_Leaf(), _Leaf()]
        observer = NativeObserver(leaves, backend=None)
        variable = _Variable(
            [np.array([]), np.array([0.0, 1.0])],
            [np.zeros((1, 0)), np.array([[1.0, 2.0]])],
        )

        np.testing.assert_allclose(observer.observe_raw(variable), [2.0, 4.0])
        assert leaves[0].trajectory_calls == []

    def test_segment_selection_is_computed_once_per_observer(self):
        observer = NativeObserver([_Leaf()], backend=None)
        variable = _Variable([np.array([0.0, 1.0])], [np.array([[1.0, 2.0]])])

        observer.observe_raw(variable)
        selector = observer._selector
        observer.observe_raw(variable)
        assert observer._selector is selector


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


class TestObserverCoercion:
    def test_a_bare_casadi_list_becomes_a_casadi_observer(self):
        observer = as_observer([None, None])
        assert isinstance(observer, CasadiObserver)
        assert observer.leaves == [None, None]

    def test_an_observer_passes_through(self):
        observer = NativeObserver([_Leaf()], backend=None)
        assert as_observer(observer) is observer

    def test_process_variable_accepts_either_form(self):
        # A bare list is only ever CasADi leaves, so drive the CasADi backend.
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "casadi"
        solution = pybamm.Simulation(model).solve([0, 600])
        name = "Terminal voltage [V]"
        base = [m.get_processed_variable_or_event(name) for m in solution.all_models]
        leaves = solution[name]._observer.leaves

        direct = pybamm.process_variable(name, base, leaves, solution)
        wrapped = pybamm.process_variable(name, base, as_observer(leaves), solution)
        np.testing.assert_allclose(direct.entries, wrapped.entries)

    def test_casadi_leaves_serialise_once_across_calls(self):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "casadi"
        solution = pybamm.Simulation(model).solve([0, 600])
        variable = solution["Terminal voltage [V]"]

        variable.entries
        serialised = dict(variable._observer._serialised)
        assert serialised
        variable(np.linspace(0, 600, 11))
        # the immutable CasADi function is not re-serialised per observe call
        assert variable._observer._serialised == serialised


class TestPackSensitivityDict:
    def test_all_block_plus_one_flat_vector_per_parameter(self):
        S_var = np.arange(6.0).reshape(3, 2)
        packed = pack_sensitivity_dict(S_var, ["a", "b"])

        assert set(packed) == {"all", "a", "b"}
        np.testing.assert_array_equal(packed["all"], S_var)
        np.testing.assert_array_equal(packed["a"], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(packed["b"], [1.0, 3.0, 5.0])


class TestSolutionCarriesOneField:
    @staticmethod
    def _solution(t_eval):
        return pybamm.Simulation(pybamm.lithium_ion.SPM()).solve(t_eval)

    def test_backend_travels_with_derived_solutions(self):
        solution = self._solution([0, 600])
        backend = solution.observation
        assert backend.n_segments == len(solution.all_ys)

        for derived in (solution.copy(), solution.first_state, solution.last_state):
            assert type(derived.observation) is type(backend)
            assert derived.observation.n_segments == len(derived.all_ys)
            assert derived.observation.compile_cache is backend.compile_cache

    def test_added_solutions_cover_every_segment(self):
        first = self._solution([0, 600])
        second = self._solution([600, 1200])
        joined = first + second

        assert joined.observation.n_segments == len(joined.all_ys)
        np.testing.assert_allclose(
            joined["Terminal voltage [V]"](300.0),
            first["Terminal voltage [V]"](300.0),
            rtol=1e-10,
        )
