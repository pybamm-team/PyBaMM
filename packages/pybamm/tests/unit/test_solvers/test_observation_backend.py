"""Tests for observation backends and variable observers."""

import pickle
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

import pybamm
from pybamm.solvers.observation import (
    CASADI_OBSERVATION,
    CasadiObservation,
    join_observations,
)
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    SegmentSelector,
    as_observer,
    pack_sensitivity_dict,
)


class _RecordingObservation(CasadiObservation):
    """A CasADi backend that records the segment slices taken of it."""

    def __init__(self):
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return self


@pytest.fixture(scope="module")
def spm_solution():
    model = pybamm.lithium_ion.SPM()
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


class TestSegmentSlicing:
    def test_the_casadi_backend_is_a_shared_stateless_value(self):
        assert isinstance(CASADI_OBSERVATION, CasadiObservation)
        assert CASADI_OBSERVATION[:1] is CASADI_OBSERVATION
        assert CASADI_OBSERVATION[-1:] is CASADI_OBSERVATION

    def test_casadi_backends_are_interchangeable(self):
        # A Solution unpickled from a derived one carries its own instance.
        restored = pickle.loads(pickle.dumps(CASADI_OBSERVATION))
        assert restored is not CASADI_OBSERVATION
        assert restored == CASADI_OBSERVATION
        assert hash(restored) == hash(CASADI_OBSERVATION)


class TestJoin:
    def test_runs_sharing_a_backend_join_to_it(self):
        joined = join_observations([CASADI_OBSERVATION, CASADI_OBSERVATION])
        assert joined is CASADI_OBSERVATION

    def test_equal_backends_join_to_the_first(self):
        restored = pickle.loads(pickle.dumps(CASADI_OBSERVATION))
        assert join_observations([restored, CASADI_OBSERVATION]) is restored

    def test_different_backends_do_not_join(self):
        with pytest.raises(pybamm.SolverError, match=r"different observation"):
            join_observations([CASADI_OBSERVATION, _RecordingObservation()])


class TestObservationCarriedThroughDerivedSolutions:
    @pytest.fixture
    def recorded(self, spm_solution):
        first, second = _split(spm_solution, 5)
        backend = _RecordingObservation()
        first._observation = backend
        second._observation = backend
        return first, second, backend

    def test_first_and_last_state_slice_it(self, recorded):
        first, _, backend = recorded
        assert first.first_state._observation is backend
        assert first.last_state._observation is backend
        assert backend.keys == [slice(None, 1), slice(-1, None)]

    def test_addition_joins_it(self, recorded):
        first, second, backend = recorded
        assert (first + second)._observation is backend

    def test_from_sub_solutions_joins_it(self, recorded):
        first, second, backend = recorded
        joined = pybamm.Solution.from_sub_solutions([first, second])
        assert joined._observation is backend

    def test_copy_keeps_it(self, recorded):
        first, _, backend = recorded
        assert first.copy()._observation is backend

    def test_solutions_read_differently_cannot_be_added(self, recorded):
        first, second, _ = recorded
        second._observation = CASADI_OBSERVATION
        with pytest.raises(pybamm.SolverError, match=r"different observation"):
            first + second


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
