"""Tests for solution observation and variable observers."""

import pickle  # nosec B403 - used in tests with trusted input
from itertools import pairwise
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import scipy

import pybamm
from pybamm.solvers import variable_observer
from pybamm.solvers.observation import ObserverCache
from pybamm.solvers.variable_observer import (
    CasadiObserver,
    SegmentSelector,
    as_observer,
    pack_sensitivity_dict,
)


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


def _decay_model(factor):
    """A discretised model whose "w" is ``factor`` times its decaying state."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    model.rhs = {u: -u}
    model.initial_conditions = {u: 1}
    model.variables = {"u": u, "w": factor * u}
    pybamm.Discretisation().process_model(model)
    return model


def _input_decay_model():
    """Discretised ``du/dt = -a u`` from ``u = 1``, with "a u" as a variable."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    a = pybamm.InputParameter("a")
    model.rhs = {u: -a * u}
    model.initial_conditions = {u: 1}
    model.variables = {"u": u, "a u": a * u}
    pybamm.Discretisation().process_model(model)
    return model


def _solve_with_sensitivities(model, t0=0, n_times=11):
    """``model`` solved over ``[t0, t0 + 1]`` with ``a = 0.5``, for sensitivities."""
    return pybamm.IDAKLUSolver(rtol=1e-10, atol=1e-12).solve(
        model,
        [t0, t0 + 1],
        t_interp=np.linspace(t0, t0 + 1, n_times),
        inputs={"a": 0.5},
        calculate_sensitivities=True,
    )


def _exact_a_u_sensitivity(tau):
    """d(a u)/da at times ``tau`` after a restart from ``u = 1``, with ``a = 0.5``."""
    u = np.exp(-0.5 * tau)
    return u - 0.5 * tau * u


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

        restored = pickle.loads(pickle.dumps(model))  # nosec B301

        assert restored._observer_cache is None
        solution = pybamm.IDAKLUSolver().solve(restored, [0, 1])
        np.testing.assert_allclose(solution["w"].entries, 2 * solution["u"].entries)
        assert "w" in ObserverCache.of(restored)._casadi_leaves

    def test_pickling_a_model_drops_its_chain_rules(self):
        model = _input_decay_model()
        expected = _solve_with_sensitivities(model)["a u"].sensitivities["a"]
        assert ObserverCache.of(model).chain_rules

        restored = pickle.loads(pickle.dumps(model))  # nosec B301

        assert "a u" in restored._observer_cache._casadi_leaves
        assert "_chain_rules" not in restored._observer_cache.__dict__
        sensitivities = _solve_with_sensitivities(restored)["a u"].sensitivities
        np.testing.assert_allclose(sensitivities["a"], expected)

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

    def test_a_time_in_a_roundoff_gap_moves_to_the_earlier_segment_end(self):
        eps = np.finfo(float).eps
        end, start = 1 - eps, 1 + eps
        selector = SegmentSelector([np.array([0.0, end]), np.array([start, 2.0])])

        t, in_gap = selector.resolve_gaps(np.array([0.5, 1.0, start, 1.5]))

        np.testing.assert_array_equal(t, [0.5, end, start, 1.5])
        assert in_gap is None

    def test_a_time_in_a_longer_gap_is_flagged(self):
        selector = SegmentSelector([np.array([0.0, 1.0]), np.array([2.0, 3.0])])

        t, in_gap = selector.resolve_gaps(np.array([2.5, 1.5, 0.5, 4.0]))

        np.testing.assert_array_equal(t, [2.5, 1.5, 0.5, 4.0])
        np.testing.assert_array_equal(in_gap, [False, True, False, False])
        assert selector.resolve_gaps(1.5)[1].ndim == 0


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

        restored = pickle.loads(pickle.dumps(observer))  # nosec B301

        for cache in ("_selector", "_selector_ts", "_serialised", "_chain_rules"):
            assert cache not in restored.__dict__
            assert cache in observer.__dict__
        variable._observer = restored
        np.testing.assert_allclose(variable(t), expected)


class TestCasadiObserverSensitivities:
    @pytest.mark.parametrize("block_size", [4, 64])
    def test_sensitivities_over_blocks_of_times_and_segments(self, block_size):
        model = _input_decay_model()
        joined = _solve_with_sensitivities(model) + _solve_with_sensitivities(model, 1)

        with mock.patch.object(variable_observer._ChainRule, "block_size", block_size):
            sensitivities = joined["a u"].sensitivities["a"]

        t_interp = np.linspace(0, 1, 11)
        tau = np.concatenate([t_interp, t_interp[1:]])
        np.testing.assert_allclose(
            sensitivities, _exact_a_u_sensitivity(tau), rtol=1e-6, atol=1e-12
        )

    def test_segments_of_many_lengths_share_a_few_maps(self):
        model = _input_decay_model()
        lengths = range(2, 20)
        joined = pybamm.Solution.from_sub_solutions(
            [
                _solve_with_sensitivities(model, t0, n_times)
                for t0, n_times in enumerate(lengths)
            ]
        )

        sensitivities = joined["a u"].sensitivities["a"]

        # Each later segment drops its first time, which repeats the previous end
        tau = np.concatenate(
            [np.linspace(0, 1, n)[0 if i == 0 else 1 :] for i, n in enumerate(lengths)]
        )
        np.testing.assert_allclose(
            sensitivities, _exact_a_u_sensitivity(tau), rtol=1e-6, atol=1e-12
        )
        ((_, chain_rule),) = ObserverCache.of(model).chain_rules.values()
        assert set(chain_rule._maps) <= {1, 2, 4, 8, 16, 32, 64}

    def test_later_solves_of_a_model_reuse_its_chain_rules(self):
        model = _input_decay_model()

        with mock.patch.object(
            variable_observer, "_ChainRule", wraps=variable_observer._ChainRule
        ) as spy:
            first = _solve_with_sensitivities(model)["a u"].sensitivities["a"]
            second = _solve_with_sensitivities(model)["a u"].sensitivities["a"]

        assert spy.call_count == 1
        np.testing.assert_array_equal(second, first)

    def test_a_variable_of_neither_states_nor_inputs_has_zero_sensitivities(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        model.rhs = {u: -pybamm.InputParameter("a") * u}
        model.initial_conditions = {u: 1}
        model.variables = {"two": pybamm.Scalar(2)}
        pybamm.Discretisation().process_model(model)

        # Five times evaluate as a block of four and a block of one
        solution = _solve_with_sensitivities(model, n_times=5)
        sensitivities = solution["two"].sensitivities["a"]

        np.testing.assert_array_equal(sensitivities, np.zeros(5))
        ((_, chain_rule),) = ObserverCache.of(model).chain_rules.values()
        assert set(chain_rule._maps) == {1, 4}
        for mapped in chain_rule._maps.values():
            assert mapped.sparsity_out(0).is_dense()


class TestPackSensitivityDict:
    def test_all_block_plus_one_flat_vector_per_parameter(self):
        sensitivity_matrix = np.arange(6.0).reshape(3, 2)
        packed = pack_sensitivity_dict(sensitivity_matrix, ["a", "b"])

        assert set(packed) == {"all", "a", "b"}
        np.testing.assert_array_equal(packed["all"], sensitivity_matrix)
        np.testing.assert_array_equal(packed["a"], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(packed["b"], [1.0, 3.0, 5.0])
