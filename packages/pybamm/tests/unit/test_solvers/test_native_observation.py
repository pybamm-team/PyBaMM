"""Unit tests for native (Rust) observation: accessor, producer, compile-once."""

import numpy as np

from pybamm.rust import ExprGraph
from pybamm.solvers.observation import (
    NativeComputedObservation,
    NativeObservation,
)
from pybamm.solvers.variable_observer import chain_rule_sensitivities

# D_n's dust rides IDAKLU's unscaled pbar = 1, splitting the two backends' step
# sequences at default tolerances. 1e-11 converges both; the parity bounds stay.
_TWO_PARAM_SOLVER_TOL = 1e-11


def _make_compiled_model():
    """Minimal 2-state model with one input (mirrors the binding-test fixture).

    rhs = [a * y1, -y0]; identity mass; one input 'a'.
    """
    from pybamm.rust import CompiledModel

    g = ExprGraph()
    a = g.input_parameter("a")
    y0 = g.state_vector(0, 1)
    y1 = g.state_vector(1, 2)
    rhs = g.concat([g.mul(a, y1), g.neg(y0)])
    mass = np.array([1.0, 1.0])
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    model = CompiledModel.from_expr(g, rhs, mass, indptr, indices, n_inputs=1)
    return g, model


class TestGraphAccessor:
    def test_graph_returns_exprgraph(self):
        _, model = _make_compiled_model()
        assert isinstance(model.graph, ExprGraph)

    def test_graph_compiles_a_new_observation_root(self):
        # Lower a NEW root (2*y0) into the retained arena and compile it.
        # The new root has no inputs, so p is the empty stacked array.
        _, model = _make_compiled_model()
        graph = model.graph
        new_root = graph.mul(graph.scalar(2.0), graph.state_vector(0, 1))
        fn = graph.compile(new_root, name="obs", n_states=model.n_states)
        out = fn(0.0, np.array([3.0, 5.0]), np.array([1.0]))
        np.testing.assert_allclose(out, [6.0])


class TestObservationContext:
    def test_set_and_propagate(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "casadi"  # no native observation to start from
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 600])
        assert not isinstance(sol.observation, NativeObservation)

        _, compiled = _make_compiled_model()
        sol.observation = NativeComputedObservation.uniform(compiled, len(sol.all_ys))
        assert sol.observation.primary_model is compiled
        assert sol.observation.compile_cache == {}

        # Context survives copy and slicing
        assert sol.copy().observation.primary_model is compiled
        assert sol.first_state.observation.primary_model is compiled
        assert sol.last_state.observation.primary_model is compiled


def _native_idaklu_sim(model, **solver_kwargs):
    """Simulation on idaklu with native observation forced on (test-only).

    The public switch is `convert_to_format == "rust"`; idaklu's
    `_observes_via_compiled_model` stays False until prereqs A + B land, so
    foundation tests set it on the solver instance. The instance attribute
    survives the shallow `solver.copy()` that `Simulation` performs.
    """
    import pybamm

    model.convert_to_format = "rust"
    solver = pybamm.IDAKLUSolver(**solver_kwargs)
    solver._observes_via_compiled_model = True
    return pybamm.Simulation(model, solver=solver)


class TestSingleVariableNative:
    def test_native_matches_casadi_0d(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        sol = _native_idaklu_sim(model).solve([0, 3600])
        v_native = sol["Terminal voltage [V]"].entries

        casadi_model = pybamm.lithium_ion.SPM()
        casadi_model.convert_to_format = "casadi"
        sim_c = pybamm.Simulation(casadi_model, solver=pybamm.IDAKLUSolver())
        sol_c = sim_c.solve([0, 3600])
        v_casadi = sol_c["Terminal voltage [V]"].entries

        np.testing.assert_allclose(v_native, v_casadi, rtol=1e-10, atol=1e-12)

    def test_hermite_native_matches_casadi_multi_tile(self):
        import pybamm

        # A dense 137-point grid off the solver's knots: the lane-batched Hermite
        # evaluator spans several tiles plus a ragged tail, and hits s=0/1 and 0<s<1.
        model = pybamm.lithium_ion.SPM()
        sol = _native_idaklu_sim(model).solve([0, 3600])

        casadi_model = pybamm.lithium_ion.SPM()
        casadi_model.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(casadi_model, solver=pybamm.IDAKLUSolver()).solve(
            [0, 3600]
        )

        t_query = np.linspace(0, 3600, 137)
        v_native = sol["Terminal voltage [V]"](t_query)
        v_casadi = sol_c["Terminal voltage [V]"](t_query)
        assert v_native.shape == t_query.shape
        np.testing.assert_allclose(v_native, v_casadi, rtol=1e-6, atol=1e-8)

    def test_compile_once(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        sol = _native_idaklu_sim(model).solve([0, 600])
        _ = sol["Terminal voltage [V]"]
        key = ("Terminal voltage [V]", id(sol.observation.primary_model))
        assert key in sol.observation.compile_cache
        fn1 = sol.observation.compile_cache[key]
        # Second access reuses the cached compiled fn; cache identity holds.
        _ = sol["Terminal voltage [V]"]
        assert sol.observation.compile_cache[key] is fn1


def _make_var_fn():
    """g(t, y, p) = [a*y0*y1, sin(y0)*b]; 2 states, 2 inputs ('a','b')."""
    g = ExprGraph()
    a = g.input_parameter("a")
    b = g.input_parameter("b")
    y0 = g.state_vector(0, 1)
    y1 = g.state_vector(1, 2)
    expr = g.concat([g.mul(g.mul(a, y0), y1), g.mul(g.sin(y0), b)])
    return g, g.compile(expr, name="var")


class TestNativeChainRuleKernel:
    def _trajectory(self):
        n_t, n_states, n_p = 8, 2, 2
        ts = np.linspace(0.0, 1.0, n_t)
        ys = np.vstack([np.linspace(0.1, 1.0, n_t), np.linspace(2.0, 3.0, n_t)])
        rng = np.random.default_rng(0)
        dy_dp = rng.standard_normal((n_t * n_states, n_p))
        p = np.array([3.0, 4.0])
        return ts, ys, dy_dp, p, n_t, n_states, n_p

    def test_matches_numpy_jacobian_oracle(self):
        # Independent oracle: per timestep, dvar_dy @ yS_k + dvar_dp @ e_k.
        _, cf = _make_var_fn()
        ts, ys, dy_dp, p, n_t, n_states, n_p = self._trajectory()
        out_len = cf.output_len
        Jy = cf.jacobian("y")
        Jp = cf.jacobian("p")
        S = chain_rule_sensitivities(cf, ts, ys, p, dy_dp, ["a", "b"])
        assert S.shape == (n_t * out_len, n_p)
        for k in range(n_p):
            e_k = np.zeros(n_p)
            e_k[k] = 1.0
            for j in range(n_t):
                dvar_dy = Jy(ts[j], ys[:, j].copy(), p).toarray()
                dvar_dp = Jp(ts[j], ys[:, j].copy(), p).toarray()
                yS_jk = dy_dp[j * n_states : (j + 1) * n_states, k]
                oracle = dvar_dy @ yS_jk + dvar_dp @ e_k
                block = S[j * out_len : (j + 1) * out_len, k]
                np.testing.assert_allclose(block, oracle, rtol=1e-12, atol=1e-12)

    def test_preserves_zero_derivative_width(self):
        # var = const_vector([1,2,3]) + a; dvar_dy ≡ 0, output width 3.
        g = ExprGraph()
        cv = g.array(np.array([1.0, 2.0, 3.0]))
        cf = g.compile(g.add(cv, g.input_parameter("a")), name="zeroy", n_states=2)
        assert cf.output_len == 3
        n_t, n_states = 5, 2
        ts = np.linspace(0.0, 1.0, n_t)
        ys = np.zeros((n_states, n_t))
        dy_dp = np.ones((n_t * n_states, 1))  # nonzero yS, but dvar_dy is 0
        S = chain_rule_sensitivities(cf, ts, ys, np.array([5.0]), dy_dp, ["a"])
        assert S.shape == (n_t * 3, 1)
        # only the direct dvar_dp = 1 term survives, in every output row
        np.testing.assert_array_equal(S, np.ones((n_t * 3, 1)))


class TestNativeSensitivities:
    # SPM (events removed) with Current function [A] as InputParameter "I";
    # native IDAKLU observation forced on (test-only), compared to CasADi.
    def _solve_both(self, output_name, calc, extra_inputs=None, solver_tol=None):
        import pybamm

        params = pybamm.ParameterValues("Chen2020")
        params["Current function [A]"] = pybamm.InputParameter("I")
        inputs = {"I": 0.5}
        if extra_inputs:
            for pybamm_name, (input_name, value) in extra_inputs.items():
                params[pybamm_name] = pybamm.InputParameter(input_name)
                inputs[input_name] = value
        t_eval = np.linspace(0, 100, 15)
        tol = {} if solver_tol is None else {"rtol": solver_tol, "atol": solver_tol}

        m_native = pybamm.lithium_ion.SPM()
        m_native.events = []
        m_native.convert_to_format = "rust"
        solver_n = pybamm.IDAKLUSolver(**tol)
        solver_n._observes_via_compiled_model = True  # force native obs (test-only)
        sol_n = pybamm.Simulation(
            m_native, parameter_values=params, solver=solver_n
        ).solve(
            t_eval,
            inputs=inputs,
            calculate_sensitivities=calc,
            t_interp=t_eval,
        )

        m_casadi = pybamm.lithium_ion.SPM()
        m_casadi.events = []
        m_casadi.convert_to_format = "casadi"
        sol_c = pybamm.Simulation(
            m_casadi, parameter_values=params, solver=pybamm.IDAKLUSolver(**tol)
        ).solve(t_eval, inputs=inputs, calculate_sensitivities=calc, t_interp=t_eval)
        return sol_n[output_name], sol_c[output_name]

    def test_sensitivities_match_casadi_0d(self):
        var_n, var_c = self._solve_both("Terminal voltage [V]", ["I"])
        sens_n = var_n.sensitivities
        sens_c = var_c.sensitivities
        assert set(sens_n) == set(sens_c)
        for key in sens_c:
            np.testing.assert_allclose(sens_n[key], sens_c[key], rtol=1e-5, atol=1e-8)

    def test_sensitivities_match_casadi_spatial(self):
        # output_len > 1 exercises the (output_len, n_t) -> time-outer/output-inner
        # flatten against the CasADi block-diagonal layout.
        name = "Negative particle concentration [mol.m-3]"
        var_n, var_c = self._solve_both(name, ["I"])
        # Assert the variable is genuinely spatial (multi-point); if it resolves to
        # 0D, the flatten logic under test would not be exercised.
        assert var_c.entries.shape[0] > 1, (
            f"{name!r} resolved to 0D (shape {var_c.entries.shape}); "
            "test premise violated"
        )
        sens_n = var_n.sensitivities
        sens_c = var_c.sensitivities
        assert set(sens_n) == set(sens_c)
        for key in sens_c:
            assert sens_n[key].shape == sens_c[key].shape
            np.testing.assert_allclose(sens_n[key], sens_c[key], rtol=1e-5, atol=1e-8)

    def test_sensitivities_all_block_two_params(self):
        # "I" plus a diffusivity input; sorted column order is ["D_n", "I"].
        var_n, var_c = self._solve_both(
            "Terminal voltage [V]",
            ["D_n", "I"],
            extra_inputs={"Negative particle diffusivity [m2.s-1]": ("D_n", 3.3e-14)},
            solver_tol=_TWO_PARAM_SOLVER_TOL,
        )
        a_n = var_n.sensitivities["all"]
        a_c = var_c.sensitivities["all"]
        assert a_n.shape == a_c.shape  # (n_t*output_len, 2)
        assert a_n.shape[1] == 2  # block must have exactly 2 parameter columns
        assert np.any(var_n.sensitivities["D_n"] != 0)
        # Sorted order is ["D_n", "I"]; pin column 0 absolutely to D_n sensitivity.
        np.testing.assert_allclose(
            a_n[:, 0], var_n.sensitivities["D_n"], rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(a_n, a_c, rtol=1e-5, atol=1e-8)

    def test_no_sensitivities_when_not_requested(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        sol = _native_idaklu_sim(model).solve([0, 600])
        assert sol["Terminal voltage [V]"].sensitivities == {}


class TestObservationCacheAcrossSolves:
    # The compiled-observation cache lives with the solver setup, 1:1 with the rust
    # model, so repeated solves reuse tapes instead of growing the retained graph.

    @staticmethod
    def _sim():
        # events removed so termination is "final time" and solve() returns the
        # post-processed Solution directly (which shares the solver's cache).
        import pybamm

        model = pybamm.lithium_ion.SPM()
        model.events = []
        return _native_idaklu_sim(model)

    def test_compiled_leaf_reused_across_solves(self):
        sim = self._sim()
        sol1 = sim.solve([0, 600])
        v1 = sol1["Terminal voltage [V]"]
        sol2 = sim.solve([0, 600])
        v2 = sol2["Terminal voltage [V]"]

        # Same solver-owned cache dict, shared by reference across solves.
        assert sol2.observation.compile_cache is sol1.observation.compile_cache
        assert (
            sol1.observation.compile_cache
            is sim._solver._setup["rust_observation_cache"]
        )
        # The compiled leaf is reused, not recompiled, on the second solve.
        assert v2._observer.leaves[0] is v1._observer.leaves[0]

    def test_retained_graph_stops_growing(self):
        sim = self._sim()
        sol1 = sim.solve([0, 600])
        _ = sol1["Terminal voltage [V]"]
        cache = sim._solver._setup["rust_observation_cache"]
        nodes_after_first = sim._solver._setup["rust_model"].graph.n_nodes
        cache_len_after_first = len(cache)

        sol2 = sim.solve([0, 600])
        _ = sol2["Terminal voltage [V]"]
        # No recompile on the second observe: neither the retained graph's node
        # count nor the cache size grows.
        assert sim._solver._setup["rust_model"].graph.n_nodes == nodes_after_first
        assert len(cache) == cache_len_after_first

    def test_discrete_time_sum_across_two_solves(self):
        import pybamm

        # DiscreteTimeSum output var exercises the time-integral memo path.
        # The 0D model auto-discretises, so the solver is driven directly.
        data_times = np.linspace(0, 1, 10)
        data = pybamm.DiscreteTimeData(data_times, np.zeros_like(data_times), "zeros")

        def _build_model(fmt):
            m = pybamm.BaseModel(name="dts_model")
            c = pybamm.Variable("c")
            m.rhs = {c: -2 * c}
            m.initial_conditions = {c: 1}
            m.variables["c"] = c
            # (c - 0)^2 summed over the data times -> sum(exp(-4 t))
            m.variables["dts"] = pybamm.DiscreteTimeSum((c - data) ** 2)
            m.convert_to_format = fmt
            return m

        model = _build_model("rust")
        solver = pybamm.IDAKLUSolver()
        solver._observes_via_compiled_model = True
        sol1 = solver.solve(model, t_eval=[0, 1], t_interp=data_times)
        val1 = sol1["dts"]()
        sol2 = solver.solve(model, t_eval=[0, 1], t_interp=data_times)
        val2 = sol2["dts"]()

        # Stable across solves (the memoised time-integral analysis is reused).
        np.testing.assert_allclose(val2, val1, rtol=1e-12, atol=0)
        # Correct against a CasADi reference.
        model_c = _build_model("casadi")
        sol_c = pybamm.IDAKLUSolver().solve(model_c, t_eval=[0, 1], t_interp=data_times)
        np.testing.assert_allclose(val1, sol_c["dts"](), rtol=1e-6, atol=1e-8)

        # The memo stored the (model, name, nstates) analysis, reused on solve 2.
        assert sol2.observation.compile_cache is sol1.observation.compile_cache
        ti_keys = [
            k
            for k in sol1.observation.compile_cache
            if isinstance(k, tuple) and k[:2] == ("__time_integral__", "dts")
        ]
        assert len(ti_keys) == 1

    def test_pickle_round_trip_after_observe(self):
        import pickle

        sim = self._sim()
        sol = sim.solve([0, 600])
        expected = sol["Terminal voltage [V]"].entries  # populate the cache

        restored = pickle.loads(pickle.dumps(sol))
        np.testing.assert_allclose(
            restored["Terminal voltage [V]"].entries, expected, rtol=1e-10, atol=1e-12
        )


class TestDiffsolObservationCache:
    def test_compiled_fn_reused_across_solves(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        model.events = []
        model.convert_to_format = "rust"
        sim = pybamm.Simulation(model, solver=pybamm.DiffsolSolver())

        sol1 = sim.solve(np.linspace(0, 600, 50))
        _ = sol1["Terminal voltage [V]"]
        cache = sim._solver._rust_observation_cache
        key = ("Terminal voltage [V]", id(sim._solver._rust_model))
        fn1 = cache[key]
        nodes_after_first = sim._solver._rust_model.graph.n_nodes

        sol2 = sim.solve(np.linspace(0, 600, 50))
        _ = sol2["Terminal voltage [V]"]
        assert sol2.observation.compile_cache is cache
        assert cache[key] is fn1  # reused, not recompiled
        assert sim._solver._rust_model.graph.n_nodes == nodes_after_first
