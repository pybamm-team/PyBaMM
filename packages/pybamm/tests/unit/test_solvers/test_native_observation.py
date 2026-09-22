"""Unit tests for native (Rust) observation: accessor, producer, compile-once."""

import numpy as np

from pybamm.rust import ExprGraph
from pybamm.solvers.observation import ObserverCache
from pybamm.solvers.variable_observer import chain_rule_sensitivities

# Default tolerances split the backends' step sequences on D_n; 1e-11 converges both.
_TWO_PARAM_SOLVER_TOL = 1e-11


def _make_compiled_model():
    """Minimal 2-state model: rhs = [a * y1, -y0], identity mass, one input 'a'."""
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
        # The new root has no inputs, so p is the empty stacked array.
        _, model = _make_compiled_model()
        graph = model.graph
        new_root = graph.mul(graph.scalar(2.0), graph.state_vector(0, 1))
        fn = graph.compile(new_root, name="obs", n_states=model.n_states)
        out = fn(0.0, np.array([3.0, 5.0]), np.array([1.0]))
        np.testing.assert_allclose(out, [6.0])


def _native_idaklu_sim(model, **solver_kwargs):
    """Simulation on idaklu of ``model`` as a rust model, so it reads natively."""
    import pybamm

    model.convert_to_format = "rust"
    return pybamm.Simulation(model, solver=pybamm.IDAKLUSolver(**solver_kwargs))


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

        # Off-knot grid spanning several Hermite lane tiles plus a ragged tail.
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
        leaf = sol["Terminal voltage [V]"]._observer.leaves[0]
        cache = ObserverCache.of(sol.all_models[0])
        assert cache._native_leaves["Terminal voltage [V]"][0] is leaf
        sol.update("Terminal voltage [V]")
        assert sol["Terminal voltage [V]"]._observer.leaves[0] is leaf


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
        g = ExprGraph()
        cv = g.array(np.array([1.0, 2.0, 3.0]))
        cf = g.compile(g.add(cv, g.input_parameter("a")), name="zeroy", n_states=2)
        assert cf.output_len == 3
        n_t, n_states = 5, 2
        ts = np.linspace(0.0, 1.0, n_t)
        ys = np.zeros((n_states, n_t))
        dy_dp = np.ones((n_t * n_states, 1))
        S = chain_rule_sensitivities(cf, ts, ys, np.array([5.0]), dy_dp, ["a"])
        assert S.shape == (n_t * 3, 1)
        np.testing.assert_array_equal(S, np.ones((n_t * 3, 1)))


class TestNativeSensitivities:
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
        # Exercises the time-outer/output-inner flatten against CasADi's layout.
        name = "Negative particle concentration [mol.m-3]"
        var_n, var_c = self._solve_both(name, ["I"])
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
        assert a_n.shape == a_c.shape
        assert a_n.shape[1] == 2
        assert np.any(var_n.sensitivities["D_n"] != 0)
        np.testing.assert_allclose(
            a_n[:, 0], var_n.sensitivities["D_n"], rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(a_n, a_c, rtol=1e-5, atol=1e-8)

    def test_no_sensitivities_when_not_requested(self):
        import pybamm

        model = pybamm.lithium_ion.SPM()
        sol = _native_idaklu_sim(model).solve([0, 600])
        assert sol["Terminal voltage [V]"].sensitivities == {}

    def test_outputs_only_solve_reads_other_variables_sensitivities(self):
        import pybamm
        from pybamm.solvers.variable_observer import NativeObserver

        # "2a" and "a t" read no states, so an outputs-only solve still serves them
        def solve(convert_to_format):
            model = pybamm.BaseModel(name="decay")
            c = pybamm.Variable("c")
            a = pybamm.InputParameter("a")
            model.rhs = {c: -a * c}
            model.initial_conditions = {c: 1}
            model.variables = {"c": c, "2a": 2 * a, "a t": a * pybamm.t}
            model.convert_to_format = convert_to_format
            solver = pybamm.IDAKLUSolver(output_variables=["c"])
            # The lowerings step differently, so t_interp puts both on one grid
            return solver.solve(
                model,
                [0, 1],
                t_interp=np.linspace(0, 1, 11),
                inputs={"a": 1.5},
                calculate_sensitivities=True,
            )

        native, casadi = solve("rust"), solve("casadi")
        for name in ("2a", "a t"):
            assert isinstance(native[name]._observer, NativeObserver)
            np.testing.assert_allclose(
                native[name].sensitivities["a"],
                casadi[name].sensitivities["a"],
                rtol=1e-12,
                atol=1e-12,
            )


class TestObservationCacheAcrossSolves:
    @staticmethod
    def _sim():
        # events removed so termination is "final time"
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

        cache = ObserverCache.of(sol1.all_models[0])
        assert sol2.all_models[0] is sol1.all_models[0]
        assert cache.rust_model is sim._solver._setup["rust_model"]
        assert v2._observer.leaves[0] is v1._observer.leaves[0]

    def test_retained_graph_stops_growing(self):
        sim = self._sim()
        sol1 = sim.solve([0, 600])
        _ = sol1["Terminal voltage [V]"]
        cache = ObserverCache.of(sol1.all_models[0])._native_leaves
        nodes_after_first = sim._solver._setup["rust_model"].graph.n_nodes
        cache_len_after_first = len(cache)

        sol2 = sim.solve([0, 600])
        _ = sol2["Terminal voltage [V]"]
        assert sim._solver._setup["rust_model"].graph.n_nodes == nodes_after_first
        assert len(cache) == cache_len_after_first

    def test_discrete_time_sum_across_two_solves(self):
        import pybamm

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
        sol1 = solver.solve(model, t_eval=[0, 1], t_interp=data_times)
        val1 = sol1["dts"]()
        sol2 = solver.solve(model, t_eval=[0, 1], t_interp=data_times)
        val2 = sol2["dts"]()

        np.testing.assert_allclose(val2, val1, rtol=1e-12, atol=0)
        model_c = _build_model("casadi")
        sol_c = pybamm.IDAKLUSolver().solve(model_c, t_eval=[0, 1], t_interp=data_times)
        np.testing.assert_allclose(val1, sol_c["dts"](), rtol=1e-6, atol=1e-8)

        time_integral = ObserverCache.of(model)._time_integrals["dts"]
        assert sol2["dts"].time_integral is time_integral
        assert sol1["dts"].time_integral is time_integral

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
        cache = ObserverCache.of(sol1.all_models[0])
        assert cache.rust_model is sim._solver._rust_model
        fn1 = cache._native_leaves["Terminal voltage [V]"][0]
        nodes_after_first = sim._solver._rust_model.graph.n_nodes

        sol2 = sim.solve(np.linspace(0, 600, 50))
        _ = sol2["Terminal voltage [V]"]
        assert ObserverCache.of(sol2.all_models[0]) is cache
        assert cache._native_leaves["Terminal voltage [V]"][0] is fn1
        assert sim._solver._rust_model.graph.n_nodes == nodes_after_first


class TestBatchedNativeObservation:
    """`update([...])` shares one tape per segment across the whole set."""

    NAMES = [
        "Terminal voltage [V]",
        "Current [A]",
        "X-averaged negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
    ]

    def _experiment_solution(self):
        import pybamm

        model = pybamm.lithium_ion.SPMe()
        model.convert_to_format = "rust"
        solver = pybamm.IDAKLUSolver()
        experiment = pybamm.Experiment(
            [("Discharge at 1C until 3.4 V", "Charge at 0.5C until 3.9 V")]
        )
        sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
        return sim.solve()

    def test_batched_matches_per_name_bitwise(self):
        batched = self._experiment_solution()
        per_name = self._experiment_solution()
        assert len(batched.all_ys) > 1

        batched.update(self.NAMES)
        for name in self.NAMES:
            np.testing.assert_array_equal(batched[name].entries, per_name[name].entries)

    def test_batched_hermite_matches_per_name_bitwise(self):
        batched = self._experiment_solution()
        per_name = self._experiment_solution()
        batched.update(self.NAMES)

        # off the solver's own knots, so every read Hermite-reconstructs
        t = np.linspace(batched.t[0] + 1.0, batched.t[-1] - 1.0, 61)
        for name in self.NAMES:
            np.testing.assert_array_equal(batched[name](t), per_name[name](t))

    def test_one_tape_per_segment_serves_every_variable(self):
        from pybamm.solvers.variable_observer import SharedOutputLeaf

        solution = self._experiment_solution()
        solution.update(self.NAMES)
        observers = [solution[name]._observer for name in self.NAMES]

        leaves = [observer.leaves[0] for observer in observers]
        assert all(isinstance(leaf, SharedOutputLeaf) for leaf in leaves)
        shared = {id(leaf._shared) for leaf in leaves}
        assert len(shared) == 1
        assert leaves[0]._shared.function.output_names == tuple(self.NAMES)

    def test_the_set_is_compiled_once_and_reused(self):
        solution = self._experiment_solution()
        cache = ObserverCache.of(solution.all_models[0])
        graph = cache.rust_model.graph

        solution.update(self.NAMES)
        sets = dict(cache._native_sets)
        nodes = graph.n_nodes
        assert tuple(self.NAMES) in sets

        solution.update(self.NAMES)
        assert cache._native_sets == sets
        assert graph.n_nodes == nodes

    def test_a_time_integral_keeps_the_per_name_path(self):
        import pybamm
        from pybamm.solvers.variable_observer import SharedOutputLeaf

        # A time integral is not a row of the set's tape, so it falls back alone.
        model = pybamm.BaseModel(name="integral_model")
        c = pybamm.Variable("c")
        model.rhs = {c: -2 * c}
        model.initial_conditions = {c: 1}
        model.variables = {
            "c": c,
            "2c": 2 * c,
            "integral": pybamm.ExplicitTimeIntegral(c, 0),
        }
        model.convert_to_format = "rust"
        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(model, t_eval=[0, 1])

        per_name = solver.solve(model, t_eval=[0, 1])
        expected = {name: per_name[name].entries for name in ("c", "2c", "integral")}

        solution.update(["c", "2c", "integral"])
        assert not isinstance(
            solution["integral"]._observer.leaves[0], SharedOutputLeaf
        )
        assert isinstance(solution["c"]._observer.leaves[0], SharedOutputLeaf)
        for name, want in expected.items():
            np.testing.assert_array_equal(solution[name].entries, want)

    def test_a_single_name_is_not_batched(self):
        from pybamm.solvers.variable_observer import SharedOutputLeaf

        solution = self._experiment_solution()
        solution.update([self.NAMES[0]])
        leaf = solution[self.NAMES[0]]._observer.leaves[0]
        assert not isinstance(leaf, SharedOutputLeaf)

    def test_a_data_export_prepares_the_requested_set(self):
        from pybamm.solvers.variable_observer import SharedOutputLeaf

        solution = self._experiment_solution()
        data = solution.get_data_dict(self.NAMES, cycles_and_steps=False)

        assert set(data) == set(self.NAMES)
        leaves = [solution[name]._observer.leaves[0] for name in self.NAMES]
        assert all(isinstance(leaf, SharedOutputLeaf) for leaf in leaves)

    def test_an_already_built_variable_is_left_alone(self):
        solution = self._experiment_solution()
        built = solution[self.NAMES[0]]
        solution._build_missing(self.NAMES)
        assert solution[self.NAMES[0]] is built

    def test_batched_sensitivities_match_per_name_bitwise(self):
        import pybamm

        def solve():
            model = pybamm.lithium_ion.SPM()
            model.convert_to_format = "rust"
            solver = pybamm.IDAKLUSolver()
            parameters = pybamm.ParameterValues("Chen2020")
            parameters.update({"Negative particle diffusivity [m2.s-1]": "[input]"})
            sim = pybamm.Simulation(model, parameter_values=parameters, solver=solver)
            return sim.solve(
                [0, 3600],
                inputs={"Negative particle diffusivity [m2.s-1]": 3.3e-14},
                calculate_sensitivities=True,
            )

        names = [
            "Terminal voltage [V]",
            "X-averaged negative particle surface concentration [mol.m-3]",
        ]
        batched, per_name = solve(), solve()
        batched.update(names)
        for name in names:
            np.testing.assert_array_equal(
                batched[name].sensitivities["all"],
                per_name[name].sensitivities["all"],
            )

    def test_batched_sensitivities_match_per_name_over_several_parameters(self):
        import pybamm

        def solve():
            model = pybamm.lithium_ion.SPM()
            model.convert_to_format = "rust"
            solver = pybamm.IDAKLUSolver()
            parameters = pybamm.ParameterValues("Chen2020")
            parameters.update(
                {
                    "Negative particle diffusivity [m2.s-1]": "[input]",
                    "Current function [A]": "[input]",
                }
            )
            sim = pybamm.Simulation(model, parameter_values=parameters, solver=solver)
            return sim.solve(
                [0, 3600],
                inputs={
                    "Negative particle diffusivity [m2.s-1]": 3.3e-14,
                    "Current function [A]": 0.68,
                },
                calculate_sensitivities=True,
            )

        names = [
            "Terminal voltage [V]",
            "Negative particle concentration [mol.m-3]",
        ]
        batched, per_name = solve(), solve()
        batched.update(names)

        spatial = per_name[names[1]].sensitivities["all"]
        assert spatial.shape[1] == 2
        assert spatial.shape[0] > len(per_name.t)
        for name in names:
            np.testing.assert_array_equal(
                batched[name].sensitivities["all"],
                per_name[name].sensitivities["all"],
            )

    def test_a_mutated_query_array_is_not_read_from_the_held_tape(self):
        import pybamm
        from pybamm.solvers.variable_observer import SharedOutputLeaf

        # Mutating the last call's array must not make the next call match it.
        model = pybamm.BaseModel(name="decay")
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c, "2c": 2 * c}
        model.convert_to_format = "rust"
        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(model, t_eval=[0, 1])
        solution.update(["c", "2c"])
        assert isinstance(solution["c"]._observer.leaves[0], SharedOutputLeaf)
        assert solution["c"].hermite_interpolation

        t = np.array([0.2, 0.3])
        solution["c"](t)
        t[:] = [0.7, 0.8]
        np.testing.assert_allclose(solution["2c"](t), 2 * np.exp(-t), rtol=1e-3)

    def test_batched_sensitivities_over_an_empty_segment(self):
        import pybamm

        def joined():
            model = pybamm.BaseModel(name="decay")
            c = pybamm.Variable("c")
            a = pybamm.InputParameter("a")
            model.rhs = {c: -a * c}
            model.initial_conditions = {c: 1}
            model.variables = {"c": c, "2c": 2 * c}
            model.convert_to_format = "rust"
            solver = pybamm.IDAKLUSolver()
            kwargs = {"inputs": {"a": 1.0}, "calculate_sensitivities": True}
            first = solver.solve(model, [0, 1], **kwargs)
            second = solver.solve(model, [1, 2], **kwargs)
            # The last state's single point is dropped as a repeat, leaving it empty
            return first + (first.last_state + second)

        batched, per_name = joined(), joined()
        assert any(t.size == 0 for t in batched.all_ts)
        batched.update(["c", "2c"])
        for name in ("c", "2c"):
            np.testing.assert_array_equal(
                batched[name].sensitivities["all"],
                per_name[name].sensitivities["all"],
            )

    def test_concurrent_reads_each_get_their_own_rows(self):
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import pybamm

        model = pybamm.BaseModel(name="decay")
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"c": c, "2c": 2 * c, "3c": 3 * c}
        model.convert_to_format = "rust"
        solution = pybamm.IDAKLUSolver().solve(model, t_eval=[0, 1])
        names = ["c", "2c", "3c"]
        solution.update(names)

        # Every thread reads the whole set at its own times, so held calls differ
        n_threads = 8
        queries = [np.linspace(0.01 + 1e-3 * k, 0.99, 40) for k in range(n_threads)]
        expected = [{name: solution[name](t) for name in names} for t in queries]
        barrier = threading.Barrier(n_threads)

        def read(k):
            barrier.wait()
            for _ in range(50):
                for name in names:
                    np.testing.assert_array_equal(
                        solution[name](queries[k]), expected[k][name]
                    )

        # Frequent thread switches expose any unguarded window in the shared hold
        switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            with ThreadPoolExecutor(n_threads) as pool:
                list(pool.map(read, range(n_threads)))
        finally:
            sys.setswitchinterval(switch_interval)
