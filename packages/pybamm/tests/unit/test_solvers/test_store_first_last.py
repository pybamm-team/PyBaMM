import json
import warnings

import numpy as np
import pytest

import pybamm


def _build_simple_dae_model():
    """Tiny DAE used by base-solver-level tests."""
    model = pybamm.BaseModel()
    u = pybamm.Variable("u")
    v = pybamm.Variable("v")
    model.rhs = {u: 0.1 * v}
    model.algebraic = {v: 1 - v}
    model.initial_conditions = {u: 0, v: 1}
    model.variables = {"u": u, "v": v}
    disc = pybamm.Discretisation()
    disc.process_model(model)
    return model


class TestStoreFirstLast:
    def test_base_solver_validates_type(self):
        with pytest.raises(TypeError, match="store_first_last must be a bool"):
            pybamm.IDAKLUSolver(store_first_last="yes")

    def test_default_is_false(self):
        solver = pybamm.IDAKLUSolver()
        assert solver.store_first_last is False

    def test_plain_solve_returns_two_samples(self):
        model = _build_simple_dae_model()
        solver = pybamm.IDAKLUSolver(store_first_last=True)

        t_eval = np.linspace(0, 1, 50)
        sol = solver.solve(model, t_eval)

        assert sol.t.size == 2
        assert sol.t[0] == pytest.approx(t_eval[0])
        assert sol.t[-1] == pytest.approx(t_eval[-1])

    def test_experiment_each_step_has_two_samples(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                "Discharge at C/10 for 30 seconds",
                "Rest for 30 seconds",
                "Charge at C/10 for 30 seconds",
            ],
        )
        solver = pybamm.IDAKLUSolver(store_first_last=True)
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve()

        assert len(sol.sub_solutions) == 3
        for sub in sol.sub_solutions:
            assert sub.t.size == 2

        # Both runs stop the integrator at the same step endpoints, so the
        # end-of-step values should match to solver precision.
        ref_solver = pybamm.IDAKLUSolver()
        ref_sim = pybamm.Simulation(model, experiment=experiment, solver=ref_solver)
        ref_sol = ref_sim.solve()
        for sub, ref_sub in zip(sol.sub_solutions, ref_sol.sub_solutions, strict=True):
            np.testing.assert_allclose(
                sub["Voltage [V]"].data[-1],
                ref_sub["Voltage [V]"].data[-1],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_overrides_per_step_period(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [pybamm.step.string("Discharge at C/10 for 30 seconds", period=1.0)],
        )
        solver = pybamm.IDAKLUSolver(store_first_last=True)
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve()

        # Without store_first_last, period=1s on a 30s step would store ~31 samples.
        assert sol.sub_solutions[0].t.size == 2

    def test_overrides_caller_t_interp_in_step(self):
        """In a non-experiment step() call, a caller-provided t_interp is overridden."""
        model = _build_simple_dae_model()
        solver = pybamm.IDAKLUSolver(store_first_last=True)
        dt = 1.0
        t_eval = np.array([0.0, dt])
        # Caller asks for 10 interpolation points; flag should override.
        t_interp = np.linspace(0.0, dt, 10)

        sol = solver.step(None, model, dt, t_eval=t_eval, t_interp=t_interp)
        assert sol.t.size == 2

    def test_composes_with_output_variables(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(["Discharge at C/10 for 30 seconds"])
        solver = pybamm.IDAKLUSolver(
            store_first_last=True,
            output_variables=["Voltage [V]"],
        )
        sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
        sol = sim.solve()

        assert sol.sub_solutions[0].t.size == 2
        # ProcessedVariableComputed path produces one value per stored sample.
        assert sol["Voltage [V]"].data.size == 2

    def test_non_idaklu_warns_and_no_ops(self):
        model = _build_simple_dae_model()
        solver = pybamm.CasadiSolver(store_first_last=True)

        t_eval = np.linspace(0, 1, 11)
        with pytest.warns(pybamm.SolverWarning, match="store_first_last has no effect"):
            sol = solver.solve(model, t_eval)

        # CasadiSolver stops at every t_eval point; flag is a no-op.
        np.testing.assert_allclose(sol.t, t_eval)

    def test_non_idaklu_warning_fires_once_per_instance(self):
        model = _build_simple_dae_model()
        solver = pybamm.CasadiSolver(store_first_last=True)
        t_eval = np.linspace(0, 1, 5)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", pybamm.SolverWarning)
            solver.solve(model, t_eval)
            solver.solve(model, t_eval)
            solver.solve(model, t_eval)

        no_op_warnings = [
            w for w in captured if "store_first_last has no effect" in str(w.message)
        ]
        assert len(no_op_warnings) == 1

    def test_round_trip_serialisation_idaklu(self):
        solver = pybamm.IDAKLUSolver(store_first_last=True, rtol=1e-5)
        config = solver.to_config()
        assert config["store_first_last"] is True
        # Wire format must use a JSON bool token.
        assert '"store_first_last": true' in json.dumps(config)
        solver2 = pybamm.BaseSolver.from_config(config)
        assert isinstance(solver2, pybamm.IDAKLUSolver)
        assert solver2.store_first_last is True

    def test_round_trip_serialisation_casadi(self):
        solver = pybamm.CasadiSolver(store_first_last=True)
        config = solver.to_config()
        assert config["store_first_last"] is True
        solver2 = pybamm.BaseSolver.from_config(config)
        assert isinstance(solver2, pybamm.CasadiSolver)
        assert solver2.store_first_last is True
