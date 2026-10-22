import numpy as np
import pytest

import pybamm


def _solve_spm(fmt, solver, t_eval, **solve_kwargs):
    model = pybamm.lithium_ion.SPM()
    model.convert_to_format = fmt
    sim = pybamm.Simulation(model, solver=solver)
    return sim.solve(t_eval, **solve_kwargs)


class TestIdakluSpmRustParity:
    def test_spm_idaklu_rust_vs_casadi(self):
        # t_interp forces both paths to output exactly on t_eval.
        t_eval = np.linspace(0, 3600, 50)
        sols = {
            fmt: _solve_spm(
                fmt,
                pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
                t_eval,
                t_interp=t_eval,
            )
            for fmt in ("casadi", "rust")
        }
        np.testing.assert_allclose(
            sols["rust"]["Voltage [V]"].entries,
            sols["casadi"]["Voltage [V]"].entries,
            rtol=1e-6,
        )


class TestAlgebraicRustParity:
    def test_pure_algebraic_rust_vs_casadi(self):
        results = {}
        for fmt in ("casadi", "rust"):
            model = pybamm.BaseModel()
            v = pybamm.Variable("v")
            model.algebraic = {v: v**2 - 3 * v + 2}
            model.initial_conditions = {v: 3.0}
            disc = pybamm.Discretisation()
            disc.process_model(model)
            model.convert_to_format = fmt
            sol = pybamm.AlgebraicSolver().solve(model, [0])
            results[fmt] = sol.y[0][0]
        assert results["rust"] == pytest.approx(results["casadi"], rel=1e-8)
        assert results["rust"] == pytest.approx(2.0, rel=1e-6)


class TestIdakluRustParity:
    def test_dfn_chen2020_rust_vs_casadi(self):
        # t_interp forces IDAKLU to output exactly on t_eval for both paths.
        t_eval = np.linspace(0, 3500, 50)
        sols = {}
        for fmt in ("casadi", "rust"):
            model = pybamm.lithium_ion.DFN()
            model.events = []
            param = pybamm.ParameterValues("Chen2020")
            model.convert_to_format = fmt
            sim = pybamm.Simulation(
                model,
                parameter_values=param,
                solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
            )
            sols[fmt] = sim.solve(t_eval, t_interp=t_eval)
        np.testing.assert_allclose(
            sols["rust"]["Voltage [V]"].entries,
            sols["casadi"]["Voltage [V]"].entries,
            rtol=1e-5,
        )


class TestRustNewton:
    @staticmethod
    def _pure_algebraic_rust():
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        a = pybamm.InputParameter("a")
        model.algebraic = {v: v**2 - a * v + 2}
        model.initial_conditions = {v: 3.0}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        model.convert_to_format = "rust"
        return model

    def test_pure_algebraic_nonlinear_solver_parity(self):
        results = {}
        for fmt in ("casadi", "rust"):
            model = pybamm.BaseModel()
            v = pybamm.Variable("v")
            a = pybamm.InputParameter("a")
            model.algebraic = {v: v**2 - a * v + 2}
            model.initial_conditions = {v: 3.0}
            disc = pybamm.Discretisation()
            disc.process_model(model)
            model.convert_to_format = fmt
            sol = pybamm.NonlinearSolver().solve(model, [0], inputs={"a": 3.0})
            results[fmt] = sol.y[0][0]
        assert results["rust"] == pytest.approx(results["casadi"], rel=1e-10)

    def test_rust_newton_setup_pickles_to_inert(self):
        # The C++ Newton setup holds a raw pointer into the CompiledModel.
        import pickle

        from pybamm.solvers.nonlinear_solver import _NonlinearSolverSetup

        solver = pybamm.NonlinearSolver()
        model = self._pure_algebraic_rust()
        setup = solver._set_up_root_solver_rust(model, {"a": 3.0})
        assert bool(setup) is True and setup._keepalive is not None
        restored = pickle.loads(pickle.dumps(setup))
        assert isinstance(restored, _NonlinearSolverSetup)
        assert restored._setup is None and restored._keepalive is None
        assert bool(restored) is False

    def test_rust_newton_solver_resolves_after_pickle(self):
        import pickle

        fresh = pybamm.NonlinearSolver().solve(
            self._pure_algebraic_rust(), [0], inputs={"a": 3.0}
        )
        solver = pybamm.NonlinearSolver()
        solver.solve(self._pure_algebraic_rust(), [0], inputs={"a": 3.0})
        revived = pickle.loads(pickle.dumps(solver))
        sol = revived.solve(self._pure_algebraic_rust(), [0], inputs={"a": 3.0})
        assert sol.y[0][0] == pytest.approx(fresh.y[0][0], rel=1e-10)

    def test_dae_rust_newton_parity_vs_casadi(self):
        # Exercises the global->local algebraic Jacobian column remap.
        t_eval = np.linspace(0, 1, 20)
        sols = {}
        for fmt in ("casadi", "rust"):
            model = pybamm.BaseModel()
            u = pybamm.Variable("u")
            w = pybamm.Variable("w")
            a = pybamm.InputParameter("a")
            model.rhs = {u: -a * u}
            model.algebraic = {w: w**2 - a * w + 2 * u}
            model.initial_conditions = {u: 1.0, w: 3.0}
            disc = pybamm.Discretisation()
            disc.process_model(model)
            model.convert_to_format = fmt
            solver = pybamm.IDAKLUSolver(
                rtol=1e-8, atol=1e-10, root_method="nonlinear_solver"
            )
            sols[fmt] = solver.solve(model, t_eval, inputs={"a": 3.0}, t_interp=t_eval)
        np.testing.assert_allclose(sols["rust"].y, sols["casadi"].y, rtol=1e-6)


class TestRustRootResolution:
    def test_default_root_method_keeps_the_rust_format(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        solver = pybamm.IDAKLUSolver(options={"calc_ic": False})
        solver._check_and_prepare_model_inplace(model)
        assert model.convert_to_format == "rust"
        assert isinstance(solver.root_method, pybamm.NonlinearSolver)

    def test_diffsol_normalises_model_to_rust(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("casadi")
        solver = pybamm.DiffsolSolver()
        solver.solve(model, np.linspace(0, 1, 10), inputs={"a": 0.5})
        assert model.convert_to_format == "rust"

    def test_dae_consistent_ic_parity_rust_newton_vs_casadi(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        y0 = {}
        for fmt in ("casadi", "rust"):
            model = _toy_dae(fmt)
            model.initial_conditions = {
                k: v for k, v in model.initial_conditions.items()
            }
            solver = pybamm.IDAKLUSolver(
                root_method="nonlinear_solver", options={"calc_ic": False}
            )
            solver.set_up(model, inputs=[{"a": 0.5}])
            solver._set_consistent_initialization(model, 0.0, [{"a": 0.5}])
            y0[fmt] = np.asarray(model.y0_list[0]).ravel()
        np.testing.assert_allclose(y0["rust"], y0["casadi"], rtol=1e-8)

    def test_casadi_root_method_solves_a_rust_model_without_being_replaced(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        a = pybamm.InputParameter("a")
        model.rhs = {u: -a * u}
        model.algebraic = {v: 2 * u - v}
        model.initial_conditions = {u: 1.0, v: 0.0}  # v is inconsistent
        pybamm.Discretisation().process_model(model)
        model.convert_to_format = "rust"

        solver = pybamm.IDAKLUSolver(root_method="casadi", options={"calc_ic": False})
        solver.solve(model, [0, 1], inputs={"a": 0.5})

        np.testing.assert_allclose(
            np.asarray(model.y0_list[0]).ravel(), [1.0, 2.0], rtol=1e-6
        )
        assert model.convert_to_format == "rust"
        assert isinstance(solver.root_method, pybamm.CasadiAlgebraicSolver)
