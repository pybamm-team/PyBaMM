import pytest

import pybamm


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


class TestRustRootResolution:
    def test_default_root_method_keeps_the_rust_format(self):
        from tests.unit.test_solvers.test_process_rust import _toy_dae

        model = _toy_dae("rust")
        solver = pybamm.IDAKLUSolver(options={"calc_ic": False})
        solver._check_and_prepare_model_inplace(model)
        assert model.convert_to_format == "rust"
        assert isinstance(solver.root_method, pybamm.NonlinearSolver)
