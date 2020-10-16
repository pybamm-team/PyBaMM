#
# Test for the evaluate-to-Julia functions
#
import pybamm

import unittest
import numpy as np

from julia import Main
from diffeqpy import de


class TestCreateSolveMTKModel(unittest.TestCase):
    def test_exponential_decay_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -2 * v}
        model.initial_conditions = {v: 0.5}

        mtk_str = pybamm.get_julia_mtk_model(model)

        Main.eval("using ModelingToolkit")
        Main.eval(mtk_str)

        Main.tspan = (0.0, 10.0)
        # this definition of prob doesn't work, so we use Main.eval instead
        # prob = de.ODEProblem(Main.sys, Main.u0, Main.tspan)

        Main.eval("prob = ODEProblem(sys, u0, tspan)")
        sol = de.solve(Main.prob, de.Tsit5())

        y_sol = np.concatenate(sol.u)
        y_exact = 0.5 * np.exp(-2 * sol.t)
        np.testing.assert_almost_equal(y_sol, y_exact, decimal=6)

    def test_lotka_volterra_model(self):
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")
        x = pybamm.Variable("x")
        y = pybamm.Variable("y")

        model.rhs = {x: a * x - b * x * y, y: c * x * y - d * y}
        model.initial_conditions = {x: 1.0, y: 1.0}

        mtk_str = pybamm.get_julia_mtk_model(model)

        # Solve using julia
        Main.eval("using ModelingToolkit")
        Main.eval(mtk_str)

        Main.tspan = (0.0, 10.0)
        Main.eval(
            """
            begin
            p = [a => 1.5, b => 1.0, c => 3.0, d => 1.0]
            prob = ODEProblem(sys, u0, tspan, p)
            end
            """
        )
        sol_julia = de.solve(Main.prob, de.Tsit5(), reltol=1e-8, abstol=1e-8)

        y_sol_julia = np.vstack(sol_julia.u).T

        # Solve using pybamm
        sol_pybamm = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8).solve(
            model, sol_julia.t, inputs={"a": 1.5, "b": 1.0, "c": 3.0, "d": 1.0}
        )

        # Compare
        np.testing.assert_almost_equal(y_sol_julia, sol_pybamm.y, decimal=5)

    def test_dae_model(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        y = pybamm.Variable("y")

        model.rhs = {x: -2 * x}
        model.algebraic = {y: x - y}
        model.initial_conditions = {x: 1.0, y: 1.0}

        mtk_str = pybamm.get_julia_mtk_model(model)

        # Solve using julia
        Main.eval("using ModelingToolkit")
        Main.eval(mtk_str)

        Main.tspan = (0.0, 10.0)
        Main.eval("prob = ODEProblem(sys, u0, tspan)")
        sol_julia = de.solve(Main.prob, de.Rodas5(), reltol=1e-8, abstol=1e-8)

        y_sol_julia = np.vstack(sol_julia.u).T

        # Solve using pybamm
        sol_pybamm = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8).solve(model, sol_julia.t)

        # Compare
        np.testing.assert_almost_equal(y_sol_julia, sol_pybamm.y, decimal=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
