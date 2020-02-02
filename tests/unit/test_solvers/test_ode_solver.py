#
# Tests for the ODE Solver class
#
import pybamm
import unittest


class TestOdeSolver(unittest.TestCase):
    def test_exceptions(self):
        solver = pybamm.OdeSolver()
        with self.assertRaises(NotImplementedError):
            solver.integrate(None, None, None)

    def test_wrong_solver(self):
        # Create model
        model = pybamm.BaseModel()
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: var1}
        model.algebraic = {var2: var2 - 1}
        model.initial_conditions = {var1: 0, var2: 1}
        disc = pybamm.Discretisation()
        disc.process_model(model)

        # test errors
        solver = pybamm.OdeSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError, "Cannot use ODE solver to solve model with DAEs"
        ):
            solver.solve(model, None)
        with self.assertRaisesRegex(
            pybamm.SolverError, "Cannot use ODE solver to solve model with DAEs"
        ):
            solver.set_up_casadi(model)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
