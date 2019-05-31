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
        var = pybamm.Variable("var")
        model.rhs = {var: var}
        model.algebraic = {var: var - 1}

        # test errors
        solver = pybamm.OdeSolver()
        with self.assertRaisesRegex(
            pybamm.SolverError, "Cannot use ODE solver to solve model with DAEs"
        ):
            solver.solve(model, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
