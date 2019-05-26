#
# Tests for the electrode submodels
#
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestOhm(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrode.Ohm(param)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_exceptions(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrode.Ohm(param)

        with self.assertRaises(pybamm.DomainError):
            phi_s = pybamm.Variable("phi_s", "not a domain")
            model.set_algebraic_system(phi_s, {})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
