#
# Tests for the electrolyte current submodel
#
import pybamm
from pybamm.solvers.scikits_ode_solver import scikits_odes_spec
import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesStefanMaxwell(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestMacInnesCapacitance(unittest.TestCase):
    def test_default_solver(self):
        param = pybamm.standard_parameters_lithium_ion
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, True)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        model = pybamm.electrolyte_current.MacInnesCapacitance(param, False)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
