#
# Tests for the electrode submodels
#
import pybamm
import unittest


@unittest.skipIf(scikits_odes_spec is None, "scikits.odes not installed")
class TestOhm(unittest.TestCase):
    def setUp(self):
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables and reactions
        phi_s_n = pybamm.standard_variables.phi_s_n
        phi_s_p = pybamm.standard_variables.phi_s_p
        onen = pybamm.Broadcast(1, ["negative electrode"])
        onep = pybamm.Broadcast(1, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": onen}, "pos": {"s_plus": 1, "aj": onep}}
        }

        # Set up model and test
        # Negative only
        self.model = pybamm.electrode.Ohm(param)

    def test_default_solver(self):
        self.assertTrue(isinstance(model.default_solver, pybamm.ScikitsDaeSolver))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
