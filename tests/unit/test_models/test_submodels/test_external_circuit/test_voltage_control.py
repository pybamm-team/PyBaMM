#
# Test voltage control submodel
#

import pybamm
import tests
import unittest


class TestVoltageControl(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        submodel = pybamm.external_circuit.VoltageFunctionControl(param)
        variables = {
            "Positive current collector potential": pybamm.Scalar(0),
            "Positive current collector potential [V]": pybamm.Scalar(0),
        }
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()

    def test_public_functions_direct(self):
        param = pybamm.standard_parameters_lithium_ion
        submodel = pybamm.external_circuit.VoltageControl(param)
        a = pybamm.FullBroadcast(
            0,
            "positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        variables = {
            "Positive electrode potential": a,
            "Positive electrode tortuosity": a,
        }
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
