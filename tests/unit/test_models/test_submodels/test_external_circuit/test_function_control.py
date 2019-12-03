#
# Test function control submodel
#
import pybamm
import tests
import unittest


class ExternalCircuitFunction:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return (
            V + I - pybamm.FunctionParameter("Current plus voltage function", pybamm.t)
        )


class TestFunctionControl(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        submodel = pybamm.external_circuit.FunctionControl(
            param, ExternalCircuitFunction()
        )
        variables = {
            "Positive current collector potential": pybamm.Scalar(0),
            "Positive current collector potential [V]": pybamm.Scalar(0),
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
