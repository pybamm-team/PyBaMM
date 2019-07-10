#
# Test full concentration submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid
        a = pybamm.Scalar(0)
        variables = {
            "Porosity": pybamm.Concatenation(
                pybamm.Broadcast(a, "negative electrode"),
                pybamm.Broadcast(a, "separator"),
                pybamm.Broadcast(a, "positive electrode"),
            ),
            "Porosity change": pybamm.Concatenation(
                pybamm.Broadcast(a, "negative electrode"),
                pybamm.Broadcast(a, "separator"),
                pybamm.Broadcast(a, "positive electrode"),
            ),
            "Volume-averaged velocity": a,
            "Negative electrode interfacial current density": pybamm.Broadcast(
                a, "negative electrode"
            ),
            "Positive electrode interfacial current density": pybamm.Broadcast(
                a, "positive electrode"
            ),
        }
        reactions = {
            "main": {
                "Negative": {
                    "s_ox": param.s_ox_Ox,
                    "aj": "Negative electrode interfacial current density",
                },
                "Positive": {
                    "s_ox": param.s_ox_Ox,
                    "aj": "Positive electrode interfacial current density",
                },
            }
        }
        submodel = pybamm.oxygen_diffusion.Full(param, reactions)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.setting.debug_mode = True
    unittest.main()
