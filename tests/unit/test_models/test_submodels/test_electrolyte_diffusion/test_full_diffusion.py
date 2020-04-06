#
# Test full concentration submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lithium_ion
        a = pybamm.Scalar(0)
        full = pybamm.FullBroadcast(
            a,
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        variables = {
            "Porosity": a,
            "Electrolyte tortuosity": a,
            "Porosity change": a,
            "Volume-averaged velocity": a,
            "Electrolyte concentration": a,
            "Electrolyte current density": full,
            "Interfacial current density": full,
            "Oxygen interfacial current density": full,
            "Cell temperature": pybamm.FullBroadcast(
                a,
                ["negative electrode", "separator", "positive electrode"],
                "current collector",
            ),
            "Transverse volume-averaged acceleration": pybamm.FullBroadcast(
                a,
                ["negative electrode", "separator", "positive electrode"],
                "current collector",
            ),
        }
        submodel = pybamm.electrolyte_diffusion.Full(param)
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
