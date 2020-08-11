#
# Test full concentration submodel
#

import pybamm
import tests
import unittest


class TestFull(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
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
            "Sum of electrolyte reaction source terms": full,
            "Cell temperature": full,
            "Transverse volume-averaged acceleration": full,
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
