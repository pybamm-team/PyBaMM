#
# Test uniform current collector submodel
#

import pybamm
import tests
import unittest


class TestUniformModel(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.standard_parameters_lead_acid

        submodel = pybamm.current_collector.Uniform(param)
        std_tests = tests.StandardSubModelTests(submodel)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
