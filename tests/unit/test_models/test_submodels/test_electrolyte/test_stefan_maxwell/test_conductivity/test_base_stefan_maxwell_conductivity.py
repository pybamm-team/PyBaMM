#
# Test base stefan maxwell electrolyte conductivity submodel
#

import pybamm
import tests
import unittest


pybamm.settings.debug_mode = True


class TestBaseModel(unittest.TestCase):
    def test_public_functions(self):
        submodel = pybamm.electrolyte.stefan_maxwell.conductivity.BaseModel(None)
        std_tests = tests.StandardSubModelTests(submodel)

        with self.assertRaises(KeyError):
            std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
