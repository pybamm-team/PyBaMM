#
# Test base kinetics submodel
#

import pybamm
import tests
import unittest


class TestBaseModel(unittest.TestCase):
    def test_not_implemented(self):
        submodel = pybamm.interface.kinetics.BaseModel(None, None)
        with self.assertRaises(NotImplementedError):
            submodel._get_exchange_current_density(None)
        with self.assertRaises(NotImplementedError):
            submodel._get_kinetics(None, None, None)
        with self.assertRaises(NotImplementedError):
            submodel._get_open_circuit_potential(None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
