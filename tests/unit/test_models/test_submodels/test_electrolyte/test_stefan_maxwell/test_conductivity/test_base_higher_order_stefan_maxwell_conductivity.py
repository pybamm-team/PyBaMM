#
# Test combined stefan maxwell electrolyte conductivity submodel
#

import pybamm
import unittest


class TestComposite(unittest.TestCase):
    def test_not_implemented(self):
        submodel = pybamm.electrolyte.stefan_maxwell.conductivity.BaseHigherOrder(None)
        with self.assertRaises(NotImplementedError):
            submodel._higher_order_macinnes_function(None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
