#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest


class TestSPMe(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lithium_ion.SPMe()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
