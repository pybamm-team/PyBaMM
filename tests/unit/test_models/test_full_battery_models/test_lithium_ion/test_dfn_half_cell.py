#
# Tests for the lithium-ion half-cell DFN model
# This is achieved by using the {"working electrode": "positive"} option
#
import pybamm
import unittest


class TestDFNHalfCell(unittest.TestCase):
    def test_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.DFN(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
