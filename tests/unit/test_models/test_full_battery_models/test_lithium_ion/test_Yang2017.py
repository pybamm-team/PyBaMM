#
# Tests for the lithium-ion DFN model
#
import pybamm
import unittest


class TestYang2017(unittest.TestCase):
    def test_deprecated(self):
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.Yang2017()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
