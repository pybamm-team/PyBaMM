#
# Test base lithium plating submodel
#

import pybamm
import unittest


class TestBasePlating(unittest.TestCase):
    def test_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "Lithium plating models are not implemented for the positive electrode",
        ):
            pybamm.lithium_plating.ReversiblePlating(None, "Positive")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
