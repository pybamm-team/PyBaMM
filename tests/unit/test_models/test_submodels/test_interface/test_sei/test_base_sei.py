#
# Test base SEI submodel
#
import pybamm
import unittest


class TestBaseSEI(unittest.TestCase):
    def test_not_implemented(self):
        with self.assertRaisesRegex(
            NotImplementedError,
            "SEI models are not implemented for the positive electrode",
        ):
            pybamm.sei.ReactionLimited(None, "Positive", True)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
