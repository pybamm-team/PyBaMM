#
# Tests the profiler.
#
import pybamm
import unittest


class TestProfile(unittest.TestCase):
    """
    Tests the profiler.
    """

    def test_profile(self):
        pybamm.profile("[2+2 for _ in range(1000000)]")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
