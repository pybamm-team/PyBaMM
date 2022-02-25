import pybamm
import unittest
import tests


class TestYang2017(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.Yang2017()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
