import pybamm
import unittest


class TestInterface(unittest.TestCase):
    def test_domain_failure(self):
        with self.assertRaises(pybamm.DomainError):
            pybamm.interface.BaseInterface(None, "not a domain")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
