#
# Test base submodel
#

import pybamm
import unittest


class TestBaseSubModel(unittest.TestCase):
    def test_domain(self):
        # Accepted string
        submodel = pybamm.BaseSubModel(None, "Negative")
        self.assertEqual(submodel.domain, "Negative")

        # None (accepted but can't be called as an attribute)
        submodel = pybamm.BaseSubModel(None, None)
        self.assertFalse(hasattr(submodel, "_domain"))

        # bad string
        with self.assertRaises(pybamm.DomainError):
            pybamm.BaseSubModel(None, "bad string")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
