#
# Tests for the PyBaMM parameters management
# command line interface
#

import pybamm
import unittest
from tests import TestCase


class TestParametersCLI(TestCase):
    def test_error(self):
        with self.assertRaisesRegex(NotImplementedError, "deprecated"):
            pybamm.add_parameter()
        with self.assertRaisesRegex(NotImplementedError, "deprecated"):
            pybamm.edit_parameter()
        with self.assertRaisesRegex(NotImplementedError, "deprecated"):
            pybamm.remove_parameter()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
