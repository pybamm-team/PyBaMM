#
# Tests for the Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestParameter(unittest.TestCase):
    def test_parameter_init(self):
        a = pybamm.Parameter("a")
        self.assertEqual(a._name, "a")
        b = pybamm.Parameter("b", family="addams")
        self.assertEqual(b.family, "addams")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
