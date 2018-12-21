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
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        c = pybamm.Parameter("c", domain=["test"])
        self.assertEqual(c.domain[0], "test")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
