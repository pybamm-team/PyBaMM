#
# Tests for the Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestIndependentVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.IndependentVariable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.IndependentVariable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        a = pybamm.IndependentVariable("a", domain="test")
        self.assertEqual(a.domain[0], "test")
        with self.assertRaises(TypeError):
            pybamm.IndependentVariable("a", domain=1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
