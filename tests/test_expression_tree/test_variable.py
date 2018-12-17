#
# Tests for the Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.Variable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.Variable("a", domain=['test'])
        self.assertEqual(a.domain[0], 'test')
        self.assertRaises(TypeError, pybamm.Variable("a", domain='test'))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
