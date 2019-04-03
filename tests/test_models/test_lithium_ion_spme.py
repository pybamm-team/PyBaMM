#
# Tests for the lithium-ion SPMe model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests
import numpy as np
import unittest


class TestSPMe(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lithium_ion.SPMe()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
