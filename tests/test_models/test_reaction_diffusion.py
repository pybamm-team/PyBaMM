#
# Tests for the Reaction diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import tests

import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_basic_processing(self):
        current_scale = 1
        current_function = pybamm.standard_current_functions.constant_current
        model = pybamm.ReactionDiffusionModel(current_scale, current_function)

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
