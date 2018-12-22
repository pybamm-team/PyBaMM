#
# Tests for the Binary Operator classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import unittest


class TestStefanMaxwellDiffusion(unittest.TestCase):
    def test_make_tree(self):
        G = pybamm.Scalar(1)
        pybamm.electrolyte.StefanMaxwellDiffusion(G)
