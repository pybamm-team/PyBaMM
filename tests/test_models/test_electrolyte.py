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

    def test_processing_parameters(self):
        G = pybamm.Scalar(1)
        pybamm.electrolyte.StefanMaxwellDiffusion(G)

        param = pybamm.ParameterValues("input/parameters/lead-acid/default.csv")
        self.assertEqual(param["R"], 8.314)

        print(param["R"])
        # just going to skip for now.
        # param.process_model(model)
